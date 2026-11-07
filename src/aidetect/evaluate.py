"""Evaluate a trained model on any labelled data (ideally images from generators it has never seen)."""
from __future__ import annotations

import csv
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
from joblib import Parallel, delayed
from PIL import Image

from .config import Config
from .data import DataError, Sample, collect_samples
from .detector import Detector, Prediction
from .extraction import Extraction, extract_from_loaded, load_for_key
from .features import get_extractor
from .imaging import ImageError
from .metrics import compute_metrics, safe_auc
from .model import _jsonable
from .perturb import apply_perturbation, describe
from .reporting import save_evaluation_plots, write_html_report

log = logging.getLogger(__name__)
Progress = Callable[[str, float, str], None]


def _robust_worker(path: str, spec: Optional[Dict[str, Any]], key: str, max_patches: int, seed: int) -> Extraction:
    try:
        loaded = load_for_key(path, key)
        if spec is not None:
            img = apply_perturbation(Image.fromarray(loaded.rgb), spec, np.random.default_rng(seed))
            loaded = load_for_key(img, key)
        return extract_from_loaded(loaded, get_extractor(key), max_patches)
    except ImageError as e:
        return Extraction(ok=False, error=str(e))
    except Exception as e:
        return Extraction(ok=False, error=f"{type(e).__name__}: {e}")


def run_robustness(det: Detector, paths: Sequence[str], labels: Sequence[int], specs: Sequence[Dict[str, Any]],
                   max_images: int, n_jobs: int, seed: int, prog: Progress) -> List[Dict[str, Any]]:
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    chosen: List[int] = []
    for c in (0, 1):
        idx = np.where(labels == c)[0]
        take = min(len(idx), max(1, max_images // 2))
        chosen += sorted(rng.choice(idx, size=take, replace=False).tolist())
    conditions: List[Optional[Dict[str, Any]]] = [None] + list(specs)
    tasks = [(ci, ii) for ci in range(len(conditions)) for ii in chosen]
    out_ex: List[Extraction] = []
    chunk = 64
    with Parallel(n_jobs=n_jobs) as par:
        for start in range(0, len(tasks), chunk):
            batch = tasks[start: start + chunk]
            out_ex += par(delayed(_robust_worker)(paths[ii], conditions[ci], det.feature_key,
                                                  det.max_patches, seed + 1009 * ci + ii) for ci, ii in batch)
            prog("robustness", min(1.0, (start + chunk) / len(tasks)),
                 f"Robustness: {min(start + chunk, len(tasks))}/{len(tasks)}")
    rows: List[Dict[str, Any]] = []
    for ci, spec in enumerate(conditions):
        ys, ps, failed = [], [], 0
        for (tci, ii), ex in zip(tasks, out_ex):
            if tci != ci:
                continue
            if not ex.ok:
                failed += 1
                continue
            ys.append(int(labels[ii]))
            ps.append(det.predict_from_extraction(ex, os.path.basename(paths[ii])).prob_ai)
        row: Dict[str, Any] = {"condition": "original (no edit)" if spec is None else describe(spec),
                               "n_ok": len(ys), "n_failed": failed}
        if ys:
            m = compute_metrics(np.array(ys), np.array(ps), det.threshold, det.margin)
            row.update({"accuracy": m["accuracy"], "balanced_accuracy": m["balanced_accuracy"],
                        "roc_auc": m["roc_auc"], "recall": m["recall"], "specificity": m["specificity"]})
        rows.append(row)
    return rows


def evaluate_model(model_path: str, data_cfg: Config, out_dir: str, *, data_dir: Optional[str] = None,
                   real_dirs: Sequence[str] = (), fake_dirs: Sequence[str] = (),
                   manifest: Optional[str] = None, only_split: Optional[str] = None,
                   robustness: bool = False, n_jobs: Optional[int] = None,
                   progress: Optional[Progress] = None) -> Dict[str, Any]:
    prog = progress or (lambda s, f, m: None)
    det = Detector.load(model_path)
    cfg = det.cfg
    jobs = data_cfg.runtime.n_jobs if n_jobs is None else n_jobs

    samples: List[Sample] = collect_samples(data_cfg, data_dir, real_dirs, fake_dirs, manifest)
    if only_split:
        samples = [s for s in samples if s.split == only_split]
        if not samples:
            raise DataError(f"No images in the '{only_split}' split of this dataset.")
    prog("predict", 0.0, f"Analysing {len(samples)} images")
    preds: List[Prediction] = det.predict_many(
        [s.path for s in samples], n_jobs=jobs,
        progress=lambda i, n: prog("predict", 0.6 * i / n, f"Analysed {i}/{n}"))

    ok = [(s, p) for s, p in zip(samples, preds) if p.ok]
    failed = [(s, p) for s, p in zip(samples, preds) if not p.ok]
    if not ok:
        raise DataError("No image could be analysed. First problems: " +
                        "; ".join(f"{os.path.basename(s.path)}: {p.error}" for s, p in failed[:3]))
    y = np.array([s.label for s, _ in ok])
    p = np.array([pr.prob_ai for _, pr in ok])
    metrics = compute_metrics(y, p, det.threshold, det.margin, cfg.evaluation.bootstrap_samples, cfg.runtime.seed)

    notes: List[str] = []
    train_hashes = det.bundle.metadata.get("_training_hashes")
    if train_hashes:
        overlap = sum(1 for _, pr in ok if pr.sha256 and pr.sha256 in train_hashes)
        if overlap:
            notes.append(f"{overlap} of the evaluated images were also in the training data, so these "
                         f"numbers are optimistic. Evaluate on images the model has never seen.")
    if failed:
        notes.append(f"{len(failed)} image(s) could not be analysed and are excluded from the metrics.")
    notes += list(det.bundle.metadata.get("load_warnings", []))

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plots = save_evaluation_plots(out, y, p, det.threshold)
    with open(out / "predictions.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["file", "true_label", "prob_ai", "verdict", "reliability", "error"])
        for s, pr in zip(samples, preds):
            wr.writerow([s.path, "ai_generated" if s.label else "real",
                         "" if pr.prob_ai is None else f"{pr.prob_ai:.6f}", pr.verdict or "",
                         pr.reliability or "", pr.error or ""])

    rob_rows: Optional[List[Dict[str, Any]]] = None
    if robustness:
        specs = cfg.evaluation.robustness
        rob_rows = run_robustness(det, [s.path for s, _ in ok], y, specs, cfg.evaluation.robustness_max_images,
                                  jobs, cfg.runtime.seed, prog)
    result = {"metrics": metrics, "notes": notes, "robustness": rob_rows,
              "n_failed": len(failed), "failed_examples": [{"file": s.path, "reason": pr.error} for s, pr in failed[:20]]}
    (out / "evaluation.json").write_text(json.dumps(_jsonable(result), indent=2), encoding="utf-8")
    write_html_report(out / "evaluation.html", "AI image detector - evaluation", metrics, out, plots,
                      notes=notes, robustness=rob_rows)
    prog("done", 1.0, "Evaluation finished")
    return result
