"""Evaluate a trained model on any labelled dataset, optionally under image degradations."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .dataset import DatasetError, Sample
from .detector import Detector
from .evaluation import compute_metrics, per_source, plot_report
from .pipeline import Task, run_tasks
from .trainer import aggregate
from .utils import dump_json, log


def _probs(det: Detector, samples: list[Sample], variant: str, ops: list | None, desc: str):
    cfg = det.cfg
    tasks = [Task(i, s.path, variant, ops) for i, s in enumerate(samples)]
    res = run_tasks(tasks, cfg, desc)
    rows, row_unit, keep = [], [], []
    for r in res:
        if r.feats is None:
            continue
        rows.append(r.feats)
        row_unit.extend([len(keep)] * len(r.feats))
        keep.append(r.idx)
    if not keep:
        raise DatasetError("no image could be analysed")
    X = np.concatenate(rows)
    z = aggregate(det._scores(X)[:, None], np.array(row_unit), len(keep))[:, 0]
    return np.array(keep), det.bundle.calibrator(z), len(samples) - len(keep)


def evaluate(det: Detector, samples: list[Sample], out_dir: str | Path | None, robustness: bool = False) -> dict:
    b, cfg = det.bundle, det.cfg
    keep, p, n_fail = _probs(det, samples, "clean", None, "evaluating")
    y = np.array([samples[i].label for i in keep])
    if len(np.unique(y)) < 2:
        raise DatasetError("evaluation data must contain both real and AI images")
    result = {"n_failed": int(n_fail), "metrics": compute_metrics(y, p, b.threshold, b.margin)}
    result["metrics"]["by_source"] = per_source(y, p, [samples[i].source for i in keep], b.threshold)
    log(f"AUC={result['metrics']['roc_auc']:.4f} acc={result['metrics']['accuracy']:.4f} "
        f"bal-acc={result['metrics']['balanced_accuracy']:.4f} (n={len(y)}, skipped {n_fail})")
    if robustness:
        base_auc = result["metrics"]["roc_auc"]
        rob = {}
        for spec in cfg.robustness:
            k2, p2, f2 = _probs(det, samples, f"pert:{spec['name']}", spec["ops"], f"robustness {spec['name']}")
            y2 = np.array([samples[i].label for i in k2])
            if len(np.unique(y2)) < 2:
                continue
            m = compute_metrics(y2, p2, b.threshold)
            rob[spec["name"]] = {"roc_auc": m["roc_auc"], "accuracy": m["accuracy"],
                                 "balanced_accuracy": m["balanced_accuracy"], "recall_ai": m["recall_ai"],
                                 "specificity_real": m["specificity_real"], "auc_drop": base_auc - m["roc_auc"],
                                 "n": m["n"]}
            log(f"  {spec['name']:<14} AUC={m['roc_auc']:.3f} (drop {base_auc - m['roc_auc']:+.3f}) "
                f"acc={m['accuracy']:.3f}")
        result["robustness"] = rob
    if out_dir:
        out_dir = Path(out_dir)
        dump_json(result, out_dir / "evaluation.json")
        plot_report(y, p, b.threshold, out_dir / "evaluation.png", "Evaluation")
    return result
