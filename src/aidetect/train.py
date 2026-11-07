"""Training pipeline.

1. discover + validate images, extract patch features (parallel)
2. drop exact duplicates, group near-duplicates, audit dataset biases
3. leak-free train / test split (group-aware, stratified)
4. out-of-fold predictions on the training images -> Platt calibration + threshold
5. fit the final patch model, evaluate on the untouched test images (bootstrap CIs,
   calibration error, signal-group permutation importance, plots, HTML report)
6. optionally refit on *all* images for deployment
"""
from __future__ import annotations

import datetime as _dt
import logging
import os
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedGroupKFold

from . import __version__
from .config import Config
from .data import (DataError, ImageMeta, Sample, assign_splits, audit_dataset, collect_samples,
                   drop_exact_duplicates, group_near_duplicates, summarize_splits)
from .extraction import extract_features
from .features import FEATURE_VERSION, GROUP_ORDER, extractor_for
from .metrics import choose_threshold, compute_metrics, safe_auc
from .model import (META_FILE, ModelBundle, ModelError, PatchModel, Platt, aggregate_by_image, versions)
from .reporting import save_evaluation_plots, write_html_report

log = logging.getLogger(__name__)
Progress = Callable[[str, float, str], None]


@dataclass
class TrainResult:
    model_dir: str
    metrics: Dict[str, Any]
    metadata: Dict[str, Any]
    warnings: List[str]


# --------------------------------------------------------------------------- #
def _noop(stage: str, frac: float, msg: str) -> None:
    return None


def _extract_all(samples: Sequence[Sample], key: str, n_jobs: int, prog: Progress):
    total = len(samples)
    workers = (os.cpu_count() or 2) if n_jobs == -1 else n_jobs
    chunk = max(16, 4 * workers)
    results = []
    with Parallel(n_jobs=n_jobs) as par:
        for start in range(0, total, chunk):
            batch = samples[start: start + chunk]
            results.extend(par(delayed(extract_features)(s.path, key, None, True) for s in batch))
            done = min(start + chunk, total)
            prog("features", 0.05 + 0.40 * done / total, f"Analysed {done}/{total} images")
    return results


def _rows(offsets: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return np.concatenate([np.arange(offsets[i], offsets[i + 1]) for i in idx]) if len(idx) else np.array([], int)


def _subtable(X: np.ndarray, offsets: np.ndarray, idx: np.ndarray):
    rows = _rows(offsets, idx)
    n_per = np.diff(offsets)[idx]
    offs = np.concatenate([[0], np.cumsum(n_per)])
    return X[rows], offs


def _weights(n_per: np.ndarray, y_img: np.ndarray) -> np.ndarray:
    """Each image counts equally (1/#patches) and both classes carry equal total weight."""
    n = len(y_img)
    cls = {c: n / (2.0 * max(int(np.sum(y_img == c)), 1)) for c in (0, 1)}
    per_img = np.array([cls[int(c)] for c in y_img]) / n_per
    w = np.repeat(per_img, n_per)
    return w / w.mean()


def _fit_stack(cfg: Config, X: np.ndarray, offsets: np.ndarray, y_img: np.ndarray, groups: np.ndarray,
               idx: np.ndarray, n_jobs: int, prog: Progress, base: float, span: float, tag: str):
    """Out-of-fold scoring -> calibrator + threshold, then the final patch model on ``idx``."""
    seed = cfg.runtime.seed
    Xs, offs = _subtable(X, offsets, idx)
    ys, gs = y_img[idx], groups[idx]
    n_per = np.diff(offs)
    y_patch = np.repeat(ys, n_per)
    w_all = _weights(n_per, ys)

    min_groups = min(len(np.unique(gs[ys == c])) for c in (0, 1))
    k = max(2, min(cfg.decision.cv_folds, min_groups))
    if min_groups < 2:
        raise DataError("Not enough distinct images in one class for cross-validated calibration.")
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    oof = np.full(len(idx), np.nan)
    for f, (tr, va) in enumerate(sgkf.split(np.zeros(len(idx)), ys, gs)):
        prog("model", base + span * f / (k + 1), f"{tag}: calibration fold {f + 1}/{k}")
        Xtr, otr = _subtable(Xs, offs, tr)
        Xva, ova = _subtable(Xs, offs, va)
        ntr = np.diff(otr)
        try:
            pm = PatchModel(cfg.model, seed + f, n_jobs).fit(Xtr, np.repeat(ys[tr], ntr), _weights(ntr, ys[tr]))
        except ModelError as e:
            raise DataError(f"Calibration fold {f + 1} failed: {e}. Add more images per class.") from e
        oof[va] = aggregate_by_image(pm.predict_proba(Xva), ova, cfg.decision.aggregation)
    if np.isnan(oof).any():
        raise ModelError("internal error: missing out-of-fold scores")

    platt = Platt.fit(oof, ys, cfg.decision.calibration_C)
    p_oof = platt(oof)
    threshold = choose_threshold(ys, p_oof, cfg.decision.threshold_strategy, cfg.decision.fixed_threshold)
    oof_metrics = compute_metrics(ys, p_oof, threshold, cfg.decision.uncertainty_margin)

    prog("model", base + span * k / (k + 1), f"{tag}: fitting final model")
    pm = PatchModel(cfg.model, seed, n_jobs).fit(Xs, y_patch, w_all)
    return pm, platt, float(threshold), oof_metrics, k


def _group_importance(pm: PatchModel, X: np.ndarray, y_patch: np.ndarray, columns: Dict[str, np.ndarray],
                      repeats: int, seed: int) -> Dict[str, float]:
    base = safe_auc(y_patch, pm.predict_proba(X))
    if base is None:
        return {}
    rng = np.random.default_rng(seed)
    out: Dict[str, float] = {}
    for g in GROUP_ORDER:
        cols = columns[g]
        drops = []
        for _ in range(repeats):
            Xp = X.copy()
            Xp[:, cols] = X[rng.permutation(len(X))][:, cols]
            a = safe_auc(y_patch, pm.predict_proba(Xp))
            if a is not None:
                drops.append(base - a)
        out[g] = float(np.mean(drops)) if drops else 0.0
    return out


def _image_metrics(pm: PatchModel, platt: Platt, X: np.ndarray, offsets: np.ndarray, y: np.ndarray,
                   threshold: float, cfg: Config):
    agg = aggregate_by_image(pm.predict_proba(X), offsets, cfg.decision.aggregation)
    p = platt(agg)
    return p, compute_metrics(y, p, threshold, cfg.decision.uncertainty_margin,
                              cfg.evaluation.bootstrap_samples, cfg.runtime.seed)


# --------------------------------------------------------------------------- #
def train_model(cfg: Config, out_dir: str, *, data_dir: Optional[str] = None,
                real_dirs: Sequence[str] = (), fake_dirs: Sequence[str] = (),
                manifest: Optional[str] = None, final_fit_on_all: bool = False,
                progress: Optional[Progress] = None) -> TrainResult:
    prog = progress or _noop
    t0 = time.time()
    seed = cfg.runtime.seed
    n_jobs = cfg.runtime.n_jobs
    warnings: List[str] = []

    # ---- 1. discover + extract -------------------------------------------------
    prog("discover", 0.0, "Scanning dataset")
    samples = collect_samples(cfg, data_dir, real_dirs, fake_dirs, manifest)
    log.info("Found %d images (%d real, %d AI)", len(samples),
             sum(s.label == 0 for s in samples), sum(s.label == 1 for s in samples))
    key = cfg.feature_key()
    results = _extract_all(samples, key, n_jobs, prog)

    failed = [(samples[i].path, r.error) for i, r in enumerate(results) if not r.ok]
    if failed:
        reasons = Counter((e or "unknown").split(" (")[0] for _, e in failed)
        warnings.append(f"{len(failed)} of {len(samples)} files could not be used "
                        f"({', '.join(f'{n}x {r}' for r, n in reasons.most_common(4))}).")
        for p, e in failed[:5]:
            log.warning("Skipped %s: %s", p, e)
    good = [i for i, r in enumerate(results) if r.ok]
    if not good:
        raise DataError("None of the images could be analysed. First problems:\n  " +
                        "\n  ".join(f"{p}: {e}" for p, e in failed[:5]))

    metas: List[ImageMeta] = []
    for i in good:
        s, r = samples[i], results[i]
        exif = r.raw_meta.get("exif", {})
        metas.append(ImageMeta(
            path=s.path, label=s.label, split=s.split, sha256=r.sha256 or "", dhash=int(r.dhash or 0),
            width=int(r.info.get("width", 0)), height=int(r.info.get("height", 0)),
            fmt=str(r.info.get("format") or "unknown"), file_size=int(r.info.get("file_size") or 0),
            has_camera_exif=bool(exif.get("Make") or exif.get("Model"))))
    feats = [results[i].features for i in good]
    del results

    # ---- 2. clean ---------------------------------------------------------------
    prog("clean", 0.46, "Removing duplicates and auditing data")
    keep, dedupe = drop_exact_duplicates(metas)
    metas = [metas[i] for i in keep]
    feats = [feats[i] for i in keep]
    if dedupe["exact_duplicates_removed"] or dedupe["conflicting_label_files_removed"]:
        warnings.append(f"Removed {dedupe['exact_duplicates_removed']} exact duplicate file(s) and "
                        f"{dedupe['conflicting_label_files_removed']} file(s) with conflicting labels.")
    labels = np.array([m.label for m in metas])
    for c, name in ((0, "real"), (1, "AI-generated")):
        n_c = int(np.sum(labels == c))
        if n_c < cfg.data.min_images_per_class:
            raise DataError(f"Only {n_c} usable {name} images after validation and de-duplication; "
                            f"at least {cfg.data.min_images_per_class} are required "
                            f"(data.min_images_per_class). Several hundred per class is recommended.")
    groups = group_near_duplicates([m.dhash for m in metas], labels, cfg.data.near_duplicate_hamming)
    n_merged = int(len(metas) - len(np.unique(groups)))
    if n_merged:
        log.info("%d near-duplicate images are kept together in the same split", n_merged)
    audit = audit_dataset(metas)
    warnings += audit["warnings"]

    # ---- 3. split ---------------------------------------------------------------
    split, split_info = assign_splits(labels, groups, [m.split for m in metas], cfg.data.test_size, seed)
    keep2 = np.where(split != "drop")[0]
    if len(keep2) < len(metas):
        warnings.append(f"Dropped {len(metas) - len(keep2)} test image(s) that are near-duplicates of "
                        f"training images (leak prevention).")
    metas = [metas[i] for i in keep2]
    feats = [feats[i] for i in keep2]
    labels, groups, split = labels[keep2], groups[keep2], split[keep2]
    split_counts = summarize_splits(labels, split)
    for s in ("train", "test"):
        for name in ("real", "ai_generated"):
            need = 6 if s == "train" else 3
            if split_counts[s][name] < need:
                raise DataError(f"The {s} split has only {split_counts[s][name]} {name} image(s) "
                                f"(minimum {need}). Provide more images.")

    n_img = len(metas)
    counts = np.array([f.shape[0] for f in feats])
    offsets = np.concatenate([[0], np.cumsum(counts)])
    X = np.concatenate(feats, axis=0).astype(np.float32)
    del feats
    train_idx = np.where(split == "train")[0]
    test_idx = np.where(split == "test")[0]
    log.info("train: %d images (%d patches) | test: %d images", len(train_idx),
             int(counts[train_idx].sum()), len(test_idx))

    ext = extractor_for(cfg)

    # ---- 4/5. fit, calibrate, evaluate ------------------------------------------
    pm, platt, thr, oof_metrics, k = _fit_stack(cfg, X, offsets, labels, groups, train_idx, n_jobs, prog,
                                                0.50, 0.30, "train split")
    prog("evaluate", 0.82, "Evaluating on held-out images")
    Xt, ot = _subtable(X, offsets, test_idx)
    yt = labels[test_idx]
    p_test, test_metrics = _image_metrics(pm, platt, Xt, ot, yt, thr, cfg)
    importance = _group_importance(pm, Xt, np.repeat(yt, np.diff(ot)), ext.group_columns,
                                   cfg.evaluation.permutation_repeats, seed)
    warnings += test_metrics["warnings"]
    if oof_metrics["roc_auc"] is not None and test_metrics["roc_auc"] is not None \
            and oof_metrics["roc_auc"] - test_metrics["roc_auc"] > 0.15:
        warnings.append("Cross-validated AUC on the training images is much higher than on the held-out "
                        "images: the model may be overfitting or the split is hard. Trust the held-out number.")
    if test_metrics["roc_auc"] is not None and test_metrics["roc_auc"] < 0.6:
        warnings.append("Held-out AUC is below 0.6: the detector found little usable signal in this data.")

    # ---- optional refit on everything ---------------------------------------------
    final_pm, final_platt, final_thr = pm, platt, thr
    final_info: Dict[str, Any] = {"trained_on": "train split only"}
    if final_fit_on_all:
        all_idx = np.arange(n_img)
        final_pm, final_platt, final_thr, all_oof, _ = _fit_stack(cfg, X, offsets, labels, groups, all_idx,
                                                                  n_jobs, prog, 0.84, 0.12, "all images")
        final_info = {"trained_on": "all images (train + test)", "oof_metrics_all": all_oof}
        warnings.append("The final model was refitted on ALL images; the reported test metrics come from the "
                        "earlier fit that never saw the test images, so they are an honest estimate for it.")

    # ---- 6. persist ----------------------------------------------------------------
    prog("save", 0.97, "Saving model and report")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plots_dir = out / "report"
    plots = save_evaluation_plots(plots_dir, yt, p_test, thr, importance)
    import csv
    with open(plots_dir / "test_predictions.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["file", "true_label", "prob_ai", "predicted"])
        for i, p in zip(test_idx, p_test):
            wr.writerow([metas[i].path, "ai_generated" if labels[i] else "real", f"{p:.6f}",
                         "ai_generated" if p >= thr else "real"])

    metadata = {
        "aidetect_version": __version__, "feature_version": FEATURE_VERSION,
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "training_seconds": round(time.time() - t0, 1), "versions": versions(),
        "n_features": ext.n_features,
        "feature_groups": {g: int(len(c)) for g, c in ext.group_columns.items()},
        "config": cfg.to_dict(),
        "data": {
            "files_found": len(samples), "files_unusable": len(failed),
            "unusable_examples": [{"file": p, "reason": e} for p, e in failed[:20]],
            **dedupe, "near_duplicates_grouped": n_merged, "split_info": split_info,
            "split_counts": split_counts, "patches_total": int(counts.sum()), "audit": audit,
        },
        "decision": {"threshold": float(final_thr), "uncertainty_margin": cfg.decision.uncertainty_margin,
                     "aggregation": cfg.decision.aggregation,
                     "calibration": {"a": final_platt.a, "b": final_platt.b},
                     "cv_folds": k, "cv_metrics_train_split": oof_metrics,
                     "note": "Probabilities assume equal class priors (class-balanced calibration)."},
        "evaluation": {"test": test_metrics, "signal_importance": importance},
        "final_fit": final_info, "warnings": warnings,
        "_training_hashes": {metas[i].sha256 for i in (range(n_img) if final_fit_on_all else train_idx)},
    }
    bundle = ModelBundle(final_pm, final_platt, final_thr, cfg.to_dict(), ext.names, ext.groups, metadata)
    bundle.save(out)
    write_html_report(plots_dir / "report.html", "AI image detector - held-out evaluation", test_metrics,
                      plots_dir, plots, notes=warnings)
    prog("done", 1.0, "Training finished")
    return TrainResult(str(out), test_metrics, metadata, warnings)
