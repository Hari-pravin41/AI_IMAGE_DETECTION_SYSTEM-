"""End-to-end training: split -> features -> grouped CV -> ensemble weights -> calibration ->
threshold & abstain band -> hold-out evaluation -> saved model + reports."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from .config import Config
from .dataset import DatasetError, Sample, scan_dataset
from .evaluation import compute_metrics, per_source, plot_report
from .features import feature_group_of, feature_names
from .model import Calibrator, Ensemble, HybridImageHead, ModelBundle, choose_margin, choose_threshold, unit_auc
from .pipeline import Task, run_tasks
from .utils import dump_json, log, resolve_jobs


@dataclass
class Table:
    X: np.ndarray            # (rows, features) crop-level feature rows
    row_unit: np.ndarray     # row -> unit (one unit = one image variant)
    unit_sample: np.ndarray  # unit -> index into the samples list
    unit_variant: list       # unit -> variant name
    y_unit: np.ndarray

    @property
    def n_units(self) -> int:
        return len(self.unit_sample)


def aggregate(vals: np.ndarray, row_unit: np.ndarray, n_units: int) -> np.ndarray:
    """Mean of per-crop values within each unit."""
    sums = np.zeros((n_units,) + vals.shape[1:], dtype=np.float64)
    np.add.at(sums, row_unit, vals)
    cnt = np.bincount(row_unit, minlength=n_units).astype(np.float64)
    return sums / cnt.reshape((-1,) + (1,) * (vals.ndim - 1))


def build_table(samples: list[Sample], idxs: list[int], variants: list[str], cfg: Config,
                desc: str) -> tuple[Table, list[int], list[str]]:
    """Extract features. Returns (table, sample indices that failed their clean variant, error messages)."""
    tasks = [Task(i, samples[i].path, v) for i in idxs for v in variants]
    results = run_tasks(tasks, cfg, desc)
    by_sample: dict[int, dict[str, object]] = {}
    for r in results:
        by_sample.setdefault(r.idx, {})[r.variant] = r
    failed, errors = [], []
    rows, row_unit, u_sample, u_var = [], [], [], []
    for i in idxs:
        got = by_sample[i]
        clean = got["clean"]
        if clean.feats is None:
            failed.append(i)
            errors.append(f"{samples[i].path}: {clean.error}")
            continue
        for v in variants:
            r = got[v]
            if r.feats is None:
                continue                      # a failed augmentation only loses that augmented copy
            uid = len(u_sample)
            rows.append(r.feats)
            row_unit.extend([uid] * len(r.feats))
            u_sample.append(i)
            u_var.append(v)
    if not rows:
        raise DatasetError("feature extraction failed for every image")
    X = np.concatenate(rows).astype(np.float32)
    ys = np.array([samples[i].label for i in u_sample])
    return Table(X, np.array(row_unit), np.array(u_sample), u_var, ys), failed, errors


def _split(samples: list[Sample], cfg: Config) -> tuple[list[int], list[int]]:
    idx = np.arange(len(samples))
    if cfg.test_size <= 0:
        return list(idx), []
    y = np.array([s.label for s in samples])
    groups = np.array([s.group for s in samples])
    k = max(2, int(round(1.0 / cfg.test_size)))
    sgk = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=cfg.seed)
    try:
        tr, te = next(iter(sgk.split(idx, y, groups)))
    except ValueError as e:
        log(f"warning: hold-out split failed ({e}); using cross-validation only")
        return list(idx), []
    if min(np.bincount(y[te], minlength=2)) < 3 or min(np.bincount(y[tr], minlength=2)) < 6:
        log("warning: dataset too small for a hold-out set; reporting cross-validation metrics only")
        return list(idx), []
    return list(tr), list(te)


def _group_importance(ens: Ensemble, tbl: Table, names: list[str], reps: int = 3, seed: int = 0) -> dict:
    """Drop in hold-out AUC when each feature group is shuffled across images."""
    rng = np.random.default_rng(seed)
    base = unit_auc(tbl.y_unit, ens.combine(aggregate(ens.member_logits(tbl.X), tbl.row_unit, tbl.n_units)))
    groups = sorted({feature_group_of(n) for n in names})
    out = {}
    for g in groups:
        cols = [i for i, n in enumerate(names) if feature_group_of(n) == g]
        drops = []
        for _ in range(reps):
            X = tbl.X.copy()
            perm = rng.permutation(len(X))
            X[:, cols] = X[perm][:, cols]
            z = ens.combine(aggregate(ens.member_logits(X), tbl.row_unit, tbl.n_units))
            drops.append(base - unit_auc(tbl.y_unit, z))
        out[g] = float(np.mean(drops))
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def train(samples: list[Sample], cfg: Config, out_dir: str | Path, audit: dict | None = None,
          refit_all: bool = False) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_jobs = resolve_jobs(cfg.n_jobs)
    names = feature_names(cfg)

    tr_idx, te_idx = _split(samples, cfg)
    aug = [f"aug{k}" for k in range(cfg.augment_copies)]
    log(f"train images: {len(tr_idx)}  hold-out images: {len(te_idx)}  features: {len(names)}  "
        f"augmented copies: {cfg.augment_copies}")

    train_tbl, failed, errs = build_table(samples, tr_idx, ["clean"] + aug, cfg, "features (train)")
    test_tbl = None
    if te_idx:
        test_tbl, f2, e2 = build_table(samples, te_idx, ["clean"], cfg, "features (hold-out)")
        failed, errs = failed + f2, errs + e2
    if failed:
        log(f"warning: {len(failed)} images failed feature extraction and were skipped (e.g. {errs[0]})")
    counts = np.bincount(np.array([samples[i].label for i in set(train_tbl.unit_sample.tolist())]), minlength=2)
    if counts.min() < 8:
        raise DatasetError(f"need at least 8 usable training images per class, got real={counts[0]} ai={counts[1]}")

    # ---- grouped out-of-fold predictions (used for weights, calibration, threshold, margin)
    tr_samples = np.array(sorted(set(train_tbl.unit_sample.tolist())))
    y_s = np.array([samples[i].label for i in tr_samples])
    g_s = np.array([samples[i].group for i in tr_samples])
    k = int(min(cfg.cv_folds, np.bincount(y_s).min()))
    if k < 2:
        raise DatasetError("not enough images per class for cross-validation")
    fold_of = {}
    for f, (_, te) in enumerate(StratifiedGroupKFold(k, shuffle=True, random_state=cfg.seed).split(tr_samples, y_s, g_s)):
        for j in te:
            fold_of[int(tr_samples[j])] = f
    row_fold = np.array([fold_of[int(train_tbl.unit_sample[u])] for u in train_tbl.row_unit])
    y_row = train_tbl.y_unit[train_tbl.row_unit]
    oof = np.zeros((len(train_tbl.X), len(cfg.models)))
    for f in range(k):
        log(f"cross-validation fold {f + 1}/{k}")
        te_m = row_fold == f
        ens = Ensemble(cfg, n_jobs).fit(train_tbl.X[~te_m], y_row[~te_m])
        oof[te_m] = ens.member_logits(train_tbl.X[te_m])
    unit_member = aggregate(oof, train_tbl.row_unit, train_tbl.n_units)
    member_aucs = [unit_auc(train_tbl.y_unit, unit_member[:, j]) for j in range(unit_member.shape[1])]

    final = Ensemble(cfg, n_jobs)
    final.set_weights_from_auc(member_aucs)
    z_oof = final.combine(unit_member)
    calibrator = Calibrator.fit(z_oof, train_tbl.y_unit)
    p_oof = calibrator(z_oof)
    thr = choose_threshold(train_tbl.y_unit, p_oof, cfg.threshold_metric)
    margin, margin_info = choose_margin(train_tbl.y_unit, p_oof, thr, cfg)
    clean_u = np.array([v == "clean" for v in train_tbl.unit_variant])
    cv_metrics = compute_metrics(train_tbl.y_unit[clean_u], p_oof[clean_u], thr, margin)
    cv_metrics["member_auc"] = dict(zip(cfg.models, member_aucs))
    cv_metrics["ensemble_weights"] = dict(zip(cfg.models, final.weights.tolist()))
    log(f"CV (clean images): AUC={cv_metrics.get('roc_auc', float('nan')):.4f} "
        f"acc={cv_metrics['accuracy']:.4f} threshold={thr:.3f} uncertain-margin={margin:.3f}")

    # ---- final model on all training rows
    log("fitting final model")
    weights = final.weights.copy()
    final.fit(train_tbl.X, y_row)
    final.weights = weights

    unit_feats = aggregate(train_tbl.X, train_tbl.row_unit, train_tbl.n_units)
    unit_offsets = np.arange(0, unit_feats.shape[0] + 1, dtype=np.int64)
    high_level = None
    if cfg.high_level.enabled:
        high_level = HybridImageHead(weight=cfg.high_level.weight, method=cfg.high_level.method,
                                    hidden_dim=cfg.high_level.hidden_dim)
        high_level.fit(unit_feats, unit_offsets, train_tbl.y_unit)
        p_cv_hybrid = high_level.predict(unit_feats, unit_offsets, forensic_probs=p_oof)
        thr = choose_threshold(train_tbl.y_unit, p_cv_hybrid, cfg.threshold_metric)
        margin, margin_info = choose_margin(train_tbl.y_unit, p_cv_hybrid, thr, cfg)
        cv_metrics = compute_metrics(train_tbl.y_unit, p_cv_hybrid, thr, margin)
        cv_metrics["member_auc"] = dict(zip(cfg.models, member_aucs))
        cv_metrics["ensemble_weights"] = dict(zip(cfg.models, final.weights.tolist()))
        log(f"CV (hybrid blend): AUC={cv_metrics.get('roc_auc', float('nan')):.4f} "
            f"acc={cv_metrics['accuracy']:.4f} threshold={thr:.3f} uncertain-margin={margin:.3f}")

    summary: dict = {"n_train_images": int(len(tr_samples)), "n_holdout_images": len(te_idx),
                     "n_features": len(names), "cv_folds": k, "cv": cv_metrics,
                     "margin_info": margin_info, "audit": audit, "high_level_enabled": bool(cfg.high_level.enabled)}
    report_dir = out_dir
    if test_tbl is not None:
        z = final.combine(aggregate(final.member_logits(test_tbl.X), test_tbl.row_unit, test_tbl.n_units))
        p = calibrator(z)
        if high_level is not None:
            test_feats = aggregate(test_tbl.X, test_tbl.row_unit, test_tbl.n_units)
            test_offsets = np.arange(0, test_feats.shape[0] + 1, dtype=np.int64)
            p = high_level.predict(test_feats, test_offsets, forensic_probs=p)
        tm = compute_metrics(test_tbl.y_unit, p, thr, margin)
        tm["by_source"] = per_source(test_tbl.y_unit, p, [samples[i].source for i in test_tbl.unit_sample], thr)
        tm["group_importance_auc_drop"] = _group_importance(final, test_tbl, names)
        summary["holdout"] = tm
        plot_report(test_tbl.y_unit, p, thr, report_dir / "evaluation_holdout.png", "Hold-out evaluation")
        log(f"HOLD-OUT: AUC={tm.get('roc_auc', float('nan')):.4f} acc={tm['accuracy']:.4f} "
            f"bal-acc={tm['balanced_accuracy']:.4f} (n={tm['n']})")
    plot_report(train_tbl.y_unit[clean_u], p_oof[clean_u], thr, report_dir / "evaluation_cv.png",
                "Cross-validation (out-of-fold)")

    if refit_all and test_tbl is not None:
        log("refitting on train + hold-out data")
        X_all = np.concatenate([train_tbl.X, test_tbl.X])
        y_all = np.concatenate([y_row, test_tbl.y_unit[test_tbl.row_unit]])
        final.fit(X_all, y_all)
        final.weights = weights
        summary["refit_all"] = True
        medians = np.median(X_all, axis=0)
    else:
        medians = np.median(train_tbl.X, axis=0)

    bundle = ModelBundle(cfg, names, final, calibrator, float(thr), float(margin), medians, high_level, summary)
    bundle.save(out_dir)
    dump_json(summary, out_dir / "metrics.json")
    if audit is not None:
        dump_json(audit, out_dir / "dataset_report.json")
    (out_dir / "config_used.yaml").write_text(cfg.to_yaml(), encoding="utf-8")
    log(f"model saved to {out_dir}")
    return summary
