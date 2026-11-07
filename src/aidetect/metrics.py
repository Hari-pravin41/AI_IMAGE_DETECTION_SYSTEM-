"""Evaluation metrics with calibration quality, abstention and bootstrap confidence intervals."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    return np.log(p / (1 - p))


def confusion(y: np.ndarray, pred: np.ndarray) -> Dict[str, int]:
    y, pred = np.asarray(y).astype(bool), np.asarray(pred).astype(bool)
    return {"tn": int(np.sum(~y & ~pred)), "fp": int(np.sum(~y & pred)),
            "fn": int(np.sum(y & ~pred)), "tp": int(np.sum(y & pred))}


def _rates(c: Dict[str, int]) -> Dict[str, Optional[float]]:
    tp, tn, fp, fn = c["tp"], c["tn"], c["fp"], c["fn"]
    n = tp + tn + fp + fn
    rec = tp / (tp + fn) if tp + fn else None
    spec = tn / (tn + fp) if tn + fp else None
    prec = tp / (tp + fp) if tp + fp else None
    f1 = 2 * prec * rec / (prec + rec) if prec and rec else (0.0 if prec is not None and rec is not None else None)
    bal = (rec + spec) / 2 if rec is not None and spec is not None else None
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / den if den > 0 else 0.0
    return {"accuracy": (tp + tn) / n if n else None, "balanced_accuracy": bal, "precision": prec,
            "recall": rec, "specificity": spec, "f1": f1, "mcc": float(mcc)}


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & ((p < edges[i + 1]) if i < bins - 1 else (p <= edges[i + 1]))
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def reliability_curve(y: np.ndarray, p: np.ndarray, bins: int = 10) -> Dict[str, List[float]]:
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    xs, ys, ns = [], [], []
    for i in range(bins):
        m = (p >= edges[i]) & ((p < edges[i + 1]) if i < bins - 1 else (p <= edges[i + 1]))
        if m.any():
            xs.append(float(p[m].mean()))
            ys.append(float(y[m].mean()))
            ns.append(int(m.sum()))
    return {"mean_predicted": xs, "fraction_positive": ys, "count": ns}


def safe_auc(y: np.ndarray, p: np.ndarray) -> Optional[float]:
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, p))


def choose_threshold(y: np.ndarray, p: np.ndarray, strategy: str, fixed: float = 0.5) -> float:
    """Pick a decision threshold on (out-of-fold) calibrated probabilities.
    Ties are broken towards 0.5 so a perfectly separable set gets a threshold in the middle of the gap."""
    if strategy == "fixed":
        return float(fixed)
    y = np.asarray(y).astype(bool)
    p = np.asarray(p, dtype=np.float64)
    uniq = np.unique(p)
    if uniq.size < 2:
        return 0.5
    cands = (uniq[:-1] + uniq[1:]) / 2.0
    if cands.size > 2000:
        cands = np.unique(np.quantile(p, np.linspace(0, 1, 2000)))
    cands = np.unique(np.concatenate([cands, [0.5]]))
    pos, neg = max(int(y.sum()), 1), max(int((~y).sum()), 1)
    pred = p[None, :] >= cands[:, None]
    tp = (pred & y[None, :]).sum(axis=1)
    fp = (pred & ~y[None, :]).sum(axis=1)
    if strategy == "f1":
        score = 2 * tp / np.maximum(2 * tp + fp + (pos - tp), 1)
    else:  # balanced_accuracy
        score = (tp / pos + (neg - fp) / neg) / 2.0
    best = np.flatnonzero(score >= score.max() - 1e-12)
    return float(cands[best[np.argmin(np.abs(cands[best] - 0.5))]])


def compute_metrics(y: np.ndarray, p: np.ndarray, threshold: float, margin: float = 0.0,
                    n_boot: int = 0, seed: int = 0) -> Dict[str, Any]:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=np.float64)
    n = len(y)
    pred = p >= threshold
    c = confusion(y, pred)
    out: Dict[str, Any] = {"n": int(n), "n_real": int(np.sum(y == 0)), "n_ai": int(np.sum(y == 1)),
                           "threshold": float(threshold)}
    out.update(_rates(c))
    out["confusion"] = c
    out["fpr"] = (1 - out["specificity"]) if out["specificity"] is not None else None
    out["fnr"] = (1 - out["recall"]) if out["recall"] is not None else None
    out["roc_auc"] = safe_auc(y, p)
    out["average_precision"] = float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else None
    out["brier"] = float(np.mean((p - y) ** 2))
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    out["log_loss"] = float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc)))
    out["ece"] = expected_calibration_error(y, p)

    unc = np.abs(p - threshold) < margin
    conf = ~unc
    out["uncertain_fraction"] = float(unc.mean()) if n else 0.0
    out["confident_accuracy"] = float(np.mean(pred[conf] == y[conf].astype(bool))) if conf.any() else None

    warnings: List[str] = []
    if n < 30:
        warnings.append(f"Only {n} evaluation images: every metric below has a very wide error bar. "
                        f"Use more data for trustworthy numbers.")
    if out["n_real"] == 0 or out["n_ai"] == 0:
        warnings.append("Only one class present: AUC, recall/specificity are undefined.")
    out["warnings"] = warnings

    if n_boot and out["n_real"] > 0 and out["n_ai"] > 0:
        rng = np.random.default_rng(seed)
        ir, ia = np.where(y == 0)[0], np.where(y == 1)[0]
        accs, aucs = [], []
        for _ in range(n_boot):
            idx = np.concatenate([rng.choice(ir, len(ir)), rng.choice(ia, len(ia))])
            accs.append(float(np.mean(pred[idx] == y[idx].astype(bool))))
            a = safe_auc(y[idx], p[idx])
            if a is not None:
                aucs.append(a)
        out["ci95"] = {"accuracy": [float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))],
                       "roc_auc": ([float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))]
                                   if aucs else None)}
    return out
