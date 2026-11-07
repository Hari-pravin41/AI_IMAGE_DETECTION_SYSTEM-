"""Metrics, per-source breakdowns and report plots."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, log_loss,
                             precision_recall_curve, roc_auc_score, roc_curve)


def _tpr_at_fpr(fpr: np.ndarray, tpr: np.ndarray, target: float) -> float:
    return float(np.interp(target, fpr, tpr))


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            tot += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(tot)


def compute_metrics(y: np.ndarray, p: np.ndarray, thr: float, margin: float | None = None) -> dict:
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    pred = (p >= thr).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    n = len(y)
    rec = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    out = {
        "n": n, "n_ai": int(y.sum()), "n_real": int(n - y.sum()), "threshold": float(thr),
        "accuracy": (tp + tn) / max(n, 1), "balanced_accuracy": (rec + spec) / 2,
        "precision": prec, "recall_ai": rec, "specificity_real": spec,
        "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }
    if len(np.unique(y)) == 2:
        fpr, tpr, _ = roc_curve(y, p)
        out.update({"roc_auc": float(roc_auc_score(y, p)), "average_precision": float(average_precision_score(y, p)),
                    "tpr_at_1pct_fpr": _tpr_at_fpr(fpr, tpr, 0.01), "tpr_at_5pct_fpr": _tpr_at_fpr(fpr, tpr, 0.05),
                    "brier": float(brier_score_loss(y, p)),
                    "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])),
                    "ece": ece(y, p)})
    if margin is not None:
        keep = np.abs(p - thr) >= margin
        out["uncertain_band"] = {"margin": float(margin), "coverage": float(keep.mean()),
                                 "accuracy_when_decided": float((pred[keep] == y[keep]).mean()) if keep.any() else None}
    return out


def per_source(y: np.ndarray, p: np.ndarray, sources: list[str], thr: float) -> dict:
    out = {}
    pred = (np.asarray(p) >= thr).astype(int)
    for s in sorted(set(sources)):
        m = np.array([x == s for x in sources])
        unique_labels = np.unique(y[m])
        label = "mixed" if len(unique_labels) > 1 else ("ai" if unique_labels[0] == 1 else "real")
        out[s] = {"n": int(m.sum()), "label": label,
                  "accuracy": float((pred[m] == y[m]).mean()),
                  "mean_prob_ai": float(np.asarray(p)[m].mean())}
    return out


def plot_report(y: np.ndarray, p: np.ndarray, thr: float, path: Path, title: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y = np.asarray(y).astype(int)
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))
    fig.suptitle(title or "Evaluation", fontsize=14)
    fpr, tpr, _ = roc_curve(y, p)
    ax[0, 0].plot(fpr, tpr, lw=2, label=f"AUC={roc_auc_score(y, p):.3f}")
    ax[0, 0].plot([0, 1], [0, 1], "--", c="gray")
    ax[0, 0].set(title="ROC", xlabel="false-positive rate", ylabel="true-positive rate")
    ax[0, 0].legend(loc="lower right")
    pr, rc, _ = precision_recall_curve(y, p)
    ax[0, 1].plot(rc, pr, lw=2, label=f"AP={average_precision_score(y, p):.3f}")
    ax[0, 1].set(title="Precision / recall (AI = positive)", xlabel="recall", ylabel="precision")
    ax[0, 1].legend(loc="lower left")
    cm = confusion_matrix(y, (p >= thr).astype(int), labels=[0, 1])
    ax[0, 2].imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax[0, 2].text(j, i, str(v), ha="center", va="center", fontsize=16,
                      color="white" if v > cm.max() / 2 else "black")
    ax[0, 2].set(title=f"Confusion @ threshold {thr:.3f}", xticks=[0, 1], yticks=[0, 1],
                 xticklabels=["real", "AI"], yticklabels=["real", "AI"], xlabel="predicted", ylabel="actual")
    bins = np.linspace(0, 1, 21)
    ax[1, 0].hist(p[y == 0], bins, alpha=.6, label="real", color="#2b8a3e")
    ax[1, 0].hist(p[y == 1], bins, alpha=.6, label="AI", color="#c92a2a")
    ax[1, 0].axvline(thr, c="k", ls="--")
    ax[1, 0].set(title="Score distribution", xlabel="P(AI)", ylabel="images")
    ax[1, 0].legend()
    edges = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, 9)
    xs = [p[idx == b].mean() for b in range(10) if (idx == b).any()]
    ys = [y[idx == b].mean() for b in range(10) if (idx == b).any()]
    ax[1, 1].plot([0, 1], [0, 1], "--", c="gray")
    ax[1, 1].plot(xs, ys, "o-")
    ax[1, 1].set(title="Calibration", xlabel="predicted P(AI)", ylabel="observed fraction AI")
    ax[1, 2].axis("off")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
