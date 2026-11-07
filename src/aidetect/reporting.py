"""Evaluation plots and a self-contained HTML report."""
from __future__ import annotations

import base64
import html
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .metrics import reliability_curve


def _plt():
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    return plt


def save_evaluation_plots(out_dir: Path, y: np.ndarray, p: np.ndarray, threshold: float,
                          group_importance: Optional[Dict[str, float]] = None) -> Dict[str, str]:
    """Write ROC, confusion matrix, score histogram, reliability diagram and group importance PNGs."""
    from sklearn.metrics import roc_curve
    plt = _plt()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    files: Dict[str, str] = {}

    def save(fig, name: str) -> None:
        fig.tight_layout()
        fig.savefig(out_dir / name, dpi=130)
        plt.close(fig)
        files[name.rsplit(".", 1)[0]] = name

    if len(np.unique(y)) == 2:
        fpr, tpr, _ = roc_curve(y, p)
        fig, ax = plt.subplots(figsize=(4.6, 4.2))
        ax.plot(fpr, tpr, lw=2)
        ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
        ax.set_xlabel("False positive rate (real flagged as AI)")
        ax.set_ylabel("True positive rate (AI caught)")
        ax.set_title("ROC curve")
        save(fig, "roc.png")

    pred = p >= threshold
    cm = np.array([[np.sum((y == 0) & ~pred), np.sum((y == 0) & pred)],
                   [np.sum((y == 1) & ~pred), np.sum((y == 1) & pred)]])
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    ax.set_xticks([0, 1], ["pred real", "pred AI"])
    ax.set_yticks([0, 1], ["true real", "true AI"])
    ax.set_title(f"Confusion matrix @ threshold {threshold:.2f}")
    save(fig, "confusion.png")

    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    bins = np.linspace(0, 1, 21)
    ax.hist(p[y == 0], bins=bins, alpha=0.65, label="real")
    ax.hist(p[y == 1], bins=bins, alpha=0.65, label="AI-generated")
    ax.axvline(threshold, color="k", ls="--", lw=1)
    ax.set_xlabel("P(AI-generated)")
    ax.set_ylabel("images")
    ax.set_title("Score distribution")
    ax.legend()
    save(fig, "scores.png")

    rc = reliability_curve(y, p, bins=10)
    if rc["mean_predicted"]:
        fig, ax = plt.subplots(figsize=(4.4, 4.2))
        ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
        ax.plot(rc["mean_predicted"], rc["fraction_positive"], "o-", lw=2)
        ax.set_xlabel("predicted probability")
        ax.set_ylabel("observed fraction AI")
        ax.set_title("Reliability (calibration)")
        save(fig, "calibration.png")

    if group_importance:
        items = sorted(group_importance.items(), key=lambda kv: kv[1])
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        ax.barh([k for k, _ in items], [v for _, v in items])
        ax.set_xlabel("drop in patch-level AUC when the group is shuffled")
        ax.set_title("Which signals the model relies on")
        save(fig, "signals.png")
    return files


def _fmt(v: Any, pct: bool = False) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v * 100:.1f}%" if pct else f"{v:.3f}"
    return html.escape(str(v))


def write_html_report(path: Path, title: str, metrics: Dict[str, Any], plots_dir: Path,
                      plots: Dict[str, str], extra_sections: Optional[List[str]] = None,
                      notes: Optional[List[str]] = None, robustness: Optional[List[Dict[str, Any]]] = None) -> Path:
    def img(name: str) -> str:
        f = Path(plots_dir) / name
        if not f.is_file():
            return ""
        b64 = base64.b64encode(f.read_bytes()).decode()
        return f'<img src="data:image/png;base64,{b64}" style="max-width:100%;height:auto">'

    rows = [("Images evaluated", f"{metrics['n']} ({metrics['n_real']} real, {metrics['n_ai']} AI)"),
            ("Accuracy", _fmt(metrics.get("accuracy"), True)),
            ("Balanced accuracy", _fmt(metrics.get("balanced_accuracy"), True)),
            ("ROC AUC", _fmt(metrics.get("roc_auc"))),
            ("Recall (AI caught)", _fmt(metrics.get("recall"), True)),
            ("Specificity (real kept)", _fmt(metrics.get("specificity"), True)),
            ("Precision", _fmt(metrics.get("precision"), True)),
            ("F1", _fmt(metrics.get("f1"))),
            ("Brier score", _fmt(metrics.get("brier"))),
            ("Calibration error (ECE)", _fmt(metrics.get("ece"))),
            ("Threshold", _fmt(metrics.get("threshold"))),
            ("Uncertain fraction", _fmt(metrics.get("uncertain_fraction"), True)),
            ("Accuracy on confident calls", _fmt(metrics.get("confident_accuracy"), True))]
    ci = metrics.get("ci95")
    if ci:
        rows.append(("95% CI accuracy", f"{ci['accuracy'][0] * 100:.1f}% - {ci['accuracy'][1] * 100:.1f}%"))
        if ci.get("roc_auc"):
            rows.append(("95% CI ROC AUC", f"{ci['roc_auc'][0]:.3f} - {ci['roc_auc'][1]:.3f}"))
    table = "".join(f"<tr><th>{html.escape(k)}</th><td>{v}</td></tr>" for k, v in rows)
    warn = "".join(f"<li>{html.escape(w)}</li>" for w in (metrics.get("warnings", []) + (notes or [])))
    rob = ""
    if robustness:
        body = "".join(
            f"<tr><td>{html.escape(str(r['condition']))}</td><td>{r['n_ok']}</td><td>{r['n_failed']}</td>"
            f"<td>{_fmt(r.get('accuracy'), True)}</td><td>{_fmt(r.get('balanced_accuracy'), True)}</td>"
            f"<td>{_fmt(r.get('roc_auc'))}</td></tr>" for r in robustness)
        rob = ("<h2>Robustness to common edits</h2><table><tr><th>Condition</th><th>OK</th><th>Failed</th>"
               f"<th>Accuracy</th><th>Balanced acc.</th><th>AUC</th></tr>{body}</table>")
    gallery = "".join(f"<div>{img(n)}</div>" for n in plots.values())
    doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>body{{font-family:system-ui,sans-serif;max-width:980px;margin:2rem auto;padding:0 1rem;color:#222}}
table{{border-collapse:collapse}}th,td{{border:1px solid #ccc;padding:.35rem .7rem;text-align:left}}
th{{background:#f4f4f4}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:1rem}}
.warn{{background:#fff7e0;border:1px solid #f0d890;padding:.5rem 1rem;border-radius:6px}}</style></head><body>
<h1>{html.escape(title)}</h1>
{('<div class="warn"><b>Read this first</b><ul>' + warn + '</ul></div>') if warn else ''}
<h2>Metrics</h2><table>{table}</table>{rob}
<h2>Plots</h2><div class="grid">{gallery}</div>
{''.join(extra_sections or [])}
</body></html>"""
    path = Path(path)
    path.write_text(doc, encoding="utf-8")
    return path
