"""Patch-score heat-map overlay."""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
from PIL import Image

_LANCZOS = getattr(getattr(Image, "Resampling", Image), "LANCZOS")


def render_heatmap(rgb: np.ndarray, positions: np.ndarray, probs: Sequence[float], patch: int,
                   max_side: int = 1024, alpha: float = 0.5) -> Tuple[Image.Image, float]:
    """Overlay per-patch P(AI) on the image (green = looks real, red = looks AI-generated).
    Overlapping patches are averaged; areas that were not analysed stay untouched.
    Returns (overlay image, fraction of the image area that was analysed)."""
    import matplotlib
    h, w = rgb.shape[:2]
    f = min(1.0, max_side / float(max(h, w)))
    H, W = max(1, int(round(h * f))), max(1, int(round(w * f)))
    acc = np.zeros((H, W), dtype=np.float64)
    cnt = np.zeros((H, W), dtype=np.float64)
    for (y, x), p in zip(np.asarray(positions), probs):
        y0, x0 = int(round(y * f)), int(round(x * f))
        y1, x1 = max(y0 + 1, int(round((y + patch) * f))), max(x0 + 1, int(round((x + patch) * f)))
        acc[y0:y1, x0:x1] += float(p)
        cnt[y0:y1, x0:x1] += 1.0
    avg = np.where(cnt > 0, acc / np.maximum(cnt, 1.0), 0.0)
    base = Image.fromarray(rgb)
    if (W, H) != (w, h):
        base = base.resize((W, H), _LANCZOS)
    rgba = (matplotlib.colormaps["RdYlGn_r"](np.clip(avg, 0, 1)) * 255).astype(np.uint8)
    rgba[..., 3] = np.where(cnt > 0, int(alpha * 255), 0).astype(np.uint8)
    out = Image.alpha_composite(base.convert("RGBA"), Image.fromarray(rgba, "RGBA")).convert("RGB")
    return out, float(np.mean(cnt > 0))
