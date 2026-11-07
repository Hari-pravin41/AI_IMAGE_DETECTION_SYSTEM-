"""Visual evidence panels: localisation heatmap, Fourier spectrum, noise residual, ELA map."""
from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image

from .detector import Prediction
from .loader import LoadedImage

_MAX_VIEW = 512


def _png(rgb: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return buf.tobytes()


def _center(rgb: np.ndarray, size: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    y, x = max((h - size) // 2, 0) // 8 * 8, max((w - size) // 2, 0) // 8 * 8
    return np.ascontiguousarray(rgb[y:y + size, x:x + size])


def heatmap_panel(img: LoadedImage, pred: Prediction) -> bytes | None:
    if not pred.crop_scores:
        return None
    h, w = img.rgb.shape[:2]
    s = min(1.0, _MAX_VIEW / max(h, w))
    vw, vh = max(int(round(w * s)), 1), max(int(round(h * s)), 1)
    base = cv2.resize(img.rgb, (vw, vh), interpolation=cv2.INTER_AREA).astype(np.float32)
    acc, cnt = np.zeros((vh, vw), np.float32), np.zeros((vh, vw), np.float32)
    for c in pred.crop_scores:
        y0, x0 = int(c["y"] * s), int(c["x"] * s)
        y1, x1 = min(int((c["y"] + c["size"]) * s) + 1, vh), min(int((c["x"] + c["size"]) * s) + 1, vw)
        acc[y0:y1, x0:x1] += c["prob"]
        cnt[y0:y1, x0:x1] += 1
    covered = cnt > 0
    prob = np.where(covered, acc / np.maximum(cnt, 1), 0)
    color = cv2.cvtColor(cv2.applyColorMap((prob * 255).astype(np.uint8), cv2.COLORMAP_JET),
                         cv2.COLOR_BGR2RGB).astype(np.float32)
    alpha = np.where(covered, 0.5, 0.0)[..., None]
    return _png(np.clip(base * (1 - alpha) + color * alpha, 0, 255).astype(np.uint8))


def spectrum_panel(img: LoadedImage) -> bytes:
    h, w = img.rgb.shape[:2]
    size = max(min(h, w, 512) // 16 * 16, 64)
    g = cv2.cvtColor(_center(img.rgb, size), cv2.COLOR_RGB2GRAY).astype(np.float32)
    win = np.outer(np.hanning(size), np.hanning(size)).astype(np.float32)
    P = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2((g - g.mean()) * win))))
    lo, hi = np.percentile(P, 1), np.percentile(P, 99.9)
    norm = np.clip((P - lo) / max(hi - lo, 1e-6), 0, 1)
    col = cv2.cvtColor(cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_MAGMA), cv2.COLOR_BGR2RGB)
    return _png(cv2.resize(col, (size * 2, size * 2), interpolation=cv2.INTER_NEAREST) if size < 256 else col)


def noise_panel(img: LoadedImage) -> bytes:
    region = _center(img.rgb, min(min(img.rgb.shape[:2]), _MAX_VIEW))
    res = region.astype(np.float32) - cv2.medianBlur(region, 3).astype(np.float32)
    scale = 110.0 / max(float(np.percentile(np.abs(res), 99.5)), 1.0)
    return _png(np.clip(128 + res * scale, 0, 255).astype(np.uint8))


def ela_panel(img: LoadedImage, quality: int = 90) -> bytes:
    region = _center(img.rgb, min(min(img.rgb.shape[:2]), _MAX_VIEW))
    buf = io.BytesIO()
    Image.fromarray(region).save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    re = np.asarray(Image.open(buf).convert("RGB"), dtype=np.float32)
    d = np.abs(region.astype(np.float32) - re)
    scale = 255.0 / max(float(np.percentile(d, 99.5)), 1.0)
    return _png(np.clip(d * scale, 0, 255).astype(np.uint8))


def render_panels(img: LoadedImage, pred: Prediction) -> dict[str, bytes]:
    panels = {"spectrum": spectrum_panel(img), "noise_residual": noise_panel(img), "ela": ela_panel(img)}
    hm = heatmap_panel(img, pred)
    if hm:
        panels = {"heatmap": hm, **panels}
    return panels
