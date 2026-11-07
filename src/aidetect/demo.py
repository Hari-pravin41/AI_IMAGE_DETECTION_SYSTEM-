"""Synthetic smoke-test dataset.

These images are procedurally generated and only exist so you can verify the whole
pipeline (train -> evaluate -> predict -> app) in a minute without downloading anything.
Scores obtained on them say NOTHING about performance on real AI images.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

_RESAMPLE = getattr(Image, "Resampling", Image)


def _fractal(rng: np.random.Generator, size: int, beta: float) -> np.ndarray:
    f = np.hypot(*np.meshgrid(np.fft.fftfreq(size), np.fft.fftfreq(size)))
    f[0, 0] = 1.0
    spec = (rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))) / f ** beta
    spec[0, 0] = 0.0
    a = np.fft.ifft2(spec).real
    a -= a.min()
    return a / max(a.max(), 1e-9)


def _scene(rng: np.random.Generator, size: int, beta: float) -> np.ndarray:
    lum = _fractal(rng, size, beta)
    chroma = np.stack([_fractal(rng, size, beta + 0.4) for _ in range(3)], -1)
    base = np.clip(0.75 * lum[..., None] + 0.25 * chroma, 0, 1)
    img = Image.fromarray((base * 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    for _ in range(int(rng.integers(3, 8))):
        x0, y0 = rng.integers(0, size - 8, 2)
        x1, y1 = x0 + rng.integers(8, size // 2), y0 + rng.integers(8, size // 2)
        color = tuple(int(c) for c in rng.integers(20, 235, 3))
        (draw.ellipse if rng.random() < 0.5 else draw.rectangle)([x0, y0, x1, y1], fill=color)
    return np.asarray(img, dtype=np.float64)


def _real_like(rng: np.random.Generator, size: int) -> np.ndarray:
    arr = _scene(rng, size, float(rng.uniform(1.0, 1.4)))
    arr = np.stack([ndimage.gaussian_filter(arr[..., c], float(rng.uniform(0.4, 0.9))) for c in range(3)], -1)
    shared = rng.normal(0, 0.4, (size, size, 1))
    std = np.sqrt(1.0 + 0.04 * arr)                       # signal-dependent "shot" noise
    return np.clip(arr + rng.normal(0, 1, arr.shape) * std + shared, 0, 255)


def _fake_like(rng: np.random.Generator, size: int) -> np.ndarray:
    scale = int(rng.choice([2, 4]))
    low = _scene(rng, size // scale, float(rng.uniform(1.4, 1.9)))
    im = Image.fromarray(np.clip(low, 0, 255).astype(np.uint8))
    if rng.random() < 0.5:
        up = im.resize((size, size), _RESAMPLE.NEAREST)
        arr = ndimage.gaussian_filter(np.asarray(up, dtype=np.float64), (0.5, 0.5, 0))
    else:
        arr = np.asarray(im.resize((size, size), _RESAMPLE.BILINEAR), dtype=np.float64)
    return np.clip(arr + rng.normal(0, 0.5, arr.shape), 0, 255)


def make_demo_dataset(out_dir: str, n_per_class: int = 60, size: int = 256, seed: int = 0) -> Dict[str, object]:
    if n_per_class < 6:
        raise ValueError("n_per_class must be at least 6")
    if size < 64 or size % 8:
        raise ValueError("size must be a multiple of 8 and at least 64")
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    for cls, maker in (("real", _real_like), ("fake", _fake_like)):
        (out / cls).mkdir(parents=True, exist_ok=True)
        for i in range(n_per_class):
            arr = maker(rng, size)
            q = int(rng.integers(88, 96))                  # identical JPEG settings for both classes
            buf = io.BytesIO()
            Image.fromarray(np.rint(arr).astype(np.uint8)).save(buf, format="JPEG", quality=q)
            (out / cls / f"{cls}_{i:04d}.jpg").write_bytes(buf.getvalue())
    return {"directory": str(out), "per_class": n_per_class, "size": size}
