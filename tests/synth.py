"""Synthetic image generator used ONLY by the test-suite / smoke test.

It does not imitate any real AI generator. It creates two statistically different image
families (camera-like: signal-dependent sensor noise; generator-like: up-sampled, noise-poor,
with a faint periodic grid) and saves *both* families as a random mix of JPEG and PNG so the
file format cannot act as a shortcut. Real-world accuracy must be measured on real data.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def _base(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.zeros((h, w, 3), np.float32) + rng.uniform(40, 200, 3)
    for _ in range(rng.integers(6, 14)):
        cy, cx = rng.uniform(0, h), rng.uniform(0, w)
        s = rng.uniform(0.08, 0.3) * max(h, w)
        g = np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * s * s)))
        img += g[..., None] * rng.uniform(-110, 110, 3)
    for _ in range(rng.integers(2, 5)):
        y0, x0 = int(rng.uniform(0, h * .7)), int(rng.uniform(0, w * .7))
        y1, x1 = y0 + int(rng.uniform(10, h * .4)), x0 + int(rng.uniform(10, w * .4))
        img[y0:y1, x0:x1] += rng.uniform(-50, 50, 3)
    return np.clip(img, 0, 255)


def _save(arr: np.ndarray, path: Path, rng: np.random.Generator) -> None:
    im = Image.fromarray(np.clip(np.rint(arr), 0, 255).astype(np.uint8))
    if rng.random() < 0.5:
        im.save(path.with_suffix(".jpg"), quality=int(rng.integers(80, 96)))
    else:
        im.save(path.with_suffix(".png"))


def make_real(rng, h, w, path):
    img = _base(rng, h, w)
    sigma = 1.0 + 0.03 * img.mean(axis=2, keepdims=True)           # signal-dependent noise
    shared = rng.normal(0, 1, (h, w, 1)) * sigma
    img = img + shared * 0.7 + rng.normal(0, 1, img.shape) * sigma * 0.7
    _save(img, path, rng)


def make_ai(rng, h, w, path, kind="a"):
    f = 2 if kind == "a" else 4
    small = _base(rng, h // f + 1, w // f + 1)
    up = np.asarray(Image.fromarray(small.astype(np.uint8)).resize((w, h), Image.BICUBIC), dtype=np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    up += (1.5 if kind == "a" else 0.8) * np.sin(2 * np.pi * xx / 8)[..., None] * np.cos(2 * np.pi * yy / 8)[..., None]
    up += rng.normal(0, 0.4, up.shape)
    _save(up, path, rng)


def make_dataset(root: Path, n_per_class: int = 40, seed: int = 0, size=(160, 240)) -> Path:
    rng = np.random.default_rng(seed)
    (root / "real").mkdir(parents=True, exist_ok=True)
    for kind in ("a", "b"):
        (root / "fake" / f"gen_{kind}").mkdir(parents=True, exist_ok=True)
    for i in range(n_per_class):
        h, w = (int(rng.integers(size[0], size[1])) for _ in range(2))
        make_real(rng, h, w, root / "real" / f"r{i:04d}")
        kind = "a" if i % 2 == 0 else "b"
        make_ai(rng, h, w, root / "fake" / f"gen_{kind}" / f"f{i:04d}", kind)
    return root
