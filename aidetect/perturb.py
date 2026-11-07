"""Image degradations used for training augmentation and robustness evaluation."""
from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageFilter


def _jpeg(img: Image.Image, quality: int) -> tuple[Image.Image, bytes]:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(quality))
    data = buf.getvalue()
    return Image.open(io.BytesIO(data)).convert("RGB"), data


def apply_ops(rgb: np.ndarray, ops: list[dict], rng: np.random.Generator, min_side: int) -> bytes:
    """Apply ``ops`` to an RGB array and return the encoded result (JPEG if the last op is jpeg,
    otherwise lossless PNG), so the result can be re-loaded through the normal loader."""
    img = Image.fromarray(rgb)
    last_jpeg: bytes | None = None
    for op in ops:
        kind = op["op"]
        last_jpeg = None
        if kind == "jpeg":
            img, last_jpeg = _jpeg(img, op["quality"])
        elif kind == "resize":
            w, h = img.size
            scale = max(float(op["scale"]), min_side / min(w, h))   # never fall below min_side
            if scale < 0.999:
                img = img.resize((max(int(round(w * scale)), 1), max(int(round(h * scale)), 1)),
                                 Image.LANCZOS)
        elif kind == "blur":
            img = img.filter(ImageFilter.GaussianBlur(float(op["sigma"])))
        elif kind == "noise":
            arr = np.asarray(img, dtype=np.float32)
            arr = arr + rng.normal(0.0, float(op["sigma"]), arr.shape).astype(np.float32)
            img = Image.fromarray(np.clip(np.rint(arr), 0, 255).astype(np.uint8))
        else:
            raise ValueError(f"unknown perturbation op {kind!r}")
    if last_jpeg is not None:
        return last_jpeg
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def sample_ops(aug: dict, rng: np.random.Generator) -> list[dict]:
    """Randomly sample a degradation chain from the ranges in ``Config.augment``."""
    p = float(aug["prob_each"])
    ops: list[dict] = []
    if rng.random() < p:
        ops.append({"op": "resize", "scale": float(rng.uniform(*aug["scale"]))})
    if rng.random() < p:
        ops.append({"op": "blur", "sigma": float(rng.uniform(*aug["blur_sigma"]))})
    if rng.random() < p:
        ops.append({"op": "noise", "sigma": float(rng.uniform(*aug["noise_sigma"]))})
    if rng.random() < p or not ops:          # always degrade at least once
        lo, hi = aug["jpeg_quality"]
        ops.append({"op": "jpeg", "quality": int(rng.integers(int(lo), int(hi) + 1))})
    return ops
