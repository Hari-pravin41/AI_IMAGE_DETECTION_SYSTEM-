"""Image edits used to test how robust a detector is to everyday processing."""
from __future__ import annotations

import io
from typing import Any, Dict

import numpy as np
from PIL import Image, ImageFilter

_LANCZOS = getattr(getattr(Image, "Resampling", Image), "LANCZOS")


def describe(spec: Dict[str, Any]) -> str:
    t = spec["type"]
    return {"jpeg": f"jpeg_q{int(spec.get('quality', 0))}", "resize": f"resize_x{spec.get('scale')}",
            "blur": f"blur_r{spec.get('radius')}", "noise": f"noise_s{spec.get('sigma')}"}[t]


def apply_perturbation(img: Image.Image, spec: Dict[str, Any], rng: np.random.Generator) -> Image.Image:
    t = spec["type"]
    img = img.convert("RGB")
    if t == "jpeg":
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=int(spec["quality"]))
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    if t == "resize":
        s = float(spec["scale"])
        size = (max(1, int(round(img.width * s))), max(1, int(round(img.height * s))))
        return img.resize(size, _LANCZOS)
    if t == "blur":
        return img.filter(ImageFilter.GaussianBlur(radius=float(spec["radius"])))
    if t == "noise":
        a = np.asarray(img, dtype=np.float32) + rng.normal(0.0, float(spec["sigma"]), (img.height, img.width, 3))
        return Image.fromarray(np.clip(np.rint(a), 0, 255).astype(np.uint8))
    raise ValueError(f"unknown perturbation type '{t}'")
