"""Robust image loading: any readable format -> 8-bit RGB array plus file-level facts."""
from __future__ import annotations

import io
import os
import struct
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import ExifTags, Image, UnidentifiedImageError

from .config import Config
from .provenance import scan as scan_provenance


class ImageError(ValueError):
    """Raised when a file cannot be used as analysis input (message is user-facing)."""


@dataclass
class LoadedImage:
    rgb: np.ndarray                       # (H, W, 3) uint8
    name: str
    format: str | None
    mode: str
    size: tuple[int, int]                 # (width, height)
    n_bytes: int | None
    has_alpha: bool
    is_grayscale: bool
    icc: bool
    exif: dict[str, Any] = field(default_factory=dict)
    quant_tables: list[list[int]] | None = None
    provenance: dict = field(default_factory=dict)


def _exif_dict(img: Image.Image) -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        ex = img.getexif()
        items = dict(ex.items())
        try:
            items.update(ex.get_ifd(0x8769))
        except Exception:  # noqa: BLE001 - sub-IFD is optional
            pass
        for tag, val in items.items():
            name = ExifTags.TAGS.get(tag, str(tag))
            if isinstance(val, bytes):
                try:
                    val = val.decode("utf-8", "ignore").strip("\x00 ")
                except Exception:  # noqa: BLE001
                    continue
            if isinstance(val, (str, int, float)) or hasattr(val, "numerator"):
                out[name] = val if isinstance(val, str) else str(val)
    except Exception:  # noqa: BLE001 - damaged EXIF must never block analysis
        pass
    return out


def _to_rgb(img: Image.Image) -> np.ndarray:
    mode = img.mode
    if mode in ("I;16", "I;16L", "I;16B", "I;16N", "I", "F"):
        arr = np.asarray(img).astype(np.float64)
        if mode.startswith("I;16"):
            arr = arr / 257.0
        else:
            lo, hi = float(arr.min()), float(arr.max())
            arr = (arr - lo) / (hi - lo) * 255.0 if hi > lo else np.zeros_like(arr)
        g = np.clip(np.rint(arr), 0, 255).astype(np.uint8)
        return np.ascontiguousarray(np.stack([g, g, g], axis=-1))
    return np.ascontiguousarray(np.asarray(img.convert("RGB"), dtype=np.uint8))


def load_image(source, cfg: Config, name: str | None = None, check_size: bool = True) -> LoadedImage:
    """Load a path / bytes / PIL image. Raises :class:`ImageError` with a clear reason."""
    raw: bytes | None = None
    pil: Image.Image | None = None
    if isinstance(source, (str, os.PathLike)):
        p = Path(source)
        name = name or p.name
        try:
            size = p.stat().st_size
        except OSError as e:
            raise ImageError(f"cannot read file: {e.strerror or e}") from e
        if size == 0:
            raise ImageError("file is empty")
        if size > cfg.max_file_mb * 1024 * 1024:
            raise ImageError(f"file larger than {cfg.max_file_mb:g} MB")
        try:
            raw = p.read_bytes()
        except OSError as e:
            raise ImageError(f"cannot read file: {e.strerror or e}") from e
    elif isinstance(source, (bytes, bytearray, memoryview)):
        raw = bytes(source)
        name = name or "upload"
        if not raw:
            raise ImageError("file is empty")
        if len(raw) > cfg.max_file_mb * 1024 * 1024:
            raise ImageError(f"file larger than {cfg.max_file_mb:g} MB")
    elif isinstance(source, Image.Image):
        pil = source
        name = name or "image"
    else:
        raise ImageError(f"unsupported input type: {type(source).__name__}")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            img = pil if pil is not None else Image.open(io.BytesIO(raw))
            fmt, (w, h) = img.format, img.size
            if w * h > cfg.max_pixels:
                raise ImageError(f"image has {w * h / 1e6:.0f} MP, limit is {cfg.max_pixels / 1e6:.0f} MP")
            info = {}
            for k, v in img.info.items():
                if isinstance(v, str):
                    info[str(k)] = v[:65536]
                elif isinstance(v, (int, float)):
                    info[str(k)] = str(v)
            icc = bool(img.info.get("icc_profile"))
            quant = getattr(img, "quantization", None)
            quant_tables = [list(map(int, t)) for t in quant.values()] if quant else None
            exif = _exif_dict(img)
            img.load()
            mode = img.mode
            has_alpha = mode in ("RGBA", "LA", "PA") or "transparency" in img.info
            rgb = _to_rgb(img)
    except ImageError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError, EOFError, struct.error,
            Image.DecompressionBombError, MemoryError) as e:
        raise ImageError(f"not a readable image ({type(e).__name__}: {str(e)[:120]})") from e

    h, w = rgb.shape[:2]
    if check_size and min(h, w) < cfg.effective_min_side:
        raise ImageError(f"image is {w}x{h}px; the shortest side must be >= {cfg.effective_min_side}px")

    is_gray = bool(np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 1], rgb[..., 2]))
    prov = scan_provenance(raw, fmt, info, exif)
    return LoadedImage(rgb=rgb, name=name or "image", format=fmt, mode=mode, size=(w, h),
                       n_bytes=len(raw) if raw is not None else None, has_alpha=has_alpha,
                       is_grayscale=is_gray, icc=icc, exif=exif, quant_tables=quant_tables,
                       provenance=prov)
