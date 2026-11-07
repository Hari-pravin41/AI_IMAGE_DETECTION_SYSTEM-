"""Robust image loading.

Handles paths, raw bytes, file-like objects and PIL images; EXIF orientation,
transparency, palette / CMYK / 16-bit / float modes, animated files (first
frame), truncated or corrupt files and decompression bombs. Everything that
can go wrong is reported as an :class:`ImageError` with a readable reason so
batch jobs keep going and the user knows exactly which file failed and why.
"""
from __future__ import annotations

import hashlib
import io
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

import numpy as np
from PIL import ExifTags, Image, ImageOps

_LANCZOS = getattr(getattr(Image, "Resampling", Image), "LANCZOS")

Source = Union[str, os.PathLike, bytes, bytearray, io.IOBase, Image.Image]


class ImageError(Exception):
    """The image could not be read or is unsuitable for analysis."""


@dataclass
class LoadedImage:
    rgb: np.ndarray                      # uint8, H x W x 3
    info: Dict[str, Any]                 # format, size, mode, ...
    raw_meta: Dict[str, Any]             # EXIF / PNG text / XMP (for provenance hints)
    sha256: Optional[str] = None
    has_content_credentials: bool = False
    warnings: list = field(default_factory=list)

    @property
    def height(self) -> int:
        return int(self.rgb.shape[0])

    @property
    def width(self) -> int:
        return int(self.rgb.shape[1])


# --------------------------------------------------------------------------- #
def load_image(source: Source, *, max_pixels: int = 120_000_000, min_side: int = 0,
               small_image_policy: str = "skip", recompress_jpeg_quality: Optional[int] = None
               ) -> LoadedImage:
    data: Optional[bytes] = None
    file_size: Optional[int] = None
    warnings: list = []

    if isinstance(source, Image.Image):
        img = source
    else:
        data = _read_bytes(source)
        file_size = len(data)
        if file_size == 0:
            raise ImageError("file is empty")
        try:
            img = Image.open(io.BytesIO(data))
        except Exception as e:  # PIL raises several unrelated types here
            raise ImageError(f"not a readable image ({_short(e)})") from e

    try:
        width, height = img.size
        if width < 1 or height < 1:
            raise ImageError("image has zero size")
        if width * height > max_pixels:
            raise ImageError(f"image too large ({width}x{height} = {width * height / 1e6:.0f} MP, "
                             f"limit {max_pixels / 1e6:.0f} MP)")
        fmt = img.format
        mode = img.mode
        n_frames = int(getattr(img, "n_frames", 1) or 1)
        if n_frames > 1:
            img.seek(0)
            warnings.append(f"animated image with {n_frames} frames: only the first frame is analysed")
        try:
            img.load()
        except Exception as e:
            raise ImageError(f"corrupt or truncated image ({_short(e)})") from e

        raw_meta = _collect_metadata(img)
        rgb_img = _to_rgb(img)
    except ImageError:
        raise
    except Exception as e:
        raise ImageError(f"could not decode image ({_short(e)})") from e

    info: Dict[str, Any] = {"format": fmt, "mode": mode, "width": width, "height": height,
                            "n_frames": n_frames, "file_size": file_size}

    if rgb_img.width != width or rgb_img.height != height:   # EXIF transpose swapped axes
        info["width"], info["height"] = rgb_img.width, rgb_img.height

    if min_side and min(rgb_img.size) < min_side:
        if small_image_policy == "upscale":
            scale = min_side / float(min(rgb_img.size))
            new = (max(min_side, int(round(rgb_img.width * scale))),
                   max(min_side, int(round(rgb_img.height * scale))))
            rgb_img = rgb_img.resize(new, _LANCZOS)
            info["upscaled_from"] = (info["width"], info["height"])
            warnings.append(f"image smaller than the analysis patch ({info['width']}x{info['height']}); "
                            f"upscaled, so the result is less reliable")
        else:
            raise ImageError(f"image too small ({info['width']}x{info['height']}); "
                             f"the model needs at least {min_side}px on each side")

    if recompress_jpeg_quality:
        buf = io.BytesIO()
        rgb_img.save(buf, format="JPEG", quality=int(recompress_jpeg_quality))
        buf.seek(0)
        rgb_img = Image.open(buf).convert("RGB")

    arr = np.ascontiguousarray(np.asarray(rgb_img, dtype=np.uint8))
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ImageError("unexpected pixel layout after conversion")

    return LoadedImage(
        rgb=arr, info=info, raw_meta=raw_meta,
        sha256=hashlib.sha256(data).hexdigest() if data is not None else None,
        has_content_credentials=bool(data is not None and (b"c2pa" in data or b"jumb" in data)),
        warnings=warnings,
    )


def dhash64(rgb: np.ndarray) -> int:
    """64-bit difference hash used to find near-duplicate images."""
    small = Image.fromarray(rgb).convert("L").resize((9, 8), _LANCZOS)
    a = np.asarray(small, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).ravel()
    out = 0
    for b in bits:
        out = (out << 1) | int(b)
    return out


def to_gray(rgb: np.ndarray) -> np.ndarray:
    """BT.601 luma in 0..255 as float64."""
    a = rgb.astype(np.float64)
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


# --------------------------------------------------------------------------- #
def _read_bytes(source: Any) -> bytes:
    if isinstance(source, (bytes, bytearray)):
        return bytes(source)
    if isinstance(source, (str, os.PathLike)):
        path = os.fspath(source)
        if not os.path.exists(path):
            raise ImageError("file not found")
        if os.path.isdir(path):
            raise ImageError("path is a directory, not an image")
        try:
            with open(path, "rb") as fh:
                return fh.read()
        except OSError as e:
            raise ImageError(f"cannot read file ({_short(e)})") from e
    if hasattr(source, "read"):
        try:
            if hasattr(source, "seek"):
                source.seek(0)
            raw = source.read()
        except Exception as e:
            raise ImageError(f"cannot read stream ({_short(e)})") from e
        if isinstance(raw, str):
            raise ImageError("stream must be opened in binary mode")
        return bytes(raw)
    raise ImageError(f"unsupported image source type: {type(source).__name__}")


def _to_rgb(img: Image.Image) -> Image.Image:
    img = ImageOps.exif_transpose(img) or img
    mode = img.mode
    if mode in ("RGBA", "LA", "PA", "RGBa", "La") or (mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(bg, rgba).convert("RGB")
    if mode in ("I;16", "I;16L", "I;16B", "I;16N", "I", "F"):
        arr = np.asarray(img).astype(np.float64)
        top = float(arr.max()) if arr.size else 0.0
        if mode == "F":
            arr = arr * 255.0 if top <= 1.0 else arr
        elif top > 255.0:
            arr = arr / 257.0 if top <= 65535.0 else arr * (255.0 / top)
        gray = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="L")
        return gray.convert("RGB")
    return img.convert("RGB")


_EXIF_KEEP = {"Make", "Model", "Software", "DateTime", "DateTimeOriginal", "Artist",
              "ImageDescription", "LensModel", "ExposureTime", "FNumber", "ISOSpeedRatings",
              "PhotographicSensitivity", "FocalLength", "UserComment", "Copyright",
              "HostComputer", "ProcessingSoftware"}


def _collect_metadata(img: Image.Image) -> Dict[str, Any]:
    meta: Dict[str, Any] = {"exif": {}, "text": {}, "xmp": ""}
    try:
        for k, v in (img.info or {}).items():
            if isinstance(v, str):
                meta["text"][str(k)] = v[:4000]
            elif isinstance(v, bytes) and str(k).lower() in ("xmp", "xml:com.adobe.xmp"):
                meta["xmp"] = v[:200_000].decode("utf-8", "ignore")
    except Exception:
        pass
    try:
        exif = img.getexif()
        merged = dict(exif.items())
        try:
            merged.update(exif.get_ifd(0x8769))      # Exif sub-IFD (lens, ISO, exposure ...)
        except Exception:
            pass
        for tag, val in merged.items():
            name = ExifTags.TAGS.get(tag, str(tag))
            if name not in _EXIF_KEEP:
                continue
            if isinstance(val, bytes):
                val = val.decode("utf-8", "ignore").replace("\x00", "").strip()
            meta["exif"][name] = str(val)[:500]
    except Exception:
        pass
    return meta


def _short(e: BaseException) -> str:
    s = str(e).strip().replace("\n", " ")
    return (s[:120] + "...") if len(s) > 120 else (s or type(e).__name__)
