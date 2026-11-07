"""Provenance hints from embedded metadata.

These are reported *next to* the classifier verdict and never mixed into it:
metadata is trivially stripped or forged, so absence proves nothing and
presence is only a hint.
"""
from __future__ import annotations

from typing import Any, Dict, List

from .config import ProvenanceConfig


def inspect_provenance(raw_meta: Dict[str, Any], has_content_credentials: bool,
                       cfg: ProvenanceConfig) -> Dict[str, Any]:
    exif: Dict[str, str] = raw_meta.get("exif", {}) or {}
    text: Dict[str, str] = raw_meta.get("text", {}) or {}
    xmp: str = raw_meta.get("xmp", "") or ""

    camera = " ".join(x for x in (exif.get("Make"), exif.get("Model")) if x).strip()
    software = exif.get("Software") or exif.get("ProcessingSoftware") or ""
    has_capture_params = any(k in exif for k in ("ExposureTime", "FNumber", "ISOSpeedRatings",
                                                 "PhotographicSensitivity", "LensModel"))

    haystacks: Dict[str, str] = {f"EXIF {k}": v for k, v in exif.items()}
    haystacks.update({f"text chunk '{k}'": v for k, v in text.items()})
    if xmp:
        haystacks["XMP"] = xmp

    hints: List[str] = []
    for where, value in haystacks.items():
        low = value.lower()
        for kw in cfg.generator_keywords:
            if kw.lower() in low:
                hints.append(f"{where} mentions '{kw}'")
                break

    wanted = {k.lower() for k in cfg.generation_metadata_keys}
    gen_keys = sorted(k for k in text if k.lower() in wanted)
    for k in gen_keys:
        v = text[k].lower()
        if k.lower() in ("parameters", "prompt", "workflow", "negative_prompt", "sd-metadata",
                         "invokeai_metadata", "dream") or any(s in v for s in ("steps:", "sampler", "cfg scale", "seed")):
            hints.append(f"text chunk '{k}' looks like embedded generation parameters")

    hints = list(dict.fromkeys(hints))            # de-duplicate, keep order

    notes: List[str] = []
    if camera or has_capture_params:
        notes.append("Camera-style EXIF found (" + (camera or "capture settings") +
                     "). It can be forged and does not prove the pixels are unedited.")
    else:
        notes.append("No camera EXIF found. Common for AI images, but also for screenshots, "
                     "social-media downloads and edited photos, so it is not evidence by itself.")
    if has_content_credentials:
        notes.append("A Content Credentials (C2PA) marker is present. Its signature was NOT "
                     "verified here; check it with a C2PA verification tool.")
    if hints:
        notes.append("Metadata contains generator-related strings (see hints).")

    return {
        "camera": camera or None,
        "software": software or None,
        "has_exif": bool(exif),
        "has_capture_parameters": has_capture_params,
        "content_credentials_marker": bool(has_content_credentials),
        "generator_hints": hints,
        "notes": notes,
    }
