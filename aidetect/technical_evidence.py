"""File-level observations that can be verified without claiming image provenance."""
from __future__ import annotations

import io
import math
from typing import Any

from PIL import Image, UnidentifiedImageError
from .provenance import scan as scan_provenance

_EXIF_NAMES = {
    271: "camera_make",
    272: "camera_model",
    306: "date_time",
    36867: "date_time_original",
    36868: "date_time_digitized",
    41729: "scene_type",
    41990: "scene_capture_type",
}


def _entropy(image: Image.Image) -> float:
    histogram = image.convert("L").histogram()
    total = sum(histogram)
    if not total:
        return 0.0
    return -sum((count / total) * math.log2(count / total) for count in histogram if count)


def inspect_image(image_bytes: bytes) -> dict[str, Any]:
    evidence: dict[str, Any] = {
        "status": "inspected",
        "pixel_facts": {},
        "metadata": {},
        "embedded_signals": [],
        "limitations": [
            "Metadata can be incomplete, copied, edited, or removed without indicating image origin.",
            "Absence of provenance data is not evidence that an image is AI-generated.",
        ],
    }
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.load()
            grayscale = image.convert("L")
            extrema = grayscale.getextrema()
            evidence["pixel_facts"] = {
                "width": image.width,
                "height": image.height,
                "format": image.format,
                "mode": image.mode,
                "grayscale_min": extrema[0],
                "grayscale_max": extrema[1],
                "estimated_entropy": round(_entropy(grayscale), 3),
            }

            exif = image.getexif()
            metadata: dict[str, Any] = {}
            for tag_id, tag_name in _EXIF_NAMES.items():
                value = exif.get(tag_id)
                if value not in (None, ""):
                    metadata[tag_name] = str(value).strip()[:200]
            software = exif.get(305)
            if software:
                metadata["software"] = str(software).strip()[:200]

            orientation = exif.get(274)
            if orientation in range(1, 9):
                metadata["orientation"] = str(orientation)
            evidence["metadata"] = metadata
            provenance_info = {
                key: value for key, value in getattr(image, "info", {}).items()
                if isinstance(value, str)
            }
            exif_values = {}
            for tag_id, value in exif.items():
                exif_values[str(tag_id)] = value
                if isinstance(value, (str, int, float)):
                    exif_values[str(tag_id)] = str(value)
            provenance = scan_provenance(image_bytes, image.format, provenance_info, exif_values)
            evidence["provenance"] = {
                "declares_ai": bool(provenance.get("declares_ai")),
                "camera_like": bool(provenance.get("camera_like")),
                "content_credentials_present": bool(provenance.get("content_credentials")),
                "software": provenance.get("software"),
                "camera_evidence": provenance.get("camera_evidence", {}),
                "ai_markers": provenance.get("ai_markers", [])[:20],
                "summary": provenance.get("summary", "No declared provenance signal was found."),
            }

            embedded = []
            marker = b"c2pa" if b"c2pa" in image_bytes.lower() else None
            if marker:
                embedded.append(
                    "C2PA Content Credentials marker detected. The embedded data is not cryptographically verified by this check."
                )
            if "parameters" in getattr(image, "info", {}):
                embedded.append("Image contains embedded generation-style parameters.")
            evidence["embedded_signals"] = embedded
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        evidence["status"] = "inspection_failed"
        evidence["error"] = "The image is readable only as an unverified upload."
    return evidence
