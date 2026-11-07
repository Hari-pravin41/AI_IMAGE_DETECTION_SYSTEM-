"""Build privacy-conscious, factual case reports for in-memory image analyses."""
from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from typing import Any

from PIL import Image

from .loader import LoadedImage

_EXIF_REPORT_FIELDS = (
    "Make",
    "Model",
    "DateTime",
    "DateTimeOriginal",
    "DateTimeDigitized",
    "Software",
    "Orientation",
    "ExposureTime",
    "FNumber",
    "ISOSpeedRatings",
    "FocalLength",
    "LensModel",
    "ImageDescription",
)


def _metadata_summary(image: LoadedImage) -> dict[str, Any]:
    exif = {
        key: str(image.exif[key])[:256]
        for key in _EXIF_REPORT_FIELDS
        if key in image.exif and str(image.exif[key]).strip()
    }
    provenance = image.provenance
    markers = [
        {
            "source": str(marker.get("source", ""))[:40],
            "kind": str(marker.get("kind", ""))[:60],
            "match": str(marker.get("match", ""))[:80],
        }
        for marker in provenance.get("ai_markers", [])[:20]
        if isinstance(marker, dict)
    ]
    return {
        "exif_status": "available" if exif else "not_present",
        "exif": exif,
        "software": str(provenance["software"])[:100] if provenance.get("software") else None,
        "camera_evidence": {
            str(key)[:40]: str(value)[:100]
            for key, value in provenance.get("camera_evidence", {}).items()
        },
        "declared_ai_metadata": {
            "status": "found" if provenance.get("declares_ai") else "not_found",
            "markers": markers,
            "content_credentials_present": bool(provenance.get("content_credentials")),
        },
        "interpretation": (
            "Metadata may be absent, stripped, copied, or forged. Its presence or absence alone "
            "does not establish whether image content is authentic."
        ),
    }


def _model_summary(detector: Any) -> dict[str, Any]:
    if hasattr(detector, "bundle"):
        info = detector.info()
        return {
            "identifier": str(getattr(detector, "model_identifier", "forensic-feature-ensemble")),
            "model_type": "forensic_feature_ensemble",
            "components": info.get("members", []),
            "feature_count": info.get("n_features"),
            "version": "aidetect 1.0.0",
        }
    return {
        "identifier": str(getattr(detector, "model_identifier", "resnet18")),
        "model_type": "gpu_image_classifier",
        "components": ["ResNet-18"],
        "version": "aidetect 1.0.0",
    }


def build_case_report(
    original_bytes: bytes,
    image: LoadedImage,
    prediction: dict[str, Any],
    detector: Any,
    filename: str,
) -> dict[str, Any]:
    """Describe original-file facts and model output; never persist uploaded bytes."""
    file_facts = {
        "original_filename": filename,
        "sha256": hashlib.sha256(original_bytes).hexdigest(),
        "byte_size": len(original_bytes),
        "detected_mime_type": Image.MIME.get(image.format, "application/octet-stream"),
        "format": image.format,
        "dimensions_px": {"width": image.size[0], "height": image.size[1]},
        "mode": image.mode,
        "has_alpha": image.has_alpha,
        "is_grayscale": image.is_grayscale,
    }
    metadata = _metadata_summary(image)
    model = _model_summary(detector)
    model_score = prediction.get("prob_ai")
    threshold = prediction.get("threshold")
    decision = prediction.get("label")
    return {
        "schema_version": 1,
        "case_id": str(uuid.uuid4()),
        "analysis_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "file_integrity": file_facts,
        "metadata_examination": metadata,
        "authenticity_analysis": {
            "task": "AI-generated versus real image classification",
            "supported_domain": "photographic images; image style is not automatically checked",
            "model": model,
            "decision": decision,
            "score_name": "AI-class model score",
            "score": model_score,
            "score_interpretation": (
                "This model score is not asserted to be a calibrated probability."
            ),
            "threshold": threshold,
            "decision_basis": prediction.get("decision_basis", "model"),
        },
        "manipulation_analysis": {
            "status": "not_implemented",
            "finding": None,
            "explanation": (
                "This application does not determine whether an image has been edited or tampered with."
            ),
        },
        "findings": [
            {
                "category": "file_integrity",
                "evidence_type": "observed",
                "observation": "SHA-256 was computed over the original uploaded bytes.",
                "value": file_facts["sha256"],
            },
            {
                "category": "metadata",
                "evidence_type": "observed",
                "observation": f"EXIF metadata status: {metadata['exif_status']}.",
                "interpretation": (
                    "Missing metadata is not proof of AI generation or image manipulation."
                ),
            },
            {
                "category": "ai_authenticity",
                "evidence_type": "model_prediction",
                "observation": f"Model decision: {decision}; AI-class score: {model_score}.",
                "interpretation": (
                    "A model prediction is an estimate, not proof that an image is authentic, "
                    "generated, manipulated, or malicious."
                ),
            },
            {
                "category": "image_manipulation",
                "evidence_type": "capability_status",
                "observation": "Manipulation detection is not implemented.",
            },
        ],
        "limitations": [
            "Intended for photographic images; non-photographic style is not automatically detected.",
            "The AI-class score is not guaranteed to be calibrated.",
            "AI-generation classification and image-manipulation detection are separate tasks.",
            "Metadata can be removed, copied, or forged.",
            "This report is not legally certified forensic evidence.",
        ],
        "retention": (
            "The server processes uploads in memory and does not save image contents or reports."
        ),
    }
