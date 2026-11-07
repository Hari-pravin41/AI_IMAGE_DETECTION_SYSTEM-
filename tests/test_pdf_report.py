import io
import json
import hashlib

import pytest
from PIL import Image

from aidetect.config import Config
from aidetect.forensics import build_case_report
from aidetect.loader import load_image
from aidetect.pdf_report import InvalidReportError, create_analysis_pdf_report, create_pdf_report
from aidetect.server import create_app, create_gpu_app


class StubDetector:
    model_identifier = "pdf-test-model"
    cfg = Config()


def make_report(decision="uncertain"):
    image = Image.new("RGB", (320, 240), "navy")
    output = io.BytesIO()
    image.save(output, format="PNG")
    original = output.getvalue()
    loaded = load_image(original, Config(), name="sample.png")
    report = build_case_report(
        original,
        loaded,
        {
            "label": decision,
            "prob_ai": 0.51,
            "threshold": 0.55,
            "decision_basis": "model",
        },
        StubDetector(),
        "sample.png",
    )
    return report, original


def test_pdf_report_generates_downloadable_pdf_with_preview():
    report, original = make_report()

    pdf = create_pdf_report(report, original)

    assert pdf.startswith(b"%PDF-")
    assert b"Image preview" in pdf
    assert b"SHA-256" in pdf
    assert b"pdf-test-model" in pdf
    assert b"0.5100" in pdf and b"0.5500" in pdf
    assert b"Declared AI metadata" in pdf
    assert b"not proof" in pdf
    assert b"not implemented" in pdf


def test_pdf_report_preserves_uncertain_decision_and_safe_filename_text():
    report, _ = make_report()
    report["file_integrity"]["original_filename"] = "</para><para>injected"

    pdf = create_pdf_report(report)

    assert pdf.startswith(b"%PDF-")
    assert b"UNCERTAIN" in pdf
    assert b"&lt;/para&gt;&lt;para&gt;injected" not in pdf
    assert b"injected" in pdf


def test_pdf_report_explains_missing_exif_metadata():
    report, _ = make_report()

    pdf = create_pdf_report(report)

    assert b"EXIF status" in pdf
    assert b"Not present" in pdf
    assert b"Missing metadata alone is not evidence" in pdf


def test_analysis_pdf_includes_explainability_without_empty_visual_section():
    image = Image.new("RGB", (320, 240), "white")
    original = io.BytesIO()
    image.save(original, format="PNG")
    result = {
        "analysis_id": "analysis-123",
        "filename": "sample.png",
        "analysis_timestamp": "2026-11-01T08:23:18+00:00",
        "assessment": "Uncertain",
        "description": "A simple test image.",
        "direct_observations": ["The background is uniform."],
        "assessment_reasoning": "The visible evidence is balanced.",
        "possible_ai_indicators": ["The uniform output could be digital."],
        "possible_real_indicators": ["A real surface can appear uniform."],
        "uncertain_interpretations": ["The source cannot be established visually."],
        "limitations": ["Visual assessment alone cannot prove provenance."],
        "image_data": {
            "width": 320,
            "height": 240,
            "file_size_bytes": len(original.getvalue()),
            "format": "PNG",
            "sha256": hashlib.sha256(original.getvalue()).hexdigest(),
        },
    }

    pdf = create_analysis_pdf_report(result, original.getvalue())

    assert pdf.startswith(b"%PDF-")
    assert len(pdf) > 1000


@pytest.mark.parametrize("invalid", [None, [], {}, {"case_id": "missing fields"}])
def test_pdf_report_rejects_invalid_input(invalid):
    with pytest.raises(InvalidReportError):
        create_pdf_report(invalid)


def test_pdf_download_endpoint_returns_attachment_and_rejects_invalid_report():
    report, original = make_report()
    for app in (create_app(StubDetector()), create_gpu_app(StubDetector(), Config())):
        client = app.test_client()
        response = client.post(
            "/api/report/pdf",
            data={
                "report": json.dumps(report),
                "image": (io.BytesIO(original), "sample.png"),
            },
            content_type="multipart/form-data",
        )

        assert response.status_code == 200
        assert response.mimetype == "application/pdf"
        assert response.headers["Content-Disposition"].startswith("attachment;")
        assert response.data.startswith(b"%PDF-")

        invalid = client.post(
            "/api/report/pdf",
            data={"report": "{not json"},
            content_type="multipart/form-data",
        )
        assert invalid.status_code == 400
        assert "valid JSON" in invalid.json["error"]
