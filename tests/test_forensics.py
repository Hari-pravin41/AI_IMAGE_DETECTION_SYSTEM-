import hashlib
import json

from PIL import Image

from aidetect.config import Config
from aidetect.forensics import build_case_report
from aidetect.loader import load_image


class StubDetector:
    model_identifier = "test-model"


def test_case_report_hashes_original_bytes_and_separates_findings():
    image = Image.new("RGB", (320, 240), "navy")
    import io

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    original = buffer.getvalue()
    loaded = load_image(original, Config(), name="upload.jpg")
    prediction = {
        "label": "uncertain",
        "prob_ai": 0.51,
        "threshold": 0.55,
        "decision_basis": "model",
    }

    report = build_case_report(
        original, loaded, prediction, StubDetector(), "upload.jpg"
    )

    assert report["file_integrity"]["sha256"] == hashlib.sha256(original).hexdigest()
    assert report["file_integrity"]["byte_size"] == len(original)
    assert report["file_integrity"]["detected_mime_type"] == "image/png"
    assert report["metadata_examination"]["exif_status"] == "not_present"
    assert "does not establish" in report["metadata_examination"]["interpretation"]
    assert report["authenticity_analysis"]["decision"] == "uncertain"
    assert report["authenticity_analysis"]["score_interpretation"].startswith(
        "This model score is not asserted"
    )
    assert report["manipulation_analysis"]["status"] == "not_implemented"
    assert report["case_id"] != build_case_report(
        original, loaded, prediction, StubDetector(), "upload.jpg"
    )["case_id"]
    json.dumps(report)


def test_case_report_includes_selected_exif_and_excludes_location_data():
    source = Image.new("RGB", (320, 240), "navy")
    import io

    buffer = io.BytesIO()
    source.save(buffer, format="JPEG")
    original = buffer.getvalue()
    loaded = load_image(original, Config(), name="image.jpg")
    loaded.exif = {
        "Make": "Camera maker",
        "Model": "Camera model",
        "GPSInfo": "sensitive location",
        "BodySerialNumber": "private serial",
    }

    report = build_case_report(
        original,
        loaded,
        {"label": "real", "prob_ai": 0.2, "threshold": 0.5},
        StubDetector(),
        "image.jpg",
    )

    exif = report["metadata_examination"]["exif"]
    assert exif == {"Make": "Camera maker", "Model": "Camera model"}
    assert report["metadata_examination"]["exif_status"] == "available"
    assert "GPSInfo" not in json.dumps(report)
    assert "private serial" not in json.dumps(report)
