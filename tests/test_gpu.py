import numpy as np
import pytest
import hashlib
from PIL import Image

torch = pytest.importorskip("torch")
torchvision = pytest.importorskip("torchvision")
from torchvision.models import resnet18

from aidetect.config import Config
from aidetect.gpu import GPUDetector, _CanonicalJPEG, _build_model, _transforms
from aidetect.scope import SCOPE_WARNING, SUPPORTED_DOMAIN
from aidetect.server import create_gpu_app


def test_transfer_model_trains_only_final_block_and_classifier():
    model = _build_model(pretrained=False)
    assert all(not parameter.requires_grad for parameter in model.layer3.parameters())
    assert all(parameter.requires_grad for parameter in model.layer4.parameters())
    assert all(parameter.requires_grad for parameter in model.fc.parameters())


def test_gpu_training_transform_normalizes_size_with_equal_quality_augmentations():
    image = Image.new("RGB", (320, 240), (20, 80, 120))

    tensor = _transforms(training=True)(image)

    assert tensor.shape == (3, 224, 224)
    assert torch.isfinite(tensor).all()


def test_canonical_jpeg_uses_fixed_quality_for_evaluation(monkeypatch):
    saved = {}
    original_save = Image.Image.save

    def record_save(image, target, format=None, **options):
        saved.update(format=format, **options)
        original_save(image, target, format=format, **options)

    monkeypatch.setattr(Image.Image, "save", record_save)
    _CanonicalJPEG(random_quality=False)(Image.new("RGB", (32, 32), "white"))

    assert saved == {"format": "JPEG", "quality": 85}


def test_canonical_jpeg_randomizes_training_quality(monkeypatch):
    saved = {}
    original_save = Image.Image.save

    def record_save(image, target, format=None, **options):
        saved.update(format=format, **options)
        original_save(image, target, format=format, **options)

    monkeypatch.setattr(Image.Image, "save", record_save)
    monkeypatch.setattr(torch, "randint", lambda low, high, size: torch.tensor(72))
    _CanonicalJPEG(random_quality=True)(Image.new("RGB", (32, 32), "white"))

    assert saved == {"format": "JPEG", "quality": 72}


def test_gpu_model_checkpoint_round_trip_on_cpu(tmp_path):
    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    checkpoint = tmp_path / "vision_model.pt"
    torch.save({"state_dict": model.state_dict(), "threshold": 0.5, "margin": 0.1}, checkpoint)

    detector = GPUDetector.load(checkpoint, device_name="cpu")
    assert not detector.canonical_jpeg
    image_path = tmp_path / "image.png"
    Image.fromarray(np.zeros((240, 320, 3), dtype=np.uint8)).save(image_path)

    prediction = detector.predict(image_path)
    assert prediction["ok"]
    assert 0.0 <= prediction["prob_ai"] <= 1.0
    assert prediction["label"] in {"real", "ai_generated", "uncertain"}
    assert prediction["warnings"] == [SCOPE_WARNING]

    client = create_gpu_app(detector).test_client()
    with image_path.open("rb") as image_file:
        response = client.post(
            "/api/predict",
            data={"files": (image_file, image_path.name)},
            content_type="multipart/form-data",
        )
    assert response.status_code == 200
    assert response.json["results"][0]["ok"]
    assert response.json["results"][0]["warnings"] == [SCOPE_WARNING]
    report = response.json["results"][0]["forensic_report"]
    assert report["file_integrity"]["sha256"] == hashlib.sha256(image_path.read_bytes()).hexdigest()
    assert report["authenticity_analysis"]["model"]["identifier"] == tmp_path.name
    assert report["manipulation_analysis"]["status"] == "not_implemented"


def test_gpu_web_app_upload_and_errors():
    class StubDetector:
        threshold = 0.5
        margin = 0.1
        device = "cpu"
        supported_domain = SUPPORTED_DOMAIN
        scope_warning = SCOPE_WARNING

        def predict_image(self, image, name):
            return {
                "name": name, "ok": True, "label": "real", "model_label": "real",
                "prob_ai": 0.1, "threshold": self.threshold, "margin": self.margin,
                "size": list(image.size), "format": image.format, "n_crops": 1,
                "decision_basis": "model", "warnings": [], "provenance": {},
            }

    client = create_gpu_app(StubDetector()).test_client()
    page = client.get("/")
    assert page.status_code == 200 and b"AI Image Detection System" in page.data
    assert b"Evidence-led image review" in page.data
    assert b"Start with an account" in page.data
    model_info = client.get("/api/model").json
    assert model_info["model_type"] == "gpu"
    assert model_info["supported_domain"] == SUPPORTED_DOMAIN
    assert model_info["scope_warning"] == SCOPE_WARNING

    image = Image.new("RGB", (240, 240), (20, 80, 120))
    from io import BytesIO
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    valid = BytesIO(buffer.getvalue())
    broken = BytesIO(b"not an image")
    response = client.post(
        "/api/predict",
        data={"files": [(valid, "photo.png"), (broken, "broken.png")]},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    results = response.json["results"]
    assert results[0]["ok"] and results[0]["size"] == [240, 240]
    assert not results[1]["ok"] and results[1]["error"]
    assert client.post("/api/predict", data={}, content_type="multipart/form-data").status_code == 400


def test_gpu_api_returns_explicit_error_for_per_file_size_limit():
    from io import BytesIO

    from aidetect.server import create_gpu_app

    class StubDetector:
        threshold = 0.5
        margin = 0.1
        device = "cpu"

    client = create_gpu_app(StubDetector(), Config(max_upload_mb=0.0005)).test_client()
    response = client.post(
        "/api/predict",
        data={"files": (BytesIO(b"x" * 1000), "large.png")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert response.json["results"][0]["ok"] is False
    assert "larger than" in response.json["results"][0]["error"]
