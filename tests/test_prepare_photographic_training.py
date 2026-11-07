import json
from urllib.error import HTTPError
import io

from PIL import Image

from scripts import prepare_photographic_training as preparation


def test_used_ids_includes_openfake_and_commons_rows(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "\n".join([
            json.dumps({"row_id": "openfake-1"}),
            json.dumps({"row_id": "commons-2"}),
            json.dumps({"row_id": "other-3", "source_dataset": "ComplexDataLab/OpenFake"}),
            json.dumps({"row_id": "unrelated-4"}),
        ]),
        encoding="utf-8",
    )

    assert preparation._load_used_ids([manifest]) == {
        "openfake-1", "commons-2", "other-3"
    }


def test_relative_paths_keep_image_format_extension(tmp_path):
    row = {
        "label": "real",
        "source": "wikimedia-commons-photographs",
        "row_id": "commons-42",
    }

    output_path, relative_path = preparation._relative_path(row, tmp_path, "PNG")

    assert output_path.suffix == ".png"
    assert relative_path.endswith(".png")
    assert relative_path.startswith("train/real/")


def test_openfake_real_record_retains_dataset_source_and_license():
    record = preparation._record_for_openfake_real({
        "row_id": "openfake-real-1",
        "model": "laion",
        "image_sha256": "source-hash",
        "image": {"src": "https://example.invalid/real.jpg"},
    })

    assert record["label"] == "real"
    assert record["source"] == "laion"
    assert record["source_dataset"] == preparation._DATASET
    assert record["source_license"] == "CC-BY-NC-4.0"
    assert record["source_image_sha256"] == "source-hash"


def test_parallel_batch_never_exceeds_requested_class_count(tmp_path, monkeypatch):
    image = Image.new("RGB", (128, 128), "white")
    encoded = io.BytesIO()
    image.save(encoded, format="JPEG")
    image_bytes = encoded.getvalue()
    candidates = [
        {
            "row_id": f"openfake-{index}",
            "label": "ai",
            "source": "test-generator",
            "image": {"src": f"https://example.invalid/{index}.jpg"},
        }
        for index in range(8)
    ]
    monkeypatch.setattr(preparation, "_download", lambda row: image_bytes)
    accepted = {}

    preparation._download_class(
        "ai",
        candidates,
        target=1,
        root=tmp_path,
        progress_path=tmp_path / "progress.jsonl",
        existing_hashes=set(),
        existing_dhashes=set(),
        accepted=accepted,
        rejected=set(),
        blocked_sources=set(),
        workers=2,
    )

    assert len(accepted) == 1
    assert len(list((tmp_path / "train" / "ai").rglob("*.jpg"))) == 1


def test_commons_rate_limit_falls_back_to_openfake_real_candidate(tmp_path, monkeypatch):
    encoded = io.BytesIO()
    Image.new("RGB", (128, 128), "white").save(encoded, format="JPEG")
    image_bytes = encoded.getvalue()
    commons_row = {
        "row_id": "commons-1",
        "label": "real",
        "source": "wikimedia-commons-photographs",
        "image": {"src": "https://example.invalid/commons.jpg"},
    }
    openfake_row = {
        "row_id": "openfake-real-2",
        "label": "real",
        "source": "laion",
        "image": {"src": "https://example.invalid/openfake.jpg"},
    }

    def download(row):
        if row["source"] == "wikimedia-commons-photographs":
            raise HTTPError(row["image"]["src"], 429, "rate limited", {}, None)
        return image_bytes

    monkeypatch.setattr(preparation, "_download", download)
    accepted = {}
    blocked_sources = set()
    preparation._download_class(
        "real",
        [commons_row, openfake_row],
        target=1,
        root=tmp_path,
        progress_path=tmp_path / "progress.jsonl",
        existing_hashes=set(),
        existing_dhashes=set(),
        accepted=accepted,
        rejected=set(),
        blocked_sources=blocked_sources,
        workers=2,
    )

    assert set(accepted) == {"openfake-real-2"}
    assert blocked_sources == {"wikimedia-commons-photographs"}
