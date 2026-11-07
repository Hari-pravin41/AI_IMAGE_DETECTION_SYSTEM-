from collections import Counter
import hashlib

from PIL import Image

from scripts.prepare_openfake import _balanced_sample, _safe_component
from scripts.prepare_commons_holdout import (
    _allowed_license,
    _download_one_commons,
    _plain_text,
)


def test_balanced_sample_excludes_starter_rows_and_caps_generator_sources():
    rows = [
        {"row_id": f"real-{source}-{index}", "label": "real", "model": source}
        for source in ("pexels", "laion")
        for index in range(4)
    ]
    rows.extend(
        {"row_id": f"fake-{source}-{index}", "label": "fake", "model": source}
        for source in ("model-a", "model-b", "model-c", "model-d")
        for index in range(4)
    )

    selected = _balanced_sample(rows, 4, seed=12, excluded_ids={"real-pexels-0"}, source_cap=1)

    assert Counter(row["label"] for row in selected) == {"real": 4, "fake": 4}
    assert "real-pexels-0" not in {row["row_id"] for row in selected}
    fake_sources = Counter(row["model"] for row in selected if row["label"] == "fake")
    assert max(fake_sources.values()) == 1


def test_safe_component_removes_path_separators():
    component = _safe_component(r"..\..\outside/model")

    assert "/" not in component
    assert "\\" not in component
    assert component not in {".", ".."}


def test_balanced_sample_can_select_one_class_from_external_source():
    rows = [
        {"row_id": f"coco-{index}", "label": "real", "source": "coco2017"}
        for index in range(3)
    ]

    selected = _balanced_sample(
        rows, 2, seed=4, excluded_ids=set(), source_cap=None, labels=("real",)
    )

    assert len(selected) == 2
    assert all(row["source"] == "coco2017" for row in selected)


def test_commons_license_filter_excludes_noncommercial_and_no_derivatives():
    assert _allowed_license("CC0")
    assert _allowed_license("CC BY 4.0")
    assert _allowed_license("CC BY-SA 4.0")
    assert _allowed_license("Public domain")
    assert not _allowed_license("CC BY-NC 4.0")
    assert not _allowed_license("CC BY-ND 4.0")
    assert not _allowed_license("All rights reserved")


def test_commons_metadata_html_is_normalized():
    assert _plain_text('<a href="/artist">A &amp; B</a>') == "A & B"


def test_commons_download_resumes_from_a_verified_existing_photo(tmp_path):
    row = {
        "row_id": "commons-123",
        "title": "File:Example.jpg",
        "image": {"src": "https://example.invalid/photo.jpg"},
    }
    image_path = (
        tmp_path / "validation" / "real" /
        "wikimedia-commons-photographs" / "commons-123.jpg"
    )
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (8, 8), color="white").save(image_path, format="JPEG")

    result, relative_path = _download_one_commons(row, tmp_path)

    assert result["download_sha256"] == hashlib.sha256(image_path.read_bytes()).hexdigest()
    assert relative_path == str(image_path.relative_to(tmp_path))
