import io
import struct
import zlib

import numpy as np
import pytest
from PIL import Image, PngImagePlugin

from aidetect.config import Config
from aidetect.dataset import DatasetError, discover, scan_dataset
from aidetect.features import crop_positions, extract, feature_names
from aidetect.loader import ImageError, load_image
from aidetect.perturb import apply_ops, sample_ops
from aidetect.provenance import scan


def noise_img(h=160, w=200, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def to_bytes(arr, fmt="PNG", **kw):
    b = io.BytesIO()
    Image.fromarray(arr).save(b, format=fmt, **kw)
    return b.getvalue()


# ---------------------------------------------------------------- config
def test_config_overrides_and_validation():
    cfg = Config.load(None, ["crop_size=96", "augment.jpeg_quality=[50,90]", "models=[hgb]"])
    assert cfg.crop_size == 96 and cfg.augment["jpeg_quality"] == [50, 90] and cfg.augment["prob_each"] == 0.5
    with pytest.raises(ValueError):
        Config.load(None, ["crop_size=100"])
    with pytest.raises(ValueError):
        Config.load(None, ["nonsense=1"])
    with pytest.raises(ValueError):
        Config.load(None, ["models=[svm]"])
    assert Config.from_dict(Config().to_dict()).to_dict() == Config().to_dict()


def test_high_level_hybrid_model_config_and_prediction():
    cfg = Config.from_dict({
        "high_level": {"enabled": True, "method": "logreg", "weight": 0.35},
        "model": {"estimators": {"hgb": {"enabled": True, "weight": 1.0}}},
    })
    assert cfg.high_level.enabled is True
    assert cfg.high_level.method == "logreg"
    arr = np.array([[0.1, 0.2], [0.8, 0.9], [0.7, 0.6], [0.2, 0.1]], dtype=np.float64)
    offsets = np.array([0, 2, 4], dtype=np.int64)
    probs = np.array([0.2, 0.8, 0.1, 0.9], dtype=np.float64)
    from aidetect.model import HybridImageHead
    model = HybridImageHead(weight=0.35)
    model.fit(arr, offsets, np.array([0, 1]))
    pred = model.predict(arr, offsets, probs)
    assert pred.shape == (2,)
    assert np.isfinite(pred).all()


def test_high_level_path_is_enabled_by_default_and_blends_scores():
    cfg = Config()
    assert cfg.high_level.enabled is True
    arr = np.array([[0.1, 0.2], [0.7, 0.8], [0.2, 0.1], [0.9, 0.95]], dtype=np.float64)
    offsets = np.array([0, 2, 4], dtype=np.int64)
    probs = np.array([0.15, 0.85, 0.12, 0.88], dtype=np.float64)
    from aidetect.model import HybridImageHead
    model = HybridImageHead(weight=0.5)
    model.fit(arr, offsets, np.array([0, 1]))
    pred = model.predict(arr, offsets, probs)
    assert pred.shape == (2,)
    assert 0.0 <= pred.min() <= pred.max() <= 1.0


# ---------------------------------------------------------------- loader
def test_loader_modes_and_errors():
    cfg = Config()
    rgba = np.dstack([noise_img(), np.full((160, 200), 255, np.uint8)])
    assert load_image(to_bytes(rgba), cfg).has_alpha
    g = load_image(to_bytes(noise_img()[..., 0]), cfg)
    assert g.is_grayscale and g.rgb.shape == (160, 200, 3)
    pal = Image.fromarray(noise_img()).convert("P")
    b = io.BytesIO(); pal.save(b, format="PNG")
    assert load_image(b.getvalue(), cfg).rgb.shape == (160, 200, 3)
    i16 = (np.random.default_rng(1).integers(0, 65535, (150, 150))).astype(np.uint16)
    b = io.BytesIO(); Image.fromarray(i16).save(b, format="PNG")
    assert load_image(b.getvalue(), cfg).rgb.dtype == np.uint8
    with pytest.raises(ImageError, match="shortest side"):
        load_image(to_bytes(noise_img(60, 60)), cfg)
    with pytest.raises(ImageError):
        load_image(b"definitely not an image", cfg)
    with pytest.raises(ImageError, match="empty"):
        load_image(b"", cfg)
    with pytest.raises(ImageError):
        load_image("/no/such/file.png", cfg)
    good = to_bytes(noise_img(), "JPEG")
    with pytest.raises(ImageError):
        load_image(good[: len(good) // 2], cfg)       # truncated file must error, not silently pass


# ---------------------------------------------------------------- features
def test_features_fixed_layout_finite_and_deterministic():
    cfg = Config()
    names = feature_names(cfg)
    assert len(names) == len(set(names)) > 100
    for arr in (noise_img(), np.zeros((160, 160, 3), np.uint8), np.full((160, 160, 3), 255, np.uint8),
                np.tile(np.arange(256, dtype=np.uint8)[:160, None, None], (1, 200, 3))):
        img = load_image(to_bytes(arr), cfg)
        a, b = extract(img, cfg), extract(img, cfg)
        assert a.names == names and a.feats.shape[1] == len(names)
        assert np.isfinite(a.feats).all()
        assert np.array_equal(a.feats, b.feats)


def test_feature_groups_and_crop_size_configurable():
    cfg = Config.from_dict({"feature_groups": ["spectral", "noise"], "crop_size": 64})
    names = feature_names(cfg)
    assert {n.split(".")[0] for n in names} == {"spectral", "noise"}
    assert extract(load_image(to_bytes(noise_img(100, 100)), cfg), cfg).feats.shape[1] == len(names)


def test_metadata_features_optional():
    cfg = Config.from_dict({"use_metadata_features": True})
    assert any(n.startswith("meta.") for n in feature_names(cfg))
    img = load_image(to_bytes(noise_img(), "JPEG"), cfg)
    assert extract(img, cfg).feats.shape[1] == len(feature_names(cfg))


def test_crop_positions_valid():
    for h, w, c, n in [(128, 128, 128, 6), (500, 300, 128, 6), (4000, 3000, 128, 36), (130, 900, 128, 5)]:
        pos = crop_positions(h, w, c, n)
        assert 1 <= len(pos) <= n and len(set(pos)) == len(pos)
        assert all(0 <= y <= h - c and 0 <= x <= w - c and y % 8 == 0 and x % 8 == 0 for y, x in pos)


# ---------------------------------------------------------------- provenance
def test_provenance_png_text_and_exif_and_false_positives():
    cfg = Config()
    meta = PngImagePlugin.PngInfo()
    meta.add_text("parameters", "a cat, Steps: 20, Sampler: Euler a, Model: sdxl")
    b = io.BytesIO(); Image.fromarray(noise_img()).save(b, "PNG", pnginfo=meta)
    p = load_image(b.getvalue(), cfg).provenance
    assert p["declares_ai"] and any(m["kind"] == "generation_parameters" for m in p["ai_markers"])

    ex = Image.Exif(); ex[0x0131] = "Midjourney V6"; ex[0x010F] = "Canon"
    b = io.BytesIO(); Image.fromarray(noise_img()).save(b, "JPEG", exif=ex)
    p = load_image(b.getvalue(), cfg).provenance
    assert p["declares_ai"] and p["software"] == "Midjourney V6"

    ex = Image.Exif(); ex[0x0131] = "Adobe Photoshop 25"; ex[0x010F] = "Canon"; ex[0x0110] = "EOS R5"
    b = io.BytesIO(); Image.fromarray(noise_img()).save(b, "JPEG", exif=ex)
    p = load_image(b.getvalue(), cfg).provenance
    assert not p["declares_ai"] and p["camera_evidence"]["Make"] == "Canon"

    # a word like "influx" must not match "flux"; random pixel bytes must never match
    assert not scan(None, None, {"Comment": "influx of fluxes"}, {})["declares_ai"]
    for seed in range(20):
        assert not load_image(to_bytes(noise_img(seed=seed)), cfg).provenance["declares_ai"]


def test_provenance_xmp_digital_source_type():
    xmp = (b"http://ns.adobe.com/xap/1.0/\x00<x:xmpmeta><rdf:Description Iptc4xmpExt:DigitalSourceType="
           b"'http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia'/></x:xmpmeta>")
    seg = b"\xff\xe1" + struct.pack(">H", len(xmp) + 2) + xmp
    jpg = to_bytes(noise_img(), "JPEG")
    raw = jpg[:2] + seg + jpg[2:]
    p = load_image(raw, Config()).provenance
    assert p["declares_ai"] and p["ai_markers"][0]["kind"] == "declared_synthetic_media"


# ---------------------------------------------------------------- perturb
def test_perturbations_roundtrip_through_loader():
    cfg, rng = Config(), np.random.default_rng(0)
    rgb = noise_img(300, 300)
    for ops in ([{"op": "jpeg", "quality": 60}], [{"op": "resize", "scale": 0.5}], [{"op": "blur", "sigma": 1.2}],
                [{"op": "noise", "sigma": 4}], sample_ops(cfg.augment, rng)):
        out = load_image(apply_ops(rgb, ops, rng, cfg.effective_min_side), cfg)
        assert out.rgb.shape[2] == 3 and min(out.rgb.shape[:2]) >= cfg.effective_min_side
    assert load_image(apply_ops(rgb, [{"op": "jpeg", "quality": 60}], rng, 128), cfg).format == "JPEG"
    with pytest.raises(ValueError):
        apply_ops(rgb, [{"op": "bogus"}], rng, 128)


# ---------------------------------------------------------------- dataset
def test_dataset_discovery_and_audit(tmp_path, data_dir):
    cfg = Config.from_dict({"n_jobs": 1})
    s = discover(cfg, root=data_dir)
    assert {x.label for x in s} == {0, 1} and {x.source for x in s if x.label} == {"gen_a", "gen_b"}
    # duplicates, conflicts, junk
    import shutil
    d = tmp_path / "ds"
    (d / "real").mkdir(parents=True); (d / "ai").mkdir()
    src = sorted((data_dir / "real").iterdir())[:3]
    for i, f in enumerate(src):
        shutil.copy(f, d / "real" / f"a{i}{f.suffix}")
    shutil.copy(src[0], d / "real" / f"dup{src[0].suffix}")                 # exact duplicate (same class)
    shutil.copy(src[1], d / "ai" / f"conflict{src[1].suffix}")             # same bytes, other class
    (d / "ai" / "broken.png").write_bytes(b"nope")
    Image.fromarray(noise_img(50, 50)).save(d / "ai" / "tiny.png")
    for f in sorted((data_dir / "fake" / "gen_a").iterdir())[:3]:
        shutil.copy(f, d / "ai" / f.name)
    res = scan_dataset(discover(cfg, root=d), cfg, show_progress=False)
    issues = res.audit["issues"]
    assert issues["unreadable"]["count"] == 1 and issues["too_small"]["count"] == 1
    assert issues["exact_duplicates"]["count"] == 1 and issues["label_conflicts"]["count"] >= 2
    assert all(x.group >= 0 for x in res.samples)


def test_dataset_errors(tmp_path):
    cfg = Config()
    (tmp_path / "a").mkdir(); (tmp_path / "b").mkdir()
    with pytest.raises(DatasetError, match="could not identify both classes"):
        discover(cfg, root=tmp_path)
    with pytest.raises(DatasetError):
        discover(cfg)
    with pytest.raises(DatasetError, match="not a directory"):
        discover(cfg, real_dirs=[tmp_path / "x"], ai_dirs=[tmp_path / "a"])
    man = tmp_path / "m.csv"
    man.write_text("path,label\nx.png,maybe\n")
    with pytest.raises(DatasetError, match="unknown label"):
        discover(cfg, manifest=man)


def test_manifest_relative_paths(tmp_path, data_dir):
    import shutil
    r = sorted((data_dir / "real").iterdir())[0]; f = sorted((data_dir / "fake" / "gen_a").iterdir())[0]
    shutil.copy(r, tmp_path / "r.jpg" if r.suffix == ".jpg" else tmp_path / "r.png")
    shutil.copy(f, tmp_path / "f.png" if f.suffix == ".png" else tmp_path / "f.jpg")
    (tmp_path / "m.csv").write_text(f"path,label,source\n{'r' + r.suffix},real,cam\n{'f' + f.suffix},1,sd\n")
    s = discover(Config(), manifest=tmp_path / "m.csv")
    assert [x.label for x in s] == [0, 1] and s[0].source == "cam" and s[0].path.startswith(str(tmp_path.resolve()))
