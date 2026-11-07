import json
import hashlib
import shutil
from pathlib import Path

import numpy as np
import pytest

from aidetect.cli import main
from aidetect.dataset import discover, scan_dataset
from aidetect.detector import Detector
from aidetect.evaluate import evaluate
from aidetect.server import create_app


def test_training_outputs_and_quality(trained):
    mdir, cfg, summary = trained
    for f in ("model.joblib", "model_info.json", "metrics.json", "dataset_report.json", "config_used.yaml",
              "evaluation_holdout.png", "evaluation_cv.png"):
        assert (mdir / f).is_file(), f
    assert summary["holdout"]["roc_auc"] > 0.9 and summary["cv"]["roc_auc"] > 0.9
    assert 0.0 <= summary["holdout"]["threshold"] <= 1.0
    assert abs(sum(summary["cv"]["ensemble_weights"].values()) - 1) < 1e-9
    assert set(summary["holdout"]["group_importance_auc_drop"]) == {"spectral", "noise", "texture", "color", "compression"}


def test_predict_roundtrip_and_errors(trained, data_dir, tmp_path):
    det = Detector.load(trained[0])
    real = sorted((data_dir / "real").iterdir())[:8]
    fake = sorted((data_dir / "fake" / "gen_b").iterdir())[:8]
    pr = [det.predict(f) for f in real]
    pf = [det.predict(f) for f in fake]
    assert all(p.ok and 0 <= p.prob_ai <= 1 for p in pr + pf)
    assert np.mean([p.prob_ai for p in pf]) > np.mean([p.prob_ai for p in pr]) + 0.5
    again = det.predict(real[0])
    assert again.prob_ai == pr[0].prob_ai                          # deterministic
    bad = det.predict(b"garbage")
    assert not bad.ok and bad.error and bad.label is None
    ex = det.predict(fake[0], explain=True)
    assert ex.prob_ai == pf[0].prob_ai                             # explain never changes the verdict
    assert ex.group_contributions and ex.crop_scores
    assert abs(sum(ex.group_contributions.values())) > 0
    # many / threaded
    many = list(det.predict_many(real[:4] + [b"x"], workers=2))
    assert [m.ok for m in many] == [True] * 4 + [False]


def test_metadata_override(trained, data_dir):
    import io
    from PIL import Image, PngImagePlugin
    det = Detector.load(trained[0])
    f = sorted((data_dir / "real").iterdir())[0]
    im = Image.open(f).convert("RGB")
    meta = PngImagePlugin.PngInfo(); meta.add_text("parameters", "Steps: 30, Sampler: DPM++")
    b = io.BytesIO(); im.save(b, "PNG", pnginfo=meta)
    p = det.predict(b.getvalue())
    assert p.label == "ai_generated" and p.decision_basis == "provenance_metadata" and p.model_label != "ai_generated"
    assert any("forged" in w for w in p.warnings)
    det.cfg.provenance_override = False
    assert det.predict(b.getvalue()).label == p.model_label
    det.cfg.provenance_override = True


def test_evaluate_with_robustness(trained, data_dir, tmp_path):
    det = Detector.load(trained[0])
    cfg = trained[1]
    scan = scan_dataset(discover(cfg, root=data_dir), cfg, show_progress=False)
    res = evaluate(det, scan.samples, tmp_path / "ev", robustness=True)
    assert res["metrics"]["roc_auc"] > 0.9
    assert set(res["robustness"]) == {s["name"] for s in det.cfg.robustness}
    assert (tmp_path / "ev" / "evaluation.json").is_file()


def test_server_api(trained, data_dir):
    det = Detector.load(trained[0])
    client = create_app(det).test_client()
    assert client.get("/").status_code == 200 and b"AI Image Detection System" in client.get("/").data
    assert client.get("/api/health").json == {"status": "ok"}
    model_info = client.get("/api/model").json
    assert model_info["n_features"] == len(det.bundle.feature_names)
    assert model_info["supported_domain"] == "AI-generated vs real photographic images"
    f = sorted((data_dir / "fake" / "gen_a").iterdir())[0]
    with open(f, "rb") as fh:
        original = fh.read()
        r = client.post("/api/predict", data={"files": [(__import__("io").BytesIO(original), f.name), (__import__("io").BytesIO(b"junk"), "bad.png")]},
                        content_type="multipart/form-data")
    res = r.json["results"]
    assert r.status_code == 200 and res[0]["ok"] and set(res[0]["panels"]) >= {"heatmap", "spectrum", "noise_residual", "ela"}
    assert res[0]["panels"]["ela"].startswith("data:image/png;base64,")
    report = res[0]["forensic_report"]
    assert report["file_integrity"]["sha256"] == hashlib.sha256(original).hexdigest()
    assert report["authenticity_analysis"]["decision"] == res[0]["label"]
    assert report["authenticity_analysis"]["model"]["identifier"] == "m"
    assert report["manipulation_analysis"]["status"] == "not_implemented"
    assert not res[1]["ok"] and "readable" in res[1]["error"]
    assert client.post("/api/predict", data={}, content_type="multipart/form-data").status_code == 400


def test_cli_flow(trained, data_dir, tmp_path, capsys):
    mdir = str(trained[0])
    f = str(sorted((data_dir / "real").iterdir())[0])
    assert main(["predict", f, "--model", mdir, "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out[0]["path"] == f and out[0]["ok"]
    assert main(["predict", str(tmp_path / "missing.png"), "--model", mdir]) == 1
    assert main(["predict", "--model", str(tmp_path / "nomodel"), f]) == 1
    assert main(["config", "--set", "crop_size=96"]) == 0
    assert main(["train", "--root", str(tmp_path), "--out", str(tmp_path / "o")]) == 1   # no classes -> clean error


def test_tiny_dataset_rejected(tmp_path, data_dir):
    cfg_args = ["--set", f"cache_dir={tmp_path / 'c'}", "--set", "n_jobs=1"]
    for kind, src in (("real", data_dir / "real"), ("fake", data_dir / "fake" / "gen_a")):
        (tmp_path / "d" / kind).mkdir(parents=True)
        for f in sorted(src.iterdir())[:4]:
            shutil.copy(f, tmp_path / "d" / kind / f.name)
    assert main(["train", "--root", str(tmp_path / "d"), "--out", str(tmp_path / "o"), *cfg_args]) == 1
