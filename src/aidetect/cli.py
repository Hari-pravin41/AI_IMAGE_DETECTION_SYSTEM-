"""Command line interface: ``aidetect <command>``."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from . import __version__
from .config import Config, ConfigError
from .data import DataError, ImageMeta, audit_dataset, collect_samples, list_images
from .imaging import ImageError
from .model import ModelError, _jsonable

log = logging.getLogger("aidetect")


# --------------------------------------------------------------------------- #
# argument helpers
# --------------------------------------------------------------------------- #
def _add_config_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("configuration")
    g.add_argument("--config", metavar="FILE", help="YAML config (see config/default.yaml)")
    g.add_argument("--set", dest="overrides", action="append", default=[], metavar="SECTION.KEY=VALUE",
                   help="override any option, repeatable, e.g. --set patch.patch_size=64")
    g.add_argument("-v", "--verbose", action="store_true", help="debug logging and full tracebacks")


def _add_data_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("dataset")
    g.add_argument("--data", metavar="DIR", help="folder with class sub-folders (real/ fake/), "
                                                 "optionally inside train/ and test/")
    g.add_argument("--real-dir", action="append", default=[], metavar="DIR", help="folder of real photos (repeatable)")
    g.add_argument("--fake-dir", action="append", default=[], metavar="DIR",
                   help="folder of AI-generated images (repeatable)")
    g.add_argument("--manifest", metavar="CSV", help="CSV with columns path,label[,split]")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="aidetect", description="Forensic AI-generated image detector.")
    ap.add_argument("--version", action="version", version=f"aidetect {__version__}")
    sub = ap.add_subparsers(dest="command", required=True, metavar="<command>")

    t = sub.add_parser("train", help="train a detector on your labelled images")
    _add_data_args(t)
    t.add_argument("--out", required=True, metavar="DIR", help="where to save the model")
    t.add_argument("--final-fit-on-all", action="store_true",
                   help="after evaluating, refit on ALL images for deployment")
    _add_config_args(t)

    e = sub.add_parser("evaluate", help="measure a trained model on labelled images")
    e.add_argument("--model", required=True, metavar="DIR")
    _add_data_args(e)
    e.add_argument("--only-test-split", action="store_true",
                   help="use only images from a 'test' folder / manifest split")
    e.add_argument("--robustness", action="store_true", help="also test JPEG / resize / blur / noise edits")
    e.add_argument("--out", required=True, metavar="DIR")
    e.add_argument("--jobs", type=int, help="parallel workers (default from config)")
    _add_config_args(e)

    pr = sub.add_parser("predict", help="analyse images or folders")
    pr.add_argument("--model", required=True, metavar="DIR")
    pr.add_argument("inputs", nargs="+", metavar="PATH", help="image files and/or folders")
    pr.add_argument("--explain", action="store_true", help="show which signal groups drove the score")
    pr.add_argument("--heatmap-dir", metavar="DIR", help="save patch-score heat-map overlays here")
    pr.add_argument("--json", action="store_true", help="print JSON instead of a table")
    pr.add_argument("--csv", metavar="FILE", help="also write results to a CSV file")
    pr.add_argument("--margin", type=float, help="override the uncertainty margin")
    pr.add_argument("--threshold", type=float, help="override the decision threshold")
    pr.add_argument("--jobs", type=int, help="parallel workers")
    pr.add_argument("--no-recursive", action="store_true", help="do not descend into sub-folders")
    _add_config_args(pr)

    a = sub.add_parser("audit", help="check a dataset for biases a model could exploit (no training)")
    _add_data_args(a)
    _add_config_args(a)

    i = sub.add_parser("info", help="show details of a trained model")
    i.add_argument("--model", required=True, metavar="DIR")
    i.add_argument("-v", "--verbose", action="store_true")

    d = sub.add_parser("make-demo-data", help="create a tiny SYNTHETIC dataset to smoke-test the pipeline")
    d.add_argument("--out", required=True, metavar="DIR")
    d.add_argument("--n-per-class", type=int, default=60)
    d.add_argument("--size", type=int, default=256)
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("-v", "--verbose", action="store_true")

    s = sub.add_parser("serve", help="launch the web app")
    s.add_argument("--model", metavar="DIR", help="default model folder shown in the app")
    s.add_argument("--port", type=int, default=8501)
    s.add_argument("--host", default="localhost")
    s.add_argument("-v", "--verbose", action="store_true")
    return ap


def _load_cfg(args: argparse.Namespace) -> Config:
    return Config.load(getattr(args, "config", None), getattr(args, "overrides", None))


def _setup_logging(level: str, verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else getattr(logging, level.upper(), logging.INFO),
                        format="%(levelname)s: %(message)s", force=True)


def _pct(v: Optional[float]) -> str:
    return "n/a" if v is None else f"{v * 100:.1f}%"


def _num(v: Optional[float]) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def _print_metrics(m: Dict[str, Any]) -> None:
    ci = m.get("ci95") or {}
    print(f"  images        : {m['n']} ({m['n_real']} real, {m['n_ai']} AI-generated)")
    acc = f"{_pct(m['accuracy'])}"
    if ci:
        acc += f"  (95% CI {_pct(ci['accuracy'][0])} - {_pct(ci['accuracy'][1])})"
    print(f"  accuracy      : {acc}")
    print(f"  balanced acc. : {_pct(m['balanced_accuracy'])}")
    auc = _num(m["roc_auc"])
    if ci.get("roc_auc"):
        auc += f"  (95% CI {ci['roc_auc'][0]:.3f} - {ci['roc_auc'][1]:.3f})"
    print(f"  ROC AUC       : {auc}")
    print(f"  AI caught     : {_pct(m['recall'])}   real kept: {_pct(m['specificity'])}   precision: {_pct(m['precision'])}")
    print(f"  calibration   : Brier {_num(m['brier'])}, ECE {_num(m['ece'])}")
    print(f"  threshold     : {m['threshold']:.3f}   uncertain: {_pct(m['uncertain_fraction'])}   "
          f"accuracy when confident: {_pct(m['confident_accuracy'])}")
    c = m["confusion"]
    print(f"  confusion     : TN={c['tn']} FP={c['fp']} FN={c['fn']} TP={c['tp']}")


class _Progress:
    def __init__(self) -> None:
        self.last_msg = ""
        self.last_frac = -1.0

    def __call__(self, stage: str, frac: float, msg: str) -> None:
        if msg != self.last_msg and (frac - self.last_frac >= 0.04 or frac >= 1.0 or stage != "features"):
            print(f"[{frac * 100:5.1f}%] {msg}", flush=True)
            self.last_msg, self.last_frac = msg, frac


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
def cmd_train(args: argparse.Namespace) -> int:
    from .train import train_model
    cfg = _load_cfg(args)
    _setup_logging(cfg.runtime.log_level, args.verbose)
    res = train_model(cfg, args.out, data_dir=args.data, real_dirs=args.real_dir, fake_dirs=args.fake_dir,
                      manifest=args.manifest, final_fit_on_all=args.final_fit_on_all, progress=_Progress())
    md = res.metadata
    print(f"\nModel saved to: {res.model_dir}")
    d = md["data"]
    print(f"Data: {d['split_counts']['train']} train / {d['split_counts']['test']} held-out test images "
          f"({d['patches_total']} patches, {md['n_features']} features each)")
    print("\nHeld-out test results (images never seen during training or calibration):")
    _print_metrics(res.metrics)
    imp = md["evaluation"]["signal_importance"]
    if imp:
        top = sorted(imp.items(), key=lambda kv: -kv[1])[:3]
        print("  top signals   : " + ", ".join(f"{k} ({v:+.3f})" for k, v in top))
    if res.warnings:
        print("\nPlease read:")
        for w in res.warnings:
            print(f"  ! {w}")
    print(f"\nFull report: {Path(res.model_dir) / 'report' / 'report.html'}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from .evaluate import evaluate_model
    cfg = _load_cfg(args)
    _setup_logging(cfg.runtime.log_level, args.verbose)
    res = evaluate_model(args.model, cfg, args.out, data_dir=args.data, real_dirs=args.real_dir,
                         fake_dirs=args.fake_dir, manifest=args.manifest,
                         only_split="test" if args.only_test_split else None,
                         robustness=args.robustness, n_jobs=args.jobs, progress=_Progress())
    print("\nResults:")
    _print_metrics(res["metrics"])
    if res["robustness"]:
        print("\nRobustness to edits (accuracy / AUC):")
        for r in res["robustness"]:
            if "accuracy" in r:
                print(f"  {r['condition']:<22} acc {_pct(r['accuracy'])}  auc {_num(r['roc_auc'])}"
                      f"   ({r['n_ok']} ok, {r['n_failed']} failed)")
            else:
                print(f"  {r['condition']:<22} no image could be analysed")
    for n in res["notes"] + res["metrics"]["warnings"]:
        print(f"  ! {n}")
    print(f"\nReport: {Path(args.out) / 'evaluation.html'}")
    return 0


def _gather_inputs(inputs: Sequence[str], extensions: Sequence[str], recursive: bool) -> List[str]:
    files: List[str] = []
    for item in inputs:
        if os.path.isdir(item):
            found = list_images(item, extensions, recursive)
            if not found:
                log.warning("No images found in %s", item)
            files += found
        elif os.path.isfile(item):
            files.append(item)
        else:
            raise DataError(f"Path does not exist: {item}")
    seen, out = set(), []
    for f in files:
        k = os.path.realpath(f)
        if k not in seen:
            seen.add(k)
            out.append(f)
    return out


def cmd_predict(args: argparse.Namespace) -> int:
    from .detector import Detector
    cfg = _load_cfg(args)
    _setup_logging(cfg.runtime.log_level, args.verbose)
    det = Detector.load(args.model, uncertainty_margin=args.margin, threshold=args.threshold)
    files = _gather_inputs(args.inputs, det.cfg.data.extensions, not args.no_recursive)
    if not files:
        raise DataError("No images to analyse.")
    names = [os.path.relpath(f) if len(files) > 1 else os.path.basename(f) for f in files]
    preds = det.predict_many(files, names=names, n_jobs=args.jobs, explain=args.explain,
                             heatmap=bool(args.heatmap_dir))
    if args.heatmap_dir:
        hd = Path(args.heatmap_dir)
        hd.mkdir(parents=True, exist_ok=True)
        for f, p in zip(files, preds):
            if p.heatmap is not None:
                p.heatmap.save(hd / f"{Path(f).stem}_heatmap.png")
    if args.csv:
        rows = [p.to_row() for p in preds]
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
    if args.json:
        print(json.dumps(_jsonable([p.to_dict() for p in preds]), indent=2))
    else:
        w = max(len(p.name) for p in preds)
        print(f"{'FILE':<{w}}  {'VERDICT':<13} {'P(AI)':>6}  {'RELIABILITY':<11} NOTES")
        for p in preds:
            if p.ok:
                notes = "; ".join(p.warnings[:2])
                print(f"{p.name:<{w}}  {p.verdict:<13} {p.prob_ai:>6.3f}  {p.reliability:<11} {notes}")
                if p.provenance.get("generator_hints"):
                    print(f"{'':<{w}}  metadata hint: {p.provenance['generator_hints'][0]}")
                if args.explain and p.signals:
                    for s in p.signals[:4]:
                        arrow = "-> AI  " if s["effect"] > 0 else "-> real"
                        print(f"{'':<{w}}    {s['group']:<9} {arrow} {abs(s['effect']):.3f}")
            else:
                print(f"{p.name:<{w}}  {'ERROR':<13} {'':>6}  {'':<11} {p.error}")
        n_ok = sum(p.ok for p in preds)
        print(f"\n{n_ok}/{len(preds)} analysed. Verdicts are probabilistic; see README 'Limits'.")
    return 0 if any(p.ok for p in preds) else 1


def cmd_audit(args: argparse.Namespace) -> int:
    from PIL import Image
    import hashlib
    cfg = _load_cfg(args)
    _setup_logging(cfg.runtime.log_level, args.verbose)
    samples = collect_samples(cfg, args.data, args.real_dir, args.fake_dir, args.manifest)
    metas, bad = [], 0
    for s in samples:
        try:
            with Image.open(s.path) as im:
                w, h = im.size
                fmt = im.format or "unknown"
                exif = {}
                try:
                    ex = im.getexif()
                    exif = {"Make": ex.get(271), "Model": ex.get(272)}
                except Exception:
                    pass
            data = Path(s.path).read_bytes()
            metas.append(ImageMeta(s.path, s.label, s.split, hashlib.sha256(data).hexdigest(), 0, w, h, fmt,
                                   len(data), bool(exif.get("Make") or exif.get("Model"))))
        except Exception:
            bad += 1
    if not metas:
        raise DataError("No readable images found.")
    res = audit_dataset(metas)
    print(f"Scanned {len(samples)} files ({bad} unreadable).\n")
    for cls, d in res["per_class"].items():
        print(f"{cls}: {d['n']} images, formats {d['formats']}")
        print(f"   JPEG {d['jpeg_fraction']:.0%} | camera EXIF {d['camera_exif_fraction']:.0%} | "
              f"median {d['median_width']:.0f}x{d['median_height']:.0f} px | {d['median_bits_per_pixel']:.2f} bits/px | "
              f"{d['distinct_sizes']} distinct sizes")
    print()
    if res["warnings"]:
        for w in res["warnings"]:
            print(f"! {w}")
    else:
        print("No obvious class-confounding differences found (this does not prove there are none).")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    from .model import ModelBundle
    _setup_logging("INFO", args.verbose)
    b = ModelBundle.load(args.model)
    md = b.metadata
    print(f"Model        : {args.model}")
    print(f"Created      : {md.get('created_utc')}  (aidetect {md.get('aidetect_version')})")
    print(f"Features     : {md.get('n_features')} in groups {md.get('feature_groups')}")
    dec = md.get("decision", {})
    print(f"Decision     : threshold {dec.get('threshold'):.3f}, uncertainty margin {dec.get('uncertainty_margin')}, "
          f"aggregation {dec.get('aggregation')}")
    data = md.get("data", {})
    print(f"Training data: {data.get('split_counts')}  | trained on: {md.get('final_fit', {}).get('trained_on')}")
    test = md.get("evaluation", {}).get("test")
    if test:
        print("Held-out test:")
        _print_metrics(test)
    for w in md.get("warnings", []):
        print(f"  ! {w}")
    return 0


def cmd_make_demo(args: argparse.Namespace) -> int:
    from .demo import make_demo_dataset
    _setup_logging("INFO", args.verbose)
    r = make_demo_dataset(args.out, args.n_per_class, args.size, args.seed)
    print(f"Created a SYNTHETIC demo dataset in {r['directory']} ({r['per_class']} images per class, "
          f"{r['size']}px). It is only for smoke-testing the pipeline; accuracy on it says nothing about "
          f"real AI-image detection.")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    _setup_logging("INFO", args.verbose)
    app = Path(__file__).resolve().parents[2] / "app" / "streamlit_app.py"
    if not app.is_file():
        raise DataError(f"Web app file not found at {app}. Run from a source checkout "
                        f"(pip install -e .) or start it manually: streamlit run app/streamlit_app.py")
    try:
        import streamlit  # noqa: F401
    except ImportError as e:
        raise DataError("The web app needs streamlit: pip install streamlit pandas") from e
    env = dict(os.environ)
    if args.model:
        env["AIDETECT_MODEL"] = os.path.abspath(args.model)
    cmd = [sys.executable, "-m", "streamlit", "run", str(app), "--server.port", str(args.port),
           "--server.address", args.host, "--browser.gatherUsageStats", "false"]
    return subprocess.call(cmd, env=env)


COMMANDS = {"train": cmd_train, "evaluate": cmd_evaluate, "predict": cmd_predict, "audit": cmd_audit,
            "info": cmd_info, "make-demo-data": cmd_make_demo, "serve": cmd_serve}


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return COMMANDS[args.command](args)
    except (ConfigError, DataError, ModelError, ImageError, ValueError) as e:
        if getattr(args, "verbose", False):
            raise
        print(f"error: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
