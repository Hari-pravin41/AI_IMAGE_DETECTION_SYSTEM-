"""Command line interface:  aidetect {inspect,train,evaluate,predict,serve,config}"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from .config import Config
from .dataset import DatasetError, discover, scan_dataset
from .utils import NumpyEncoder, dump_json, log


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", help="YAML config file (see `aidetect config`)")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override a config value, e.g. --set crop_size=96 --set augment.jpeg_quality=[50,90]")


def _add_data(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("dataset (choose one way)")
    g.add_argument("--root", help="folder containing class sub-folders named like real/ and fake/ (auto-detected)")
    g.add_argument("--real", nargs="+", metavar="DIR", help="folder(s) of real photographs")
    g.add_argument("--ai", nargs="+", metavar="DIR", help="folder(s) of AI-generated images")
    g.add_argument("--manifest", help="CSV with columns path,label[,source] (label: ai/real or 1/0)")


def _samples(args, cfg: Config):
    return discover(cfg, real_dirs=args.real, ai_dirs=args.ai, root=args.root, manifest=args.manifest)


def _print_audit(audit: dict) -> None:
    pc = audit["per_class"]
    log(f"usable images: {audit['n_usable']}  (real={pc['real']}, ai={pc['ai']})")
    for k, v in audit["sources"].items():
        log(f"  source {k}: {v}")
    if audit.get("file_cue_auc") is not None:
        log(f"file-cue-only AUC (shortcut risk): {audit['file_cue_auc']:.3f}")
    for w in audit["warnings"]:
        log(f"warning: {w}")


def cmd_inspect(args) -> int:
    cfg = Config.load(args.config, args.set)
    scan = scan_dataset(_samples(args, cfg), cfg)
    _print_audit(scan.audit)
    if args.out:
        dump_json(scan.audit, Path(args.out))
        log(f"audit written to {args.out}")
    return 0


def cmd_train(args) -> int:
    from .trainer import train
    cfg = Config.load(args.config, args.set)
    if args.no_cache:
        cfg.use_cache = False
    scan = scan_dataset(_samples(args, cfg), cfg)
    _print_audit(scan.audit)
    pc = scan.audit["per_class"]
    if min(pc.values()) < 10:
        raise DatasetError(f"need at least 10 usable images per class (real={pc['real']}, ai={pc['ai']})")
    summary = train(scan.samples, cfg, args.out, audit=scan.audit, refit_all=args.refit_all)
    ho = summary.get("holdout") or summary["cv"]
    print(json.dumps({"model_dir": str(args.out), "auc": ho.get("roc_auc"), "accuracy": ho["accuracy"],
                      "evaluated_on": "hold-out" if "holdout" in summary else "cross-validation"}, indent=2))
    return 0


def cmd_train_gpu(args) -> int:
    from .gpu import train_gpu
    cfg = Config.load(args.config, args.set)
    scan = scan_dataset(_samples(args, cfg), cfg)
    _print_audit(scan.audit)
    pc = scan.audit["per_class"]
    if min(pc.values()) < 10:
        raise DatasetError(f"need at least 10 usable images per class (real={pc['real']}, ai={pc['ai']})")
    summary = train_gpu(scan.samples, cfg, args.out, args.epochs, args.batch_size,
                        args.learning_rate, args.device, not args.no_pretrained, scan.audit)
    print(json.dumps({"model_dir": str(args.out), "validation": summary["validation"],
                      "trained_on": summary["n_images"]}, indent=2, cls=NumpyEncoder))
    return 0


def cmd_evaluate(args) -> int:
    from .detector import Detector
    from .evaluate import evaluate
    det = Detector.load(args.model)
    cfg = det.cfg
    if args.set or args.config:           # only data-discovery options may differ from the model's config
        user = Config.load(args.config, args.set)
        cfg.class_aliases, cfg.image_extensions = user.class_aliases, user.image_extensions
        cfg.n_jobs, cfg.use_cache, cfg.cache_dir = user.n_jobs, user.use_cache, user.cache_dir
    scan = scan_dataset(_samples(args, cfg), cfg)
    _print_audit(scan.audit)
    res = evaluate(det, scan.samples, args.out, robustness=args.robustness)
    print(json.dumps({k: v for k, v in res["metrics"].items() if k not in ("by_source",)}, indent=2, cls=NumpyEncoder))
    return 0


def cmd_evaluate_gpu(args) -> int:
    from .gpu import GPUDetector, evaluate_gpu
    cfg = Config.load(args.config, args.set)
    scan = scan_dataset(_samples(args, cfg), cfg)
    _print_audit(scan.audit)
    detector = GPUDetector.load(args.model, args.device)
    result = evaluate_gpu(detector, scan.samples, args.out)
    print(json.dumps(result["metrics"], indent=2, cls=NumpyEncoder))
    return 0


def _gather(paths: list[str], recursive: bool, exts: set[str]) -> list[Path]:
    out: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            it = p.rglob("*") if recursive else p.iterdir()
            out.extend(sorted(f for f in it if f.is_file() and f.suffix.lower() in exts and not f.name.startswith(".")))
        elif p.exists():
            out.append(p)
        else:
            raise FileNotFoundError(f"no such file or directory: {raw}")
    return out


def cmd_predict(args) -> int:
    from .detector import Detector
    from .explain import render_panels
    from .loader import ImageError, load_image
    det = Detector.load(args.model)
    files = _gather(args.paths, args.recursive, set(det.cfg.image_extensions))
    if not files:
        log("no images found")
        return 1
    explain_dir = Path(args.explain) if args.explain else None
    rows = []
    if explain_dir:
        explain_dir.mkdir(parents=True, exist_ok=True)
        results = []
        for f in files:
            try:
                img = load_image(f, det.cfg)
            except ImageError as e:
                from .detector import Prediction
                results.append(Prediction(name=f.name, ok=False, error=str(e)))
                continue
            pred = det.predict_loaded(img, explain=True)
            results.append(pred)
            for k, data in render_panels(img, pred).items():
                (explain_dir / f"{f.stem}__{k}.png").write_bytes(data)
    else:
        results = list(det.predict_many(files, workers=args.workers))
    for f, r in zip(files, results):
        d = r.to_dict()
        d["path"] = str(f)
        rows.append(d)
    if args.json:
        json.dump(rows, sys.stdout, indent=2, cls=NumpyEncoder)
        print()
    else:
        for d in rows:
            if not d["ok"]:
                print(f"{d['path']}: ERROR {d['error']}")
            else:
                note = "  [metadata declares AI]" if d["decision_basis"] == "provenance_metadata" else ""
                print(f"{d['path']}: {d['label'].upper():<13} P(AI)={d['prob_ai']:.3f}{note}")
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["path", "label", "prob_ai", "model_label", "decision_basis", "declares_ai_metadata", "error"])
            for d in rows:
                w.writerow([d["path"], d.get("label") or "", "" if d["prob_ai"] is None else f"{d['prob_ai']:.6f}",
                            d.get("model_label") or "", d.get("decision_basis", ""),
                            d["provenance"].get("declares_ai", "") if d["provenance"] else "", d.get("error") or ""])
        log(f"wrote {args.csv}")
    return 0 if all(d["ok"] for d in rows) else 2



def cmd_predict_gpu(args) -> int:
    from .gpu import GPUDetector

    detector = GPUDetector.load(args.model, args.device)
    files = _gather(
        args.paths,
        args.recursive,
        set(Config().image_extensions)
    )

    if not files:
        log("no images found")
        return 1

    rows = []

    for path in files:
        try:
            prediction = detector.predict(path)

            # Add the file path expected by the CLI.
            prediction["path"] = str(path)

            rows.append(prediction)

        except OSError as e:
            rows.append({
                "path": str(path),
                "ok": False,
                "error": str(e)
            })

    if args.json:
        json.dump(rows, sys.stdout, indent=2, cls=NumpyEncoder)
        print()
    else:
        for row in rows:
            if row["ok"]:
                print(
                    f"{row['path']}: "
                    f"{row['label'].upper():<13} "
                    f"P(AI)={row['prob_ai']:.3f}"
                )
            else:
                print(f"{row['path']}: ERROR {row['error']}")

    return 0 if all(row["ok"] for row in rows) else 2


def cmd_serve(args) -> int:
    from .server import serve
    serve(args.model, args.host, args.port)
    return 0


def cmd_serve_gpu(args) -> int:
    from .server import serve_gpu
    serve_gpu(args.model, args.host, args.port, args.device)
    return 0


def cmd_config(args) -> int:
    cfg = Config.load(args.config, args.set)
    text = cfg.to_yaml()
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        log(f"wrote {args.out}")
    else:
        print(text)
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="aidetect", description="Detect AI-generated images; train on your own data.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("inspect", help="audit a dataset (counts, duplicates, shortcut risk) without training")
    _add_common(p), _add_data(p)
    p.add_argument("--out", help="write the audit as JSON")
    p.set_defaults(fn=cmd_inspect)

    p = sub.add_parser("train", help="train a model on your labelled images")
    _add_common(p), _add_data(p)
    p.add_argument("--out", required=True, help="output model directory")
    p.add_argument("--refit-all", action="store_true", help="after evaluating, refit on train + hold-out data")
    p.add_argument("--no-cache", action="store_true", help="do not read/write the feature cache")
    p.set_defaults(fn=cmd_train)

    p = sub.add_parser("train-gpu", help="fine-tune a pretrained image model on an NVIDIA GPU")
    _add_common(p), _add_data(p)
    p.add_argument("--out", required=True, help="output GPU model directory")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=8, help="reduce this if CUDA runs out of memory")
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--no-pretrained", action="store_true", help="do not download ImageNet weights")
    p.set_defaults(fn=cmd_train_gpu)

    p = sub.add_parser("evaluate", help="evaluate a model on a labelled dataset (e.g. a new generator)")
    _add_common(p), _add_data(p)
    p.add_argument("--model", required=True)
    p.add_argument("--out", help="directory for evaluation.json / evaluation.png")
    p.add_argument("--robustness", action="store_true", help="also test under JPEG/resize/blur/noise")
    p.set_defaults(fn=cmd_evaluate)

    p = sub.add_parser("evaluate-gpu", help="evaluate a GPU model on a labelled dataset")
    _add_common(p), _add_data(p)
    p.add_argument("--model", required=True)
    p.add_argument("--out", help="directory for evaluation.json / evaluation.png")
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.set_defaults(fn=cmd_evaluate_gpu)

    p = sub.add_parser("predict", help="classify image files or folders")
    p.add_argument("paths", nargs="+")
    p.add_argument("--model", required=True)
    p.add_argument("-r", "--recursive", action="store_true")
    p.add_argument("--json", action="store_true")
    p.add_argument("--csv", help="write results to a CSV file")
    p.add_argument("--explain", metavar="DIR", help="save heatmap/spectrum/noise/ELA panels into DIR")
    p.add_argument("--workers", type=int, default=1)
    p.set_defaults(fn=cmd_predict)

    p = sub.add_parser("predict-gpu", help="classify images with a saved GPU model")
    p.add_argument("paths", nargs="+")
    p.add_argument("--model", required=True)
    p.add_argument("-r", "--recursive", action="store_true")
    p.add_argument("--json", action="store_true")
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.set_defaults(fn=cmd_predict_gpu)

    p = sub.add_parser("serve", help="run the web UI and JSON API")
    p.add_argument("--model", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(fn=cmd_serve)

    p = sub.add_parser("serve-gpu", help="run the web UI with a saved GPU model")
    p.add_argument("--model", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.set_defaults(fn=cmd_serve_gpu)

    p = sub.add_parser("config", help="print or write the full default configuration")
    _add_common(p)
    p.add_argument("--out")
    p.set_defaults(fn=cmd_config)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except (DatasetError, ValueError, FileNotFoundError) as e:
        log(f"error: {e}")
        return 1
    except KeyboardInterrupt:
        log("interrupted")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
