import csv
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image, ImageFile
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, confusion_matrix
)

ROOT = Path(".")
DATA = ROOT / "datasets/ai_detector_arena_v0_1_source/benchmark-v0.1/images"
MODEL = ROOT / "models/dinov2_vits14_clean_v1"
BATCH_SIZE = 2
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

ImageFile.LOAD_TRUNCATED_IMAGES = True

def folder_label(name):
    name = name.lower().replace("_", "-").replace(" ", "-")
    tokens = set(name.split("-"))
    if "real" in tokens or "real" in name:
        return 0
    if tokens.intersection({"ai", "fake", "synthetic", "generated"}):
        return 1
    return None

def main():
    if not DATA.is_dir():
        raise SystemExit(f"Benchmark folder not found: {DATA.resolve()}")
    classifier_path = MODEL / "classifier.joblib"
    if not classifier_path.is_file():
        raise SystemExit(f"Model not found: {classifier_path.resolve()}")

    files = []
    for folder in sorted(p for p in DATA.iterdir() if p.is_dir()):
        label = folder_label(folder.name)
        print(f"Folder: {folder.name} -> label: {label}")
        if label is None:
            continue
        for p in folder.rglob("*"):
            if p.is_file() and p.suffix.lower() in EXTENSIONS:
                files.append((p, label))

    if not files or len({label for _, label in files}) != 2:
        raise SystemExit(
            "Could not identify both AI and real folders. "
            f"Folders found: {[p.name for p in DATA.iterdir() if p.is_dir()]}"
        )

    print(f"\nBenchmark images: {len(files)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    classifier = joblib.load(classifier_path)

    print("Loading DINOv2 encoder...")
    encoder = torch.hub.load(
        "facebookresearch/dinov2", "dinov2_vits14",
        trust_repo=True, verbose=False
    ).to(device).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)

    transform = transforms.Compose([
        transforms.Resize(
            (224, 224),
            interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225)
        )
    ])

    y_true, probabilities, paths = [], [], []
    skipped = []

    for start in range(0, len(files), BATCH_SIZE):
        batch = files[start:start + BATCH_SIZE]
        tensors, valid_items = [], []

        for path, label in batch:
            try:
                with Image.open(path) as image:
                    image.load()
                    tensors.append(transform(image.convert("RGB")))
                    valid_items.append((path, label))
            except Exception as exc:
                skipped.append({"path": str(path), "error": str(exc)})

        if not tensors:
            continue

        images = torch.stack(tensors).to(device)
        with torch.inference_mode():
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=(device.type == "cuda")
            ):
                output = encoder(images)
                if isinstance(output, dict):
                    features = output.get(
                        "x_norm_clstoken", output.get("x_prenorm")
                    )
                    if features is None:
                        raise RuntimeError("Unexpected DINOv2 output dictionary")
                    if features.ndim == 3:
                        features = features[:, 0]
                elif isinstance(output, (tuple, list)):
                    features = output[0]
                    if features.ndim == 3:
                        features = features[:, 0]
                else:
                    features = output

        probs = classifier.predict_proba(
            features.float().cpu().numpy()
        )[:, 1]

        for (path, label), prob in zip(valid_items, probs):
            paths.append(str(path))
            y_true.append(label)
            probabilities.append(float(prob))

        done = min(start + len(batch), len(files))
        if done % 100 < BATCH_SIZE or done == len(files):
            print(f"Processed {done}/{len(files)} images")

    del encoder
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    y_pred = (probabilities >= 0.5).astype(int)

    if len(y_true) == 0 or len(np.unique(y_true)) != 2:
        raise SystemExit("Evaluation failed: insufficient valid images or labels.")

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    metrics = {
        "model": str(MODEL.resolve()),
        "benchmark_folder": str(DATA.resolve()),
        "evaluated_images": int(len(y_true)),
        "skipped_images": len(skipped),
        "threshold": 0.5,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_ai": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_ai": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity_real": float(tn / (tn + fp)) if tn + fp else None,
        "f1_ai": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp),
            "fn": int(fn), "tp": int(tp)
        }
    }

    output_dir = MODEL
    (output_dir / "benchmark_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    with (output_dir / "benchmark_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "image", "actual_label", "probability_ai", "predicted_label"
        ])
        for path, actual, prob, pred in zip(
            paths, y_true, probabilities, y_pred
        ):
            writer.writerow([
                path, "AI" if actual == 1 else "REAL",
                round(float(prob), 6),
                "AI" if pred == 1 else "REAL"
            ])

    if skipped:
        (output_dir / "benchmark_skipped.json").write_text(
            json.dumps(skipped, indent=2), encoding="utf-8"
        )

    print("\n=== BENCHMARK RESULTS ===")
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved metrics: {(output_dir / 'benchmark_metrics.json').resolve()}")
    print(f"Saved predictions: {(output_dir / 'benchmark_predictions.csv').resolve()}")

if __name__ == "__main__":
    main()
