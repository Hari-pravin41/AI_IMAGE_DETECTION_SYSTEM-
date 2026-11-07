import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, confusion_matrix
)

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ai-dir", required=True)
    parser.add_argument("--real-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    ai_dir = Path(args.ai_dir)
    real_dir = Path(args.real_dir)
    model_dir = Path(args.model_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    classifier_file = model_dir / "classifier.joblib"
    if not classifier_file.exists():
        raise FileNotFoundError(f"Missing classifier: {classifier_file}")
    classifier = joblib.load(classifier_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("Loading DINOv2 ViT-S/14...")
    encoder = torch.hub.load(
        "facebookresearch/dinov2", "dinov2_vits14",
        trust_repo=True, verbose=True
    )
    encoder.eval().to(device)

    transform = transforms.Compose([
        transforms.Resize(
            (224, 224),
            interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225)
        ),
    ])

    results = []
    skipped = []

    for folder, actual_label in [(ai_dir, 1), (real_dir, 0)]:
        if not folder.is_dir():
            raise FileNotFoundError(f"Image folder not found: {folder}")

        files = sorted(
            p for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in EXTENSIONS
        )
        label_name = "AI" if actual_label == 1 else "REAL"
        print(f"\n{label_name}: {len(files)} images")

        for i, path in enumerate(files, 1):
            try:
                with Image.open(path) as im:
                    image = im.convert("RGB")

                tensor = transform(image).unsqueeze(0).to(device)

                with torch.inference_mode():
                    output = encoder(tensor)
                    if isinstance(output, dict):
                        features = output.get("x_norm_clstoken")
                        if features is None:
                            features = output.get("x_prenorm")
                        if features is None:
                            raise RuntimeError("DINOv2 feature output not found.")
                    elif isinstance(output, (tuple, list)):
                        features = output[0]
                    else:
                        features = output

                    if features.ndim == 3:
                        features = features[:, 0]
                    features = features.detach().float().cpu().numpy()

                probability = float(classifier.predict_proba(features)[0, 1])
                predicted = 1 if probability >= 0.5 else 0

                results.append({
                    "path": str(path.resolve()),
                    "actual": label_name,
                    "predicted": "AI" if predicted else "REAL",
                    "probability_ai": probability,
                    "correct": predicted == actual_label,
                    "subfolder": str(path.parent.relative_to(folder))
                })

            except (OSError, ValueError, UnidentifiedImageError) as exc:
                skipped.append({"path": str(path), "reason": str(exc)})

            if i % 100 == 0 or i == len(files):
                print(f"Processed {i}/{len(files)}")

    if not results:
        raise RuntimeError("No images were evaluated.")

    y_true = np.array([1 if r["actual"] == "AI" else 0 for r in results])
    y_pred = np.array([1 if r["predicted"] == "AI" else 0 for r in results])
    probabilities = np.array([r["probability_ai"] for r in results])
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    metrics = {
        "total_evaluated": len(results),
        "skipped_images": len(skipped),
        "device": str(device),
        "threshold": 0.5,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_ai": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_ai": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_ai": float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity_real": float(tn / (tn + fp)) if tn + fp else None,
        "roc_auc": float(roc_auc_score(y_true, probabilities))
            if len(np.unique(y_true)) == 2 else None,
        "confusion_matrix": {
            "true_real_pred_real": int(tn),
            "false_ai_alarm": int(fp),
            "missed_ai": int(fn),
            "true_ai_pred_ai": int(tp)
        },
        "warning": "Check dataset overlap and duplicate images before claiming independent benchmark performance."
    }

    with (out_dir / "predictions.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (out_dir / "skipped_images.json").write_text(
        json.dumps(skipped, indent=2), encoding="utf-8"
    )

    groups = {}
    for row in results:
        key = f'{row["actual"]}/{row["subfolder"]}'
        groups.setdefault(key, []).append(row)

    summary = []
    for key, rows in sorted(groups.items()):
        summary.append({
            "group": key,
            "count": len(rows),
            "accuracy": sum(r["correct"] for r in rows) / len(rows)
        })

    (out_dir / "subfolder_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\n===== EVALUATION RESULTS =====")
    for key, value in metrics.items():
        if key not in {"warning", "confusion_matrix"}:
            print(f"{key}: {value}")
    print("Confusion matrix:", metrics["confusion_matrix"])
    print("Results saved to:", out_dir.resolve())

if __name__ == "__main__":
    main()
