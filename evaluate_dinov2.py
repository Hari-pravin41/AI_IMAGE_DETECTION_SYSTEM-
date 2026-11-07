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
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

SUPPORTED = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

def get_features(encoder, image_tensor):
    with torch.inference_mode():
        output = encoder(image_tensor)
        if isinstance(output, dict):
            feat = output.get("x_norm_clstoken")
            if feat is None:
                feat = output.get("x_prenorm")
            if feat is None:
                raise RuntimeError("DINOv2 output has no x_norm_clstoken or x_prenorm.")
        elif isinstance(output, (tuple, list)):
            feat = output[0]
        else:
            feat = output

        if feat.ndim == 3:
            feat = feat[:, 0]
        if feat.ndim != 2:
            raise RuntimeError(f"Unexpected feature shape: {tuple(feat.shape)}")
        return feat.detach().float().cpu().numpy()

def main():
    parser = argparse.ArgumentParser(description="Evaluate the local DINOv2 AI-image detector on AI and real image folders.")
    parser.add_argument("--ai-dir", required=True)
    parser.add_argument("--real-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    ai_dir = Path(args.ai_dir).resolve()
    real_dir = Path(args.real_dir).resolve()
    model_dir = Path(args.model_dir).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    classifier_path = model_dir / "classifier.joblib"
    if not classifier_path.exists():
        raise FileNotFoundError(f"Classifier not found: {classifier_path}")
    classifier = joblib.load(classifier_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("Loading DINOv2 ViT-S/14 encoder...")
    encoder = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", trust_repo=True, verbose=True)
    encoder.eval().to(device)

    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    records = []
    skipped = []

    # Process one image at a time to fit a 4 GB GPU.
    for folder, label, label_name in [(ai_dir, 1, "AI"), (real_dir, 0, "REAL")]:
        if not folder.is_dir():
            raise NotADirectoryError(f"Folder not found: {folder}")
        paths = sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED)
        print(f"\n{label_name} folder: {len(paths)} supported images")
        for index, path in enumerate(paths, start=1):
            try:
                with Image.open(path) as im:
                    image = im.convert("RGB")
                tensor = transform(image).unsqueeze(0).to(device)
                features = get_features(encoder, tensor)
                probability_ai = float(classifier.predict_proba(features)[0, 1])
                prediction = 1 if probability_ai >= 0.5 else 0
                records.append({
                    "path": str(path),
                    "actual": label_name,
                    "predicted": "AI" if prediction == 1 else "REAL",
                    "probability_ai": probability_ai,
                    "correct": prediction == label,
                    "subfolder": str(path.parent.relative_to(folder)) if path.parent != folder else ".",
                })
            except (UnidentifiedImageError, OSError, ValueError, RuntimeError) as exc:
                skipped.append({"path": str(path), "reason": str(exc)})
            if index % 100 == 0 or index == len(paths):
                print(f"  Processed {index}/{len(paths)}")

    if not records:
        raise RuntimeError("No images were evaluated. Check the folder paths and image files.")

    y_true = np.array([1 if r["actual"] == "AI" else 0 for r in records], dtype=int)
    y_pred = np.array([1 if r["predicted"] == "AI" else 0 for r in records], dtype=int)
    probs = np.array([r["probability_ai"] for r in records], dtype=float)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics = {
        "model": "DINOv2 ViT-S/14 frozen encoder + LogisticRegression",
        "device": str(device),
        "total_evaluated": len(records),
        "skipped_images": len(skipped),
        "threshold_probability_ai": 0.5,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_ai": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_ai": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_ai": float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity_real": float(tn / (tn + fp)) if (tn + fp) else None,
        "roc_auc": float(roc_auc_score(y_true, probs)) if len(np.unique(y_true)) == 2 else None,
        "average_precision_ai": float(average_precision_score(y_true, probs)) if len(np.unique(y_true)) == 2 else None,
        "confusion_matrix": {"TN_real_pred_real": int(tn), "FP_real_pred_ai": int(fp), "FN_ai_pred_real": int(fn), "TP_ai_pred_ai": int(tp)},
        "classification_report": classification_report(y_true, y_pred, labels=[0, 1], target_names=["REAL", "AI"], output_dict=True, zero_division=0),
        "warning": "Treat these results as diagnostic until you verify that this benchmark has no overlap or near-duplicates with training/validation data and confirm labels/source independence."
    }

    predictions_csv = out_dir / "predictions.csv"
    with predictions_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "actual", "predicted", "probability_ai", "correct", "subfolder"])
        writer.writeheader()
        writer.writerows(records)

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / "skipped_images.json").write_text(json.dumps(skipped, indent=2), encoding="utf-8")

    # Per-subfolder summary is useful for spotting weak source categories.
    groups = {}
    for r in records:
        key = f'{r["actual"]}/{r["subfolder"]}'
        groups.setdefault(key, []).append(r)
    source_summary = []
    for key, group in sorted(groups.items()):
        source_summary.append({
            "group": key,
            "count": len(group),
            "accuracy": sum(bool(x["correct"]) for x in group) / len(group),
            "false_ai_predictions": sum(x["actual"] == "REAL" and x["predicted"] == "AI" for x in group),
            "missed_ai_predictions": sum(x["actual"] == "AI" and x["predicted"] == "REAL" for x in group),
        })
    (out_dir / "subfolder_summary.json").write_text(json.dumps(source_summary, indent=2), encoding="utf-8")

    print("\n=== EVALUATION RESULTS ===")
    for key in ["total_evaluated", "skipped_images", "accuracy", "balanced_accuracy", "precision_ai", "recall_ai", "specificity_real", "f1_ai", "roc_auc", "average_precision_ai"]:
        print(f"{key}: {metrics[key]}")
    print("Confusion matrix:", metrics["confusion_matrix"])
    print(f"\nSaved files in: {out_dir}")
    print(" - predictions.csv")
    print(" - metrics.json")
    print(" - subfolder_summary.json")
    print(" - skipped_images.json")
    print("\nNote: this is diagnostic until benchmark independence/duplicates are checked.")

if __name__ == "__main__":
    main()
