"""Predict with the standalone frozen DINOv2 + LogisticRegression experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import torch
from PIL import Image, ImageFile
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict whether an image is AI-generated using the saved DINOv2 experiment.")
    parser.add_argument("--image", required=True, help="Path to one image")
    parser.add_argument("--model-dir", default="models/dinov2_vits14_v1", help="Directory containing classifier.joblib")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--margin", type=float, default=0.10,
                        help="Report UNCERTAIN when probability is close to threshold (default: 0.10)")
    args = parser.parse_args()

    image_path = Path(args.image)
    model_dir = Path(args.model_dir)
    classifier_path = model_dir / "classifier.joblib"
    if not image_path.is_file():
        parser.error(f"Image not found: {image_path}")
    if not classifier_path.is_file():
        parser.error(f"Classifier not found: {classifier_path}")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Re-run with --device cpu")

    device = torch.device(args.device)
    print("Loading pretrained DINOv2 ViT-S/14 encoder...")
    encoder = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", trust_repo=True, verbose=False)
    encoder = encoder.to(device).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)

    classifier = joblib.load(classifier_path)
    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    try:
        with Image.open(image_path) as image:
            image.load()
            original_format = image.format
            original_size = list(image.size)
            tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    except Exception as exc:
        parser.error(f"Could not read image: {exc}")

    use_amp = device.type == "cuda"
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            output = encoder(tensor)
            if isinstance(output, dict):
                features = output.get("x_norm_clstoken", output.get("x_prenorm"))
                if features is None:
                    parser.error("Unexpected DINOv2 output dictionary.")
                if features.ndim == 3:
                    features = features[:, 0]
            elif isinstance(output, (tuple, list)):
                features = output[0]
                if features.ndim == 3:
                    features = features[:, 0]
            else:
                features = output
            if features.ndim != 2:
                parser.error(f"Unexpected feature shape: {tuple(features.shape)}")
            feature_array = features.float().cpu().numpy()

    prob_ai = float(classifier.predict_proba(feature_array)[0, 1])
    if abs(prob_ai - args.threshold) < args.margin:
        label = "UNCERTAIN"
    elif prob_ai >= args.threshold:
        label = "AI_GENERATED"
    else:
        label = "REAL"

    result = {
        "image": str(image_path.resolve()),
        "format": original_format,
        "size": original_size,
        "prediction": label,
        "probability_ai": round(prob_ai, 6),
        "threshold": args.threshold,
        "uncertain_margin": args.margin,
        "model": "DINOv2 ViT-S/14 frozen encoder + LogisticRegression",
        "warning": "Experimental detector. A probability is not a guarantee; performance on independent real-world images may differ.",
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
