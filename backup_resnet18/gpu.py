"""Optional GPU image classifier for AI-generated-image detection."""
from __future__ import annotations

import contextlib
import io
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

from .config import Config
from .dataset import DatasetError, Sample
from .evaluation import compute_metrics, per_source, plot_report
from .model import choose_threshold
from .scope import SCOPE_WARNING, SUPPORTED_DOMAIN
from .utils import dump_json, log

_MODEL_FILE = "vision_model.pt"
_IMAGE_SIZE = 224
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


class _RandomImageDegradation:
    def __call__(self, image: Image.Image) -> Image.Image:
        if torch.rand(()) < 0.5:
            scale = float(torch.empty(1).uniform_(0.5, 0.95))
            small_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
            image = image.resize(small_size, Image.Resampling.BILINEAR).resize(
                image.size, Image.Resampling.BILINEAR
            )
        return image


class _CanonicalJPEG:
    def __init__(self, random_quality: bool):
        self.random_quality = random_quality

    def __call__(self, image: Image.Image) -> Image.Image:
        buffer = io.BytesIO()
        quality = int(torch.randint(45, 96, ()).item()) if self.random_quality else 85
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            return compressed.convert("RGB")


def _transforms(training: bool, canonical_jpeg: bool = False):
    ops = []
    if training:
        ops.extend([
            transforms.Resize(256),
            transforms.RandomResizedCrop(_IMAGE_SIZE, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            _RandomImageDegradation(),
        ])
    else:
        ops.extend([transforms.Resize(256), transforms.CenterCrop(_IMAGE_SIZE)])
    if canonical_jpeg:
        ops.append(_CanonicalJPEG(random_quality=training))
    ops.extend([transforms.ToTensor(), transforms.Normalize(_MEAN, _STD)])
    return transforms.Compose(ops)


class _ImageDataset(Dataset):
    def __init__(
        self, samples: list[Sample], indices: np.ndarray, training: bool,
        canonical_jpeg: bool = False,
    ):
        self.samples = [samples[int(i)] for i in indices]
        self.transform = _transforms(training, canonical_jpeg)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        with Image.open(sample.path) as image:
            image = image.convert("RGB")
            image.load()
        return self.transform(image), sample.label


def _build_model(pretrained: bool) -> nn.Module:
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 2)
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.layer4.parameters():
        parameter.requires_grad = True
    for parameter in model.fc.parameters():
        parameter.requires_grad = True
    return model


def _train_epoch(model: nn.Module, loader: DataLoader, optimizer, scaler, device: torch.device) -> None:
    model.train()
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
    try:
        for images, labels in loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            labels = labels.to(device, non_blocking=device.type == "cuda")
            optimizer.zero_grad(set_to_none=True)
            autocast = torch.autocast("cuda") if device.type == "cuda" else contextlib.nullcontext()
            with autocast:
                loss = nn.functional.cross_entropy(model(images), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        raise DatasetError("CUDA ran out of memory; retry with a smaller --batch-size (for example 4)") from exc


def _probabilities(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    result = []
    with torch.inference_mode():
        for images, _ in loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            autocast = torch.autocast("cuda") if device.type == "cuda" else contextlib.nullcontext()
            with autocast:
                probs = model(images).softmax(dim=1)[:, 1]
            result.extend(probs.float().cpu().tolist())
    return np.asarray(result, dtype=np.float64)


def _loader(samples: list[Sample], indices: np.ndarray, batch_size: int,
            training: bool, device: torch.device,
            canonical_jpeg: bool = False) -> DataLoader:
    return DataLoader(
        _ImageDataset(samples, indices, training, canonical_jpeg),
        batch_size=batch_size,
        shuffle=training,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )


def _source_folds(samples: list[Sample], labels: np.ndarray, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    by_source: dict[str, list[int]] = {}
    for index, sample in enumerate(samples):
        by_source.setdefault(sample.source, []).append(index)
    real_sources = [source for source, indices in by_source.items() if labels[indices[0]] == 0]
    ai_sources = [source for source, indices in by_source.items() if labels[indices[0]] == 1]
    n_folds = min(5, len(real_sources), len(ai_sources))
    if n_folds < 2:
        raise DatasetError("source-held-out validation needs at least two sources per class")

    fold_sources: list[list[str]] = [[] for _ in range(n_folds)]
    class_counts = {0: [0] * n_folds, 1: [0] * n_folds}
    for label, sources in ((0, real_sources), (1, ai_sources)):
        rng = np.random.default_rng(seed + label)
        shuffled = list(sources)
        rng.shuffle(shuffled)
        for source in sorted(shuffled, key=lambda item: -len(by_source[item])):
            fold = min(range(n_folds), key=lambda index: class_counts[label][index])
            fold_sources[fold].append(source)
            class_counts[label][fold] += len(by_source[source])

    indices = np.arange(len(samples))
    folds = []
    for sources in fold_sources:
        val = np.asarray([i for source in sources for i in by_source[source]], dtype=np.int64)
        train_mask = np.ones(len(samples), dtype=bool)
        train_mask[val] = False
        folds.append((indices[train_mask], val))
    return folds


def _fit(model: nn.Module, samples: list[Sample], indices: np.ndarray, epochs: int,
         batch_size: int, learning_rate: float, device: torch.device) -> None:
    train_loader = _loader(samples, indices, batch_size, True, device, canonical_jpeg=True)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=learning_rate,
        weight_decay=1e-3,
    )
    model.to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    for epoch in range(epochs):
        _train_epoch(model, train_loader, optimizer, scaler, device)
        log(f"vision training epoch {epoch + 1}/{epochs}")


def train_gpu(
    samples: list[Sample],
    cfg: Config,
    out_dir: str | Path,
    epochs: int = 12,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    device_name: str = "cuda",
    pretrained: bool = True,
    audit: dict | None = None,
) -> dict:
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise DatasetError("CUDA is unavailable; install a CUDA-enabled PyTorch build and check the NVIDIA driver")
    if device_name not in {"cuda", "cpu"}:
        raise ValueError("device must be 'cuda' or 'cpu'")
    device = torch.device(device_name)
    y = np.asarray([sample.label for sample in samples], dtype=np.int64)
    if min(np.bincount(y, minlength=2)) < 10:
        raise DatasetError("GPU training requires at least 10 usable images per class")
    torch.manual_seed(cfg.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)
    folds = _source_folds(samples, y, cfg.seed)
    n_splits = len(folds)
    oof_probs = np.full(len(samples), np.nan, dtype=np.float64)
    fold_epochs = []
    for fold, (train_idx, val_idx) in enumerate(folds):
        if len(np.unique(y[val_idx])) != 2:
            raise DatasetError(f"source-held-out validation fold {fold + 1} lacks one label")
        torch.manual_seed(cfg.seed + fold)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(cfg.seed + fold)
        log(f"source-held-out fold {fold + 1}/{n_splits}: "
            f"{len(train_idx)} train / {len(val_idx)} validation images")
        validation_model = _build_model(pretrained)
        validation_model.to(device)
        validation_loader = _loader(
            samples, val_idx, batch_size, False, device, canonical_jpeg=True
        )
        train_loader = _loader(
            samples, train_idx, batch_size, True, device, canonical_jpeg=True
        )
        optimizer = torch.optim.AdamW(
            (p for p in validation_model.parameters() if p.requires_grad),
            lr=learning_rate,
            weight_decay=1e-3,
        )
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        best_loss, best_epoch, patience = float("inf"), 1, 0
        best_state = None
        for epoch in range(epochs):
            _train_epoch(validation_model, train_loader, optimizer, scaler, device)
            fold_probs = _probabilities(validation_model, validation_loader, device)
            val_y = y[val_idx]
            val_loss = float(-np.mean(
                val_y * np.log(np.clip(fold_probs, 1e-7, 1.0))
                + (1 - val_y) * np.log(np.clip(1 - fold_probs, 1e-7, 1.0))
            ))
            log(f"fold {fold + 1}, epoch {epoch + 1}/{epochs}: log-loss={val_loss:.4f}")
            if val_loss < best_loss:
                best_loss, best_epoch, patience = val_loss, epoch + 1, 0
                best_state = {key: value.detach().cpu().clone()
                              for key, value in validation_model.state_dict().items()}
            else:
                patience += 1
                if patience >= 2:
                    break
        if best_state is None:
            raise DatasetError(f"source-held-out fold {fold + 1} produced no checkpoint")
        validation_model.load_state_dict(best_state)
        oof_probs[val_idx] = _probabilities(validation_model, validation_loader, device)
        fold_epochs.append(best_epoch)
        del validation_model, best_state
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not np.isfinite(oof_probs).all():
        raise DatasetError("source-held-out validation did not produce a score for every image")
    threshold = choose_threshold(y, oof_probs, "balanced_accuracy")
    validation = compute_metrics(y, oof_probs, threshold, margin=0.15)
    validation["by_source"] = per_source(y, oof_probs, [sample.source for sample in samples], threshold)
    selected_epochs = max(1, int(np.median(fold_epochs)))
    log(f"source-held-out validation: AUC={validation.get('roc_auc', 0):.3f} "
        f"accuracy={validation['accuracy']:.3f} threshold={threshold:.3f}; "
        f"refit epochs={selected_epochs}")

    torch.manual_seed(cfg.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)
    final_model = _build_model(pretrained)
    _fit(final_model, samples, np.arange(len(samples)), selected_epochs, batch_size,
         learning_rate, device)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": final_model.cpu().state_dict(),
        "threshold": threshold,
        "margin": 0.15,
        "image_size": _IMAGE_SIZE,
        "mean": _MEAN,
        "std": _STD,
        "architecture": "resnet18",
        "pretrained": pretrained,
        "canonical_jpeg": True,
    }, out_path / _MODEL_FILE)
    summary = {
        "architecture": "resnet18",
        "pretrained": pretrained,
        "device": str(device),
        "n_images": len(samples),
        "n_train_images": len(samples),
        "n_validation_images": len(samples),
        "validation_method": "source_grouped_out_of_fold",
        "validation_folds": n_splits,
        "fold_epochs": fold_epochs,
        "epochs_trained": selected_epochs,
        "batch_size": batch_size,
        "validation": validation,
        "audit": audit or {},
    }
    dump_json(summary, out_path / "metrics.json")
    dump_json({
        "threshold": threshold,
        "margin": 0.15,
        "architecture": "resnet18",
        "canonical_jpeg": True,
    },
              out_path / "model_info.json")
    plot_report(y, oof_probs, threshold, out_path / "validation.png", "Source-held-out GPU validation")
    return summary


class GPUDetector:
    supported_domain = SUPPORTED_DOMAIN
    scope_warning = SCOPE_WARNING

    def __init__(
        self, model: nn.Module, threshold: float, margin: float, device: torch.device,
        canonical_jpeg: bool = False, model_identifier: str = "resnet18",
    ):
        self.model = model.eval()
        self.threshold = threshold
        self.margin = margin
        self.device = device
        self.canonical_jpeg = canonical_jpeg
        self.model_identifier = model_identifier
        self.transform = _transforms(False, canonical_jpeg)

    @classmethod
    def load(cls, model_path: str | Path, device_name: str = "cuda") -> "GPUDetector":
        if device_name == "cuda" and not torch.cuda.is_available():
            raise DatasetError("CUDA is unavailable; use --device cpu or install a CUDA-enabled PyTorch build")
        if device_name not in {"cuda", "cpu"}:
            raise ValueError("device must be 'cuda' or 'cpu'")
        device = torch.device(device_name)
        path = Path(model_path)
        if path.is_dir():
            path = path / _MODEL_FILE
        if not path.is_file():
            raise FileNotFoundError(f"GPU model not found: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
        model.load_state_dict(payload["state_dict"])
        return cls(
            model.to(device),
            float(payload["threshold"]),
            float(payload["margin"]),
            device,
            bool(payload.get("canonical_jpeg", False)),
            path.parent.name,
        )

    def predict_image(self, image: Image.Image, name: str = "image") -> dict:
        image_format = image.format
        image = image.convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            autocast = torch.autocast("cuda") if self.device.type == "cuda" else contextlib.nullcontext()
            with autocast:
                prob_ai = float(self.model(tensor).softmax(dim=1)[0, 1].cpu())
        if abs(prob_ai - self.threshold) < self.margin:
            label = "uncertain"
        else:
            label = "ai_generated" if prob_ai >= self.threshold else "real"
        return {"name": name, "ok": True, "label": label, "model_label": label, "prob_ai": prob_ai,
                "threshold": self.threshold, "margin": self.margin, "size": list(image.size),
                "format": image_format, "n_crops": 1, "decision_basis": "model",
                "warnings": [self.scope_warning],
                "provenance": {}}

    def predict(self, path: str | Path) -> dict:
        image_path = Path(path)
        with Image.open(image_path) as image:
            image.load()
            return self.predict_image(image, str(image_path))


def evaluate_gpu(detector: GPUDetector, samples: list[Sample], out_dir: str | Path | None = None) -> dict:
    indices = np.arange(len(samples))
    loader = _loader(
        samples, indices, 8, False, detector.device, detector.canonical_jpeg
    )
    probabilities = _probabilities(detector.model, loader, detector.device)
    labels = np.asarray([sample.label for sample in samples], dtype=np.int64)
    if len(np.unique(labels)) != 2:
        raise DatasetError("evaluation data must contain both real and AI images")
    metrics = compute_metrics(labels, probabilities, detector.threshold, detector.margin)
    metrics["by_source"] = per_source(
        labels, probabilities, [sample.source for sample in samples], detector.threshold
    )
    result = {"metrics": metrics}
    log(f"GPU evaluation: AUC={metrics.get('roc_auc', 0):.3f} "
        f"accuracy={metrics['accuracy']:.3f} (n={len(labels)})")
    if out_dir:
        out_path = Path(out_dir)
        dump_json(result, out_path / "evaluation.json")
        plot_report(labels, probabilities, detector.threshold, out_path / "evaluation.png",
                    "GPU model evaluation")
    return result
