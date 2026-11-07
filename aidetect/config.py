"""Configuration: one dataclass, overridable from YAML files and ``key=value`` strings."""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

FEATURE_GROUPS = ("spectral", "noise", "texture", "color", "compression")
MEMBERS = ("hgb", "extra_trees", "logreg")
THRESHOLD_METRICS = ("balanced_accuracy", "f1", "youden")


def _aug_default() -> dict:
    return {
        "prob_each": 0.5,
        "jpeg_quality": [55, 95],
        "scale": [0.5, 1.0],
        "blur_sigma": [0.3, 1.2],
        "noise_sigma": [0.5, 3.0],
    }


def _aliases_default() -> dict:
    return {
        "ai": ["ai", "fake", "fakes", "generated", "synthetic", "artificial", "gan", "diffusion",
               "sd", "sdxl", "midjourney", "dalle", "dall", "flux", "deepfake", "aigc", "machine"],
        "real": ["real", "reals", "authentic", "natural", "genuine", "photo", "photos", "photograph",
                 "human", "camera", "original", "nature"],
    }


def _robustness_default() -> list:
    return [
        {"name": "jpeg_q90", "ops": [{"op": "jpeg", "quality": 90}]},
        {"name": "jpeg_q70", "ops": [{"op": "jpeg", "quality": 70}]},
        {"name": "resize_0.75", "ops": [{"op": "resize", "scale": 0.75}]},
        {"name": "blur_1.0", "ops": [{"op": "blur", "sigma": 1.0}]},
        {"name": "noise_3", "ops": [{"op": "noise", "sigma": 3.0}]},
        {"name": "social_media", "ops": [{"op": "resize", "scale": 0.75}, {"op": "jpeg", "quality": 75}]},
    ]


@dataclass
class HighLevelConfig:
    enabled: bool = True
    method: str = "logreg"              # logreg | mlp
    weight: float = 0.35                  # blend between forensic score and high-level score
    hidden_dim: int = 64


@dataclass
class Config:
    # --- analysis window -------------------------------------------------
    crop_size: int = 128                 # native-resolution analysis window (multiple of 16)
    crops_per_image: int = 6             # windows analysed per image
    heatmap_max_crops: int = 36          # windows used when building an explanation heatmap
    min_side: int | None = None          # smallest accepted image side (None -> crop_size)
    max_pixels: int = 100_000_000
    max_file_mb: float = 64.0
    # --- features --------------------------------------------------------
    feature_groups: list = field(default_factory=lambda: list(FEATURE_GROUPS))
    radial_bins: int = 24
    use_metadata_features: bool = False  # file-level cues (format/size/EXIF) are shortcut-prone
    # --- training --------------------------------------------------------
    augment_copies: int = 1              # degraded copies of each training image
    augment: dict = field(default_factory=_aug_default)
    test_size: float = 0.2               # 0 disables the hold-out split (CV metrics only)
    cv_folds: int = 5
    seed: int = 42
    n_jobs: int = -1
    cache_dir: str | None = None
    use_cache: bool = True
    models: list = field(default_factory=lambda: list(MEMBERS))
    model: dict = field(default_factory=dict)
    model_params: dict = field(default_factory=dict)
    high_level: HighLevelConfig = field(default_factory=HighLevelConfig)
    threshold_metric: str = "balanced_accuracy"
    target_accuracy: float = 0.95        # accuracy wanted outside the "uncertain" band
    min_coverage: float = 0.5            # fraction of images that must still get a verdict
    fallback_margin: float = 0.15
    near_dup_hamming: int = 4            # perceptual-hash distance treated as near-duplicate
    # --- inference -------------------------------------------------------
    provenance_override: bool = True     # declared AI metadata forces the AI verdict
    # --- data discovery --------------------------------------------------
    class_aliases: dict = field(default_factory=_aliases_default)
    image_extensions: list = field(default_factory=lambda: [
        ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif", ".jfif"])
    robustness: list = field(default_factory=_robustness_default)
    # --- server ----------------------------------------------------------
    max_upload_mb: float = 25.0
    max_batch_files: int = 20

    # ------------------------------------------------------------------
    @property
    def effective_min_side(self) -> int:
        return int(self.min_side) if self.min_side else int(self.crop_size)

    def validate(self) -> "Config":
        if self.crop_size < 64 or self.crop_size % 16:
            raise ValueError("crop_size must be a multiple of 16 and >= 64")
        if self.crops_per_image < 1:
            raise ValueError("crops_per_image must be >= 1")
        if self.heatmap_max_crops < self.crops_per_image:
            self.heatmap_max_crops = self.crops_per_image
        if self.min_side is not None and self.min_side < self.crop_size:
            raise ValueError("min_side cannot be smaller than crop_size")
        bad = [g for g in self.feature_groups if g not in FEATURE_GROUPS]
        if bad or not self.feature_groups:
            raise ValueError(f"feature_groups must be a non-empty subset of {FEATURE_GROUPS}; got {bad or 'nothing'}")
        bad = [m for m in self.models if m not in MEMBERS]
        if bad or not self.models:
            raise ValueError(f"models must be a non-empty subset of {MEMBERS}; got {bad or 'nothing'}")
        if not 0.0 <= self.test_size < 0.9:
            raise ValueError("test_size must be in [0, 0.9)")
        if self.cv_folds < 2:
            raise ValueError("cv_folds must be >= 2")
        if self.augment_copies < 0:
            raise ValueError("augment_copies must be >= 0")
        if self.threshold_metric not in THRESHOLD_METRICS:
            raise ValueError(f"threshold_metric must be one of {THRESHOLD_METRICS}")
        if not 0.5 <= self.target_accuracy <= 1.0 or not 0.0 < self.min_coverage <= 1.0:
            raise ValueError("target_accuracy must be in [0.5,1] and min_coverage in (0,1]")
        if not 0.0 <= self.fallback_margin < 0.5:
            raise ValueError("fallback_margin must be in [0, 0.5)")
        if not 0 <= self.near_dup_hamming <= 16:
            raise ValueError("near_dup_hamming must be in [0, 16]")
        if isinstance(self.model, dict):
            self.model_params = dict(self.model.get("params", self.model.get("estimators", self.model_params)))
        if not isinstance(self.high_level, HighLevelConfig):
            self.high_level = HighLevelConfig(**self.high_level)
        if self.high_level.method not in {"logreg", "mlp"}:
            raise ValueError("high_level.method must be 'logreg' or 'mlp'")
        if not 0.0 <= self.high_level.weight <= 1.0:
            raise ValueError("high_level.weight must be in [0, 1]")
        if self.radial_bins < 8:
            raise ValueError("radial_bins must be >= 8")
        for key in ("ai", "real"):
            if not self.class_aliases.get(key):
                raise ValueError(f"class_aliases.{key} must be a non-empty list")
        self.image_extensions = [e.lower() if e.startswith(".") else "." + e.lower() for e in self.image_extensions]
        return self

    # ------------------------------------------------------------------
    @classmethod
    def from_dict(cls, data: dict | None) -> "Config":
        base = cls()
        known = {f.name for f in fields(cls)}
        data = dict(data or {})
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"unknown config keys: {unknown}")
        for k, v in data.items():
            cur = getattr(base, k)
            if isinstance(cur, dict) and isinstance(v, dict):
                merged = copy.deepcopy(cur)
                merged.update(v)
                v = merged
            setattr(base, k, v)
        return base.validate()

    @classmethod
    def load(cls, path: str | Path | None = None, overrides: list[str] | None = None) -> "Config":
        data: dict[str, Any] = {}
        if path:
            with open(path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            if not isinstance(data, dict):
                raise ValueError(f"{path}: top level must be a mapping")
        for item in overrides or []:
            if "=" not in item:
                raise ValueError(f"--set expects key=value, got {item!r}")
            key, _, raw = item.partition("=")
            value = yaml.safe_load(raw)
            node = data
            parts = key.strip().split(".")
            for p in parts[:-1]:
                node = node.setdefault(p, {})
            node[parts[-1]] = value
        return cls.from_dict(data)

    def to_dict(self) -> dict:
        return copy.deepcopy(asdict(self))

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False)

    # signatures used for cache keys / model compatibility ------------------
    def feature_signature(self) -> str:
        order = [g for g in FEATURE_GROUPS if g in self.feature_groups]
        blob = json.dumps({"c": self.crop_size, "g": order, "nb": self.radial_bins,
                           "meta": bool(self.use_metadata_features)}, sort_keys=True)
        return hashlib.sha1(blob.encode()).hexdigest()[:12]

    def augment_signature(self) -> str:
        blob = json.dumps({"a": self.augment, "s": self.seed}, sort_keys=True)
        return hashlib.sha1(blob.encode()).hexdigest()[:12]
