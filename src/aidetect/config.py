"""Central configuration.

Every tunable value in the project lives here (and in ``config/default.yaml``,
which mirrors these defaults). Nothing downstream hardcodes paths, class names,
thresholds, patch sizes or model hyper-parameters: they all come from a
``Config`` object that is validated once, stored inside every trained model and
reused at inference time so training and prediction can never drift apart.
"""
from __future__ import annotations

import copy
import dataclasses
import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml


class ConfigError(ValueError):
    """Raised for invalid or unknown configuration values."""


# --------------------------------------------------------------------------- #
# sections
# --------------------------------------------------------------------------- #
@dataclass
class PreprocessConfig:
    # Re-encode every image as JPEG at this quality (1-100) before analysis.
    # Useful when one class is mostly PNG and the other mostly JPEG, because it
    # removes the file-format shortcut. null = analyse pixels exactly as stored.
    recompress_jpeg_quality: Optional[int] = None
    # Images with more pixels than this are rejected (decompression-bomb guard).
    max_pixels: int = 120_000_000
    # What to do with images smaller than the patch size: "skip" or "upscale".
    small_image_policy: str = "skip"


@dataclass
class PatchConfig:
    # Images are analysed as square patches at native resolution (no resizing).
    patch_size: int = 128           # multiple of 8, 32..1024
    max_patches: int = 24           # patches per image used for training / verdict
    min_patch_std: float = 1.0      # patches flatter than this (grey-level std) are ignored


@dataclass
class FeatureConfig:
    n_radial_bins: int = 24         # radial power-spectrum resolution
    cooc_truncation: int = 2        # residual co-occurrence truncation T (bins = 2T+1)
    cooc_quant_step: float = 1.0    # residual quantisation step
    median_kernel: int = 3          # denoiser kernel for noise residuals (odd)
    wavelet_levels: int = 3         # Haar decomposition depth


def _default_aliases() -> Dict[str, List[str]]:
    return {
        "real": ["real", "authentic", "genuine", "human", "natural", "photo", "photos",
                 "camera", "nature", "original", "realart", "humanmade"],
        "fake": ["fake", "ai", "ai_generated", "aigenerated", "generated", "synthetic",
                 "artificial", "gan", "diffusion", "aiart", "aiartdata", "aimade"],
    }


def _default_split_aliases() -> Dict[str, List[str]]:
    return {
        "train": ["train", "training"],
        "val": ["val", "valid", "validation", "dev"],
        "test": ["test", "testing"],
    }


@dataclass
class DataConfig:
    class_aliases: Dict[str, List[str]] = field(default_factory=_default_aliases)
    split_aliases: Dict[str, List[str]] = field(default_factory=_default_split_aliases)
    extensions: List[str] = field(default_factory=lambda: [
        ".jpg", ".jpeg", ".jfif", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif"])
    test_size: float = 0.2                  # held-out fraction when no train/test split exists
    min_images_per_class: int = 20          # after validation + de-duplication
    near_duplicate_hamming: int = 3         # dHash distance treated as "same image"; -1 disables
    max_images_per_class: Optional[int] = None   # sub-sample for quick experiments


def _default_estimators() -> Dict[str, Dict[str, Any]]:
    return {
        "hgb": {"enabled": True, "weight": 1.0, "max_iter": 250, "learning_rate": 0.06,
                "max_leaf_nodes": 15, "l2_regularization": 1.0},
        "et": {"enabled": True, "weight": 1.0, "n_estimators": 300, "min_samples_leaf": 2,
               "max_features": "sqrt"},
        "lr": {"enabled": True, "weight": 1.0, "C": 0.3},
    }


@dataclass
class ModelConfig:
    estimators: Dict[str, Dict[str, Any]] = field(default_factory=_default_estimators)
    feature_clip: float = 8.0               # standardised features are clipped to +-this


@dataclass
class HighLevelConfig:
    enabled: bool = False
    method: str = "logreg"                  # logreg | mlp
    weight: float = 0.35                     # blend between forensic score and high-level score
    hidden_dim: int = 64


@dataclass
class DecisionConfig:
    aggregation: str = "mean"               # mean | median | trimmed_mean | top_quartile
    threshold_strategy: str = "balanced_accuracy"   # balanced_accuracy | f1 | fixed
    fixed_threshold: float = 0.5
    uncertainty_margin: float = 0.15        # |p - threshold| below this => "uncertain"
    cv_folds: int = 5                       # out-of-fold folds used for calibration
    calibration_C: float = 1.0              # regularisation of the Platt calibrator


@dataclass
class InferenceConfig:
    max_patches: Optional[int] = None       # null = same as patch.max_patches
    heatmap_max_patches: int = 100
    explain_top_k: int = 7
    reliable_min_patches: int = 6
    max_patch_disagreement: float = 2.5     # std of patch logits above this => low agreement


def _default_robustness() -> List[Dict[str, Any]]:
    return [
        {"type": "jpeg", "quality": 90},
        {"type": "jpeg", "quality": 70},
        {"type": "jpeg", "quality": 50},
        {"type": "resize", "scale": 0.75},
        {"type": "blur", "radius": 1.0},
        {"type": "noise", "sigma": 3.0},
    ]


@dataclass
class EvaluationConfig:
    bootstrap_samples: int = 500
    permutation_repeats: int = 3
    robustness_max_images: int = 60
    robustness: List[Dict[str, Any]] = field(default_factory=_default_robustness)


@dataclass
class RuntimeConfig:
    n_jobs: int = -1                        # -1 = all cores
    seed: int = 42
    log_level: str = "INFO"


def _default_generator_keywords() -> List[str]:
    return ["stable diffusion", "stablediffusion", "midjourney", "dall-e", "dalle", "dall·e",
            "comfyui", "automatic1111", "invokeai", "novelai", "firefly", "imagen", "leonardo.ai",
            "flux", "sdxl", "openai", "gemini", "ideogram", "playground", "nightcafe",
            "craiyon", "bing image creator", "generative ai", "ai generated", "ai-generated"]


@dataclass
class ProvenanceConfig:
    generator_keywords: List[str] = field(default_factory=_default_generator_keywords)
    generation_metadata_keys: List[str] = field(default_factory=lambda: [
        "parameters", "prompt", "workflow", "negative_prompt", "sd-metadata",
        "invokeai_metadata", "dream", "comment"])


@dataclass
class Config:
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    patch: PatchConfig = field(default_factory=PatchConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    high_level: HighLevelConfig = field(default_factory=HighLevelConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    provenance: ProvenanceConfig = field(default_factory=ProvenanceConfig)

    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(dataclasses.asdict(self))

    def feature_key(self) -> str:
        """Canonical JSON of everything that influences extracted features."""
        d = {"preprocess": dataclasses.asdict(self.preprocess),
             "patch": dataclasses.asdict(self.patch),
             "features": dataclasses.asdict(self.features)}
        return json.dumps(d, sort_keys=True)

    @property
    def effective_max_patches(self) -> int:
        return int(self.inference.max_patches or self.patch.max_patches)

    # ------------------------------------------------------------------ #
    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "Config":
        cfg = _build(cls, data or {}, "")
        cfg.validate()
        return cfg

    @classmethod
    def load(cls, path: Optional[str] = None,
             overrides: Optional[Iterable[str]] = None) -> "Config":
        """Defaults <- YAML file <- ``section.key=value`` overrides."""
        data: Dict[str, Any] = {}
        if path:
            p = Path(path)
            if not p.is_file():
                raise ConfigError(f"Config file not found: {p}")
            try:
                loaded = yaml.safe_load(p.read_text(encoding="utf-8"))
            except yaml.YAMLError as e:
                raise ConfigError(f"Invalid YAML in {p}: {e}") from e
            if loaded is not None:
                if not isinstance(loaded, dict):
                    raise ConfigError(f"Top level of {p} must be a mapping")
                data = loaded
        if overrides:
            data = _deep_merge(data, parse_overrides(overrides))
        return cls.from_dict(data)

    # ------------------------------------------------------------------ #
    def validate(self) -> None:
        def need(cond: bool, msg: str) -> None:
            if not cond:
                raise ConfigError(msg)

        pre, pc, fc, dc = self.preprocess, self.patch, self.features, self.data
        q = pre.recompress_jpeg_quality
        need(q is None or (isinstance(q, int) and not isinstance(q, bool) and 1 <= q <= 100),
             "preprocess.recompress_jpeg_quality must be null or an integer 1-100")
        need(_is_int(pre.max_pixels) and pre.max_pixels >= 1024,
             "preprocess.max_pixels must be an integer >= 1024")
        need(pre.small_image_policy in ("skip", "upscale"),
             "preprocess.small_image_policy must be 'skip' or 'upscale'")

        need(_is_int(pc.patch_size) and 32 <= pc.patch_size <= 1024 and pc.patch_size % 8 == 0,
             "patch.patch_size must be a multiple of 8 between 32 and 1024")
        need(_is_int(pc.max_patches) and pc.max_patches >= 1, "patch.max_patches must be >= 1")
        need(_is_num(pc.min_patch_std) and pc.min_patch_std >= 0, "patch.min_patch_std must be >= 0")

        need(_is_int(fc.n_radial_bins) and 8 <= fc.n_radial_bins <= pc.patch_size // 2,
             "features.n_radial_bins must be between 8 and patch_size/2")
        need(_is_int(fc.cooc_truncation) and 1 <= fc.cooc_truncation <= 3,
             "features.cooc_truncation must be 1, 2 or 3")
        need(_is_num(fc.cooc_quant_step) and fc.cooc_quant_step > 0,
             "features.cooc_quant_step must be > 0")
        need(_is_int(fc.median_kernel) and fc.median_kernel >= 3 and fc.median_kernel % 2 == 1,
             "features.median_kernel must be an odd integer >= 3")
        need(_is_int(fc.wavelet_levels) and 1 <= fc.wavelet_levels <= 4
             and pc.patch_size // (2 ** fc.wavelet_levels) >= 4,
             "features.wavelet_levels must be 1-4 and leave >= 4 px at the coarsest level")

        for label in ("real", "fake"):
            need(label in dc.class_aliases and len(dc.class_aliases[label]) > 0,
                 f"data.class_aliases must define a non-empty list for '{label}'")
        need(not (_norm_set(dc.class_aliases["real"]) & _norm_set(dc.class_aliases["fake"])),
             "data.class_aliases: the same name cannot be both a real and a fake alias")
        need(all(isinstance(e, str) and e.startswith(".") for e in dc.extensions) and dc.extensions,
             "data.extensions must be a non-empty list of strings like '.jpg'")
        need(_is_num(dc.test_size) and 0.05 <= dc.test_size <= 0.5,
             "data.test_size must be between 0.05 and 0.5")
        need(_is_int(dc.min_images_per_class) and dc.min_images_per_class >= 6,
             "data.min_images_per_class must be an integer >= 6")
        need(_is_int(dc.near_duplicate_hamming) and -1 <= dc.near_duplicate_hamming <= 16,
             "data.near_duplicate_hamming must be an integer between -1 and 16")
        need(dc.max_images_per_class is None or (_is_int(dc.max_images_per_class)
                                                 and dc.max_images_per_class >= 6),
             "data.max_images_per_class must be null or an integer >= 6")

        ests = self.model.estimators
        known = {"hgb", "et", "lr"}
        need(set(ests) <= known, f"model.estimators: unknown estimator(s) {sorted(set(ests) - known)}; "
                                 f"choose from {sorted(known)}")
        need(any(e.get("enabled", True) for e in ests.values()),
             "model.estimators: at least one estimator must be enabled")
        for name, e in ests.items():
            need(_is_num(e.get("weight", 1.0)) and e.get("weight", 1.0) > 0,
                 f"model.estimators.{name}.weight must be > 0")
        need(_is_num(self.model.feature_clip) and self.model.feature_clip > 0,
             "model.feature_clip must be > 0")

        hl = self.high_level
        need(isinstance(hl.enabled, bool), "high_level.enabled must be a boolean")
        need(hl.method in ("logreg", "mlp"), "high_level.method must be 'logreg' or 'mlp'")
        need(_is_num(hl.weight) and 0 <= hl.weight <= 1,
             "high_level.weight must be between 0 and 1")
        need(_is_int(hl.hidden_dim) and hl.hidden_dim >= 8,
             "high_level.hidden_dim must be an integer >= 8")

        d = self.decision
        need(d.aggregation in ("mean", "median", "trimmed_mean", "top_quartile"),
             "decision.aggregation must be mean, median, trimmed_mean or top_quartile")
        need(d.threshold_strategy in ("balanced_accuracy", "f1", "fixed"),
             "decision.threshold_strategy must be balanced_accuracy, f1 or fixed")
        need(_is_num(d.fixed_threshold) and 0 < d.fixed_threshold < 1,
             "decision.fixed_threshold must be between 0 and 1")
        need(_is_num(d.uncertainty_margin) and 0 <= d.uncertainty_margin < 0.5,
             "decision.uncertainty_margin must be in [0, 0.5)")
        need(_is_int(d.cv_folds) and d.cv_folds >= 2, "decision.cv_folds must be an integer >= 2")
        need(_is_num(d.calibration_C) and d.calibration_C > 0, "decision.calibration_C must be > 0")

        i = self.inference
        need(i.max_patches is None or (_is_int(i.max_patches) and i.max_patches >= 1),
             "inference.max_patches must be null or >= 1")
        need(_is_int(i.heatmap_max_patches) and i.heatmap_max_patches >= 1,
             "inference.heatmap_max_patches must be >= 1")
        need(_is_int(i.explain_top_k) and i.explain_top_k >= 1, "inference.explain_top_k must be >= 1")
        need(_is_int(i.reliable_min_patches) and i.reliable_min_patches >= 1,
             "inference.reliable_min_patches must be >= 1")
        need(_is_num(i.max_patch_disagreement) and i.max_patch_disagreement > 0,
             "inference.max_patch_disagreement must be > 0")

        e = self.evaluation
        need(_is_int(e.bootstrap_samples) and e.bootstrap_samples >= 0,
             "evaluation.bootstrap_samples must be >= 0")
        need(_is_int(e.permutation_repeats) and e.permutation_repeats >= 1,
             "evaluation.permutation_repeats must be >= 1")
        need(_is_int(e.robustness_max_images) and e.robustness_max_images >= 2,
             "evaluation.robustness_max_images must be >= 2")
        for spec in e.robustness:
            need(isinstance(spec, dict) and spec.get("type") in ("jpeg", "resize", "blur", "noise"),
                 f"evaluation.robustness entries need type jpeg|resize|blur|noise, got {spec!r}")
            key = {"jpeg": "quality", "resize": "scale", "blur": "radius", "noise": "sigma"}[spec["type"]]
            need(key in spec and _is_num(spec[key]) and spec[key] > 0,
                 f"evaluation.robustness '{spec['type']}' needs a positive '{key}'")

        r = self.runtime
        need(_is_int(r.n_jobs) and (r.n_jobs == -1 or r.n_jobs >= 1),
             "runtime.n_jobs must be -1 or a positive integer")
        need(_is_int(r.seed), "runtime.seed must be an integer")
        need(str(r.log_level).upper() in ("DEBUG", "INFO", "WARNING", "ERROR"),
             "runtime.log_level must be DEBUG, INFO, WARNING or ERROR")

        need(all(isinstance(k, str) for k in self.provenance.generator_keywords),
             "provenance.generator_keywords must be a list of strings")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _norm_set(names: Iterable[str]) -> set:
    import re
    return {re.sub(r"[^a-z0-9]", "", str(n).lower()) for n in names}


def _deep_merge(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _build(cls: Any, data: Any, path: str) -> Any:
    if not isinstance(data, dict):
        raise ConfigError(f"'{path or 'config'}' must be a mapping, got {type(data).__name__}")
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ConfigError(f"Unknown option(s) in '{path or 'config root'}': {sorted(unknown)}. "
                          f"Valid: {sorted(known)}")
    defaults = cls()
    kwargs: Dict[str, Any] = {}
    for name in known:
        default = getattr(defaults, name)
        sub = f"{path}.{name}" if path else name
        if name not in data:
            kwargs[name] = default
        elif dataclasses.is_dataclass(default):
            kwargs[name] = _build(type(default), data[name] if data[name] is not None else {}, sub)
        elif isinstance(default, dict):
            if data[name] is not None and not isinstance(data[name], dict):
                raise ConfigError(f"'{sub}' must be a mapping")
            kwargs[name] = _deep_merge(default, data[name] or {})
        else:
            kwargs[name] = data[name]
    return cls(**kwargs)


def parse_overrides(items: Iterable[str]) -> Dict[str, Any]:
    """Turn ``["patch.patch_size=64", "decision.aggregation=median"]`` into a nested dict."""
    out: Dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ConfigError(f"Override '{item}' must look like section.key=value")
        key, raw = item.split("=", 1)
        parts = [p.strip() for p in key.strip().split(".")]
        if len(parts) < 2 or not all(parts):
            raise ConfigError(f"Override '{item}' must look like section.key=value")
        try:
            value = yaml.safe_load(raw)
        except yaml.YAMLError as e:
            raise ConfigError(f"Cannot parse value in override '{item}': {e}") from e
        node = out
        for p in parts[:-1]:
            node = node.setdefault(p, {})
            if not isinstance(node, dict):
                raise ConfigError(f"Override '{item}' conflicts with another override")
        node[parts[-1]] = value
    return out
