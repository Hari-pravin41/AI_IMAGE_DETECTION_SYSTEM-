"""Patch classifier, image-level aggregation, calibration and the saved model bundle."""
from __future__ import annotations

import json
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .config import Config, DecisionConfig, ModelConfig
from .features import FEATURE_VERSION
from .metrics import logit, sigmoid

BUNDLE_FORMAT = 1
MODEL_FILE = "model.joblib"
META_FILE = "metadata.json"
HASH_FILE = "training_hashes.txt"


class ModelError(Exception):
    """A saved model is missing, incompatible or unusable."""


# --------------------------------------------------------------------------- #
# patch-level classifier
# --------------------------------------------------------------------------- #
def _build_estimators(cfg: ModelConfig, seed: int, n_jobs: int):
    members = []
    for name, spec in cfg.estimators.items():
        if not spec.get("enabled", True):
            continue
        s = {k: v for k, v in spec.items() if k not in ("enabled", "weight")}
        weight = float(spec.get("weight", 1.0))
        if name == "hgb":
            est = HistGradientBoostingClassifier(random_state=seed, early_stopping=False, **s)
        elif name == "et":
            est = ExtraTreesClassifier(random_state=seed, n_jobs=n_jobs, **s)
        elif name == "lr":
            est = LogisticRegression(max_iter=5000, **s)
        else:  # guarded by Config.validate
            raise ModelError(f"unknown estimator '{name}'")
        members.append((name, est, weight))
    return members


class PatchModel:
    """Standardise -> clip -> weighted soft-vote of several classifiers. Outputs P(AI) per patch."""

    def __init__(self, model_cfg: ModelConfig, seed: int = 0, n_jobs: int = 1):
        self.model_cfg = model_cfg
        self.seed = seed
        self.n_jobs = n_jobs
        self.scaler = StandardScaler()
        self.members: List[Tuple[str, Any, float]] = []

    def transform(self, X: np.ndarray) -> np.ndarray:
        Z = self.scaler.transform(np.asarray(X, dtype=np.float64))
        clip = self.model_cfg.feature_clip
        return np.clip(np.nan_to_num(Z, nan=0.0, posinf=clip, neginf=-clip), -clip, clip)

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> "PatchModel":
        y = np.asarray(y).astype(int)
        if len(np.unique(y)) < 2:
            raise ModelError("training data contains a single class")
        self.scaler.fit(np.asarray(X, dtype=np.float64))
        Z = self.transform(X)
        self.members = _build_estimators(self.model_cfg, self.seed, self.n_jobs)
        for _, est, _ in self.members:
            est.fit(Z, y, sample_weight=sample_weight)
        return self

    def predict_scaled(self, Z: np.ndarray) -> np.ndarray:
        if not self.members:
            raise ModelError("model is not fitted")
        total = sum(w for _, _, w in self.members)
        out = np.zeros(len(Z), dtype=np.float64)
        for _, est, w in self.members:
            col = list(est.classes_).index(1)
            out += w * est.predict_proba(Z)[:, col]
        return np.clip(out / total, 0.0, 1.0)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """P(AI-generated) for every row of ``X`` (raw, unscaled features)."""
        return self.predict_scaled(self.transform(X))


class HybridImageHead:
    """Very small image-level fusion head over the forensic patch model.

    It turns per-patch feature rows into image-level features, then blends the
    high-level score with the forensic branch using a configurable weight.
    """

    def __init__(self, weight: float = 0.35, method: str = "logreg", hidden_dim: int = 64):
        self.weight = float(weight)
        self.method = method
        self.hidden_dim = int(hidden_dim)
        self.model = None

    def _make_image_matrix(self, X: np.ndarray, offsets: np.ndarray) -> np.ndarray:
        if len(offsets) < 2:
            return np.asarray(X, dtype=np.float64).reshape(1, -1)
        mats = []
        for i in range(len(offsets) - 1):
            start, stop = int(offsets[i]), int(offsets[i + 1])
            patch_rows = np.asarray(X[start:stop], dtype=np.float64)
            if patch_rows.size == 0:
                mats.append(np.zeros(X.shape[1], dtype=np.float64))
            else:
                mats.append(patch_rows.mean(axis=0))
        return np.vstack(mats)

    def fit(self, X: np.ndarray, offsets: np.ndarray, y: np.ndarray) -> "HybridImageHead":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=int)
        if len(np.unique(y)) < 2:
            raise ModelError("high-level head requires both classes in training data")
        image_X = self._make_image_matrix(X, offsets)
        if self.method == "logreg":
            self.model = LogisticRegression(class_weight="balanced", max_iter=2000)
        elif self.method == "mlp":
            from sklearn.neural_network import MLPClassifier
            self.model = MLPClassifier(hidden_layer_sizes=(self.hidden_dim,), max_iter=500, random_state=0)
        else:
            raise ModelError(f"unsupported high-level method '{self.method}'")
        self.model.fit(image_X, y)
        return self

    def predict(self, X: np.ndarray, offsets: np.ndarray, forensic_probs: np.ndarray | None = None) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        image_X = self._make_image_matrix(X, offsets)
        if self.model is None:
            raise ModelError("high-level head is not fitted")
        high_level = self.model.predict_proba(image_X)[:, 1]
        if forensic_probs is None:
            return high_level

        probs = np.asarray(forensic_probs, dtype=np.float64)
        if probs.size == image_X.shape[0]:
            forensic = probs
        elif probs.size == X.shape[0]:
            forensic = np.array([
                probs[int(offsets[i]): int(offsets[i + 1])].mean()
                for i in range(len(offsets) - 1)
            ], dtype=np.float64)
        else:
            forensic = np.repeat(probs.mean(), image_X.shape[0])
        return np.clip((1.0 - self.weight) * forensic + self.weight * high_level, 0.0, 1.0)

    def predict_proba(self, X: np.ndarray, offsets: np.ndarray, forensic_probs: np.ndarray | None = None):
        p = self.predict(X, offsets, forensic_probs)
        return np.column_stack([1.0 - p, p])


# --------------------------------------------------------------------------- #
# aggregation and calibration
# --------------------------------------------------------------------------- #
def aggregate_logits(logits: np.ndarray, method: str) -> float:
    x = np.sort(np.asarray(logits, dtype=np.float64))
    if x.size == 0:
        raise ModelError("no patches to aggregate")
    if method == "mean":
        return float(x.mean())
    if method == "median":
        return float(np.median(x))
    if method == "trimmed_mean":
        k = int(x.size * 0.1)
        return float(x[k: x.size - k].mean()) if x.size - 2 * k > 0 else float(x.mean())
    if method == "top_quartile":
        k = max(1, int(np.ceil(x.size * 0.25)))
        return float(x[-k:].mean())
    raise ModelError(f"unknown aggregation '{method}'")


def aggregate_by_image(patch_probs: np.ndarray, offsets: np.ndarray, method: str) -> np.ndarray:
    lg = logit(patch_probs)
    return np.array([aggregate_logits(lg[offsets[i]: offsets[i + 1]], method)
                     for i in range(len(offsets) - 1)])


@dataclass
class Platt:
    """p = sigmoid(a * aggregated_logit + b); plain floats, no sklearn dependency at inference."""
    a: float
    b: float

    def __call__(self, agg_logit):
        return sigmoid(self.a * np.asarray(agg_logit, dtype=np.float64) + self.b)

    @classmethod
    def fit(cls, agg_logits: np.ndarray, y: np.ndarray, C: float) -> "Platt":
        lr = LogisticRegression(C=C, class_weight="balanced", max_iter=2000)
        lr.fit(np.asarray(agg_logits).reshape(-1, 1), np.asarray(y).astype(int))
        a, b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
        if a <= 0:
            raise ModelError("calibration failed: aggregated scores are not positively related to the labels "
                             "(the detector found no usable signal in this data)")
        return cls(a, b)


# --------------------------------------------------------------------------- #
# bundle
# --------------------------------------------------------------------------- #
@dataclass
class ModelBundle:
    patch_model: PatchModel
    platt: Platt
    threshold: float
    config: Dict[str, Any]
    feature_names: List[str]
    feature_groups: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def cfg(self) -> Config:
        return Config.from_dict(self.config)

    def save(self, directory: str) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        payload = {
            "bundle_format": BUNDLE_FORMAT, "feature_version": FEATURE_VERSION,
            "patch_model": self.patch_model, "platt": self.platt, "threshold": float(self.threshold),
            "config": self.config, "feature_names": self.feature_names,
            "feature_groups": self.feature_groups,
            "metadata": {k: v for k, v in self.metadata.items() if not k.startswith("_")},
            "env": {"sklearn": sklearn.__version__, "numpy": np.__version__, "python": platform.python_version()},
        }
        joblib.dump(payload, d / MODEL_FILE, compress=3)
        public = {k: v for k, v in self.metadata.items() if not k.startswith("_")}
        (d / META_FILE).write_text(json.dumps(_jsonable(public), indent=2), encoding="utf-8")
        hashes = self.metadata.get("_training_hashes")
        if hashes:
            (d / HASH_FILE).write_text("\n".join(sorted(hashes)), encoding="utf-8")
        return d

    @classmethod
    def load(cls, path: str) -> "ModelBundle":
        p = Path(path)
        f = p / MODEL_FILE if p.is_dir() else p
        if not f.is_file():
            raise ModelError(f"No trained model found at '{path}'. Train one first with: "
                             f"aidetect train --data <folder> --out <model_dir>")
        try:
            payload = joblib.load(f)
        except Exception as e:
            raise ModelError(f"Could not load model file {f}: {e}") from e
        if not isinstance(payload, dict) or payload.get("bundle_format") != BUNDLE_FORMAT:
            raise ModelError(f"{f} is not a compatible aidetect model (bundle format mismatch). Retrain it.")
        if payload.get("feature_version") != FEATURE_VERSION:
            raise ModelError(f"Model was built with feature version {payload.get('feature_version')} but this "
                             f"code uses {FEATURE_VERSION}. Retrain the model.")
        env = payload.get("env", {})
        meta = dict(payload["metadata"])
        hash_file = f.parent / HASH_FILE
        if hash_file.is_file():
            meta["_training_hashes"] = set(hash_file.read_text(encoding="utf-8").split())
        if env.get("sklearn") and env["sklearn"] != sklearn.__version__:
            meta.setdefault("load_warnings", []).append(
                f"Model trained with scikit-learn {env['sklearn']}, running {sklearn.__version__}. "
                f"If predictions look wrong, retrain.")
        return cls(payload["patch_model"], payload["platt"], payload["threshold"], payload["config"],
                   payload["feature_names"], payload["feature_groups"], meta)


def _jsonable(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (set, frozenset)):
        return sorted(_jsonable(v) for v in o)
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, float):
        return None if not np.isfinite(o) else o
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    return o


def versions() -> Dict[str, str]:
    import PIL
    import scipy
    return {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__, "Pillow": PIL.__version__}
