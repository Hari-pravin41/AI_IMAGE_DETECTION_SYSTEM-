"""Ensemble classifier, calibration, decision threshold and the on-disk model bundle."""
from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import sklearn
from scipy.special import expit
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer

from . import __version__
from .config import Config
from .features import FEATURE_VERSION, feature_names
from .utils import dump_json

MEMBER_DEFAULTS = {
    "hgb": dict(max_iter=250, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=1.0),
    "extra_trees": dict(n_estimators=300, min_samples_leaf=2, max_features="sqrt"),
    "logreg": dict(C=0.5, max_iter=3000),
}
_CLIP = 1e-6


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _CLIP, 1 - _CLIP)
    return np.log(p / (1 - p))


def _build_member(name: str, cfg: Config, n_rows: int, n_jobs: int):
    params = {**MEMBER_DEFAULTS[name], **cfg.model_params.get(name, {})}
    if name == "hgb":
        return HistGradientBoostingClassifier(class_weight="balanced", early_stopping=False,
                                              random_state=cfg.seed, **params)
    if name == "extra_trees":
        return ExtraTreesClassifier(class_weight="balanced_subsample", random_state=cfg.seed,
                                    n_jobs=n_jobs, **params)
    return Pipeline([
        ("q", QuantileTransformer(n_quantiles=int(min(256, max(n_rows, 10))), output_distribution="normal",
                                  random_state=cfg.seed)),
        ("lr", LogisticRegression(class_weight="balanced", **params)),
    ])


class HybridImageHead:
    """Small image-level fusion head used to combine forensic evidence with a high-level visual model."""

    def __init__(self, weight: float = 0.35, method: str = "logreg", hidden_dim: int = 64):
        self.weight = float(weight)
        self.method = method
        self.hidden_dim = int(hidden_dim)
        self.model = None

    def fit(self, X: np.ndarray, offsets: np.ndarray, y: np.ndarray) -> "HybridImageHead":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=int)
        if len(np.unique(y)) < 2:
            raise ValueError("HybridImageHead requires both classes for training")
        image_X = np.vstack([
            X[int(offsets[i]):int(offsets[i + 1])].mean(axis=0)
            for i in range(len(offsets) - 1)
        ])
        if self.method == "logreg":
            self.model = LogisticRegression(class_weight="balanced", max_iter=2000)
        elif self.method == "mlp":
            from sklearn.neural_network import MLPClassifier
            self.model = MLPClassifier(hidden_layer_sizes=(self.hidden_dim,), max_iter=500, random_state=0)
        else:
            raise ValueError(f"unsupported high-level method '{self.method}'")
        self.model.fit(image_X, y)
        return self

    def predict(self, X: np.ndarray, offsets: np.ndarray, forensic_probs: np.ndarray | None = None) -> np.ndarray:
        if self.model is None:
            raise ValueError("HybridImageHead is not fitted")
        X = np.asarray(X, dtype=np.float64)
        image_X = np.vstack([
            X[int(offsets[i]):int(offsets[i + 1])].mean(axis=0)
            for i in range(len(offsets) - 1)
        ])
        high = self.model.predict_proba(image_X)[:, 1]
        if forensic_probs is None:
            return high
        forensic = np.asarray(forensic_probs, dtype=np.float64)
        if forensic.shape[0] == X.shape[0]:
            forensic = np.array([
                forensic[int(offsets[i]):int(offsets[i + 1])].mean()
                for i in range(len(offsets) - 1)
            ], dtype=np.float64)
        elif forensic.shape[0] != image_X.shape[0]:
            forensic = np.full(image_X.shape[0], float(forensic.mean()), dtype=np.float64)
        return np.clip((1.0 - self.weight) * forensic + self.weight * high, 0.0, 1.0)

    def predict_proba(self, X: np.ndarray, offsets: np.ndarray, forensic_probs: np.ndarray | None = None):
        p = self.predict(X, offsets, forensic_probs)
        return np.column_stack([1.0 - p, p])


class Ensemble:
    """Weighted logit average of heterogeneous classifiers, trained on crop-level rows."""

    def __init__(self, cfg: Config, n_jobs: int = 1):
        self.names = list(cfg.models)
        self.cfg = cfg
        self.n_jobs = n_jobs
        self.members: list = []
        self.weights = np.full(len(self.names), 1.0 / len(self.names))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Ensemble":
        if len(np.unique(y)) < 2:
            raise ValueError("training data contains a single class")
        self.members = [_build_member(n, self.cfg, len(X), self.n_jobs).fit(X, y) for n in self.names]
        return self

    def member_logits(self, X: np.ndarray) -> np.ndarray:
        cols = []
        for m in self.members:
            cols.append(logit(m.predict_proba(X)[:, list(m.classes_).index(1)]))
        return np.stack(cols, axis=1)

    def combine(self, member_logits: np.ndarray) -> np.ndarray:
        return member_logits @ self.weights

    def set_weights_from_auc(self, aucs: list[float]) -> None:
        w = np.clip(np.asarray(aucs, dtype=float) - 0.5, 0.01, None)
        self.weights = w / w.sum()


@dataclass
class Calibrator:
    a: float = 1.0
    b: float = 0.0

    @classmethod
    def fit(cls, z: np.ndarray, y: np.ndarray) -> "Calibrator":
        lr = LogisticRegression(C=1e4, max_iter=1000).fit(z.reshape(-1, 1), y)
        return cls(float(lr.coef_[0, 0]), float(lr.intercept_[0]))

    def __call__(self, z):
        return expit(self.a * np.asarray(z, dtype=np.float64) + self.b)


def choose_threshold(y: np.ndarray, p: np.ndarray, metric: str) -> float:
    cands = np.unique(np.quantile(p, np.linspace(0, 1, 501)))
    cands = np.concatenate([[0.0], (cands[:-1] + cands[1:]) / 2, [1.0]]) if len(cands) > 1 else np.array([0.5])
    pos, neg = (y == 1), (y == 0)
    best, best_t = -np.inf, 0.5
    for t in cands:
        pred = p >= t
        tpr = (pred & pos).sum() / max(pos.sum(), 1)
        tnr = (~pred & neg).sum() / max(neg.sum(), 1)
        if metric == "f1":
            prec = (pred & pos).sum() / max(pred.sum(), 1)
            score = 2 * prec * tpr / (prec + tpr) if prec + tpr > 0 else 0.0
        else:  # balanced_accuracy and youden share an optimum
            score = (tpr + tnr) / 2
        if score > best + 1e-12 or (abs(score - best) <= 1e-12 and abs(t - 0.5) < abs(best_t - 0.5)):
            best, best_t = score, float(t)
    return best_t


def choose_margin(y: np.ndarray, p: np.ndarray, thr: float, cfg: Config) -> tuple[float, dict]:
    """Smallest half-width of an 'uncertain' band around thr that reaches the target accuracy."""
    pred = (p >= thr).astype(int)
    for m in np.linspace(0.0, 0.49, 99):
        keep = np.abs(p - thr) >= m
        if keep.mean() < cfg.min_coverage:
            break
        if (pred[keep] == y[keep]).mean() >= cfg.target_accuracy:
            return float(m), {"source": "oof", "coverage": float(keep.mean()),
                              "accuracy_outside": float((pred[keep] == y[keep]).mean())}
    return float(cfg.fallback_margin), {"source": "fallback", "reason": "target accuracy not reachable "
                                        "with the required coverage"}


def unit_auc(y: np.ndarray, z: np.ndarray) -> float:
    return float(roc_auc_score(y, z)) if len(np.unique(y)) == 2 else 0.5


@dataclass
class ModelBundle:
    cfg: Config
    feature_names: list
    ensemble: Ensemble
    calibrator: Calibrator
    threshold: float
    margin: float
    medians: np.ndarray
    high_level: HybridImageHead | None = None
    summary: dict = field(default_factory=dict)

    MODEL_FILE = "model.joblib"

    def save(self, directory: str | Path) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": 1, "aidetect": __version__, "feature_version": FEATURE_VERSION,
            "sklearn": sklearn.__version__, "created": dt.datetime.now(dt.timezone.utc).isoformat(),
            "cfg": self.cfg.to_dict(), "feature_names": self.feature_names, "ensemble": self.ensemble,
            "calibrator": {"a": self.calibrator.a, "b": self.calibrator.b},
            "threshold": self.threshold, "margin": self.margin, "medians": self.medians,
            "high_level": self.high_level, "summary": self.summary,
        }
        tmp = d / (self.MODEL_FILE + ".tmp")
        joblib.dump(payload, tmp, compress=3)
        tmp.replace(d / self.MODEL_FILE)
        dump_json({"aidetect": __version__, "created": payload["created"], "threshold": self.threshold,
                   "margin": self.margin, "members": self.ensemble.names,
                   "weights": dict(zip(self.ensemble.names, self.ensemble.weights.tolist())),
                   "n_features": len(self.feature_names), "config": self.cfg.to_dict(),
                   "summary": self.summary}, d / "model_info.json")
        return d / self.MODEL_FILE

    @classmethod
    def load(cls, directory: str | Path) -> "ModelBundle":
        p = Path(directory)
        f = p / cls.MODEL_FILE if p.is_dir() else p
        if not f.is_file():
            raise FileNotFoundError(f"no model found at {p} (expected {cls.MODEL_FILE}; run `aidetect train` first)")
        # NOTE: joblib/pickle can execute code - only load model files you trained or trust.
        payload = joblib.load(f)
        if payload.get("format") != 1:
            raise ValueError("unsupported model file format")
        if payload["feature_version"] != FEATURE_VERSION:
            raise ValueError("model was trained with a different feature version; retrain it")
        cfg = Config.from_dict(payload["cfg"])
        if feature_names(cfg) != payload["feature_names"]:
            raise ValueError("feature layout mismatch between model and installed aidetect; retrain the model")
        if payload.get("sklearn") != sklearn.__version__:
            import warnings
            warnings.warn(f"model trained with scikit-learn {payload.get('sklearn')} but {sklearn.__version__} "
                          "is installed; results may differ", RuntimeWarning, stacklevel=2)
        ens = payload["ensemble"]
        ens.n_jobs = 1
        for m in ens.members:               # inference is done per image: avoid thread-pool overhead
            if hasattr(m, "n_jobs"):
                m.n_jobs = 1
        high = payload.get("high_level")
        return cls(cfg, payload["feature_names"], ens, Calibrator(**payload["calibrator"]),
                   payload["threshold"], payload["margin"], payload["medians"], high, payload["summary"])
