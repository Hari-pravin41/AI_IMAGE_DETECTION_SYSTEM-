"""Inference API: load a trained model, score images, explain the verdict."""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from .config import Config
from .features import extract, feature_group_of
from .loader import ImageError, LoadedImage, load_image
from .model import ModelBundle

LABELS = ("real", "ai_generated", "uncertain")


@dataclass
class Prediction:
    name: str
    ok: bool = True
    error: str | None = None
    label: str | None = None             # final verdict: ai_generated | real | uncertain
    model_label: str | None = None       # verdict from pixel analysis alone
    prob_ai: float | None = None         # calibrated probability from the pixel model
    threshold: float | None = None
    margin: float | None = None
    decision_basis: str = "model"        # model | provenance_metadata
    size: tuple | None = None
    format: str | None = None
    n_crops: int = 0
    crop_scores: list = field(default_factory=list)      # only with explain=True
    provenance: dict = field(default_factory=dict)
    group_contributions: dict | None = None               # log-odds pushed towards AI per feature group
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items()}
        d["size"] = list(self.size) if self.size else None
        return d


class Detector:
    def __init__(self, bundle: ModelBundle, model_identifier: str = "forensic-feature-ensemble"):
        self.bundle = bundle
        self.cfg: Config = bundle.cfg
        self.model_identifier = model_identifier
        self._groups = np.array([feature_group_of(n) for n in bundle.feature_names])

    @classmethod
    def load(cls, path: str | Path) -> "Detector":
        model_path = Path(path)
        identifier = model_path.name if model_path.is_dir() else model_path.parent.name
        return cls(ModelBundle.load(path), identifier)

    # ------------------------------------------------------------------
    def _scores(self, feats: np.ndarray) -> np.ndarray:
        """Per-crop ensemble logit."""
        ens = self.bundle.ensemble
        return ens.combine(ens.member_logits(feats))

    def predict_loaded(self, img: LoadedImage, explain: bool = False) -> Prediction:
        cfg, b = self.cfg, self.bundle
        ex = extract(img, cfg)
        row_z = self._scores(ex.feats)
        z = float(row_z.mean())
        p = float(b.calibrator(z))
        if cfg.high_level.enabled and b.high_level is not None:
            image_x = ex.feats.mean(axis=0, keepdims=True)
            offsets = np.array([0, 1], dtype=np.int64)
            p = float(b.high_level.predict_proba(image_x, offsets, np.array([p], dtype=np.float64))[:, 1][0])
        model_label = "uncertain" if abs(p - b.threshold) < b.margin else ("ai_generated" if p >= b.threshold else "real")

        pred = Prediction(name=img.name, label=model_label, model_label=model_label, prob_ai=p,
                          threshold=b.threshold, margin=b.margin, size=img.size, format=img.format,
                          n_crops=len(ex.positions), provenance=img.provenance)
        if cfg.provenance_override and img.provenance.get("declares_ai"):
            pred.label, pred.decision_basis = "ai_generated", "provenance_metadata"

        w = pred.warnings
        if img.is_grayscale:
            w.append("grayscale image: colour-based cues carry no information")
        if img.has_alpha:
            w.append("alpha channel was discarded before analysis")
        if len(ex.positions) < cfg.crops_per_image:
            w.append(f"small image: only {len(ex.positions)} analysis windows fit (configured {cfg.crops_per_image})")
        if img.provenance.get("camera_like") and pred.model_label == "ai_generated":
            w.append("camera EXIF is present but pixel analysis suggests AI; EXIF can be copied or forged")
        if pred.decision_basis == "provenance_metadata" and pred.model_label == "real":
            w.append("metadata declares AI generation but pixel analysis looks real; metadata can be forged")

        if explain:
            a = b.calibrator.a
            contrib = {}
            for g in sorted(set(self._groups)):
                cols = self._groups == g
                mod = ex.feats.copy()
                mod[:, cols] = b.medians[cols]
                contrib[g] = float(a * (z - self._scores(mod).mean()))
            pred.group_contributions = dict(sorted(contrib.items(), key=lambda kv: -abs(kv[1])))
            dense = extract(img, cfg, cfg.heatmap_max_crops)
            pz = self._scores(dense.feats)
            probs = b.calibrator(pz)
            pred.crop_scores = [{"y": int(y), "x": int(x), "size": cfg.crop_size, "prob": float(q)}
                                for (y, x), q in zip(dense.positions, probs)]
        return pred

    def predict(self, source, explain: bool = False, name: str | None = None) -> Prediction:
        """Never raises for bad images: returns ``Prediction(ok=False, error=...)`` instead."""
        label = name or (Path(source).name if isinstance(source, (str, os.PathLike)) else "image")
        try:
            img = load_image(source, self.cfg, name=label)
        except ImageError as e:
            return Prediction(name=label, ok=False, error=str(e))
        return self.predict_loaded(img, explain)

    def predict_many(self, sources: Iterable, explain: bool = False, workers: int = 1) -> Iterator[Prediction]:
        items = list(sources)
        if workers <= 1 or len(items) < 2:
            for s in items:
                yield self.predict(s, explain)
            return
        with ThreadPoolExecutor(max_workers=workers) as ex:
            yield from ex.map(lambda s: self.predict(s, explain), items)

    def info(self) -> dict:
        s = self.bundle.summary
        hold = s.get("holdout") or {}
        cv = s.get("cv") or {}
        return {
            "members": self.bundle.ensemble.names, "threshold": self.bundle.threshold,
            "margin": self.bundle.margin, "crop_size": self.cfg.crop_size,
            "crops_per_image": self.cfg.crops_per_image, "min_side": self.cfg.effective_min_side,
            "n_features": len(self.bundle.feature_names),
            "holdout_auc": hold.get("roc_auc"), "holdout_accuracy": hold.get("accuracy"),
            "cv_auc": cv.get("roc_auc"), "cv_accuracy": cv.get("accuracy"),
            "n_train_images": s.get("n_train_images"),
        }
