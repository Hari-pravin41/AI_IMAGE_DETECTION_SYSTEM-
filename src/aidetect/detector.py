"""Inference: image(s) in, structured verdict out."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
from joblib import Parallel, delayed

from .config import Config
from .extraction import Extraction, extract_features, extract_from_loaded, load_for_key
from .features import GROUP_DESCRIPTIONS, FeatureExtractor, extractor_for
from .imaging import ImageError, Source
from .metrics import logit
from .model import ModelBundle, ModelError, aggregate_logits
from .provenance import inspect_provenance

VERDICT_AI, VERDICT_REAL, VERDICT_UNCERTAIN = "ai_generated", "real", "uncertain"
VERDICT_TEXT = {VERDICT_AI: "Likely AI-generated", VERDICT_REAL: "Likely real (camera photo)",
                VERDICT_UNCERTAIN: "Uncertain"}


def decide(prob: float, threshold: float, margin: float) -> str:
    if prob >= threshold + margin:
        return VERDICT_AI
    if prob <= threshold - margin:
        return VERDICT_REAL
    return VERDICT_UNCERTAIN


@dataclass
class Prediction:
    name: str
    status: str = "ok"                         # "ok" | "error"
    error: Optional[str] = None
    prob_ai: Optional[float] = None
    verdict: Optional[str] = None
    threshold: Optional[float] = None
    margin: Optional[float] = None
    reliability: Optional[str] = None          # high | medium | low
    n_patches: int = 0
    n_tiles: int = 0
    n_flat: int = 0
    patch_disagreement: Optional[float] = None  # std of per-patch logits
    image_info: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)
    signals: Optional[List[Dict[str, Any]]] = None     # explain=True
    heatmap: Any = None                                # PIL image, heatmap=True
    heatmap_coverage: Optional[float] = None
    patch_scores: Optional[List[Dict[str, Any]]] = None
    sha256: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    @property
    def verdict_text(self) -> str:
        return VERDICT_TEXT.get(self.verdict or "", "Not analysed")

    def to_dict(self) -> Dict[str, Any]:
        """JSON-friendly view (no images)."""
        return {
            "name": self.name, "status": self.status, "error": self.error,
            "verdict": self.verdict, "prob_ai": None if self.prob_ai is None else round(self.prob_ai, 6),
            "threshold": self.threshold, "uncertainty_margin": self.margin,
            "reliability": self.reliability, "n_patches": self.n_patches,
            "n_flat_patches_skipped": self.n_flat,
            "patch_disagreement": None if self.patch_disagreement is None else round(self.patch_disagreement, 4),
            "width": self.image_info.get("width"), "height": self.image_info.get("height"),
            "format": self.image_info.get("format"),
            "warnings": list(self.warnings), "provenance": self.provenance,
            "signals": self.signals, "patch_scores": self.patch_scores,
        }

    def to_row(self) -> Dict[str, Any]:
        """Flat row for CSV / tables."""
        prov = self.provenance or {}
        return {
            "file": self.name, "status": self.status, "verdict": self.verdict,
            "prob_ai": None if self.prob_ai is None else round(self.prob_ai, 4),
            "reliability": self.reliability, "patches": self.n_patches,
            "width": self.image_info.get("width"), "height": self.image_info.get("height"),
            "format": self.image_info.get("format"),
            "camera_exif": prov.get("camera"), "generator_hints": "; ".join(prov.get("generator_hints", [])) or None,
            "warnings": "; ".join(self.warnings) or None, "error": self.error,
        }


def _name_of(source: Any, name: Optional[str]) -> str:
    if name:
        return name
    if isinstance(source, (str, os.PathLike)):
        return os.path.basename(os.fspath(source))
    return "image"


class Detector:
    """Loads a trained model directory and analyses images with it."""

    def __init__(self, bundle: ModelBundle, uncertainty_margin: Optional[float] = None,
                 threshold: Optional[float] = None, max_patches: Optional[int] = None):
        self.bundle = bundle
        self.cfg: Config = bundle.cfg
        self.extractor: FeatureExtractor = extractor_for(self.cfg)
        if self.extractor.names != bundle.feature_names:
            raise ModelError("Feature layout of the model does not match the code (different feature "
                             "definitions). Retrain the model.")
        self.feature_key = self.cfg.feature_key()
        self.threshold = float(bundle.threshold if threshold is None else threshold)
        self.margin = float(self.cfg.decision.uncertainty_margin if uncertainty_margin is None
                            else uncertainty_margin)
        if not 0.0 <= self.margin < 0.5:
            raise ValueError("uncertainty_margin must be in [0, 0.5)")
        if not 0.0 < self.threshold < 1.0:
            raise ValueError("threshold must be between 0 and 1")
        self.max_patches = int(max_patches or self.cfg.effective_max_patches)

    @classmethod
    def load(cls, path: str, **kwargs: Any) -> "Detector":
        return cls(ModelBundle.load(path), **kwargs)

    # ------------------------------------------------------------------ #
    def _score(self, feats: np.ndarray):
        pm = self.bundle.patch_model
        p_patch = pm.predict_proba(feats)
        lg = logit(p_patch)
        agg = aggregate_logits(lg, self.cfg.decision.aggregation)
        return float(self.bundle.platt(agg)), agg, p_patch, lg

    def _reliability(self, n: int, disagreement: float, upscaled: bool) -> str:
        c = self.cfg.inference
        if upscaled or n < max(2, c.reliable_min_patches // 2):
            return "low"
        if n >= c.reliable_min_patches and disagreement <= c.max_patch_disagreement:
            return "high"
        return "medium"

    def _signals(self, feats: np.ndarray, base_p: float) -> List[Dict[str, Any]]:
        pm = self.bundle.patch_model
        Z = pm.transform(feats)
        out = []
        for group, cols in self.extractor.group_columns.items():
            Zg = Z.copy()
            Zg[:, cols] = 0.0                      # neutralise the group (= training mean)
            agg = aggregate_logits(logit(pm.predict_scaled(Zg)), self.cfg.decision.aggregation)
            out.append({"group": group, "description": GROUP_DESCRIPTIONS[group],
                        "effect": float(base_p - float(self.bundle.platt(agg)))})
        out.sort(key=lambda d: -abs(d["effect"]))
        return out[: self.cfg.inference.explain_top_k]

    def predict_from_extraction(self, ex: Extraction, name: str, explain: bool = False) -> Prediction:
        if not ex.ok:
            return Prediction(name=name, status="error", error=ex.error, image_info=ex.info,
                              n_tiles=ex.n_tiles, n_flat=ex.n_flat, warnings=list(ex.warnings),
                              sha256=ex.sha256)
        p, agg, p_patch, lg = self._score(ex.features)
        dis = float(np.std(lg)) if lg.size > 1 else 0.0
        warnings = list(ex.warnings)
        reliability = self._reliability(len(lg), dis, "upscaled_from" in ex.info)
        if reliability == "low":
            warnings.append("Low reliability: very few usable patches or an up-scaled image.")
        if dis > self.cfg.inference.max_patch_disagreement:
            warnings.append("Patches disagree strongly (possible mixed or edited content).")
        pred = Prediction(
            name=name, prob_ai=p, verdict=decide(p, self.threshold, self.margin),
            threshold=self.threshold, margin=self.margin, reliability=reliability,
            n_patches=int(len(lg)), n_tiles=ex.n_tiles, n_flat=ex.n_flat, patch_disagreement=dis,
            image_info=ex.info, warnings=warnings, sha256=ex.sha256,
            provenance=inspect_provenance(ex.raw_meta, ex.has_content_credentials, self.cfg.provenance))
        if explain:
            pred.signals = self._signals(ex.features, p)
        return pred

    # ------------------------------------------------------------------ #
    def predict(self, source: Source, *, name: Optional[str] = None, explain: bool = False,
                heatmap: bool = False) -> Prediction:
        """Analyse one image (path, bytes, file object or PIL image). Never raises for bad images."""
        nm = _name_of(source, name)
        try:
            loaded = load_for_key(source, self.feature_key)
        except ImageError as e:
            return Prediction(name=nm, status="error", error=str(e))
        try:
            ex = extract_from_loaded(loaded, self.extractor, self.max_patches)
            pred = self.predict_from_extraction(ex, nm, explain)
            if heatmap and pred.ok:
                self._attach_heatmap(pred, loaded)
        except Exception as e:  # last line of defence: report instead of crashing the caller
            return Prediction(name=nm, status="error", error=f"analysis failed ({type(e).__name__}: {e})",
                              image_info=loaded.info)
        return pred

    def _attach_heatmap(self, pred: Prediction, loaded) -> None:
        from .visual import render_heatmap
        dense = self.extractor.extract_image(loaded.rgb, self.cfg.inference.heatmap_max_patches)
        if dense.features.shape[0] == 0:
            return
        probs = self.bundle.patch_model.predict_proba(dense.features)
        pred.heatmap, pred.heatmap_coverage = render_heatmap(
            loaded.rgb, dense.positions, probs, self.cfg.patch.patch_size)
        pred.patch_scores = [{"y": int(y), "x": int(x), "prob_ai": round(float(p), 4)}
                             for (y, x), p in zip(dense.positions, probs)]

    def predict_many(self, sources: Sequence[Source], names: Optional[Sequence[str]] = None,
                     n_jobs: Optional[int] = None, explain: bool = False, heatmap: bool = False,
                     progress: Optional[Callable[[int, int], None]] = None) -> List[Prediction]:
        """Analyse many images. Feature extraction runs in parallel; order of results == order of input."""
        total = len(sources)
        names = list(names) if names is not None else [None] * total
        if len(names) != total:
            raise ValueError("names must have the same length as sources")
        jobs = self.cfg.runtime.n_jobs if n_jobs is None else n_jobs
        results: List[Prediction] = []
        if explain or heatmap or jobs == 1 or total <= 2:
            for i, (s, n) in enumerate(zip(sources, names)):
                results.append(self.predict(s, name=n, explain=explain, heatmap=heatmap))
                if progress:
                    progress(i + 1, total)
            return results
        chunk = max(8, 4 * (os.cpu_count() or 2)) if jobs == -1 else max(8, 4 * jobs)
        with Parallel(n_jobs=jobs) as par:
            for start in range(0, total, chunk):
                batch = list(sources[start: start + chunk])
                exs = par(delayed(extract_features)(s, self.feature_key, self.max_patches) for s in batch)
                for s, n, ex in zip(batch, names[start: start + chunk], exs):
                    try:
                        results.append(self.predict_from_extraction(ex, _name_of(s, n)))
                    except Exception as e:
                        results.append(Prediction(name=_name_of(s, n), status="error",
                                                  error=f"analysis failed ({type(e).__name__}: {e})"))
                if progress:
                    progress(min(start + chunk, total), total)
        return results
