"""Image -> patch features, in a form that is safe to run in worker processes."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .features import FeatureExtractor, get_extractor
from .imaging import ImageError, LoadedImage, Source, dhash64, load_image
import json


@dataclass
class Extraction:
    ok: bool
    error: Optional[str] = None
    features: Optional[np.ndarray] = None      # (n_patches, n_features)
    positions: Optional[np.ndarray] = None     # (n_patches, 2) top-left (y, x)
    n_tiles: int = 0
    n_flat: int = 0
    info: Dict[str, Any] = field(default_factory=dict)
    raw_meta: Dict[str, Any] = field(default_factory=dict)
    has_content_credentials: bool = False
    sha256: Optional[str] = None
    dhash: Optional[int] = None
    warnings: List[str] = field(default_factory=list)


def load_for_key(source: Source, feature_key: str) -> LoadedImage:
    d = json.loads(feature_key)
    return load_image(
        source,
        max_pixels=d["preprocess"]["max_pixels"],
        min_side=d["patch"]["patch_size"],
        small_image_policy=d["preprocess"]["small_image_policy"],
        recompress_jpeg_quality=d["preprocess"]["recompress_jpeg_quality"],
    )


def extract_from_loaded(loaded: LoadedImage, extractor: FeatureExtractor, max_patches: Optional[int] = None,
                        want_dhash: bool = False) -> Extraction:
    feats = extractor.extract_image(loaded.rgb, max_patches)
    warnings = list(loaded.warnings)
    if feats.features.shape[0] == 0:
        return Extraction(
            ok=False,
            error=("image has no usable texture (all analysed patches are flat/uniform)"
                   if feats.n_tiles else "image too small for analysis"),
            n_tiles=feats.n_tiles, n_flat=feats.n_flat, info=loaded.info, raw_meta=loaded.raw_meta,
            has_content_credentials=loaded.has_content_credentials, sha256=loaded.sha256, warnings=warnings)
    if feats.n_flat:
        warnings.append(f"{feats.n_flat} flat/uniform patch(es) were skipped")
    return Extraction(
        ok=True, features=feats.features, positions=feats.positions, n_tiles=feats.n_tiles,
        n_flat=feats.n_flat, info=loaded.info, raw_meta=loaded.raw_meta,
        has_content_credentials=loaded.has_content_credentials, sha256=loaded.sha256,
        dhash=dhash64(loaded.rgb) if want_dhash else None, warnings=warnings)


def extract_features(source: Source, feature_key: str, max_patches: Optional[int] = None,
                     want_dhash: bool = False) -> Extraction:
    """Top-level (picklable) worker: never raises for bad images, returns ``ok=False`` instead."""
    try:
        loaded = load_for_key(source, feature_key)
    except ImageError as e:
        return Extraction(ok=False, error=str(e))
    try:
        return extract_from_loaded(loaded, get_extractor(feature_key), max_patches, want_dhash)
    except Exception as e:  # defensive: a feature bug on one odd image must not kill a batch
        return Extraction(ok=False, error=f"feature extraction failed ({type(e).__name__}: {e})",
                          info=loaded.info)
