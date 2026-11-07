"""Forensic feature extraction.

Generative models (GANs, diffusion models, VAEs) leave statistical traces that
camera pipelines do not: periodic up-sampling peaks and a different
fall-off in the Fourier spectrum, unusually smooth or unusually uniform sensor
noise, different local residual patterns, different multi-scale wavelet
energy, and so on. This module turns an image patch (analysed at native
resolution, never resized) into ~190 numbers organised in seven named groups:

* ``spectrum``  radial Fourier power profile, slope, peakiness, anisotropy
* ``noise``     denoising-residual moments, channel correlation, uniformity
* ``cooc``      steganalysis-style residual co-occurrence histograms
* ``dct``       8x8 block-DCT statistics (Benford deviation, band energy, blockiness)
* ``color``     saturation, channel correlation, chroma noise, colour diversity
* ``texture``   gradient / Laplacian statistics, LBP histogram, local variance
* ``wavelet``   multi-scale Haar sub-band energy and kurtosis

Names and groups are produced by the same code that produces the values, so
they can never get out of sync.
"""
from __future__ import annotations

import functools
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import fft as sfft
from scipy import ndimage

from .config import Config, FeatureConfig, PatchConfig
from .imaging import to_gray

FEATURE_VERSION = "1"
EPS = 1e-8
LOG_FLOOR = 1e-3
GROUP_ORDER = ["spectrum", "noise", "cooc", "dct", "color", "texture", "wavelet"]

GROUP_DESCRIPTIONS: Dict[str, str] = {
    "spectrum": "Fourier spectrum shape (up-sampling peaks, high-frequency fall-off)",
    "noise": "Noise residual statistics (sensor-like vs. synthetic noise, uniformity)",
    "cooc": "Local residual patterns (steganalysis-style co-occurrences)",
    "dct": "Block-DCT statistics (compression and generation fingerprints)",
    "color": "Colour statistics (saturation, channel correlation, chroma noise)",
    "texture": "Gradient and texture statistics (edges, LBP, local variance)",
    "wavelet": "Multi-scale wavelet energy (detail consistency across scales)",
}

_BENFORD_P = np.log10(1.0 + 1.0 / np.arange(1, 10))     # Benford leading-digit probabilities
_SQUARE3 = np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], dtype=np.float64) / 4.0
_FLAT_GRADIENT = 1.0
_SAT_HIGH = 0.7


# --------------------------------------------------------------------------- #
# small numeric helpers
# --------------------------------------------------------------------------- #
def _slog(x: float) -> float:
    return float(np.sign(x) * np.log1p(abs(x)))


def _log10p(x: float) -> float:
    return float(np.log10(max(float(x), 0.0) + LOG_FLOOR))


def _moments(x: np.ndarray) -> Tuple[float, float, float]:
    """(std, skew, excess kurtosis) that never produce NaN/inf."""
    x = np.asarray(x, dtype=np.float64).ravel()
    d = x - x.mean()
    var = float(np.mean(d * d))
    if var < 1e-12:
        return 0.0, 0.0, 0.0
    sd = var ** 0.5
    skew = float(np.clip(np.mean(d ** 3) / sd ** 3, -20.0, 20.0))
    kurt = float(np.mean(d ** 4) / var ** 2 - 3.0)
    return sd, skew, kurt


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    sa, sb = np.sqrt(np.mean(a * a)), np.sqrt(np.mean(b * b))
    if sa < 1e-9 or sb < 1e-9:
        return 0.0
    return float(np.clip(np.mean(a * b) / (sa * sb), -1.0, 1.0))


def _autocorr(r: np.ndarray, dy: int, dx: int) -> float:
    h, w = r.shape
    if dx >= 0:
        a, b = r[: h - dy, : w - dx], r[dy:, dx:]
    else:
        a, b = r[: h - dy, -dx:], r[dy:, : w + dx]
    var = float(np.mean(r * r))
    if var < 1e-12:
        return 0.0
    return float(np.clip(np.mean(a * b) / var, -1.0, 1.0))


@functools.lru_cache(maxsize=16)
def _window(h: int, w: int) -> np.ndarray:
    win = np.outer(np.hanning(h), np.hanning(w))
    win.setflags(write=False)
    return win


@functools.lru_cache(maxsize=16)
def _radial_index(h: int, w: int, nb: int):
    yy, xx = np.indices((h, w))
    cy, cx = h // 2, w // 2
    r = np.sqrt(((xx - cx) / (w / 2.0)) ** 2 + ((yy - cy) / (h / 2.0)) ** 2)
    rpix = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    valid = (r <= 1.0) & (rpix >= 1.5)
    bins = np.minimum((r * nb).astype(np.int64), nb - 1)
    idx = np.where(valid, bins, -1)
    counts = np.bincount(idx[valid], minlength=nb)
    for a in (idx, counts, r):
        a.setflags(write=False)
    return idx, counts, r


# --------------------------------------------------------------------------- #
# feature groups: each returns (names, values)
# --------------------------------------------------------------------------- #
def _spectrum(gray: np.ndarray, nb: int):
    h, w = gray.shape
    g = (gray - gray.mean()) * _window(h, w)
    p = np.abs(np.fft.fftshift(np.fft.fft2(g))) ** 2
    lp = np.log10(p + 1e-3)
    idx, counts, r = _radial_index(h, w, nb)
    valid = idx >= 0
    prof = np.bincount(idx[valid], weights=lp[valid], minlength=nb) / np.maximum(counts, 1)

    names = [f"spec_r{i:02d}" for i in range(nb)]
    vals = list(prof - prof.mean())

    centers = (np.arange(nb) + 0.5) / nb
    slope, intercept = np.polyfit(np.log10(centers), prof, 1)
    fit = slope * np.log10(centers) + intercept
    names += ["spec_slope", "spec_fit_rmse"]
    vals += [float(slope), float(np.sqrt(np.mean((prof - fit) ** 2)))]

    total = float(p[valid].sum())
    hf = float(p[valid & (r > 0.5)].sum())
    resid = (lp - prof[np.maximum(idx, 0)])[valid & (r > 0.25)]
    cy, cx = h // 2, w // 2
    ex = p[cy - 1: cy + 2, :]
    ex_sum = float(ex.sum() - ex[:, cx - 1: cx + 2].sum())
    ey = p[:, cx - 1: cx + 2]
    ey_sum = float(ey.sum() - ey[cy - 1: cy + 2, :].sum())
    names += ["spec_hf_ratio", "spec_resid_std", "spec_resid_max", "spec_resid_p999", "spec_axis_ratio"]
    vals += [np.log10(hf + EPS) - np.log10(total + EPS), float(resid.std()), float(resid.max()),
             float(np.percentile(resid, 99.9)), np.log10(ex_sum + EPS) - np.log10(ey_sum + EPS)]
    return names, vals


def _noise(rgb: np.ndarray, gray: np.ndarray, kernel: int):
    res_g = gray - ndimage.median_filter(gray, size=kernel, mode="reflect")
    res_g = res_g - res_g.mean()
    sd, skew, kurt = _moments(res_g)
    names = ["noise_logstd", "noise_logmad", "noise_skew", "noise_kurt"]
    vals = [_log10p(sd), _log10p(np.mean(np.abs(res_g))), skew, _slog(kurt)]

    chans = [rgb[..., c].astype(np.float64) for c in range(3)]
    res_c = [c - ndimage.median_filter(c, size=kernel, mode="reflect") for c in chans]
    names += ["noise_corr_rg", "noise_corr_rb", "noise_corr_gb"]
    vals += [_corr(res_c[0], res_c[1]), _corr(res_c[0], res_c[2]), _corr(res_c[1], res_c[2])]
    sds = [_log10p(float(np.std(r))) for r in res_c]
    names += ["noise_std_ratio_rg", "noise_std_ratio_bg"]
    vals += [sds[0] - sds[1], sds[2] - sds[1]]

    for dy, dx, tag in [(0, 1, "h1"), (1, 0, "v1"), (1, 1, "d1"), (1, -1, "a1"), (0, 2, "h2"), (2, 0, "v2")]:
        names.append(f"noise_ac_{tag}")
        vals.append(_autocorr(res_g, dy, dx))

    bs = 16
    nbh, nbw = gray.shape[0] // bs, gray.shape[1] // bs
    if nbh >= 2 and nbw >= 2:
        rb = res_g[: nbh * bs, : nbw * bs].reshape(nbh, bs, nbw, bs)
        gb = gray[: nbh * bs, : nbw * bs].reshape(nbh, bs, nbw, bs)
        bstd = rb.std(axis=(1, 3)).ravel()
        bmean = gb.mean(axis=(1, 3)).ravel()
        cv = float(bstd.std() / (bstd.mean() + EPS))
        p10, p90 = np.percentile(bstd, [10, 90])
        vals += [cv, _log10p(p90) - _log10p(p10), _corr(bmean, bstd)]
    else:
        vals += [0.0, 0.0, 0.0]
    names += ["noise_blk_cv", "noise_blk_p90_p10", "noise_int_corr"]
    return names, vals


def _pair_hist(res: np.ndarray, axis: str, trunc: int, step: float) -> np.ndarray:
    k = 2 * trunc + 1
    q = np.clip(np.rint(res / step), -trunc, trunc).astype(np.int64) + trunc
    a, b = (q[:, :-1], q[:, 1:]) if axis == "h" else (q[:-1, :], q[1:, :])
    return np.bincount((a * k + b).ravel(), minlength=k * k).astype(np.float64)


def _cooc(gray: np.ndarray, trunc: int, step: float):
    k = 2 * trunc + 1
    f1h, f1v = gray[:, 1:] - gray[:, :-1], gray[1:, :] - gray[:-1, :]
    f2h = gray[:, 2:] - 2 * gray[:, 1:-1] + gray[:, :-2]
    f2v = gray[2:, :] - 2 * gray[1:-1, :] + gray[:-2, :]
    f3 = ndimage.convolve(gray, _SQUARE3, mode="nearest")[1:-1, 1:-1]
    hists = {
        "f1": _pair_hist(f1h, "h", trunc, step) + _pair_hist(f1v, "v", trunc, step),
        "f2": _pair_hist(f2h, "h", trunc, step) + _pair_hist(f2v, "v", trunc, step),
        "f3": _pair_hist(f3, "h", trunc, step) + _pair_hist(f3, "v", trunc, step),
    }
    names, vals = [], []
    for tag, h in hists.items():
        h = h / max(h.sum(), 1.0)
        for i in range(k * k):
            names.append(f"cooc_{tag}_{i:02d}")
            vals.append(float(h[i]))
    return names, vals


def _dct(gray: np.ndarray):
    h, w = gray.shape
    hb, wb = h // 8, w // 8
    g = gray[: hb * 8, : wb * 8]
    blocks = g.reshape(hb, 8, wb, 8).transpose(0, 2, 1, 3).reshape(-1, 8, 8) - 128.0
    coef = sfft.dctn(blocks, axes=(1, 2), norm="ortho")
    uv = np.add.outer(np.arange(8), np.arange(8))
    ac = coef[:, uv > 0]
    mag = np.abs(np.rint(ac))
    mag = mag[mag >= 1]
    if mag.size >= 10:
        e = np.floor(np.log10(mag))
        lead = np.floor(mag / 10.0 ** e).astype(np.int64)
        lead = np.where(lead >= 10, lead // 10, lead)
        lead = np.clip(lead, 1, 9)
        dist = np.bincount(lead, minlength=10)[1:10].astype(np.float64)
        dist /= dist.sum()
        kl = float(np.sum(dist * np.log((dist + 1e-9) / (_BENFORD_P + 1e-9))))
    else:
        dist, kl = np.zeros(9), 0.0
    names = [f"dct_lead{d}" for d in range(1, 10)] + ["dct_benford_kl", "dct_zero_frac"]
    vals = list(dist) + [kl, float(np.mean(np.abs(ac) < 0.5))]

    absc = np.abs(coef)
    for tag, lo, hi in [("low", 1, 3), ("mid", 4, 8), ("high", 9, 14)]:
        m = (uv >= lo) & (uv <= hi)
        names.append(f"dct_band_{tag}")
        vals.append(_log10p(float(absc[:, m].mean())))

    for tag, axis in [("h", 1), ("v", 0)]:
        d = np.abs(np.diff(gray, axis=axis)).mean(axis=1 - axis)
        pos = np.arange(d.size)
        boundary = (pos % 8) == 7
        ratio = (d[boundary].mean() + EPS) / (d[~boundary].mean() + EPS) if boundary.any() and (~boundary).any() else 1.0
        names.append(f"dct_blockiness_{tag}")
        vals.append(float(np.log(ratio)))
    return names, vals


def _color(rgb: np.ndarray, gray: np.ndarray, kernel: int):
    a = rgb.astype(np.float64)
    mx, mn = a.max(axis=2), a.min(axis=2)
    sat = (mx - mn) / (mx + 1.0)
    names = ["color_sat_mean", "color_sat_std", "color_sat_p95", "color_sat_high_frac"]
    vals = [float(sat.mean()), float(sat.std()), float(np.percentile(sat, 95)), float(np.mean(sat > _SAT_HIGH))]

    names += ["color_corr_rg", "color_corr_rb", "color_corr_gb"]
    vals += [_corr(a[..., 0], a[..., 1]), _corr(a[..., 0], a[..., 2]), _corr(a[..., 1], a[..., 2])]

    cb = -0.168736 * a[..., 0] - 0.331264 * a[..., 1] + 0.5 * a[..., 2]
    cr = 0.5 * a[..., 0] - 0.418688 * a[..., 1] - 0.081312 * a[..., 2]
    ncb = _log10p(float(np.std(cb - ndimage.median_filter(cb, size=kernel, mode="reflect"))))
    ncr = _log10p(float(np.std(cr - ndimage.median_filter(cr, size=kernel, mode="reflect"))))
    nl = _log10p(float(np.std(gray - ndimage.median_filter(gray, size=kernel, mode="reflect"))))
    names += ["color_chroma_noise_cb", "color_chroma_noise_cr", "color_chroma_luma_ratio"]
    vals += [ncb, ncr, 0.5 * (ncb + ncr) - nl]

    u8 = rgb.astype(np.int64)
    packed = (u8[..., 0] << 16) | (u8[..., 1] << 8) | u8[..., 2]
    names.append("color_unique_ratio")
    vals.append(float(np.unique(packed).size / packed.size))

    hist = np.bincount((np.clip(gray, 0, 255).astype(np.int64) // 4).ravel(), minlength=64).astype(np.float64)
    pr = hist[hist > 0] / hist.sum()
    names += ["color_luma_entropy", "color_clip_frac"]
    vals += [float(-(pr * np.log(pr)).sum() / np.log(64)), float(np.mean((gray <= 0.5) | (gray >= 254.5)))]
    return names, vals


def _lbp_riu2(gray: np.ndarray) -> np.ndarray:
    p = np.pad(gray, 1, mode="edge")
    c = p[1:-1, 1:-1]
    offs = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    h, w = gray.shape
    bits = np.stack([(p[1 + dy: 1 + dy + h, 1 + dx: 1 + dx + w] >= c) for dy, dx in offs]).astype(np.int8)
    trans = np.abs(bits - np.roll(bits, 1, axis=0)).sum(axis=0)
    ones = bits.sum(axis=0)
    code = np.where(trans <= 2, ones, 9)
    hist = np.bincount(code.ravel(), minlength=10).astype(np.float64)
    return hist / hist.sum()


def _texture(gray: np.ndarray):
    gy = ndimage.sobel(gray, axis=0, mode="reflect") / 8.0
    gx = ndimage.sobel(gray, axis=1, mode="reflect") / 8.0
    mag = np.hypot(gx, gy)
    m = float(mag.mean())
    p50, p99 = np.percentile(mag, [50, 99])
    names = ["tex_grad_logmean", "tex_grad_cv", "tex_grad_p99_p50", "tex_flat_frac", "tex_gx_kurt",
             "tex_lap_logvar", "tex_orient_entropy"]
    lap = ndimage.laplace(gray, mode="reflect")
    theta = np.mod(np.arctan2(gy, gx), np.pi)
    ob = np.minimum((theta / np.pi * 8).astype(np.int64), 7)
    oh = np.bincount(ob.ravel(), weights=mag.ravel(), minlength=8)
    po = oh[oh > 0] / max(oh.sum(), EPS)
    vals = [_log10p(m), float(mag.std() / (m + EPS)), _log10p(p99) - _log10p(p50),
            float(np.mean(mag < _FLAT_GRADIENT)), _slog(_moments(gx)[2]),
            _log10p(float(lap.var())), float(-(po * np.log(po)).sum() / np.log(8)) if po.size else 0.0]

    lbp = _lbp_riu2(gray)
    names += [f"tex_lbp_{i}" for i in range(10)]
    vals += list(lbp)

    m1 = ndimage.uniform_filter(gray, 5, mode="reflect")
    m2 = ndimage.uniform_filter(gray * gray, 5, mode="reflect")
    lv = np.maximum(m2 - m1 * m1, 0.0)
    q10, q50, q90 = np.percentile(lv, [10, 50, 90])
    names += ["tex_localvar_logmed", "tex_localvar_p90_p10"]
    vals += [_log10p(q50), _log10p(q90) - _log10p(q10)]
    return names, vals


def _haar(a: np.ndarray):
    a = a[: a.shape[0] // 2 * 2, : a.shape[1] // 2 * 2]
    p, q, r, s = a[0::2, 0::2], a[0::2, 1::2], a[1::2, 0::2], a[1::2, 1::2]
    return (p + q + r + s) / 4.0, (p + q - r - s) / 4.0, (p - q + r - s) / 4.0, (p - q - r + s) / 4.0


def _wavelet(gray: np.ndarray, levels: int):
    names, vals = [], []
    ll = gray
    prev_hh = None
    for lv in range(1, levels + 1):
        ll, lh, hl, hh = _haar(ll)
        for tag, band in (("lh", lh), ("hl", hl), ("hh", hh)):
            sd, _, kurt = _moments(band)
            names += [f"wav{lv}_{tag}_logstd", f"wav{lv}_{tag}_kurt"]
            vals += [_log10p(sd), _slog(kurt)]
        cur = _log10p(_moments(hh)[0])
        if prev_hh is not None:
            names.append(f"wav{lv}_hh_ratio")
            vals.append(prev_hh - cur)
        prev_hh = cur
    return names, vals


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
@dataclass
class ImageFeatures:
    features: np.ndarray              # (n_patches, n_features) float32
    positions: np.ndarray             # (n_patches, 2) int  -> (y, x) top-left
    n_tiles: int                      # tiles considered
    n_flat: int                       # tiles rejected as flat
    height: int
    width: int


def patch_positions(h: int, w: int, patch: int) -> List[Tuple[int, int]]:
    """Grid of top-left corners covering the image; coordinates are multiples of 8
    so that 8x8 JPEG block alignment is preserved."""
    def axis(n: int) -> np.ndarray:
        k = max(1, int(np.ceil(n / patch)))
        starts = np.linspace(0, n - patch, k)
        return np.unique((np.floor(starts / 8.0) * 8).astype(np.int64))

    return [(int(y), int(x)) for y in axis(h) for x in axis(w)]


class FeatureExtractor:
    def __init__(self, patch_cfg: PatchConfig, feat_cfg: FeatureConfig):
        self.patch_cfg = patch_cfg
        self.feat_cfg = feat_cfg
        probe = np.random.default_rng(0).integers(0, 256, (patch_cfg.patch_size, patch_cfg.patch_size, 3),
                                                  dtype=np.uint8)
        named = self._compute(probe)
        self.names: List[str] = [n for _, n, _ in named]
        self.groups: List[str] = [g for g, _, _ in named]
        self.n_features = len(self.names)
        self.group_columns: Dict[str, np.ndarray] = {
            g: np.array([i for i, gg in enumerate(self.groups) if gg == g], dtype=np.int64)
            for g in GROUP_ORDER}

    # ------------------------------------------------------------------ #
    def _compute(self, patch: np.ndarray) -> List[Tuple[str, str, float]]:
        fc = self.feat_cfg
        gray = to_gray(patch)
        parts = [
            ("spectrum", _spectrum(gray, fc.n_radial_bins)),
            ("noise", _noise(patch, gray, fc.median_kernel)),
            ("cooc", _cooc(gray, fc.cooc_truncation, fc.cooc_quant_step)),
            ("dct", _dct(gray)),
            ("color", _color(patch, gray, fc.median_kernel)),
            ("texture", _texture(gray)),
            ("wavelet", _wavelet(gray, fc.wavelet_levels)),
        ]
        out: List[Tuple[str, str, float]] = []
        for group, (names, vals) in parts:
            assert len(names) == len(vals), f"feature group {group}: names/values length mismatch"
            out.extend((group, n, float(v)) for n, v in zip(names, vals))
        return out

    def patch_vector(self, patch: np.ndarray) -> np.ndarray:
        vals = np.array([v for _, _, v in self._compute(patch)], dtype=np.float64)
        vals = np.nan_to_num(vals, nan=0.0, posinf=1e6, neginf=-1e6)
        return vals.astype(np.float32)

    # ------------------------------------------------------------------ #
    def select_patches(self, rgb: np.ndarray, max_patches: int) -> Tuple[List[Tuple[int, int]], int, int]:
        """Deterministically pick up to ``max_patches`` non-flat tiles spread over the image.
        Returns (positions, n_tiles_total, n_flat_rejected)."""
        h, w = rgb.shape[:2]
        p = self.patch_cfg.patch_size
        if h < p or w < p:
            return [], 0, 0
        tiles = patch_positions(h, w, p)
        order = np.random.default_rng([h, w, p]).permutation(len(tiles))
        gray = to_gray(rgb)
        chosen: List[Tuple[int, int]] = []
        flat = 0
        for i in order:
            y, x = tiles[i]
            if float(gray[y: y + p, x: x + p].std()) < self.patch_cfg.min_patch_std:
                flat += 1
                continue
            chosen.append((y, x))
            if len(chosen) >= max_patches:
                break
        chosen.sort()
        return chosen, len(tiles), flat

    def extract_image(self, rgb: np.ndarray, max_patches: Optional[int] = None) -> ImageFeatures:
        max_patches = int(max_patches or self.patch_cfg.max_patches)
        chosen, n_tiles, n_flat = self.select_patches(rgb, max_patches)
        p = self.patch_cfg.patch_size
        if chosen:
            feats = np.stack([self.patch_vector(rgb[y: y + p, x: x + p]) for y, x in chosen])
            pos = np.array(chosen, dtype=np.int64)
        else:
            feats = np.zeros((0, self.n_features), dtype=np.float32)
            pos = np.zeros((0, 2), dtype=np.int64)
        return ImageFeatures(feats, pos, n_tiles, n_flat, rgb.shape[0], rgb.shape[1])


@functools.lru_cache(maxsize=8)
def get_extractor(feature_key: str) -> FeatureExtractor:
    """Cached extractor keyed by ``Config.feature_key()`` (cheap to call per image / per worker)."""
    d = json.loads(feature_key)
    return FeatureExtractor(PatchConfig(**d["patch"]), FeatureConfig(**d["features"]))


def extractor_for(cfg: Config) -> FeatureExtractor:
    return get_extractor(cfg.feature_key())
