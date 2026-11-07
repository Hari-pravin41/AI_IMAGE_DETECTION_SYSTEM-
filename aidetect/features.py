"""Forensic feature extraction.

Every feature is computed on a *native-resolution* window of the image (no resizing, which
would destroy the very traces we look for). Five groups:

spectral     radial Fourier profile, power-law fit, up-sampling peaks, axis / angular energy
noise        median-filter residual statistics, cross-channel & lag correlations, noise-vs-signal
texture      gradient / Laplacian statistics, edge density, LBP histogram, intensity entropy
color        saturation / value stats, clipping, channel entropy & correlation, histogram combing
compression  8x8 blockiness, error-level analysis, DCT band energies, Benford first-digit law
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np
import scipy.fft
from PIL import Image
from skimage.feature import local_binary_pattern

from .config import FEATURE_GROUPS, Config
from .loader import LoadedImage

FEATURE_VERSION = 1
EPS = 1e-8
_BENFORD_P = np.log10(1.0 + 1.0 / np.arange(1, 10))   # Benford first-digit probabilities

META_NAMES = (
    "meta.log_pixels", "meta.aspect", "meta.w_mod8", "meta.h_mod8", "meta.w_mod64", "meta.h_mod64",
    "meta.is_jpeg", "meta.is_png", "meta.is_webp", "meta.is_other_fmt", "meta.log_bpp",
    "meta.has_exif", "meta.has_camera", "meta.has_software", "meta.has_icc", "meta.has_alpha",
    "meta.jpeg_q_mean", "meta.jpeg_q_max",
)


# ----------------------------------------------------------------------------- helpers
def _skew_kurt(x: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, np.float64).ravel()
    s = x.std()
    if s < 1e-9:
        return 0.0, 0.0
    z = (x - x.mean()) / s
    return float(np.mean(z ** 3)), float(np.mean(z ** 4) - 3.0)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 1e-9 else 0.0


def _entropy(hist: np.ndarray) -> float:
    p = hist.astype(np.float64)
    s = p.sum()
    if s <= 0:
        return 0.0
    p = p[p > 0] / s
    return float(-(p * np.log2(p)).sum())


@lru_cache(maxsize=16)
def _radial_maps(c: int, nb: int):
    y, x = np.indices((c, c))
    cy = cx = c // 2
    r = np.hypot(y - cy, x - cx)
    idx = np.minimum((r / (c / 2) * nb).astype(np.int64), nb)   # == nb -> outside the inscribed disc
    ang = np.arctan2(y - cy, x - cx) % np.pi
    return r, idx, ang


@lru_cache(maxsize=16)
def _window(c: int) -> np.ndarray:
    h = np.hanning(c).astype(np.float32)
    return h[:, None] * h[None, :]


def _log_spectrum(img: np.ndarray, win: np.ndarray) -> np.ndarray:
    return np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(img * win)))).astype(np.float32)


# ----------------------------------------------------------------------------- groups
def spectral_features(gray: np.ndarray, nb: int) -> dict[str, float]:
    c = gray.shape[0]
    r, idx, ang = _radial_maps(c, nb)
    win = _window(c)
    P = _log_spectrum(gray - gray.mean(), win)
    inside = idx < nb
    cnt = np.bincount(idx[inside], minlength=nb).astype(np.float64)
    prof = np.bincount(idx[inside], weights=P[inside], minlength=nb) / np.maximum(cnt, 1)

    out: dict[str, float] = {"spectral.level": float(prof.mean())}
    for i, v in enumerate(prof - prof.mean()):
        out[f"spectral.radial_{i:02d}"] = float(v)

    rc = (np.arange(nb) + 0.5) / nb * (c / 2)
    A = np.stack([np.log(rc[1:]), np.ones(nb - 1)], axis=1)
    coef = np.linalg.lstsq(A, prof[1:], rcond=None)[0]
    out["spectral.slope"] = float(coef[0])
    out["spectral.intercept"] = float(coef[1])
    out["spectral.fit_rmse"] = float(np.sqrt(np.mean((prof[1:] - A @ coef) ** 2)))
    q = max(nb // 4, 1)
    out["spectral.hf_drop"] = float(prof[-q:].mean() - prof[q:2 * q].mean())
    out["spectral.hf_tail"] = float(prof[-2:].mean() - prof[-4:-2].mean())

    # periodic peaks (up-sampling / grid artefacts) live in the high-pass residual spectrum
    res = gray - cv2.GaussianBlur(gray, (0, 0), 1.0)
    Pr = _log_spectrum(res, win)
    diff = Pr - cv2.blur(Pr, (7, 7))
    mask = (r > max(0.06 * c, 4.0)) & (r < c / 2)
    d = diff[mask].astype(np.float64)
    sd = d.std() + EPS
    out["spectral.peak_max"] = float(d.max())
    out["spectral.peak_p999"] = float(np.percentile(d, 99.9))
    out["spectral.peak_frac3"] = float((d > 3 * sd).mean())
    out["spectral.peak_z"] = float((d.max() - d.mean()) / sd)
    out["spectral.peak_kurt"] = _skew_kurt(d)[1]

    cy = c // 2
    for name, line, m in (("h", diff[cy, :], mask[cy, :]), ("v", diff[:, cy], mask[:, cy])):
        out[f"spectral.axis_{name}"] = float(line[m].mean()) if m.any() else 0.0

    band = (r > 0.5 * (c / 2)) & (r < c / 2)
    ai = np.minimum((ang[band] / np.pi * 8).astype(np.int64), 7)
    sums = np.bincount(ai, weights=Pr[band], minlength=8)
    vals = sums / np.maximum(np.bincount(ai, minlength=8), 1)
    out["spectral.aniso_range"] = float((vals.max() - vals.min()) / (abs(vals.mean()) + EPS))
    out["spectral.aniso_std"] = float(vals.std())
    return out


def noise_features(rgb: np.ndarray) -> dict[str, float]:
    f = rgb.astype(np.float32)
    res = f - cv2.medianBlur(rgb, 3).astype(np.float32)
    out: dict[str, float] = {}
    for i, ch in enumerate("rgb"):
        sk, ku = _skew_kurt(res[..., i])
        out[f"noise.std_{ch}"] = float(res[..., i].std())
        out[f"noise.skew_{ch}"] = sk
        out[f"noise.kurt_{ch}"] = ku
    out["noise.corr_rg"] = _corr(res[..., 0], res[..., 1])
    out["noise.corr_gb"] = _corr(res[..., 1], res[..., 2])
    out["noise.corr_rb"] = _corr(res[..., 0], res[..., 2])

    rg = res.mean(axis=2)
    rg = rg - rg.mean()
    var = float(rg.var()) + EPS
    for k in (1, 2, 3, 4, 8):
        out[f"noise.ac_h{k}"] = float((rg[:, :-k] * rg[:, k:]).mean() / var)
        out[f"noise.ac_v{k}"] = float((rg[:-k, :] * rg[k:, :]).mean() / var)
    out["noise.ac_d1"] = float((rg[:-1, :-1] * rg[1:, 1:]).mean() / var)

    c = rg.shape[0]
    bs = 16
    nbk = c // bs
    gray = f.mean(axis=2)
    bstd = rg[:nbk * bs, :nbk * bs].reshape(nbk, bs, nbk, bs).std(axis=(1, 3)).ravel()
    bmean = gray[:nbk * bs, :nbk * bs].reshape(nbk, bs, nbk, bs).mean(axis=(1, 3)).ravel()
    m = bstd.mean()
    out["noise.blk_mean"] = float(m)
    out["noise.blk_cv"] = float(bstd.std() / (m + EPS))
    out["noise.blk_p10_p90"] = float((np.percentile(bstd, 10) + EPS) / (np.percentile(bstd, 90) + EPS))
    out["noise.blk_min_ratio"] = float(bstd.min() / (m + EPS))
    out["noise.sig_corr"] = _corr(bmean, bstd)
    out["noise.sig_slope"] = (float(np.polyfit(bmean, bstd, 1)[0]) if bmean.std() > 1e-6 else 0.0)
    return out


def texture_features(rgb: np.ndarray) -> dict[str, float]:
    gray_u8 = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = gray_u8.astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    out = {
        "texture.grad_mean": float(np.log1p(mag.mean())),
        "texture.grad_cv": float(mag.std() / (mag.mean() + EPS)),
        "texture.grad_kurt": _skew_kurt(mag)[1],
        "texture.lap_var": float(np.log1p(cv2.Laplacian(gray, cv2.CV_32F).var())),
    }
    med = float(np.median(gray_u8))
    lo = int(max(0, 0.66 * med))
    hi = int(min(255, max(1.33 * med, lo + 1)))
    out["texture.edge_density"] = float(cv2.Canny(gray_u8, lo, hi).mean() / 255.0)
    lbp = local_binary_pattern(gray_u8, 8, 1, method="uniform")
    h = np.bincount(lbp.astype(np.int64).ravel(), minlength=10)[:10].astype(np.float64)
    h /= max(h.sum(), 1)
    for i, v in enumerate(h):
        out[f"texture.lbp_{i}"] = float(v)
    out["texture.gray_entropy"] = _entropy(np.bincount(gray_u8.ravel(), minlength=256))
    return out


def color_features(rgb: np.ndarray) -> dict[str, float]:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    s = hsv[..., 1].astype(np.float32) / 255.0
    v = hsv[..., 2].astype(np.float32) / 255.0
    out = {
        "color.sat_mean": float(s.mean()), "color.sat_std": float(s.std()),
        "color.sat_p95": float(np.percentile(s, 95)),
        "color.val_mean": float(v.mean()), "color.val_std": float(v.std()),
        "color.clip_low": float((rgb.max(axis=2) <= 0).mean()),
        "color.clip_high": float((rgb.min(axis=2) >= 255).mean()),
    }
    for i, ch in enumerate("rgb"):
        out[f"color.entropy_{ch}"] = _entropy(np.bincount(rgb[..., i].ravel(), minlength=256))
    out["color.corr_rg"] = _corr(rgb[..., 0], rgb[..., 1])
    out["color.corr_gb"] = _corr(rgb[..., 1], rgb[..., 2])
    out["color.corr_rb"] = _corr(rgb[..., 0], rgb[..., 2])
    packed = (rgb[..., 0].astype(np.uint32) << 16) | (rgb[..., 1].astype(np.uint32) << 8) | rgb[..., 2]
    out["color.unique_ratio"] = float(np.unique(packed).size / packed.size)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hist = np.bincount(gray.ravel(), minlength=256)
    nz = np.flatnonzero(hist)
    span = hist[nz[0]:nz[-1] + 1].astype(np.float64)
    out["color.hist_gaps"] = float((span == 0).mean())
    out["color.hist_comb"] = float(np.abs(np.diff(span)).mean() / (span.mean() + EPS)) if span.size > 2 else 0.0
    return out


def _jpeg_roundtrip(rgb: np.ndarray, quality: int) -> np.ndarray:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"))


def compression_features(rgb: np.ndarray) -> dict[str, float]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    c = gray.shape[0]
    out: dict[str, float] = {}
    on = (np.arange(c - 1) % 8) == 7          # differences that straddle an 8x8 block boundary
    dh = np.abs(np.diff(gray, axis=1))
    dv = np.abs(np.diff(gray, axis=0))
    bh = dh[:, on].mean() / (dh[:, ~on].mean() + EPS)
    bv = dv[on, :].mean() / (dv[~on, :].mean() + EPS)
    out["compression.block_h"], out["compression.block_v"] = float(bh), float(bv)
    out["compression.block_mean"] = float((bh + bv) / 2)

    for q in (90, 75):
        ela = np.abs(rgb.astype(np.float32) - _jpeg_roundtrip(rgb, q).astype(np.float32)).mean(axis=2)
        out[f"compression.ela{q}_mean"] = float(ela.mean())
        if q == 90:
            out["compression.ela90_std"] = float(ela.std())
            out["compression.ela90_p99"] = float(np.percentile(ela, 99))
            blk = ela[:c // 8 * 8, :c // 8 * 8].reshape(c // 8, 8, c // 8, 8).mean(axis=(1, 3))
            out["compression.ela90_blk_cv"] = float(blk.std() / (blk.mean() + EPS))
    out["compression.ela_ratio"] = float(out["compression.ela75_mean"] / (out["compression.ela90_mean"] + EPS))

    nb = c // 8
    blocks = gray[:nb * 8, :nb * 8].reshape(nb, 8, nb, 8).transpose(0, 2, 1, 3).reshape(-1, 8, 8)
    D = scipy.fft.dctn(blocks - 128.0, axes=(1, 2), norm="ortho")
    u = np.add.outer(np.arange(8), np.arange(8))
    mag = np.abs(D)
    for s in range(1, 15):
        out[f"compression.dct_band_{s:02d}"] = float(np.log1p(mag[:, u == s]).mean())
    ac = mag[:, u > 0].ravel()
    ac = ac[ac >= 1.0]
    if ac.size >= 50:
        lead = np.floor(ac / 10.0 ** np.floor(np.log10(ac))).astype(np.int64)
        p = np.bincount(np.clip(lead, 1, 9), minlength=10)[1:10].astype(np.float64)
        p /= p.sum()
    else:
        p = np.full(9, 1.0 / 9)
    for i, v in enumerate(p):
        out[f"compression.benford_{i + 1}"] = float(v)
    out["compression.benford_dev"] = float(np.abs(p - _BENFORD_P).sum())
    return out


# ----------------------------------------------------------------------------- assembly
def crop_features(rgb: np.ndarray, cfg: Config) -> tuple[list[str], np.ndarray]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    feats: dict[str, float] = {}
    groups = [g for g in FEATURE_GROUPS if g in cfg.feature_groups]
    for g in groups:
        if g == "spectral":
            feats.update(spectral_features(gray, cfg.radial_bins))
        elif g == "noise":
            feats.update(noise_features(rgb))
        elif g == "texture":
            feats.update(texture_features(rgb))
        elif g == "color":
            feats.update(color_features(rgb))
        elif g == "compression":
            feats.update(compression_features(rgb))
    names = list(feats)
    vec = np.nan_to_num(np.array([feats[n] for n in names], dtype=np.float64),
                        nan=0.0, posinf=1e6, neginf=-1e6)
    return names, np.clip(vec, -1e6, 1e6).astype(np.float32)


def meta_vector(img: LoadedImage) -> np.ndarray:
    w, h = img.size
    fmt = (img.format or "").upper()
    q = img.quant_tables
    bpp = (img.n_bytes / (w * h)) if img.n_bytes else 0.0
    v = [
        np.log(w * h), w / h, float(w % 8 == 0), float(h % 8 == 0), float(w % 64 == 0), float(h % 64 == 0),
        float(fmt in ("JPEG", "MPO")), float(fmt == "PNG"), float(fmt == "WEBP"),
        float(fmt not in ("JPEG", "MPO", "PNG", "WEBP")), np.log1p(bpp),
        float(bool(img.exif)), float(img.provenance.get("camera_like", False)),
        float(bool(img.provenance.get("software"))), float(img.icc), float(img.has_alpha),
        float(np.mean(q)) if q else 0.0, float(np.max(q)) if q else 0.0,
    ]
    return np.array(v, dtype=np.float32)


@lru_cache(maxsize=32)
def _feature_names_cached(sig: str, groups: tuple, c: int, nb: int, meta: bool) -> tuple[str, ...]:
    cfg = Config(crop_size=c, feature_groups=list(groups), radial_bins=nb, use_metadata_features=meta)
    rng = np.random.default_rng(0)
    dummy = rng.integers(0, 256, size=(c, c, 3), dtype=np.uint8)
    names, _ = crop_features(dummy, cfg)
    return tuple(names) + (META_NAMES if meta else ())


def feature_names(cfg: Config) -> list[str]:
    groups = tuple(g for g in FEATURE_GROUPS if g in cfg.feature_groups)
    return list(_feature_names_cached(cfg.feature_signature(), groups, cfg.crop_size,
                                      cfg.radial_bins, bool(cfg.use_metadata_features)))


def feature_group_of(name: str) -> str:
    return name.split(".", 1)[0]


def _axis(n: int, extent: int) -> np.ndarray:
    if n == 1:
        return np.array([extent / 2.0])
    return np.linspace(0, extent, n)


def crop_positions(h: int, w: int, c: int, n: int) -> list[tuple[int, int]]:
    """Deterministic grid of up to ``n`` distinct window origins aligned to the 8-px JPEG grid."""
    if h < c or w < c:
        raise ValueError("image smaller than crop")
    ymax, xmax = (h - c) // 8 * 8, (w - c) // 8 * 8
    if n <= 1:
        return [(int(ymax / 2) // 8 * 8, int(xmax / 2) // 8 * 8)]
    ny = max(1, int(round(np.sqrt(n * h / w))))
    nx = int(np.ceil(n / ny))
    ys = np.clip(np.rint(_axis(ny, ymax) / 8) * 8, 0, ymax).astype(int)
    xs = np.clip(np.rint(_axis(nx, xmax) / 8) * 8, 0, xmax).astype(int)
    pos = list(dict.fromkeys((int(y), int(x)) for y in ys for x in xs))
    if len(pos) > n:
        keep = np.unique(np.rint(np.linspace(0, len(pos) - 1, n)).astype(int))
        pos = [pos[i] for i in keep]
    return pos


@dataclass
class Extraction:
    feats: np.ndarray                 # (n_crops, n_features) float32
    positions: list[tuple[int, int]]
    names: list[str]


def extract(img: LoadedImage, cfg: Config, n_crops: int | None = None) -> Extraction:
    h, w = img.rgb.shape[:2]
    c = cfg.crop_size
    pos = crop_positions(h, w, c, n_crops or cfg.crops_per_image)
    meta = meta_vector(img) if cfg.use_metadata_features else None
    rows, names = [], None
    for y, x in pos:
        crop = np.ascontiguousarray(img.rgb[y:y + c, x:x + c])
        names, vec = crop_features(crop, cfg)
        rows.append(vec if meta is None else np.concatenate([vec, meta]))
    if meta is not None:
        names = names + list(META_NAMES)
    return Extraction(np.stack(rows).astype(np.float32), pos, names)
