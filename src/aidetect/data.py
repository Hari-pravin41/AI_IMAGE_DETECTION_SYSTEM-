"""Dataset discovery, cleaning, leak-free splitting and bias auditing.

Supported layouts (all combinable, none hardcoded: class names and split names
come from ``config.data``):

    data/real/*.jpg  data/fake/*.png            class folders
    data/train/real  data/train/fake  data/test/...   pre-split dataset
    --real-dir A --real-dir B --fake-dir C       explicit folders (repeatable)
    --manifest labels.csv                        CSV with columns path,label[,split]
"""
from __future__ import annotations

import csv
import logging
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .config import Config

log = logging.getLogger(__name__)

REAL, FAKE = 0, 1
LABEL_NAMES = {REAL: "real", FAKE: "ai_generated"}


class DataError(Exception):
    """Dataset problem that the user can fix (wrong folders, too few images, ...)."""


@dataclass
class Sample:
    path: str
    label: int                      # 0 = real, 1 = AI-generated
    split: Optional[str] = None     # "train" | "test" | None (decide automatically)
    origin: str = ""                # where the label came from (folder name / manifest)


@dataclass
class ImageMeta:
    """What we learn about each image while extracting its features."""
    path: str
    label: int
    split: Optional[str]
    sha256: str
    dhash: int
    width: int
    height: int
    fmt: str
    file_size: int
    has_camera_exif: bool


# --------------------------------------------------------------------------- #
# discovery
# --------------------------------------------------------------------------- #
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def label_from_name(name: str, aliases: Dict[str, List[str]], allow_digits: bool = False) -> Optional[int]:
    n = _norm(name)
    if n in {_norm(a) for a in aliases["real"]}:
        return REAL
    if n in {_norm(a) for a in aliases["fake"]}:
        return FAKE
    if allow_digits and n in ("0", "1"):
        return int(n)
    return None


def list_images(directory: os.PathLike, extensions: Iterable[str], recursive: bool = True) -> List[str]:
    exts = {e.lower() for e in extensions}
    out: List[str] = []
    root = os.fspath(directory)
    if not os.path.isdir(root):
        raise DataError(f"Not a directory: {root}")
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for fn in sorted(filenames):
            if fn.startswith("."):
                continue
            if os.path.splitext(fn)[1].lower() in exts:
                out.append(os.path.join(dirpath, fn))
        if not recursive:
            break
    return out


def _split_lookup(cfg: Config) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for split, names in cfg.data.split_aliases.items():
        target = "test" if split == "test" else "train"      # validation data joins train (CV handles tuning)
        for n in names:
            lookup[_norm(n)] = target
    return lookup


def discover_from_directory(root: os.PathLike, cfg: Config) -> List[Sample]:
    rootp = Path(root)
    if not rootp.is_dir():
        raise DataError(f"Data directory not found: {rootp}")
    lookup = _split_lookup(cfg)
    samples: List[Sample] = []
    ignored: List[str] = []
    for child in sorted(rootp.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        lab = label_from_name(child.name, cfg.data.class_aliases)
        if lab is not None:
            samples += [Sample(p, lab, None, child.name) for p in list_images(child, cfg.data.extensions)]
        elif _norm(child.name) in lookup:
            split = lookup[_norm(child.name)]
            matched = False
            for sub in sorted(child.iterdir()):
                if not sub.is_dir() or sub.name.startswith("."):
                    continue
                sl = label_from_name(sub.name, cfg.data.class_aliases)
                if sl is None:
                    ignored.append(f"{child.name}/{sub.name}")
                    continue
                matched = True
                samples += [Sample(p, sl, split, f"{child.name}/{sub.name}")
                            for p in list_images(sub, cfg.data.extensions)]
            if not matched:
                ignored.append(child.name)
        else:
            ignored.append(child.name)
    if ignored:
        log.warning("Ignored folders that are not a known class or split name: %s", ", ".join(ignored))
    if not samples:
        found = sorted(c.name for c in rootp.iterdir() if c.is_dir()) or ["<no sub-folders>"]
        raise DataError(
            f"No labelled images found under '{rootp}'. Folders found: {', '.join(found)}.\n"
            f"Expected class folders such as real/ and fake/ (known names: "
            f"real={cfg.data.class_aliases['real']}, fake={cfg.data.class_aliases['fake']}), optionally inside "
            f"train/ and test/. Otherwise pass --real-dir / --fake-dir or --manifest.")
    return samples


def discover_from_dirs(real_dirs: Sequence[str], fake_dirs: Sequence[str], cfg: Config) -> List[Sample]:
    out: List[Sample] = []
    for label, dirs in ((REAL, real_dirs), (FAKE, fake_dirs)):
        for d in dirs:
            found = list_images(d, cfg.data.extensions)
            if not found:
                log.warning("No images found in %s", d)
            out += [Sample(p, label, None, os.fspath(d)) for p in found]
    return out


def discover_from_manifest(manifest: os.PathLike, cfg: Config) -> List[Sample]:
    mp = Path(manifest)
    if not mp.is_file():
        raise DataError(f"Manifest not found: {mp}")
    out: List[Sample] = []
    missing = 0
    with open(mp, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        cols = {(c or "").strip().lower(): c for c in (reader.fieldnames or [])}
        if "path" not in cols or "label" not in cols:
            raise DataError(f"Manifest {mp} needs a header with columns 'path' and 'label' "
                            f"(optional: 'split'); found {list(cols)}")
        lookup = _split_lookup(cfg)
        for n, row in enumerate(reader, start=2):
            rel = (row[cols["path"]] or "").strip()
            raw_label = (row[cols["label"]] or "").strip()
            if not rel and not raw_label:
                continue
            lab = label_from_name(raw_label, cfg.data.class_aliases, allow_digits=True)
            if lab is None:
                raise DataError(f"Manifest {mp} line {n}: unknown label '{raw_label}'. Use 0/1 or a name "
                                f"from data.class_aliases.")
            p = Path(rel)
            if not p.is_absolute():
                p = mp.parent / p
            if not p.is_file():
                missing += 1
                continue
            split = None
            if "split" in cols and (row[cols["split"]] or "").strip():
                split = lookup.get(_norm(row[cols["split"]]))
                if split is None:
                    raise DataError(f"Manifest {mp} line {n}: unknown split '{row[cols['split']]}'")
            out.append(Sample(os.fspath(p), lab, split, "manifest"))
    if missing:
        log.warning("Manifest %s: %d listed files do not exist and were skipped", mp, missing)
    if not out:
        raise DataError(f"Manifest {mp} did not yield any existing image files")
    return out


def collect_samples(cfg: Config, data_dir: Optional[str] = None, real_dirs: Sequence[str] = (),
                    fake_dirs: Sequence[str] = (), manifest: Optional[str] = None) -> List[Sample]:
    if not (data_dir or real_dirs or fake_dirs or manifest):
        raise DataError("Provide a dataset: --data DIR, or --real-dir/--fake-dir, or --manifest FILE")
    if bool(real_dirs) != bool(fake_dirs) and not (data_dir or manifest):
        raise DataError("--real-dir and --fake-dir must be given together")
    samples: List[Sample] = []
    if data_dir:
        samples += discover_from_directory(data_dir, cfg)
    if real_dirs or fake_dirs:
        samples += discover_from_dirs(real_dirs, fake_dirs, cfg)
    if manifest:
        samples += discover_from_manifest(manifest, cfg)

    by_path: Dict[str, Sample] = {}
    for s in samples:
        key = os.path.realpath(s.path)
        if key in by_path:
            if by_path[key].label != s.label:
                raise DataError(f"The same file is labelled both real and AI-generated: {s.path}")
            continue
        by_path[key] = s
    samples = list(by_path.values())

    cap = cfg.data.max_images_per_class
    if cap:
        rng = np.random.default_rng(cfg.runtime.seed)
        kept: List[Sample] = []
        for label in (REAL, FAKE):
            group = [s for s in samples if s.label == label]
            if len(group) > cap:
                idx = np.sort(rng.choice(len(group), size=cap, replace=False))
                group = [group[i] for i in idx]
            kept += group
        samples = kept
    counts = Counter(s.label for s in samples)
    if counts[REAL] == 0 or counts[FAKE] == 0:
        raise DataError(f"Need both classes; found {counts[REAL]} real and {counts[FAKE]} AI-generated images.")
    return samples


# --------------------------------------------------------------------------- #
# cleaning: exact + near duplicates
# --------------------------------------------------------------------------- #
def drop_exact_duplicates(metas: Sequence[ImageMeta]) -> Tuple[List[int], Dict[str, int]]:
    """Indices to keep. Identical files are kept once (train copy preferred); identical files
    carrying *conflicting* labels are removed entirely."""
    by_hash: Dict[str, List[int]] = defaultdict(list)
    for i, m in enumerate(metas):
        by_hash[m.sha256].append(i)
    keep: List[int] = []
    n_dup = n_conflict = 0
    for idxs in by_hash.values():
        labels = {metas[i].label for i in idxs}
        if len(labels) > 1:
            n_conflict += len(idxs)
            continue
        idxs = sorted(idxs, key=lambda i: (metas[i].split == "test", i))
        keep.append(idxs[0])
        n_dup += len(idxs) - 1
    return sorted(keep), {"exact_duplicates_removed": n_dup, "conflicting_label_files_removed": n_conflict}


_POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _popcount(x: np.ndarray) -> np.ndarray:
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x).astype(np.int32)
    x = np.ascontiguousarray(x)
    return _POP8[x.view(np.uint8).reshape(x.shape + (8,))].sum(axis=-1).astype(np.int32)


def group_near_duplicates(dhashes: Sequence[int], labels: Sequence[int], threshold: int,
                          chunk: int = 512) -> np.ndarray:
    """Union-find over same-class images whose dHash differs by <= threshold bits.
    Returns an integer group id per image (singletons get their own id)."""
    n = len(dhashes)
    parent = np.arange(n)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    if threshold >= 0 and n > 1:
        h_all = np.array([int(h) & 0xFFFFFFFFFFFFFFFF for h in dhashes], dtype=np.uint64)
        lab = np.asarray(labels)
        for label in np.unique(lab):
            idx = np.where(lab == label)[0]
            h = h_all[idx]
            for start in range(0, len(idx), chunk):
                block = h[start:start + chunk, None] ^ h[None, :]
                rows, cols = np.nonzero(_popcount(block) <= threshold)
                for r, c in zip(rows, cols):
                    i, j = start + int(r), int(c)
                    if i < j:
                        ra, rb = find(int(idx[i])), find(int(idx[j]))
                        if ra != rb:
                            parent[max(ra, rb)] = min(ra, rb)
    roots = np.array([find(i) for i in range(n)])
    _, groups = np.unique(roots, return_inverse=True)
    return groups


# --------------------------------------------------------------------------- #
# splitting
# --------------------------------------------------------------------------- #
def assign_splits(labels: np.ndarray, groups: np.ndarray, predefined: Sequence[Optional[str]],
                  test_size: float, seed: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Return an array of 'train' / 'test' / 'drop' per image plus a description.

    * A pre-defined split (train/ + test/ folders or a manifest column) is respected when both
      parts exist and contain both classes; test images that are near-duplicates of training
      images are dropped to prevent leakage.
    * Otherwise a stratified, group-aware split keeps near-duplicates together.
    """
    labels = np.asarray(labels)
    n = len(labels)
    pre = np.array([p or "" for p in predefined])
    has_pre = (all(((pre == s) & (labels == c)).any() for s in ("train", "test") for c in (REAL, FAKE)))
    info: Dict[str, Any] = {}
    if has_pre:
        split = np.where(pre == "test", "test", "train").astype(object)
        leaked = 0
        for g in np.unique(groups):
            members = np.where(groups == g)[0]
            if (split[members] == "train").any() and (split[members] == "test").any():
                for m in members:
                    if split[m] == "test":
                        split[m] = "drop"
                        leaked += 1
        info = {"mode": "predefined", "test_dropped_for_leakage": leaked}
        return split, info

    from sklearn.model_selection import StratifiedGroupKFold
    n_splits = max(2, int(round(1.0 / test_size)))
    try:
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        _, test_idx = next(iter(sgkf.split(np.zeros(n), labels, groups)))
    except ValueError as e:
        raise DataError(f"Cannot create a stratified split: {e}") from e
    split = np.full(n, "train", dtype=object)
    split[test_idx] = "test"
    info = {"mode": "stratified_group_split", "n_splits_used": n_splits}
    return split, info


# --------------------------------------------------------------------------- #
# audit: detect dataset biases a classifier could exploit instead of the real signal
# --------------------------------------------------------------------------- #
def audit_dataset(metas: Sequence[ImageMeta]) -> Dict[str, Any]:
    per: Dict[str, Dict[str, Any]] = {}
    for label in (REAL, FAKE):
        ms = [m for m in metas if m.label == label]
        if not ms:
            continue
        fmts = Counter((m.fmt or "unknown") for m in ms)
        w = np.array([m.width for m in ms], dtype=float)
        h = np.array([m.height for m in ms], dtype=float)
        bpp = np.array([m.file_size * 8.0 / max(m.width * m.height, 1) for m in ms])
        per[LABEL_NAMES[label]] = {
            "n": len(ms),
            "formats": dict(fmts.most_common()),
            "jpeg_fraction": float(np.mean([(m.fmt or "").upper() == "JPEG" for m in ms])),
            "camera_exif_fraction": float(np.mean([m.has_camera_exif for m in ms])),
            "median_width": float(np.median(w)), "median_height": float(np.median(h)),
            "median_megapixels": float(np.median(w * h) / 1e6),
            "median_bits_per_pixel": float(np.median(bpp)),
            "distinct_sizes": int(len({(m.width, m.height) for m in ms})),
        }
    warnings: List[str] = []
    if len(per) == 2:
        r, f = per["real"], per["ai_generated"]
        if abs(r["jpeg_fraction"] - f["jpeg_fraction"]) > 0.5:
            warnings.append(
                f"File-format imbalance: {r['jpeg_fraction']:.0%} of real images are JPEG vs "
                f"{f['jpeg_fraction']:.0%} of AI images. The model may learn compression instead of "
                f"generation artifacts. Consider preprocess.recompress_jpeg_quality (e.g. 90) or matching formats.")
        if abs(r["camera_exif_fraction"] - f["camera_exif_fraction"]) > 0.5:
            warnings.append(
                f"Camera-EXIF imbalance ({r['camera_exif_fraction']:.0%} real vs "
                f"{f['camera_exif_fraction']:.0%} AI). EXIF is never used as a model feature, but this "
                f"signals the two classes come from different pipelines.")
        ratio = (r["median_megapixels"] + 1e-9) / (f["median_megapixels"] + 1e-9)
        if ratio > 2.0 or ratio < 0.5:
            warnings.append(
                f"Image-size imbalance (median {r['median_megapixels']:.2f} MP real vs "
                f"{f['median_megapixels']:.2f} MP AI). Sizes are not model inputs, but different "
                f"resolutions usually mean different processing histories.")
        bratio = (r["median_bits_per_pixel"] + 1e-9) / (f["median_bits_per_pixel"] + 1e-9)
        if bratio > 3.0 or bratio < 1 / 3.0:
            warnings.append(
                f"Compression-level imbalance (median {r['median_bits_per_pixel']:.2f} vs "
                f"{f['median_bits_per_pixel']:.2f} bits/pixel). Strongly different compression between "
                f"classes is a classic shortcut; consider preprocess.recompress_jpeg_quality.")
        n_r, n_f = r["n"], f["n"]
        if max(n_r, n_f) / max(min(n_r, n_f), 1) > 3:
            warnings.append(f"Class imbalance ({n_r} real vs {n_f} AI). Class weights compensate, "
                            f"but more minority-class images will help.")
    return {"per_class": per, "warnings": warnings}


def summarize_splits(labels: np.ndarray, split: np.ndarray) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for s in ("train", "test"):
        m = split == s
        out[s] = {"real": int(np.sum(m & (labels == REAL))), "ai_generated": int(np.sum(m & (labels == FAKE)))}
    return out
