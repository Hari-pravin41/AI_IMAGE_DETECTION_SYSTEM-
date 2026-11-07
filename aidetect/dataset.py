"""Dataset discovery, validation, de-duplication and auditing."""
from __future__ import annotations

import csv
import hashlib
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from PIL import Image

from .config import Config
from .utils import Progress, log, resolve_jobs

LABELS = {"real": 0, "ai": 1}


class DatasetError(ValueError):
    """Dataset problem with a user-actionable message."""


@dataclass
class Sample:
    path: str
    label: int          # 1 = AI-generated, 0 = real
    source: str         # first sub-folder under the class folder (e.g. generator name)
    sha1: str = ""
    group: int = -1     # duplicate / near-duplicate cluster id (kept together when splitting)


# ----------------------------------------------------------------------------- discovery
def _walk_images(root: Path, exts: set[str]) -> list[Path]:
    out = []
    for dp, dn, fn in os.walk(root):
        dn[:] = sorted(d for d in dn if not d.startswith("."))
        for f in sorted(fn):
            if not f.startswith(".") and Path(f).suffix.lower() in exts:
                out.append(Path(dp) / f)
    return out


def _tokens(name: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", name.lower()) if t]


def _classify_dir(name: str, cfg: Config) -> str | None:
    toks = set(_tokens(name))
    ai = toks & {a.lower() for a in cfg.class_aliases["ai"]}
    real = toks & {a.lower() for a in cfg.class_aliases["real"]}
    if ai and not real:
        return "ai"
    if real and not ai:
        return "real"
    return None


def _source_of(path: Path, class_dir: Path) -> str:
    rel = path.relative_to(class_dir).parts
    return rel[0] if len(rel) > 1 else class_dir.name


def _find_class_dirs(root: Path, cfg: Config) -> dict[str, list[Path]]:
    found: dict[str, list[Path]] = {"ai": [], "real": []}
    unknown: list[str] = []

    def visit(d: Path, depth: int) -> None:
        for child in sorted(p for p in d.iterdir() if p.is_dir() and not p.name.startswith(".")):
            kind = _classify_dir(child.name, cfg)
            if kind:
                found[kind].append(child)
            elif depth < 3:
                visit(child, depth + 1)
            else:
                unknown.append(str(child))

    visit(root, 0)
    return found


def discover(cfg: Config, real_dirs=None, ai_dirs=None, root=None, manifest=None) -> list[Sample]:
    """Build the labelled sample list from explicit folders, an auto-detected root, or a CSV."""
    exts = set(cfg.image_extensions)
    samples: list[Sample] = []
    if manifest:
        mp = Path(manifest)
        if not mp.is_file():
            raise DatasetError(f"manifest not found: {mp}")
        with open(mp, newline="", encoding="utf-8-sig") as fh:
            rd = csv.DictReader(fh)
            cols = {(c or "").strip().lower(): c for c in (rd.fieldnames or [])}
            if "path" not in cols or "label" not in cols:
                raise DatasetError("manifest needs columns: path,label[,source]")
            for i, row in enumerate(rd, start=2):
                lab = str(row[cols["label"]]).strip().lower()
                if lab in {"1", "ai", "fake", "generated", "synthetic"}:
                    y = 1
                elif lab in {"0", "real", "authentic", "natural"}:
                    y = 0
                else:
                    raise DatasetError(f"manifest line {i}: unknown label {lab!r} (use ai/real or 1/0)")
                p = Path(row[cols["path"]].strip())
                if not p.is_absolute():
                    p = (mp.parent / p).resolve()
                src = row[cols["source"]].strip() if "source" in cols and row[cols["source"]] else \
                    ("ai" if y else "real")
                samples.append(Sample(str(p), y, src))
    else:
        class_dirs: dict[str, list[Path]] = {"ai": [], "real": []}
        if real_dirs or ai_dirs:
            for kind, dirs in (("real", real_dirs), ("ai", ai_dirs)):
                for d in dirs or []:
                    dp = Path(d)
                    if not dp.is_dir():
                        raise DatasetError(f"not a directory: {dp}")
                    class_dirs[kind].append(dp)
        elif root:
            rp = Path(root)
            if not rp.is_dir():
                raise DatasetError(f"not a directory: {rp}")
            class_dirs = _find_class_dirs(rp, cfg)
            if not class_dirs["ai"] or not class_dirs["real"]:
                subs = sorted(p.name for p in rp.iterdir() if p.is_dir())
                raise DatasetError(
                    f"could not identify both classes under {rp} (sub-folders: {subs}). Name the folders "
                    f"with words like real/fake/ai/generated, extend class_aliases in the config, or pass "
                    f"--real DIR and --ai DIR explicitly.")
        else:
            raise DatasetError("provide --root, or --real and --ai, or --manifest")
        if not class_dirs["ai"] or not class_dirs["real"]:
            raise DatasetError("both a real and an AI image folder are required")
        for kind, dirs in class_dirs.items():
            for d in dirs:
                for p in _walk_images(d, exts):
                    samples.append(Sample(str(p), LABELS[kind], _source_of(p, d)))
    if not samples:
        raise DatasetError("no images found")
    return samples


# ----------------------------------------------------------------------------- scanning
def _scan_file(path: str) -> dict:
    rec = {"path": path, "ok": False, "err": "", "sha1": "", "dhash": 0, "w": 0, "h": 0, "fmt": "", "bytes": 0}
    try:
        h = hashlib.sha1()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        rec["sha1"], rec["bytes"] = h.hexdigest(), os.path.getsize(path)
        with Image.open(path) as im:
            rec["fmt"], (rec["w"], rec["h"]) = im.format or "", im.size
            if im.format == "JPEG":
                im.draft("L", (128, 128))
            g = np.asarray(im.convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
        bits = (g[:, 1:] > g[:, :-1]).ravel()
        rec["dhash"] = int("".join("1" if b else "0" for b in bits), 2)
        rec["ok"] = True
    except Exception as e:  # noqa: BLE001 - any failure just marks the file unusable
        rec["err"] = f"{type(e).__name__}: {str(e)[:100]}"
    return rec


def _popcount(x: int) -> int:
    return bin(x).count("1")


def _cluster(sha: list[str], dh: list[int], hamming: int) -> list[int]:
    n = len(sha)
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    first: dict[str, int] = {}
    for i, s in enumerate(sha):
        if s in first:
            union(i, first[s])
        else:
            first[s] = i
    if hamming > 0:
        bands = hamming + 1                      # pigeonhole: <=hamming differing bits leaves a clean band
        edges = np.linspace(0, 64, bands + 1).astype(int)
        for b in range(bands):
            lo, hi = int(edges[b]), int(edges[b + 1])
            mask = ((1 << (hi - lo)) - 1) << lo
            buckets: dict[int, list[int]] = defaultdict(list)
            for i, d in enumerate(dh):
                if _popcount(d) in (0, 64):      # flat images all hash alike; only exact matches count
                    continue
                buckets[(d & mask) >> lo].append(i)
            for members in buckets.values():
                for a_pos, a in enumerate(members):
                    for b_idx in members[a_pos + 1:a_pos + 1 + 60]:
                        if find(a) != find(b_idx) and _popcount(dh[a] ^ dh[b_idx]) <= hamming:
                            union(a, b_idx)
    roots: dict[int, int] = {}
    return [roots.setdefault(find(i), len(roots)) for i in range(n)]


@dataclass
class ScanResult:
    samples: list[Sample]
    audit: dict


def scan_dataset(samples: list[Sample], cfg: Config, show_progress: bool = True) -> ScanResult:
    """Validate files, remove exact duplicates/conflicts, cluster near-duplicates, audit biases."""
    n_jobs = resolve_jobs(cfg.n_jobs)
    prog = Progress(len(samples), "scanning", show_progress)
    recs = []
    gen = Parallel(n_jobs=n_jobs, return_as="generator", batch_size=16)(delayed(_scan_file)(s.path) for s in samples)
    for r in gen:
        recs.append(r)
        prog.update()
    prog.close()

    issues: dict[str, list[str]] = {"unreadable": [], "too_small": [], "exact_duplicates": [], "label_conflicts": []}
    keep: list[tuple[Sample, dict]] = []
    min_side = cfg.effective_min_side
    for s, r in zip(samples, recs):
        if not r["ok"]:
            issues["unreadable"].append(f"{s.path}: {r['err']}")
        elif min(r["w"], r["h"]) < min_side:
            issues["too_small"].append(f"{s.path}: {r['w']}x{r['h']}")
        else:
            s.sha1 = r["sha1"]
            keep.append((s, r))

    by_sha: dict[str, list[int]] = defaultdict(list)
    for i, (s, _) in enumerate(keep):
        by_sha[s.sha1].append(i)
    drop: set[int] = set()
    for idxs in by_sha.values():
        labels = {keep[i][0].label for i in idxs}
        if len(labels) > 1:                       # same bytes labelled both ways -> unusable
            drop.update(idxs)
            issues["label_conflicts"].extend(keep[i][0].path for i in idxs)
        elif len(idxs) > 1:
            drop.update(idxs[1:])
            issues["exact_duplicates"].extend(keep[i][0].path for i in idxs[1:])
    keep = [k for i, k in enumerate(keep) if i not in drop]

    groups = _cluster([s.sha1 for s, _ in keep], [r["dhash"] for _, r in keep], cfg.near_dup_hamming)
    # a near-duplicate cluster containing both classes is a labelling conflict as well
    cls_in_group: dict[int, set[int]] = defaultdict(set)
    for (s, _), g in zip(keep, groups):
        cls_in_group[g].add(s.label)
    conflict_groups = {g for g, c in cls_in_group.items() if len(c) > 1}
    final: list[tuple[Sample, dict]] = []
    for (s, r), g in zip(keep, groups):
        if g in conflict_groups:
            issues["label_conflicts"].append(s.path)
        else:
            s.group = g
            final.append((s, r))
    out = [s for s, _ in final]
    audit = build_audit(final, issues, cfg)
    return ScanResult(out, audit)


def _shortcut_auc(final: list[tuple[Sample, dict]], cfg: Config) -> float | None:
    """How well can file-level cues alone (format, size, bytes/pixel) separate the classes?"""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold

    y = np.array([s.label for s, _ in final])
    if min(np.bincount(y, minlength=2)) < 10:
        return None
    fmts = sorted({r["fmt"] for _, r in final})
    X = np.array([[np.log(r["w"] * r["h"]), r["w"] / r["h"], np.log1p(r["bytes"] / (r["w"] * r["h"])),
                   float(r["w"] % 8 == 0), float(r["w"] % 64 == 0), float(r["h"] % 64 == 0), fmts.index(r["fmt"])]
                  for _, r in final])
    groups = np.array([s.group for s, _ in final])
    k = int(min(5, np.bincount(y).min()))
    if k < 2:
        return None
    oof = np.zeros(len(y))
    for tr, te in StratifiedGroupKFold(k, shuffle=True, random_state=cfg.seed).split(X, y, groups):
        if len(set(y[tr])) < 2:
            return None
        m = HistGradientBoostingClassifier(max_iter=60, random_state=cfg.seed, categorical_features=[6]).fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return float(roc_auc_score(y, oof))


def build_audit(final: list[tuple[Sample, dict]], issues: dict, cfg: Config) -> dict:
    y = np.array([s.label for s, _ in final]) if final else np.array([], dtype=int)
    audit: dict = {
        "n_usable": len(final),
        "per_class": {"real": int((y == 0).sum()), "ai": int((y == 1).sum())},
        "sources": {},
        "issues": {k: {"count": len(v), "examples": v[:5]} for k, v in issues.items()},
        "n_duplicate_clusters": len({s.group for s, _ in final if s.group >= 0}),
        "formats": {}, "size_stats": {}, "warnings": [],
    }
    src = Counter((("ai" if s.label else "real"), s.source) for s, _ in final)
    audit["sources"] = {f"{k[0]}/{k[1]}": v for k, v in sorted(src.items())}
    for name, lab in (("real", 0), ("ai", 1)):
        rs = [r for s, r in final if s.label == lab]
        if not rs:
            continue
        audit["formats"][name] = dict(Counter(r["fmt"] for r in rs))
        audit["size_stats"][name] = {
            "median_width": float(np.median([r["w"] for r in rs])),
            "median_height": float(np.median([r["h"] for r in rs])),
            "median_bytes_per_pixel": float(np.median([r["bytes"] / (r["w"] * r["h"]) for r in rs])),
        }
    w = audit["warnings"]
    n_real, n_ai = audit["per_class"]["real"], audit["per_class"]["ai"]
    if min(n_real, n_ai) < 10:
        w.append("fewer than 10 usable images in one class: training is not possible")
    elif min(n_real, n_ai) < 100:
        w.append("fewer than 100 images in one class: expect unstable metrics; more data helps a lot")
    if n_real and n_ai and max(n_real, n_ai) / min(n_real, n_ai) > 3:
        w.append("classes are imbalanced (>3:1); class weights are applied but balanced data is better")
    for k, v in issues.items():
        if v:
            w.append(f"{len(v)} files skipped: {k.replace('_', ' ')}")
    try:
        auc = _shortcut_auc(final, cfg) if final else None
    except Exception as e:  # noqa: BLE001 - the audit must never block training
        auc = None
        w.append(f"shortcut audit failed: {type(e).__name__}")
    audit["file_cue_auc"] = auc
    if auc is not None and auc > 0.85:
        w.append(f"file-level cues alone (format/size/bytes-per-pixel) separate your classes with AUC={auc:.2f}. "
                 "A model could learn those shortcuts instead of image content. Keep use_metadata_features off "
                 "(default), keep augment_copies>=1, and ideally re-encode both classes the same way.")
    return audit
