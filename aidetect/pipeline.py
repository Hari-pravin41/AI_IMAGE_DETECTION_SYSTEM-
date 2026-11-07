"""Parallel, cached feature extraction over many files and image variants."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from joblib import Parallel, delayed

from .config import Config
from .features import FEATURE_VERSION, extract
from .loader import ImageError, load_image
from .perturb import apply_ops, sample_ops
from .utils import Progress, resolve_jobs


@dataclass
class Task:
    idx: int                      # index of the sample in the caller's list
    path: str
    variant: str                  # "clean" | "aug<k>" | "pert:<name>"
    ops: list | None = None       # explicit perturbation chain (for "pert:*")


@dataclass
class Result:
    idx: int
    variant: str
    feats: np.ndarray | None
    positions: list | None
    error: str | None = None


def _cache_dir(cfg: Config) -> Path:
    return Path(cfg.cache_dir) if cfg.cache_dir else Path.home() / ".cache" / "aidetect"


def _key(cfg: Config, task: Task, st: os.stat_result) -> str:
    extra = cfg.augment_signature() if task.variant.startswith("aug") else ""
    blob = "|".join([str(FEATURE_VERSION), os.path.abspath(task.path), str(st.st_size), str(st.st_mtime_ns),
                     task.variant, json.dumps(task.ops, sort_keys=True), str(cfg.crops_per_image),
                     cfg.feature_signature(), extra, str(cfg.max_pixels), str(cfg.effective_min_side)])
    return hashlib.sha1(blob.encode()).hexdigest()


def _rng_for(task: Task, cfg: Config) -> np.random.Generator:
    seed = int.from_bytes(hashlib.sha1(f"{task.path}|{task.variant}|{cfg.seed}".encode()).digest()[:8], "big")
    return np.random.default_rng(seed)


def _run_task(task: Task, cfg: Config) -> Result:
    cv2.setNumThreads(1)
    cache_file = None
    try:
        if cfg.use_cache:
            st = os.stat(task.path)
            key = _key(cfg, task, st)
            cache_file = _cache_dir(cfg) / key[:2] / f"{key}.npz"
            if cache_file.is_file():
                try:
                    with np.load(cache_file) as z:
                        return Result(task.idx, task.variant, z["feats"], [tuple(p) for p in z["pos"].tolist()])
                except Exception:  # noqa: BLE001 - corrupt cache entry: recompute
                    pass
        img = load_image(task.path, cfg)
        if task.variant != "clean":
            rng = _rng_for(task, cfg)
            ops = task.ops if task.ops is not None else sample_ops(cfg.augment, rng)
            data = apply_ops(img.rgb, ops, rng, cfg.effective_min_side)
            img = load_image(data, cfg, name=img.name)
        ex = extract(img, cfg)
        if cache_file is not None:
            try:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                tmp = cache_file.with_name(f"{cache_file.stem}.{os.getpid()}.tmp.npz")
                np.savez(tmp, feats=ex.feats, pos=np.array(ex.positions, dtype=np.int32))
                os.replace(tmp, cache_file)
            except OSError:
                pass  # read-only cache dir is fine
        return Result(task.idx, task.variant, ex.feats, ex.positions)
    except ImageError as e:
        return Result(task.idx, task.variant, None, None, str(e))
    except Exception as e:  # noqa: BLE001 - one bad file must not kill a long run
        return Result(task.idx, task.variant, None, None, f"{type(e).__name__}: {str(e)[:120]}")


def run_tasks(tasks: list[Task], cfg: Config, desc: str = "extracting features",
              show_progress: bool = True) -> list[Result]:
    n_jobs = resolve_jobs(cfg.n_jobs)
    prog = Progress(len(tasks), desc, show_progress)
    out: list[Result] = []
    if n_jobs == 1 or len(tasks) < 4:
        for t in tasks:
            out.append(_run_task(t, cfg))
            prog.update()
    else:
        gen = Parallel(n_jobs=n_jobs, return_as="generator", batch_size=4)(delayed(_run_task)(t, cfg) for t in tasks)
        for r in gen:
            out.append(r)
            prog.update()
    prog.close()
    return out
