"""Small shared helpers: logging, progress, JSON encoding."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np


def log(msg: str = "") -> None:
    print(msg, file=sys.stderr, flush=True)


class Progress:
    """Minimal single-line progress indicator (no external dependency)."""

    def __init__(self, total: int, desc: str, enabled: bool = True):
        self.total, self.desc, self.n = max(int(total), 1), desc, 0
        self.enabled = enabled and sys.stderr.isatty()
        self.t0 = time.time()
        self._last = 0.0
        if not self.enabled:
            log(f"{desc}: {total} items")

    def update(self, k: int = 1) -> None:
        self.n += k
        now = time.time()
        if self.enabled and (now - self._last > 0.2 or self.n >= self.total):
            self._last = now
            rate = self.n / max(now - self.t0, 1e-9)
            eta = (self.total - self.n) / max(rate, 1e-9)
            sys.stderr.write(
                f"\r{self.desc}: {self.n}/{self.total} ({100 * self.n / self.total:5.1f}%) "
                f"{rate:6.1f}/s ETA {eta:5.0f}s   "
            )
            sys.stderr.flush()

    def close(self) -> None:
        if self.enabled:
            sys.stderr.write("\n")
            sys.stderr.flush()


def resolve_jobs(n_jobs: int | None) -> int:
    """joblib-style job count: -1 -> all CPUs, -2 -> all but one, 0/None -> 1."""
    cpu = os.cpu_count() or 1
    if not n_jobs:
        return 1
    if n_jobs < 0:
        return max(cpu + 1 + n_jobs, 1)
    return int(n_jobs)


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):  # noqa: D401
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            f = float(o)
            return f if np.isfinite(f) else None
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def dump_json(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_clean(obj), fh, indent=2, cls=NumpyEncoder, ensure_ascii=False)


def _clean(o):
    if isinstance(o, float) and not np.isfinite(o):
        return None
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.floating):
        return _clean(float(o))
    return o
