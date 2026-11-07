"""Download a balanced, source-diverse OpenFake sample without streaming the full dataset."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from PIL import Image, UnidentifiedImageError

_BASE = "https://datasets-server.huggingface.co"
_DATASET = "ComplexDataLab/OpenFake"
_TRAIN_ROWS = 2_300_034
_REDDIT_ROWS = 36_087
_PAGE_SIZE = 100


def _get_json(url: str, params: dict[str, str | int]) -> dict:
    request_url = f"{url}?{urlencode(params)}"
    request = Request(request_url, headers={"User-Agent": "aidetect-openfake-sampler/1.0"})
    for attempt in range(6):
        try:
            with urlopen(request, timeout=90) as response:
                return json.load(response)
        except HTTPError as exc:
            if exc.code not in {429, 500, 502, 503, 504} or attempt == 5:
                raise
            retry_after = exc.headers.get("Retry-After")
            delay = float(retry_after) if retry_after and retry_after.isdigit() else min(2 ** attempt, 60)
            time.sleep(delay)
        except (URLError, TimeoutError):
            if attempt == 5:
                raise
            time.sleep(min(2 ** attempt, 60))
    raise RuntimeError("unreachable retry state")


def _fetch_page(dataset: str, config: str, split: str, offset: int) -> list[dict]:
    result = _get_json(f"{_BASE}/rows", {
        "dataset": dataset,
        "config": config,
        "split": split,
        "offset": offset,
        "length": _PAGE_SIZE,
    })
    return [entry["row"] for entry in result["rows"]]


def _collect(dataset: str, config: str, split: str, total_rows: int,
             requests: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    max_offset = max(total_rows - _PAGE_SIZE, 0)
    offsets = sorted({rng.randrange(max_offset + 1) for _ in range(requests)})
    rows = []
    for page, offset in enumerate(offsets, start=1):
        rows.extend(_fetch_page(dataset, config, split, offset))
        print(f"{split}: sampled metadata page {page}/{len(offsets)}", flush=True)
        if page < len(offsets):
            time.sleep(2)
    unique = {}
    for row in rows:
        row_id = row.get("row_id", row.get("image_id", row.get("key")))
        if row_id is None:
            raise ValueError(f"dataset row has no stable ID; available keys: {sorted(row)}")
        unique[row_id] = row
    return list(unique.values())


def _balanced_sample(rows: list[dict], per_class: int, seed: int,
                     excluded_ids: set[str], source_cap: int | None,
                     labels: tuple[str, ...] = ("real", "fake")) -> list[dict]:
    rng = random.Random(seed)
    selected = []
    for label in labels:
        candidates = [row for row in rows if row["label"] == label and row["row_id"] not in excluded_ids]
        groups: dict[str, list[dict]] = {}
        for row in candidates:
            source = str(row.get("source", row.get("model", "unknown")))
            groups.setdefault(source, []).append(row)
        for group in groups.values():
            rng.shuffle(group)
        sources = list(groups)
        rng.shuffle(sources)
        if source_cap is not None and label == "fake":
            for source in sources:
                groups[source] = groups[source][:source_cap]
        result = []
        while len(result) < per_class:
            progressed = False
            for source in sources:
                if groups[source]:
                    result.append(groups[source].pop())
                    progressed = True
                    if len(result) == per_class:
                        break
            if not progressed:
                counts = {
                    source: sum(
                        1 for row in candidates
                        if str(row.get("source", row.get("model", "unknown"))) == source
                    )
                    for source in sources
                }
                raise RuntimeError(
                    f"only {len(result)} usable {label} rows found for requested {per_class}; "
                    f"source counts={counts}"
                )
        selected.extend(result)
    rng.shuffle(selected)
    return selected


def _safe_component(value: str | None) -> str:
    component = re.sub(r"[^A-Za-z0-9._-]+", "_", value or "unknown").strip("._")
    return (component or "unknown")[:100]


def _download_one(row: dict, split: str, root: Path) -> tuple[dict, str]:
    label = str(row["label"])
    source = _safe_component(row.get("source", row.get("model")))
    relpath = Path(split) / label / source / f"{row['row_id']}.jpg"
    path = root / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        try:
            with Image.open(path) as image:
                image.verify()
            row["download_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            return row, str(relpath)
        except (OSError, UnidentifiedImageError):
            pass

    request = Request(row["image"]["src"], headers={"User-Agent": "aidetect-openfake-sampler/1.0"})
    for attempt in range(6):
        try:
            with urlopen(request, timeout=90) as response:
                data = response.read()
            actual_hash = hashlib.sha256(data).hexdigest()
            with Image.open(io.BytesIO(data)) as image:
                image.verify()
            path.write_bytes(data)
            row["download_sha256"] = actual_hash
            return row, str(relpath)
        except HTTPError as exc:
            if exc.code not in {429, 500, 502, 503, 504} or attempt == 5:
                raise
            retry_after = exc.headers.get("Retry-After")
            delay = float(retry_after) if retry_after and retry_after.isdigit() else min(2 ** attempt, 60)
            time.sleep(delay)
        except (URLError, TimeoutError):
            if attempt == 5:
                raise
            time.sleep(min(2 ** attempt, 60))
    raise RuntimeError(f"download failed for {row['row_id']}")


def _write_split(rows: list[dict], split: str, root: Path, workers: int) -> list[dict]:
    manifest = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_download_one, row, split, root) for row in rows]
        for completed, future in enumerate(as_completed(futures), start=1):
            row, relative_path = future.result()
            manifest.append({
                "split": split,
                "path": relative_path,
                "label": row["label"],
                "source": row.get("source", row.get("model", "unknown")),
                "row_id": row["row_id"],
                "source_dataset": row.get("source_dataset", _DATASET),
                "source_license_note": row.get("source_license_note", "CC-BY-NC-4.0; non-commercial use only"),
                "source_image_sha256": row.get("image_sha256", ""),
                "download_sha256": row["download_sha256"],
                "prompt_kind": row.get("prompt_kind", ""),
                "type": row.get("type", ""),
            })
            if completed % 50 == 0 or completed == len(futures):
                print(f"{split}: downloaded/verified {completed}/{len(futures)}", flush=True)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("datasets/openfake_expanded"))
    parser.add_argument("--train-per-class", type=int, default=600)
    parser.add_argument("--reddit-per-class", type=int, default=200)
    parser.add_argument("--train-pages", type=int, default=30,
                        help="random 100-row pages sampled from the full training split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--exclude-manifest", type=Path,
                        default=Path("datasets/openfake_v2_starter/manifest.jsonl"))
    args = parser.parse_args()
    if min(args.train_per_class, args.reddit_per_class, args.train_pages, args.workers) < 1:
        parser.error("sample sizes, train pages, and workers must be positive")

    excluded_ids = set()
    if args.exclude_manifest.is_file():
        with args.exclude_manifest.open(encoding="utf-8") as stream:
            excluded_ids = {json.loads(line)["row_id"] for line in stream if line.strip()}

    print("Sampling random pages from the full OpenFake core training split.", flush=True)
    train_rows = _collect(_DATASET, "core", "train", _TRAIN_ROWS, args.train_pages, args.seed)
    train_sample = _balanced_sample(train_rows, args.train_per_class, args.seed, excluded_ids,
                                    max(10, math.ceil(args.train_per_class / 20)))
    print(f"Selected {args.train_per_class} train images per class from {len(train_rows)} rows.", flush=True)

    reddit_rows = _collect(_DATASET, "reddit", "test", _REDDIT_ROWS,
                           max(12, math.ceil(args.reddit_per_class * 2 / _PAGE_SIZE)), args.seed + 1)
    reddit_sample = _balanced_sample(reddit_rows, args.reddit_per_class, args.seed + 1, set(), None)
    print(f"Selected {args.reddit_per_class} unseen Reddit images per class.", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = _write_split(train_sample, "train", args.out, args.workers)
    manifest.extend(_write_split(reddit_sample, "reddit_test", args.out, args.workers))
    manifest.sort(key=lambda row: (
        row["split"], row["label"], str(row["source"] or ""), row["row_id"]
    ))
    with (args.out / "manifest.jsonl").open("w", encoding="utf-8", newline="") as stream:
        for row in manifest:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.out / "DATASET_INFO.json").write_text(json.dumps({
        "dataset": _DATASET,
        "license": "CC-BY-NC-4.0",
        "source": "Hugging Face datasets-server rows API",
        "sampling": "random pages; balanced by class; capped across AI generator sources",
        "excluded_row_ids_from_starter_manifest": len(excluded_ids),
        "warning": "Non-commercial license. Do not use the Reddit split for training.",
    }, indent=2), encoding="utf-8")
    print(f"Saved balanced dataset and manifest to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
