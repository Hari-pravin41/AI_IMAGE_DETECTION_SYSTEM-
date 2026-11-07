"""Prepare a small, resumable, licensed training supplement for photo authenticity."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import io
import json
import math
import os
import random
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from PIL import Image, ImageOps, UnidentifiedImageError

if __package__:
    from .prepare_commons_holdout import (
        _PHOTO_SEARCH_TERMS,
        _allowed_license,
        _commons_candidates,
    )
    from .prepare_openfake import (
        _DATASET,
        _TRAIN_ROWS,
        _balanced_sample,
        _collect,
        _safe_component,
    )
else:
    from prepare_commons_holdout import (
        _PHOTO_SEARCH_TERMS,
        _allowed_license,
        _commons_candidates,
    )
    from prepare_openfake import (
        _DATASET,
        _TRAIN_ROWS,
        _balanced_sample,
        _collect,
        _safe_component,
    )

_MAX_IMAGE_BYTES = 12 * 1024 * 1024
_MAX_PIXELS = 60_000_000
_MAX_DATASET_BYTES = 6 * 1024 * 1024 * 1024
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
_DEFAULT_EXCLUSIONS = (
    Path("datasets/openfake_v2_starter/manifest.jsonl"),
    Path("datasets/openfake_expanded/manifest.jsonl"),
    Path("datasets/openfake_external_holdout/manifest.jsonl"),
    Path("datasets/openfake_commons_holdout/manifest.jsonl"),
)


def _load_used_ids(paths: list[Path]) -> set[str]:
    used = set()
    for manifest in paths:
        if not manifest.is_file():
            continue
        with manifest.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                row_id = str(row.get("row_id", ""))
                source_dataset = str(row.get("source_dataset", ""))
                if (
                    source_dataset == _DATASET
                    or row_id.startswith("openfake-")
                    or row_id.startswith("commons-")
                ):
                    used.add(row_id)
    return used


def _dhash(data: bytes) -> int:
    with Image.open(io.BytesIO(data)) as image:
        image = ImageOps.exif_transpose(image).convert("L").resize(
            (9, 8), Image.Resampling.LANCZOS
        )
        pixels = image.tobytes()
    value = 0
    for y in range(8):
        for x in range(8):
            value = (value << 1) | int(pixels[y * 9 + x + 1] > pixels[y * 9 + x])
    return value


def _image_bytes(path: Path) -> bytes | None:
    try:
        with path.open("rb") as stream:
            data = stream.read(_MAX_IMAGE_BYTES + 1)
    except OSError as exc:
        print(f"Could not read existing image for duplicate check {path}: {exc}", flush=True)
        return None
    if len(data) > _MAX_IMAGE_BYTES:
        return None
    return data


def _existing_image_signatures(dataset_root: Path, output_root: Path) -> tuple[set[str], set[int]]:
    hashes: set[str] = set()
    dhashes: set[int] = set()
    output_resolved = output_root.resolve()
    for path in dataset_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in _IMAGE_EXTENSIONS:
            continue
        if output_resolved == path.resolve() or output_resolved in path.resolve().parents:
            continue
        data = _image_bytes(path)
        if data is None:
            continue
        hashes.add(hashlib.sha256(data).hexdigest())
        try:
            dhashes.add(_dhash(data))
        except (OSError, UnidentifiedImageError, ValueError):
            print(f"Skipping unreadable existing image during duplicate check: {path}", flush=True)
    return hashes, dhashes


def _download(row: dict) -> bytes:
    request = Request(
        row["image"]["src"],
        headers={"User-Agent": "aidetect-training-set/1.0 (research; resumable dataset sampler)"},
    )
    for attempt in range(4):
        try:
            with urlopen(request, timeout=30) as response:
                data = response.read(_MAX_IMAGE_BYTES + 1)
            if len(data) > _MAX_IMAGE_BYTES:
                raise ValueError(f"image exceeds {_MAX_IMAGE_BYTES // (1024 * 1024)} MB")
            if row.get("source") == "wikimedia-commons-photographs":
                time.sleep(1)
            return data
        except HTTPError as exc:
            if (
                exc.code == 429
                and row.get("source") == "wikimedia-commons-photographs"
            ):
                raise
            if exc.code not in {429, 500, 502, 503, 504} or attempt == 3:
                raise
            retry_after = exc.headers.get("Retry-After")
            delay = min(2 ** attempt, 15)
            if retry_after:
                try:
                    delay = float(retry_after)
                except ValueError:
                    try:
                        retry_at = parsedate_to_datetime(retry_after)
                        if retry_at.tzinfo is None:
                            retry_at = retry_at.replace(tzinfo=timezone.utc)
                        delay = max(0, (retry_at - datetime.now(timezone.utc)).total_seconds())
                    except (TypeError, ValueError, OverflowError):
                        pass
            time.sleep(delay)
        except (URLError, TimeoutError):
            if attempt == 3:
                raise
            time.sleep(min(2 ** attempt, 15))
    raise RuntimeError("unreachable download retry state")


def _validate_image(data: bytes) -> tuple[str, tuple[int, int]]:
    with Image.open(io.BytesIO(data)) as image:
        image.verify()
    with Image.open(io.BytesIO(data)) as image:
        if image.format not in {"JPEG", "PNG", "WEBP"}:
            raise ValueError(f"unsupported image format: {image.format}")
        if image.width * image.height > _MAX_PIXELS:
            raise ValueError("image exceeds the 60-megapixel limit")
        if min(image.size) < 128:
            raise ValueError("image shortest side is smaller than the 128-pixel model window")
        return str(image.format), image.size


def _record_for(row: dict, label: str) -> dict:
    if label == "ai":
        source = str(row.get("model", "unknown"))
        return {
            "row_id": str(row["row_id"]),
            "label": "ai",
            "source": source,
            "source_dataset": _DATASET,
            "source_split": "core/train",
            "source_license": "CC-BY-NC-4.0",
            "source_page": "https://huggingface.co/datasets/ComplexDataLab/OpenFake",
            "source_image_sha256": str(row.get("image_sha256", "")),
            "prompt_kind": str(row.get("prompt_kind", "")),
            "image_type": str(row.get("type", "")),
            "image": row["image"],
        }
    return {
        "row_id": str(row["row_id"]),
        "label": "real",
        "source": "wikimedia-commons-photographs",
        "source_dataset": "Wikimedia Commons Category:Photographs",
        "source_split": "licensed photograph search",
        "source_license": row["license"],
        "license_url": row["license_url"],
        "artist": row["artist"],
        "source_page": row["source_page"],
        "title": row["title"],
        "categories": row["categories"],
        "image": row["image"],
    }


def _record_for_openfake_real(row: dict) -> dict:
    return {
        "row_id": str(row["row_id"]),
        "label": "real",
        "source": str(row.get("model", "unknown")),
        "source_dataset": _DATASET,
        "source_split": "core/train",
        "source_license": "CC-BY-NC-4.0",
        "source_page": "https://huggingface.co/datasets/ComplexDataLab/OpenFake",
        "source_image_sha256": str(row.get("image_sha256", "")),
        "prompt_kind": str(row.get("prompt_kind", "")),
        "image_type": str(row.get("type", "")),
        "image": row["image"],
    }


def _relative_path(row: dict, root: Path, image_format: str) -> tuple[Path, str]:
    source = _safe_component(row["source"])
    row_id = _safe_component(row["row_id"])
    extension = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}[image_format]
    relative = Path("train") / row["label"] / source / f"{row_id}{extension}"
    return root / relative, str(PurePosixPath(*relative.parts))


def _append_progress(path: Path, event: dict) -> None:
    with path.open("a", encoding="utf-8", newline="") as stream:
        stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _read_progress(
    path: Path, root: Path
) -> tuple[dict[str, dict], set[str], set[str]]:
    accepted: dict[str, dict] = {}
    rejected: set[str] = set()
    blocked_sources: set[str] = set()
    if not path.is_file():
        return accepted, rejected, blocked_sources
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            event = json.loads(line)
            row_id = str(event["row_id"])
            if event["status"] == "source_blocked":
                blocked_sources.add(str(event["source"]))
            elif event["status"] == "duplicate":
                rejected.add(row_id)
            elif event["status"] == "accepted":
                image_path = root / event["path"]
                try:
                    data = image_path.read_bytes()
                except OSError:
                    continue
                if hashlib.sha256(data).hexdigest() == event["download_sha256"]:
                    accepted[row_id] = event
    return accepted, rejected, blocked_sources


def _candidate_plan(
    per_class: int, pages: int, commons_pages: int, seed: int, used_ids: set[str]
) -> list[dict]:
    rows = _collect(
        _DATASET, "core", "train", _TRAIN_ROWS, pages, seed
    )
    available_ai = [
        row for row in rows
        if row.get("label") == "fake" and str(row.get("row_id")) not in used_ids
    ]
    available_real = [
        row for row in rows
        if row.get("label") == "real" and str(row.get("row_id")) not in used_ids
    ]
    ai_candidates_count = per_class + max(100, math.ceil(per_class * 0.5))
    real_candidates_count = per_class + max(100, math.ceil(per_class * 0.5))
    ai_candidates = _balanced_sample(
        available_ai,
        ai_candidates_count,
        seed,
        set(),
        source_cap=max(10, math.ceil(ai_candidates_count / 20)),
        labels=("fake",),
    )
    openfake_real_candidates = _balanced_sample(
        available_real,
        real_candidates_count,
        seed + 2,
        set(),
        source_cap=None,
        labels=("real",),
    )

    commons_rows = _commons_candidates(commons_pages, seed + 1)
    commons_candidates = [
        row for row in commons_rows
        if str(row["row_id"]) not in used_ids
        and _allowed_license(str(row.get("license", "")))
        and row.get("source_page")
        and (
            not str(row.get("license", "")).casefold().startswith("cc by")
            or (row.get("artist") and row.get("license_url"))
        )
    ]
    if len(commons_candidates) < per_class:
        raise RuntimeError(
            f"only {len(commons_candidates)} unused licensed Commons photos found; "
            f"need {per_class}. Increase --commons-pages or --pages."
        )
    return (
        [_record_for(row, "ai") for row in ai_candidates]
        + [_record_for(row, "real") for row in commons_candidates]
        + [_record_for_openfake_real(row) for row in openfake_real_candidates]
    )


def _download_class(
    label: str,
    candidates: list[dict],
    target: int,
    root: Path,
    progress_path: Path,
    existing_hashes: set[str],
    existing_dhashes: set[int],
    accepted: dict[str, dict],
    rejected: set[str],
    blocked_sources: set[str],
    workers: int,
) -> None:
    already = sum(event["label"] == label for event in accepted.values())
    if already >= target:
        return
    pending = [
        row for row in candidates
        if row["label"] == label
        and row["row_id"] not in accepted
        and row["row_id"] not in rejected
        and row.get("source") not in blocked_sources
    ]
    while pending and already < target:
        is_commons = pending[0].get("source") == "wikimedia-commons-photographs"
        batch_workers = 1 if is_commons else workers
        batch_size = batch_workers if is_commons else workers * 4
        batch = pending[:batch_size]
        pending = pending[batch_size:]
        with ThreadPoolExecutor(max_workers=batch_workers) as pool:
            future_to_row = {pool.submit(_download, row): row for row in batch}
            for future in as_completed(future_to_row):
                if already >= target:
                    continue
                row = future_to_row[future]
                try:
                    data = future.result()
                    image_format, dimensions = _validate_image(data)
                except HTTPError as exc:
                    if (
                        exc.code == 429
                        and row.get("source") == "wikimedia-commons-photographs"
                    ):
                        _append_progress(progress_path, {
                            "row_id": row["row_id"],
                            "label": label,
                            "status": "source_blocked",
                            "source": row["source"],
                        })
                        blocked_sources.add(row["source"])
                        pending = [
                            candidate for candidate in pending
                            if candidate.get("source") != row["source"]
                        ]
                        print(
                            "Wikimedia Commons rate-limited this run; "
                            "continuing with other licensed training candidates.",
                            flush=True,
                        )
                    else:
                        print(f"Skipping {row['row_id']}: {exc}", flush=True)
                    continue
                except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
                    print(f"Skipping {row['row_id']}: {exc}", flush=True)
                    continue

                digest = hashlib.sha256(data).hexdigest()
                source_digest = row.get("source_image_sha256", "")
                image_hash = _dhash(data)
                duplicate = (
                    digest in existing_hashes
                    or bool(source_digest and source_digest in existing_hashes)
                    or any((image_hash ^ old_hash).bit_count() <= 4 for old_hash in existing_dhashes)
                )
                if duplicate:
                    _append_progress(progress_path, {
                        "row_id": row["row_id"],
                        "label": label,
                        "status": "duplicate",
                    })
                    rejected.add(row["row_id"])
                    continue
                used_bytes = sum(event.get("bytes", 0) for event in accepted.values())
                if used_bytes + len(data) > _MAX_DATASET_BYTES:
                    raise RuntimeError(
                        f"dataset reached the {_MAX_DATASET_BYTES // (1024 ** 3)} GB disk budget"
                    )

                output_path, relative_path = _relative_path(row, root, image_format)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = output_path.with_suffix(output_path.suffix + ".part")
                temporary.write_bytes(data)
                temporary.replace(output_path)
                event = {
                    "row_id": row["row_id"],
                    "label": label,
                    "status": "accepted",
                    "path": relative_path,
                    "download_sha256": digest,
                    "source_image_sha256": source_digest,
                    "format": image_format,
                    "width": dimensions[0],
                    "height": dimensions[1],
                    "bytes": len(data),
                }
                _append_progress(progress_path, event)
                accepted[row["row_id"]] = event
                existing_hashes.add(digest)
                existing_dhashes.add(image_hash)
                if source_digest:
                    existing_hashes.add(source_digest)
                already += 1
                if already % 25 == 0 or already == target:
                    print(f"{label}: verified and saved {already}/{target}", flush=True)
    if already < target:
        raise RuntimeError(
            f"only {already}/{target} unique, valid {label} images saved. "
            "The download plan and progress log are preserved; rerun the same command to resume."
        )


def _finish(root: Path, plan: dict, accepted: dict[str, dict], used_ids: set[str]) -> dict:
    selected = [
        event for event in accepted.values()
        if event["label"] in {"ai", "real"}
    ]
    selected.sort(key=lambda row: (row["label"], row["row_id"]))
    manifests: list[dict] = []
    plan_by_id = {row["row_id"]: row for row in plan["candidates"]}
    for event in selected:
        source = plan_by_id[event["row_id"]]
        row = {key: value for key, value in source.items() if key != "image"}
        row.update({key: value for key, value in event.items() if key not in {"status"}})
        row["split"] = "train"
        row["download_sha256"] = event["download_sha256"]
        manifests.append(row)

    manifest_path = root / "manifest.jsonl"
    temporary_manifest = manifest_path.with_suffix(".jsonl.tmp")
    with temporary_manifest.open("w", encoding="utf-8", newline="") as stream:
        for row in manifests:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary_manifest.replace(manifest_path)

    counts = {
        "ai": sum(row["label"] == "ai" for row in manifests),
        "real": sum(row["label"] == "real" for row in manifests),
    }
    summary = {
        "dataset_name": "OpenFake core-train + licensed Wikimedia Commons photographic training supplement",
        "purpose": "Training-only, balanced supplemental images for AI-generated versus real photographic image classification.",
        "source_records": {
            "ai": {
                "dataset": _DATASET,
                "split": "core/train only",
                "license": "CC-BY-NC-4.0",
                "source_url": "https://huggingface.co/datasets/ComplexDataLab/OpenFake",
                "revision": plan["openfake_revision"],
            },
            "real": {
                "datasets": [
                    {
                        "dataset": "Wikimedia Commons Category:Photographs",
                        "split": "licensed JPEG photographs",
                        "allowed_licenses": ["CC0", "CC-BY", "CC-BY-SA", "Public Domain"],
                        "source_url": "https://commons.wikimedia.org/wiki/Commons:Licensing",
                        "reuse_guidance": "https://commons.wikimedia.org/wiki/Commons:Reusing_content_outside_Wikimedia",
                    },
                    {
                        "dataset": _DATASET,
                        "split": "core/train",
                        "license": "CC-BY-NC-4.0",
                        "source_url": "https://huggingface.co/datasets/ComplexDataLab/OpenFake",
                    },
                ],
            },
        },
        "license_note": (
            "OpenFake images in both classes are CC-BY-NC-4.0; this combined training supplement "
            "is non-commercial. Wikimedia image-specific license, artist, and source page are in manifest.jsonl. "
            "Observe each image's attribution/share-alike terms."
        ),
        "class_counts": counts,
        "total_images": len(manifests),
        "split_counts": {"train": len(manifests), "validation": 0, "test": 0},
        "unique_generator_sources": len({
            row["source"] for row in manifests if row["label"] == "ai"
        }),
        "generator_source_counts": {
            source: sum(row["source"] == source for row in manifests if row["label"] == "ai")
            for source in sorted({row["source"] for row in manifests if row["label"] == "ai"})
        },
        "real_license_counts": {
            license_name: sum(
                row.get("source_license") == license_name
                for row in manifests if row["label"] == "real"
            )
            for license_name in sorted({
                row.get("source_license", "") for row in manifests if row["label"] == "real"
            })
        },
        "real_dataset_counts": {
            source: sum(
                row.get("source_dataset") == source
                for row in manifests if row["label"] == "real"
            )
            for source in sorted({
                row.get("source_dataset", "") for row in manifests if row["label"] == "real"
            })
        },
        "duplicate_policy": (
            "Excluded IDs from existing starter/training/holdout manifests; excluded SHA-256 exact "
            "matches and perceptual difference-hash matches within Hamming distance 4 against "
            "existing dataset images and previously accepted supplement images."
        ),
        "excluded_existing_row_ids_checked": len(used_ids),
        "image_integrity": "Each downloaded image was decoded, dimension-checked, and SHA-256 hashed.",
        "not_used_for": [
            "validation",
            "threshold tuning",
            "holdout evaluation",
            "retraining without the user's explicit approval",
        ],
        "resource_limit": {
            "max_image_bytes": _MAX_IMAGE_BYTES,
            "max_pixels": _MAX_PIXELS,
            "max_dataset_bytes": _MAX_DATASET_BYTES,
            "default_images_per_class": plan["target_per_class"],
            "disk_free_gb_before_download": plan["disk_free_gb_before_download"],
        },
    }
    (root / "DATASET_INFO.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = [
        "# Photographic authenticity training supplement",
        "",
        f"Prepared **{len(manifests)} verified images**: {counts['real']} real photographs and {counts['ai']} AI-generated images.",
        "",
        "## Sources and licenses",
        "",
        "- AI images: OpenFake `core/train` only, sampled across generator labels. License: CC-BY-NC-4.0; the combined dataset is non-commercial.",
        "- Real photographs: unused OpenFake `core/train` real images plus Wikimedia Commons photos. Commons items use only CC0, CC-BY, CC-BY-SA, or Public Domain licenses; per-image source, license, artist, and source-page details are in `manifest.jsonl`.",
        "",
        "## Verification",
        "",
        f"- Training-only split; validation/test counts are both zero.",
        f"- AI generator labels represented: {summary['unique_generator_sources']}.",
        "- Existing manifest row IDs were excluded; SHA-256 and perceptual hashes were checked against existing dataset images and accepted supplement images.",
        "- Every included image decoded successfully, met the image-size limit, and has a recorded download SHA-256.",
        "- `manifest.jsonl` is the source of truth for labels, source IDs, and attribution.",
        "- Dataset use does not authorize model retraining; retraining remains pending user approval.",
        "",
        "## Limitations",
        "",
        "- Wikimedia search and category metadata do not guarantee perfect photographic content; review a sample manually before training.",
        "- The model's existing held-out datasets were not downloaded into this dataset or used as examples.",
        "- Verify attribution and share-alike obligations before redistribution; OpenFake's non-commercial terms apply to this combined supplement.",
    ]
    (root / "DATASET_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("datasets/openfake_photo_training_v1"))
    parser.add_argument("--per-class", type=int, default=400)
    parser.add_argument("--pages", type=int, default=35)
    parser.add_argument("--commons-pages", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--exclude-manifests", nargs="*", type=Path, default=list(_DEFAULT_EXCLUSIONS))
    args = parser.parse_args()
    if min(args.per_class, args.pages, args.commons_pages, args.workers) < 1:
        parser.error("sample sizes, pages, and workers must be positive")
    if args.commons_pages > len(_PHOTO_SEARCH_TERMS):
        parser.error(f"--commons-pages cannot exceed {len(_PHOTO_SEARCH_TERMS)} search terms")
    if args.workers > 4:
        parser.error("--workers must not exceed 4 to limit load on public data services")
    if (args.out / "manifest.jsonl").exists():
        parser.error(f"refusing to overwrite existing dataset manifest: {args.out / 'manifest.jsonl'}")

    args.out.mkdir(parents=True, exist_ok=True)
    plan_path = args.out / "download_plan.json"
    progress_path = args.out / "download_progress.jsonl"
    exclusion_paths = sorted({
        *args.exclude_manifests,
        *Path("datasets").rglob("manifest.jsonl"),
    })
    used_ids = _load_used_ids(exclusion_paths)
    if plan_path.exists():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if plan["target_per_class"] != args.per_class or plan["seed"] != args.seed:
            parser.error("resume parameters differ from saved plan; rerun with its original --per-class and --seed")
    else:
        existing_files = list(args.out.iterdir())
        if existing_files:
            parser.error(f"output directory contains unrecognized files; refusing to use it: {args.out}")
        print("Preparing AI candidates only from OpenFake core/train; existing IDs and holdouts are excluded.", flush=True)
        candidates = _candidate_plan(
            args.per_class, args.pages, args.commons_pages, args.seed, used_ids
        )
        info_url = "https://huggingface.co/api/datasets/ComplexDataLab/OpenFake"
        request = Request(info_url, headers={"User-Agent": "aidetect-training-set/1.0"})
        with urlopen(request, timeout=30) as response:
            repository = json.load(response)
        drive = args.out.resolve().anchor
        free_bytes = shutil.disk_usage(drive).free
        plan = {
            "target_per_class": args.per_class,
            "seed": args.seed,
            "plan_version": 2,
            "openfake_revision": repository["sha"],
            "disk_free_gb_before_download": round(free_bytes / (1024 ** 3), 2),
            "excluded_manifests": [str(path) for path in exclusion_paths if path.is_file()],
            "candidates": candidates,
        }
        plan_path.write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")
    if plan["disk_free_gb_before_download"] * (1024 ** 3) < _MAX_DATASET_BYTES + 1024 ** 3:
        raise RuntimeError(
            "less than 7 GB free at plan time; refusing to reserve space for this dataset"
        )

    if plan.get("plan_version") != 2:
        print("Adding unused OpenFake core/train real-photo fallback candidates to the saved plan.", flush=True)
        sampled_rows = _collect(
            _DATASET, "core", "train", _TRAIN_ROWS, args.pages, args.seed
        )
        already_planned = {row["row_id"] for row in plan["candidates"]}
        real_rows = [
            row for row in sampled_rows
            if row.get("label") == "real"
            and str(row.get("row_id")) not in used_ids
            and str(row.get("row_id")) not in already_planned
        ]
        real_candidates_count = args.per_class + max(100, math.ceil(args.per_class * 0.5))
        real_sample = _balanced_sample(
            real_rows,
            real_candidates_count,
            args.seed + 2,
            set(),
            source_cap=None,
            labels=("real",),
        )
        openfake_real_candidates = [
            _record_for_openfake_real(row) for row in real_sample
        ]
        ai_candidates = [row for row in plan["candidates"] if row["label"] == "ai"]
        commons_candidates = [
            row for row in plan["candidates"]
            if row["label"] == "real" and row["source"] == "wikimedia-commons-photographs"
        ]
        plan["candidates"] = ai_candidates + openfake_real_candidates + commons_candidates
        plan["plan_version"] = 2
        plan_path.write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")
    accepted, rejected, blocked_sources = _read_progress(progress_path, args.out)
    for temporary in args.out.rglob("*.part"):
        temporary.unlink()
    print(f"Resuming dataset preparation: {len(accepted)} verified images already present.", flush=True)
    existing_hashes, existing_dhashes = _existing_image_signatures(
        args.out.parent, args.out
    )
    for event in accepted.values():
        data = (args.out / event["path"]).read_bytes()
        existing_hashes.add(event["download_sha256"])
        existing_dhashes.add(_dhash(data))

    for label in ("ai", "real"):
        _download_class(
            label,
            plan["candidates"],
            args.per_class,
            args.out,
            progress_path,
            existing_hashes,
            existing_dhashes,
            accepted,
            rejected,
            blocked_sources,
            args.workers,
        )
    summary = _finish(args.out, plan, accepted, used_ids)
    print(
        f"Prepared {summary['total_images']} training-only images "
        f"(real={summary['class_counts']['real']}, ai={summary['class_counts']['ai']}) "
        f"at {args.out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
