"""Prepare a fresh non-training holdout: OpenFake core validation AI + licensed Commons photos."""
from __future__ import annotations

import argparse
import hashlib
import html
import io
import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from PIL import Image, UnidentifiedImageError

if __package__:
    from .prepare_openfake import (
        _DATASET,
        _balanced_sample,
        _collect,
        _get_json,
        _write_split,
    )
else:
    from prepare_openfake import (
        _DATASET,
        _balanced_sample,
        _collect,
        _get_json,
        _write_split,
    )

_VALIDATION_ROWS = 58_453
_COMMONS_API = "https://commons.wikimedia.org/w/api.php"
_PHOTO_SEARCH_TERMS = (
    "landscape photograph",
    "wildlife photograph",
    "portrait photograph",
    "architecture photograph",
    "street photography",
    "nature photograph",
    "bird photography",
    "mountain photograph",
    "ocean photograph",
    "forest photograph",
    "flower photograph",
    "travel photograph",
    "sports photograph",
    "vehicle photograph",
    "people photograph",
    "rural photograph",
    "city photograph",
    "waterfall photograph",
    "desert photograph",
    "beach photograph",
    "animal photography",
    "night photography",
    "macro photography",
    "underwater photography",
)
_NON_PHOTO_CATEGORY_TERMS = (
    "ai-generated",
    "computer-generated",
    "digital art",
    "illustrations",
    "paintings",
    "synthetic images",
    "renders",
)


def _plain_text(value: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", value))).strip()


def _allowed_license(name: str) -> bool:
    normalized = name.casefold().replace("-", " ").strip()
    if normalized.startswith(("cc0", "public domain")):
        return True
    if not normalized.startswith(("cc by ", "cc by sa ")):
        return False
    return not re.search(r"\b(?:nc|nd)\b", normalized)


def _commons_candidates(requests: int, seed: int) -> list[dict]:
    candidates = []
    terms = _PHOTO_SEARCH_TERMS[:requests]
    for page_number, term in enumerate(terms, start=1):
        params: dict[str, str | int] = {
            "action": "query",
            "format": "json",
            "formatversion": 2,
            "generator": "search",
            "gsrsearch": f"filemime:image/jpeg {term}",
            "gsrnamespace": 6,
            "gsrlimit": 50,
            "prop": "imageinfo|categories",
            "cllimit": 50,
            "iiprop": "url|mime|size|extmetadata",
            "iiurlwidth": 1024,
        }
        payload = _get_json(_COMMONS_API, params)
        pages = payload.get("query", {}).get("pages", [])
        for page in pages:
            image_info = (page.get("imageinfo") or [{}])[0]
            metadata = image_info.get("extmetadata", {})
            license_name = _plain_text(metadata.get("LicenseShortName", {}).get("value", ""))
            if image_info.get("mime") != "image/jpeg" or not _allowed_license(license_name):
                continue
            if min(int(image_info.get("width", 0)), int(image_info.get("height", 0))) < 320:
                continue
            categories = [category["title"] for category in page.get("categories", [])]
            if any(
                term in category.casefold()
                for category in categories
                for term in _NON_PHOTO_CATEGORY_TERMS
            ):
                continue
            artist = _plain_text(metadata.get("Artist", {}).get("value", ""))
            license_url = _plain_text(metadata.get("LicenseUrl", {}).get("value", ""))
            page_url = image_info.get("descriptionurl", "")
            image_url = image_info.get("thumburl") or image_info.get("url")
            if not image_url or not page_url:
                continue
            row_id = f"commons-{page.get('pageid', page['title'])}"
            candidates.append({
                "row_id": row_id,
                "label": "real",
                "source": "wikimedia-commons-photographs",
                "source_dataset": "Wikimedia Commons Category:Photographs",
                "source_license_note": f"{license_name}; verify attribution and license terms at {page_url}",
                "license": license_name,
                "license_url": license_url,
                "artist": artist,
                "source_page": page_url,
                "title": page["title"],
                "categories": categories,
                "image": {"src": image_url},
            })
        print(
            f"Commons photo searches {page_number}/{len(terms)}: "
            f"{len(candidates)} usable JPEG photos",
            flush=True,
        )
        if page_number < len(terms):
            time.sleep(1.25)

    unique = {row["row_id"]: row for row in candidates}
    rows = list(unique.values())
    random.Random(seed).shuffle(rows)
    return rows


def _download_one_commons(row: dict, root: Path) -> tuple[dict, str]:
    source = "wikimedia-commons-photographs"
    path = root / "validation" / "real" / source / f"{row['row_id']}.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        try:
            with Image.open(path) as image:
                image.verify()
            row["download_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            return row, str(path.relative_to(root))
        except (OSError, UnidentifiedImageError):
            pass

    request = Request(
        row["image"]["src"],
        headers={"User-Agent": "aidetect-research/1.0 (AI image detection project)"},
    )
    for attempt in range(2):
        try:
            with urlopen(request, timeout=15) as response:
                data = response.read(20_000_001)
            if len(data) > 20_000_000:
                raise ValueError(f"thumbnail exceeds 20 MB: {row['title']}")
            with Image.open(io.BytesIO(data)) as image:
                image.verify()
            path.write_bytes(data)
            row["download_sha256"] = hashlib.sha256(data).hexdigest()
            return row, str(path.relative_to(root))
        except HTTPError as exc:
            if exc.code not in {429, 500, 502, 503, 504} or attempt == 1:
                raise
            time.sleep(1)
        except (URLError, TimeoutError):
            if attempt == 1:
                raise
            time.sleep(1)
    raise RuntimeError(f"failed to download Commons image: {row['title']}")


def _download_photos(
    rows: list[dict], root: Path, workers: int, target_count: int
) -> list[dict]:
    manifest = []
    start = 0
    while start < len(rows) and len(manifest) < target_count:
        batch_size = min(workers * 4, target_count - len(manifest))
        batch = rows[start:start + batch_size]
        start += len(batch)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_download_one_commons, row, root) for row in batch]
            for future in as_completed(futures):
                try:
                    row, relative_path = future.result()
                except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
                    print(f"Skipping inaccessible Commons image: {exc}", flush=True)
                    continue
                manifest.append({
                    "split": "validation",
                    "path": relative_path,
                    "label": "real",
                    "source": row["source"],
                    "row_id": row["row_id"],
                    "source_dataset": row["source_dataset"],
                    "source_license_note": row["source_license_note"],
                    "license": row["license"],
                    "license_url": row["license_url"],
                    "artist": row["artist"],
                    "source_page": row["source_page"],
                    "title": row["title"],
                    "categories": row["categories"],
                    "download_sha256": row["download_sha256"],
                })
        print(f"Commons photos downloaded/verified {len(manifest)}/{target_count}", flush=True)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("datasets/openfake_commons_holdout"))
    parser.add_argument("--per-class", type=int, default=200)
    parser.add_argument("--openfake-pages", type=int, default=10)
    parser.add_argument("--commons-pages", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--exclude-manifests",
        nargs="*",
        type=Path,
        default=[
            Path("datasets/openfake_v2_starter/manifest.jsonl"),
            Path("datasets/openfake_expanded/manifest.jsonl"),
            Path("datasets/openfake_external_holdout/manifest.jsonl"),
        ],
    )
    args = parser.parse_args()
    if min(args.per_class, args.openfake_pages, args.commons_pages, args.workers) < 1:
        parser.error("sample sizes, page counts, and workers must be positive")
    manifest_path = args.out / "manifest.jsonl"
    if manifest_path.exists():
        parser.error(f"refusing to overwrite existing holdout manifest: {manifest_path}")

    excluded_ids = set()
    for manifest in args.exclude_manifests:
        if manifest.is_file():
            with manifest.open(encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if row.get("source_dataset", _DATASET) == _DATASET:
                        excluded_ids.add(row["row_id"])

    print("Sampling OpenFake core/validation AI images not used in prior sets.", flush=True)
    openfake_rows = _collect(
        _DATASET, "core", "validation", _VALIDATION_ROWS, args.openfake_pages, args.seed
    )
    fake_sample = _balanced_sample(
        openfake_rows,
        args.per_class,
        args.seed,
        excluded_ids,
        source_cap=None,
        labels=("fake",),
    )
    for row in fake_sample:
        row["source"] = row["model"]
        row["source_dataset"] = _DATASET
        row["source_license_note"] = "CC-BY-NC-4.0; non-commercial use only"

    print("Finding CC0/CC-BY/CC-BY-SA/Public Domain Commons photographs.", flush=True)
    commons_candidates = _commons_candidates(args.commons_pages, args.seed + 1)
    if len(commons_candidates) < args.per_class:
        raise RuntimeError(
            f"found only {len(commons_candidates)} licensed Commons JPEG photos; "
            f"need {args.per_class}. Increase --commons-pages."
        )
    args.out.mkdir(parents=True, exist_ok=True)
    commons_manifest = _download_photos(
        commons_candidates, args.out, args.workers, args.per_class
    )
    if len(commons_manifest) < args.per_class:
        raise RuntimeError(
            f"downloaded only {len(commons_manifest)} licensed Commons photos; "
            f"need {args.per_class}. Rerun to resume or increase --commons-pages."
        )
    manifest = _write_split(fake_sample, "validation", args.out, args.workers)
    manifest.extend(commons_manifest)
    manifest.sort(key=lambda row: (row["label"], str(row["source"]), row["row_id"]))
    with manifest_path.open("w", encoding="utf-8", newline="") as stream:
        for row in manifest:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.out / "DATASET_INFO.json").write_text(json.dumps({
        "purpose": "Fresh one-time evaluation only; never use for training or threshold tuning.",
        "ai_source": "OpenFake core/validation, excluding previously recorded OpenFake row IDs",
        "ai_license": "CC-BY-NC-4.0; non-commercial use only",
        "real_source": "Wikimedia Commons Category:Photographs",
        "real_license_policy": "CC0, CC-BY, CC-BY-SA, or Public Domain JPEGs; individual attribution and page URLs are in manifest.jsonl",
        "warning": "Review each Commons source page and satisfy its attribution/share-alike terms. The overall model workflow remains non-commercial due to OpenFake.",
    }, indent=2), encoding="utf-8")
    print(f"Prepared {len(manifest)} holdout images at {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
