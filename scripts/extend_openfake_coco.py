"""Add balanced COCO real photos and extra OpenFake generators to an expanded training set."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from prepare_openfake import (
    _DATASET,
    _PAGE_SIZE,
    _TRAIN_ROWS,
    _balanced_sample,
    _collect,
    _write_split,
)

_COCO_DATASET = "rafaelpadilla/coco2017"
_COCO_TRAIN_ROWS = 118_287
_COCO_LICENSE_NOTE = (
    "COCO annotation license is CC-BY-4.0; image rights belong to original owners. "
    "Check each image's original source and COCO terms before redistribution or commercial use."
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("datasets/openfake_expanded"))
    parser.add_argument("--extra-per-class", type=int, default=600)
    parser.add_argument("--openfake-pages", type=int, default=30)
    parser.add_argument("--coco-pages", type=int, default=12)
    parser.add_argument("--seed", type=int, default=314)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if min(args.extra_per_class, args.openfake_pages, args.coco_pages, args.workers) < 1:
        parser.error("sample sizes, page counts, and workers must be positive")

    manifest_path = args.out / "manifest.jsonl"
    if not manifest_path.is_file():
        parser.error(f"manifest not found: {manifest_path}; prepare the initial OpenFake sample first")
    existing = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for row in existing:
        row.setdefault("source_dataset", _DATASET)
        row.setdefault("source_license_note", "CC-BY-NC-4.0; non-commercial use only")
    openfake_ids = {
        row["row_id"] for row in existing
        if row.get("source_dataset", _DATASET) == _DATASET
    }
    coco_ids = {
        row["row_id"] for row in existing
        if row.get("source_dataset") == _COCO_DATASET
    }

    print("Sampling additional OpenFake AI rows.", flush=True)
    openfake_rows = _collect(
        _DATASET, "core", "train", _TRAIN_ROWS, args.openfake_pages, args.seed
    )
    openfake_sample = _balanced_sample(
        openfake_rows,
        args.extra_per_class,
        args.seed,
        openfake_ids,
        max(10, math.ceil(args.extra_per_class / 20)),
        labels=("fake",),
    )
    for row in openfake_sample:
        row["source"] = row["model"]
        row["source_dataset"] = _DATASET
        row["source_license_note"] = "CC-BY-NC-4.0; non-commercial use only"

    print("Sampling COCO real photos.", flush=True)
    coco_rows = _collect(
        _COCO_DATASET,
        "default",
        "train",
        _COCO_TRAIN_ROWS,
        args.coco_pages,
        args.seed + 1,
    )
    coco_rows = [
        {
            "row_id": f"coco2017-{row['image_id']}",
            "label": "real",
            "source": "coco2017",
            "source_dataset": _COCO_DATASET,
            "source_license_note": _COCO_LICENSE_NOTE,
            "image": row["image"],
        }
        for row in coco_rows
    ]
    coco_sample = _balanced_sample(
        coco_rows,
        args.extra_per_class,
        args.seed + 1,
        coco_ids,
        source_cap=None,
        labels=("real",),
    )

    added = _write_split(openfake_sample, "train", args.out, args.workers)
    added.extend(_write_split(coco_sample, "train", args.out, args.workers))
    combined = existing + added
    combined.sort(key=lambda row: (
        row["split"], row["label"], str(row["source"]), row["row_id"]
    ))
    temporary_manifest = manifest_path.with_suffix(".jsonl.tmp")
    with temporary_manifest.open("w", encoding="utf-8", newline="") as stream:
        for row in combined:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary_manifest.replace(manifest_path)

    info_path = args.out / "DATASET_INFO.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.is_file() else {}
    info["additional_training_source"] = {
        "dataset": _COCO_DATASET,
        "split": "train",
        "images_added": len(coco_sample),
        "license_note": _COCO_LICENSE_NOTE,
        "citation": "Microsoft COCO: Common Objects in Context (Lin et al., ECCV 2014)",
    }
    info["warning"] = (
        "OpenFake is non-commercial only. COCO annotations use CC-BY-4.0, but image rights "
        "belong to original owners; verify image source terms before redistribution or commercial use. "
        "Do not use the Reddit test split for training."
    )
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(
        f"Added {len(openfake_sample)} AI images and {len(coco_sample)} real images. "
        f"Updated manifest: {manifest_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
