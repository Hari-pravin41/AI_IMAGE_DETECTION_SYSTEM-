"""Prepare a separate OpenFake-core/COCO holdout without adding it to training data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from extend_openfake_coco import _COCO_DATASET, _COCO_LICENSE_NOTE
from prepare_openfake import _DATASET, _balanced_sample, _collect, _write_split

_OPENFAKE_VALIDATION_ROWS = 58_453
_COCO_VALIDATION_ROWS = 5_000


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("datasets/openfake_external_holdout"))
    parser.add_argument("--per-class", type=int, default=200)
    parser.add_argument("--openfake-pages", type=int, default=6)
    parser.add_argument("--coco-pages", type=int, default=4)
    parser.add_argument("--seed", type=int, default=9001)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if min(args.per_class, args.openfake_pages, args.coco_pages, args.workers) < 1:
        parser.error("sample sizes, page counts, and workers must be positive")
    if (args.out / "manifest.jsonl").exists():
        parser.error(f"refusing to overwrite existing holdout manifest: {args.out / 'manifest.jsonl'}")

    print("Sampling held-out OpenFake core/validation images.", flush=True)
    openfake_rows = _collect(
        _DATASET,
        "core",
        "validation",
        _OPENFAKE_VALIDATION_ROWS,
        args.openfake_pages,
        args.seed,
    )
    fake_rows = _balanced_sample(
        openfake_rows,
        args.per_class,
        args.seed,
        set(),
        source_cap=None,
        labels=("fake",),
    )
    for row in fake_rows:
        row["source"] = row["model"]
        row["source_dataset"] = _DATASET
        row["source_license_note"] = "CC-BY-NC-4.0; non-commercial use only"

    print("Sampling held-out COCO validation photos.", flush=True)
    coco_rows = _collect(
        _COCO_DATASET,
        "default",
        "val",
        _COCO_VALIDATION_ROWS,
        args.coco_pages,
        args.seed + 1,
    )
    real_rows = [
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
    real_sample = _balanced_sample(
        real_rows,
        args.per_class,
        args.seed + 1,
        set(),
        source_cap=None,
        labels=("real",),
    )

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = _write_split(fake_rows, "validation", args.out, args.workers)
    manifest.extend(_write_split(real_sample, "validation", args.out, args.workers))
    manifest.sort(key=lambda row: (row["label"], str(row["source"]), row["row_id"]))
    with (args.out / "manifest.jsonl").open("w", encoding="utf-8", newline="") as stream:
        for row in manifest:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.out / "DATASET_INFO.json").write_text(json.dumps({
        "purpose": "One-time evaluation only; do not train or tune thresholds on this set.",
        "openfake_split": "core/validation",
        "coco_split": "validation",
        "license_note": _COCO_LICENSE_NOTE,
        "openfake_license": "CC-BY-NC-4.0; non-commercial use only",
        "coco_citation": "Microsoft COCO: Common Objects in Context (Lin et al., ECCV 2014)",
    }, indent=2), encoding="utf-8")
    print(f"Prepared {len(manifest)} holdout images at {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
