import csv, hashlib, json, sys
from collections import Counter
from pathlib import Path
from PIL import Image, ImageOps

def resolve_path(value, manifest):
    p = Path(value.strip())
    if p.is_absolute() and p.is_file():
        return p.resolve()
    candidates = [Path.cwd() / p, Path(manifest).resolve().parent / p]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(value)

def convert_manifest(manifest, split, root):
    image_dir = root / split / "images"
    image_dir.mkdir(parents=True)
    output_manifest = root / f"{split}_manifest.csv"
    rows = []

    with open(manifest, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        columns = {c.strip().lower(): c for c in (reader.fieldnames or [])}
        if "path" not in columns or "label" not in columns:
            raise ValueError(f"{manifest} must contain path and label columns")

        for number, row in enumerate(reader, 1):
            original = resolve_path(row[columns["path"]], manifest)
            label = row[columns["label"]].strip()
            source = row.get(columns.get("source", ""), "") if "source" in columns else ""
            target = image_dir / f"{number:06d}.jpg"

            with Image.open(original) as im:
                im = ImageOps.exif_transpose(im)
                if im.mode == "P" and "transparency" in im.info:
                    im = im.convert("RGBA")
                if "A" in im.getbands():
                    rgba = im.convert("RGBA")
                    background = Image.new("RGBA", rgba.size, (127, 127, 127, 255))
                    im = Image.alpha_composite(background, rgba).convert("RGB")
                else:
                    im = im.convert("RGB")

                im.thumbnail((512, 512), Image.Resampling.LANCZOS)
                canvas = Image.new("RGB", (512, 512), (127, 127, 127))
                x = (512 - im.width) // 2
                y = (512 - im.height) // 2
                canvas.paste(im, (x, y))
                canvas.save(
                    target, format="JPEG", quality=92, subsampling=0,
                    optimize=False, progressive=False
                )

            rows.append({
                "path": str(target.resolve()),
                "label": label,
                "source": source
            })

            if number % 250 == 0:
                print(f"{split}: processed {number} images")

    with open(output_manifest, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "source"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"{split}: {len(rows)} images -> {output_manifest}")
    return rows

def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    if len(sys.argv) != 4:
        raise SystemExit("Usage: script.py TRAIN.csv TEST.csv OUTPUT_FOLDER")

    train_file, test_file, output = sys.argv[1:]
    root = Path(output).resolve()
    if root.exists():
        raise FileExistsError(f"Output already exists: {root}")

    train = convert_manifest(train_file, "train", root)
    test = convert_manifest(test_file, "test", root)

    train_hashes = {digest(r["path"]) for r in train}
    test_hashes = {digest(r["path"]) for r in test}
    overlap = train_hashes & test_hashes

    audit = {
        "train_images": len(train),
        "test_images": len(test),
        "train_labels": dict(Counter(r["label"] for r in train)),
        "test_labels": dict(Counter(r["label"] for r in test)),
        "exact_canonical_image_overlap": len(overlap),
        "image_size": "512x512",
        "format": "JPEG",
        "quality": 92
    }
    (root / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))

    if overlap:
        raise SystemExit("STOP: identical canonical images occur in both splits.")
    if not train or not test:
        raise SystemExit("STOP: training or test split is empty.")

if __name__ == "__main__":
    main()
