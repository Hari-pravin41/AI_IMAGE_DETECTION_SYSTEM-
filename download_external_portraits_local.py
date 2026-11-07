import csv
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen
from io import BytesIO

from PIL import Image

ROOT = Path.cwd()
DATASETS = ROOT / "datasets"
CHALLENGE = DATASETS / "portrait_external_challenge"
DATASET_ID = "TheKernel01/140k-Real-and-Fake-Faces"
BASE = "https://datasets-server.huggingface.co/"
LIMIT = 50
MAX_SCAN = 10000
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

TARGETS = {
    "real": CHALLENGE / "real" / "hf_140k_stylegan_test",
    "fake": CHALLENGE / "fake" / "hf_140k_stylegan_test",
}

def get_bytes(url):
    req = Request(url, headers={"User-Agent": "PortraitResearchBenchmark/1.0"})
    with urlopen(req, timeout=45) as response:
        return response.read()

def get_json(url):
    return json.loads(get_bytes(url).decode("utf-8"))

def sha(data):
    return hashlib.sha256(data).hexdigest()

def file_sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def label_name(value):
    if isinstance(value, dict):
        value = value.get("name", value.get("id"))
    if value in (0, "0", "real"):
        return "real"
    if value in (1, "1", "fake"):
        return "fake"
    if isinstance(value, str):
        if value.lower() in ("ai", "stylegan", "generated"):
            return "fake"
    return None

def resolve(raw, manifest):
    p = Path(raw)
    if not p.is_absolute():
        rooted = ROOT / p
        relative = manifest.parent / p
        p = rooted if rooted.exists() else relative
    return p.resolve(strict=False)

def existing_hashes():
    hashes = set()
    paths = set()

    for manifest in DATASETS.rglob("*.csv"):
        if "manifest" not in manifest.name.lower():
            continue
        if not any(w in manifest.name.lower() for w in ("train", "test")):
            continue
        try:
            with manifest.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                col = next(
                    (c for c in (reader.fieldnames or [])
                     if c.lower() == "path"), None
                )
                if not col:
                    continue
                for row in reader:
                    raw = (row.get(col) or "").strip()
                    if not raw:
                        continue
                    p = resolve(raw, manifest)
                    paths.add(str(p).casefold())
                    if p.is_file() and p.suffix.lower() in EXTS:
                        try:
                            hashes.add(file_sha256(p))
                        except (OSError, ValueError):
                            pass
        except (OSError, csv.Error) as e:
            print("Warning: manifest skipped:", manifest, e)

    for name in ("openfake_external_holdout", "openfake_commons_holdout"):
        folder = DATASETS / name / "validation"
        if folder.is_dir():
            for p in folder.rglob("*"):
                if p.is_file() and p.suffix.lower() in EXTS:
                    paths.add(str(p.resolve()).casefold())
                    try:
                        hashes.add(file_sha256(p))
                    except OSError:
                        pass

    for folder in TARGETS.values():
        if folder.is_dir():
            for p in folder.rglob("*"):
                if p.is_file() and p.suffix.lower() in EXTS:
                    try:
                        hashes.add(file_sha256(p))
                    except OSError:
                        pass

    print("Existing project image hashes checked:", len(hashes))
    return hashes, paths

def image_bytes(field):
    if not isinstance(field, dict):
        return None

    inline = field.get("bytes")
    if isinstance(inline, str) and inline:
        import base64
        try:
            data = base64.b64decode(inline)
            with Image.open(BytesIO(data)) as im:
                im.verify()
            return data
        except Exception:
            pass

    src = field.get("src") or field.get("url")
    if not src:
        return None
    src = urljoin(BASE, src)

    try:
        data = get_bytes(src)
        with Image.open(BytesIO(data)) as im:
            im.verify()
        return data
    except Exception:
        return None

def main():
    if not (ROOT / "aidetect").is_dir():
        raise SystemExit("Run from the ai-image-detector project folder.")

    for folder in TARGETS.values():
        folder.mkdir(parents=True, exist_ok=True)

    counts = {
        label: sum(1 for p in folder.rglob("*")
                   if p.is_file() and p.suffix.lower() in EXTS)
        for label, folder in TARGETS.items()
    }

    if all(counts[x] >= LIMIT for x in ("real", "fake")):
        print("Already have enough images:", counts)
        return

    seen, protected_paths = existing_hashes()
    receipts = CHALLENGE / "external_source_receipts.csv"
    downloaded = {"real": 0, "fake": 0}

    def request_rows(offset, config):
        query = urlencode({
            "dataset": DATASET_ID,
            "config": config,
            "split": "test",
            "offset": offset,
            "length": 100,
        })
        return get_json(BASE + "rows?" + query)

    print("Checking the public dataset test split...")
    info = get_json(BASE + "splits?" + urlencode({"dataset": DATASET_ID}))
    tests = [s for s in info.get("splits", []) if s.get("split") == "test"]
    if not tests:
        raise RuntimeError("No public test split was found.")

    config = tests[0].get("config") or "default"
    print("Dataset:", DATASET_ID)
    print("Split: test; config:", config)
    print("Goal: 50 real and 50 StyleGAN-generated faces")
    print("These are for external evaluation, NOT training or threshold tuning.")

    receipt_exists = receipts.exists()
    scanned = 0

    for offset in range(0, MAX_SCAN, 100):
        if all(counts[x] >= LIMIT for x in ("real", "fake")):
            break
        page = request_rows(offset, config)
        rows = page.get("rows", [])
        if not rows:
            break

        for entry in rows:
            scanned += 1
            row = entry.get("row", {})
            label = label_name(row.get("label"))
            if label not in TARGETS or counts[label] >= LIMIT:
                continue

            data = image_bytes(row.get("image"))
            if not data:
                continue

            digest = sha(data)
            if digest in seen:
                continue

            try:
                with Image.open(BytesIO(data)) as im:
                    im.verify()
            except Exception:
                continue

            name = f"external_{label}_{counts[label]+1:03d}.jpg"
            dest = TARGETS[label] / name
            if dest.exists():
                print("Refusing to overwrite:", dest)
                continue

            dest.write_bytes(data)
            seen.add(digest)
            counts[label] += 1
            downloaded[label] += 1

            receipt = {
                "file": str(dest.relative_to(ROOT)),
                "label": label,
                "sha256": digest,
                "dataset": DATASET_ID,
                "split": "test",
                "row_idx": entry.get("row_idx", ""),
                "source_url": "https://huggingface.co/datasets/" + DATASET_ID,
                "downloaded_utc": datetime.now(timezone.utc).isoformat(),
                "license_note": "Check dataset and underlying FFHQ terms; non-commercial research only unless rights are verified."
            }
            with receipts.open("a", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(receipt.keys()))
                if not receipt_exists:
                    writer.writeheader()
                    receipt_exists = True
                writer.writerow(receipt)

        print(f"Scanned {scanned} rows; real={counts['real']}, fake={counts['fake']}")
        time.sleep(0.1)

    print("\n=== DOWNLOAD SUMMARY ===")
    print("Total real images:", counts["real"])
    print("Total AI images:", counts["fake"])
    print("New downloads:", downloaded)
    print("Rows scanned:", scanned)
    print("Source receipts:", receipts)
    print("Original models and training manifests were not changed.")
    print("Note: this benchmark tests StyleGAN faces; it does not prove performance on modern generators.")

    if min(counts.values()) >= 20:
        print("\nNext: run your portrait_external_benchmark.py script.")
    else:
        print("\nNot enough images yet. Check internet/API errors and rerun to resume.")

if __name__ == "__main__":
    main()

