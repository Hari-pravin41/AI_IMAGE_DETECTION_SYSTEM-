import argparse, csv, hashlib, json, sys
from collections import Counter
from pathlib import Path

def load_manifest(filename):
    mp = Path(filename).resolve()
    records, missing = [], []
    with mp.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        cols = {c.strip().lower(): c for c in (reader.fieldnames or [])}
        if "path" not in cols or "label" not in cols:
            raise ValueError(f"{mp} needs path,label[,source] columns")
        for line, row in enumerate(reader, start=2):
            label = row[cols["label"]].strip().lower()
            p = Path(row[cols["path"]].strip())
            if not p.is_absolute():
                p = (mp.parent / p).resolve()
            if not p.is_file():
                missing.append({"line": line, "path": str(p)})
                continue
            h = hashlib.sha256()
            with p.open("rb") as image:
                for chunk in iter(lambda: image.read(1024 * 1024), b""):
                    h.update(chunk)
            records.append({"path": str(p), "label": label, "sha256": h.hexdigest()})
    return records, missing

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    args = parser.parse_args()

    train, missing_train = load_manifest(args.train)
    test, missing_test = load_manifest(args.test)
    train_hashes = {r["sha256"] for r in train}
    test_hashes = {r["sha256"] for r in test}
    overlap = sorted(train_hashes & test_hashes)

    result = {
        "train_rows_hashed": len(train),
        "test_rows_hashed": len(test),
        "train_label_counts": dict(Counter(r["label"] for r in train)),
        "test_label_counts": dict(Counter(r["label"] for r in test)),
        "missing_train": missing_train,
        "missing_test": missing_test,
        "unique_sha256_overlap_count": len(overlap),
        "overlap_examples": overlap[:10],
    }
    out = Path("reports/source_holdout_v4_hash_audit.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))

    if missing_train or missing_test:
        print("FAIL: missing images detected. Training stopped.")
        return 2
    if overlap:
        print("FAIL: identical image contents appear in both splits. Training stopped.")
        return 3
    if not train or not test:
        print("FAIL: an empty split was found.")
        return 4
    print("PASS: no missing images and no exact SHA-256 overlap.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
