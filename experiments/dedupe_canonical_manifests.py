import csv, hashlib, json, sys
from collections import defaultdict, Counter
from pathlib import Path

def label_key(s):
    s = s.strip().lower()
    if s in ("ai", "1", "fake", "synthetic"):
        return "ai"
    if s in ("real", "0", "genuine"):
        return "real"
    raise ValueError(f"Unrecognized label: {s}")

def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def load(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        cols = {x.strip().lower(): x for x in (r.fieldnames or [])}
        if "path" not in cols or "label" not in cols:
            raise ValueError(f"{path}: requires path and label columns")
        rows = []
        for row in r:
            p = Path(row[cols["path"]]).resolve()
            if not p.is_file():
                raise FileNotFoundError(p)
            rows.append({
                "path": str(p),
                "label": label_key(row[cols["label"]]),
                "source": row.get(cols.get("source", ""), "") if "source" in cols else "",
                "sha256": digest(p)
            })
    return rows

def dedupe(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["sha256"]].append(row)

    kept, conflicts, duplicates = [], [], 0
    for sha, group in groups.items():
        labels = {r["label"] for r in group}
        if len(labels) > 1:
            conflicts.append({
                "sha256": sha,
                "labels": sorted(labels),
                "paths": [r["path"] for r in group]
            })
            continue
        kept.append(group[0])
        duplicates += len(group) - 1
    return kept, conflicts, duplicates

def save(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "source"])
        w.writeheader()
        w.writerows({k:r[k] for k in ("path","label","source")} for r in rows)

def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True)
    train0 = load(src / "train_manifest.csv")
    test0 = load(src / "test_manifest.csv")
    train, train_conflicts, train_dupes = dedupe(train0)
    test, test_conflicts, test_dupes = dedupe(test0)

    # Remove any identical files that cross between training and test.
    overlap = {r["sha256"] for r in train} & {r["sha256"] for r in test}
    train = [r for r in train if r["sha256"] not in overlap]
    test = [r for r in test if r["sha256"] not in overlap]

    audit = {
        "train_before": len(train0),
        "test_before": len(test0),
        "train_after": len(train),
        "test_after": len(test),
        "train_same_label_duplicates_removed": train_dupes,
        "test_same_label_duplicates_removed": test_dupes,
        "train_conflicting_groups_removed": train_conflicts,
        "test_conflicting_groups_removed": test_conflicts,
        "cross_split_hashes_removed": len(overlap),
        "train_labels": dict(Counter(r["label"] for r in train)),
        "test_labels": dict(Counter(r["label"] for r in test))
    }

    save(out / "train_manifest.csv", train)
    save(out / "test_manifest.csv", test)
    (out / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))

    for split, rows in (("train", train), ("test", test)):
        if not rows or {r["label"] for r in rows} != {"ai", "real"}:
            raise SystemExit(f"STOP: {split} split is empty or missing a class.")

if __name__ == "__main__":
    main()
