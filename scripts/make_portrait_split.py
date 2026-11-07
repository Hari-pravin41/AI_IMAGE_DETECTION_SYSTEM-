import csv
import random
from pathlib import Path

root = Path("datasets/ai_detector_arena_v0_1_source/benchmark-v0.1/images")
out_dir = Path("datasets/portrait_experiment")
train_path = out_dir / "train_manifest.csv"
test_path = out_dir / "test_manifest.csv"

if train_path.exists() or test_path.exists():
    raise SystemExit("STOP: manifest already exists; no files were changed.")

ai_dir = root / "ai" / "portrait"
real_dir = root / "real" / "portrait"

ai_files = sorted(p for p in ai_dir.iterdir() if p.is_file())
real_files = sorted(p for p in real_dir.iterdir() if p.is_file())

if len(ai_files) != 170 or len(real_files) != 170:
    raise SystemExit(f"Unexpected counts: AI={len(ai_files)}, real={len(real_files)}")

def generator(path):
    return path.stem.split("_portrait")[0]

groups = {}
for path in ai_files:
    groups.setdefault(generator(path), []).append(path)

if len(groups) != 17 or any(len(paths) != 10 for paths in groups.values()):
    raise SystemExit("Unexpected AI generator grouping; no manifests created.")

rng = random.Random(42)
held_out = set(rng.sample(sorted(groups), 5))

train_rows, test_rows = [], []

for name, paths in sorted(groups.items()):
    target = test_rows if name in held_out else train_rows
    for path in paths:
        target.append({
            "path": str(path.resolve()),
            "label": "ai",
            "source": f"ai/{name}",
        })

real_shuffled = real_files.copy()
rng.shuffle(real_shuffled)
real_test = set(real_shuffled[:34])

for path in real_files:
    target = test_rows if path in real_test else train_rows
    target.append({
        "path": str(path.resolve()),
        "label": "real",
        "source": "real/arena",
    })

out_dir.mkdir(parents=True, exist_ok=True)
for target, rows in ((train_path, train_rows), (test_path, test_rows)):
    with target.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "source"])
        writer.writeheader()
        writer.writerows(rows)

print("TRAIN:", train_path, len(train_rows))
print("TEST :", test_path, len(test_rows))
print("Held-out AI generators:", ", ".join(sorted(held_out)))
print("Train counts:", sum(r["label"] == "ai" for r in train_rows), "AI /",
      sum(r["label"] == "real" for r in train_rows), "real")
print("Test counts :", sum(r["label"] == "ai" for r in test_rows), "AI /",
      sum(r["label"] == "real" for r in test_rows), "real")
