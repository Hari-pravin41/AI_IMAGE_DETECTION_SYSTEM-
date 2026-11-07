import csv
import json
from pathlib import Path

import numpy as np
import joblib
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    roc_auc_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

ROOT = Path(".")
OUT = ROOT / "models/dinov2_vits14_source_holdout_v1"
FEATURES = OUT / "features.npz"

def read_manifest(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def binary_label(value):
    value = str(value).strip().lower()
    if value in {"ai", "1", "fake", "synthetic"}:
        return 1
    if value in {"real", "0"}:
        return 0
    raise ValueError(f"Unknown label: {value}")

train_rows = read_manifest(ROOT / "datasets/openfake_source_holdout_train.csv")
test_rows = read_manifest(ROOT / "datasets/openfake_source_holdout_test.csv")

with np.load(FEATURES) as data:
    X_train = data["train_features"].astype(np.float32)
    X_test = data["test_features"].astype(np.float32)

y_train = np.array([binary_label(r["label"]) for r in train_rows])
y_test = np.array([binary_label(r["label"]) for r in test_rows])
groups = np.array([r["source"].strip() for r in train_rows])

if len(X_train) != len(y_train) or len(X_test) != len(y_test):
    raise SystemExit("Feature/manifest row mismatch. Stop; do not tune.")

cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
splits = list(cv.split(X_train, y_train, groups))

for fold, (tr, va) in enumerate(splits, 1):
    if len(np.unique(y_train[tr])) < 2 or len(np.unique(y_train[va])) < 2:
        raise SystemExit(f"Fold {fold} has only one class; stop and inspect split.")

candidates = {}

for c in [0.01, 0.1, 1.0, 10.0]:
    candidates[f"logistic_C{c}"] = lambda c=c: make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c, class_weight="balanced", max_iter=3000,
            solver="liblinear", random_state=42
        )
    )

for c in [0.1, 1.0, 10.0]:
    candidates[f"linear_svc_C{c}"] = lambda c=c: make_pipeline(
        StandardScaler(),
        LinearSVC(C=c, class_weight="balanced", max_iter=10000, random_state=42)
    )

for c in [0.1, 1.0, 10.0]:
    candidates[f"rbf_svc_C{c}"] = lambda c=c: make_pipeline(
        StandardScaler(),
        SVC(C=c, kernel="rbf", class_weight="balanced")
    )

results = []

print("=== SOURCE-GROUPED CROSS-VALIDATION ===")
print("Training images:", len(y_train))
print("Unique training sources:", len(np.unique(groups)))
print("CV folds:", len(splits))

for name, make_model in candidates.items():
    oof_pred = np.full(len(y_train), -1, dtype=int)
    oof_score = np.full(len(y_train), np.nan, dtype=float)

    for tr, va in splits:
        model = make_model()
        model.fit(X_train[tr], y_train[tr])
        oof_pred[va] = model.predict(X_train[va])

        if hasattr(model, "decision_function"):
            oof_score[va] = model.decision_function(X_train[va])
        else:
            oof_score[va] = model.predict_proba(X_train[va])[:, 1]

    valid = oof_pred >= 0
    result = {
        "model": name,
        "accuracy": float(accuracy_score(y_train[valid], oof_pred[valid])),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_train[valid], oof_pred[valid])
        ),
        "roc_auc": float(roc_auc_score(y_train[valid], oof_score[valid]))
    }
    results.append(result)
    print(json.dumps(result))

# Select using training cross-validation only.
best = max(results, key=lambda r: (r["roc_auc"], r["balanced_accuracy"]))
print("\nSelected model:", best["model"])

model = candidates[best["model"]]()
model.fit(X_train, y_train)

pred = model.predict(X_test)
if hasattr(model, "decision_function"):
    score = model.decision_function(X_test)
else:
    score = model.predict_proba(X_test)[:, 1]

tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()

test_result = {
    "selected_model": best["model"],
    "evaluated_images": int(len(y_test)),
    "accuracy": float(accuracy_score(y_test, pred)),
    "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
    "precision_ai": float(precision_score(y_test, pred, zero_division=0)),
    "recall_ai": float(recall_score(y_test, pred, zero_division=0)),
    "specificity_real": float(tn / (tn + fp)) if tn + fp else None,
    "f1_ai": float(f1_score(y_test, pred, zero_division=0)),
    "roc_auc": float(roc_auc_score(y_test, score)),
    "confusion_matrix": {
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
    },
    "warning": "Held-out source test. Do not select models or thresholds using these test results."
}

joblib.dump(model, OUT / "classifier_group_cv_tuned.joblib")
report = {
    "cross_validation": results,
    "selected_by_training_cv": best["model"],
    "held_out_source_test": test_result
}
(OUT / "group_cv_tuning_metrics.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8"
)

print("\n=== HELD-OUT SOURCE TEST ===")
print(json.dumps(test_result, indent=2))
print("\nSaved separate classifier:", (OUT / "classifier_group_cv_tuned.joblib").resolve())
print("Saved report:", (OUT / "group_cv_tuning_metrics.json").resolve())
