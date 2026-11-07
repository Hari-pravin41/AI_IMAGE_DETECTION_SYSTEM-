"""Standalone frozen DINOv2 feature + linear classifier experiment.
Does not modify aidetect/, existing models, or the live app.
"""
from __future__ import annotations
import argparse, csv, json, random, time
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score, average_precision_score,
                             brier_score_loss, log_loss, confusion_matrix)
from sklearn.model_selection import train_test_split

ImageFile.LOAD_TRUNCATED_IMAGES = True
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def parse_label(value):
    s = str(value).strip().lower()
    if s in {'ai', 'fake', 'synthetic', 'generated', 'ai_generated', '1'}:
        return 1
    if s in {'real', 'human', 'authentic', '0'}:
        return 0
    raise ValueError(f'Unrecognized label: {value!r}')


def read_manifest(file):
    file = Path(file)
    rows = []
    with file.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not {'path', 'label'}.issubset(set(reader.fieldnames)):
            raise ValueError(f'{file} must have path,label columns')
        for row in reader:
            p = Path(row['path'].strip().strip('"'))
            if not p.is_absolute():
                p = (file.parent / p).resolve()
            if p.is_file():
                try:
                    y = parse_label(row['label'])
                except ValueError:
                    continue
                rows.append((str(p), y, row.get('source', 'unknown')))
    if not rows:
        raise RuntimeError(f'No usable rows found in {file}')
    return rows


class ImageRows(Dataset):
    def __init__(self, rows):
        self.rows = rows
        self.tf = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        path, label, source = self.rows[i]
        try:
            with Image.open(path) as im:
                im.load()
                x = self.tf(im.convert('RGB'))
            return x, label, i
        except Exception as e:
            raise RuntimeError(f'Could not read image {path}: {e}') from e


def extract_features(rows, encoder, device, batch_size):
    loader = DataLoader(ImageRows(rows), batch_size=batch_size, shuffle=False, num_workers=0)
    feats = np.empty((len(rows), 384), dtype=np.float32)
    labels = np.asarray([r[1] for r in rows], dtype=np.int64)
    use_amp = device.type == 'cuda'
    encoder.eval()
    started = time.time()
    with torch.inference_mode():
        for step, (images, _, indices) in enumerate(loader, 1):
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                out = encoder(images)
                if isinstance(out, dict):
                    out = out.get('x_norm_clstoken', out.get('x_prenorm'))
                    if out is None:
                        raise RuntimeError('Unexpected DINOv2 output dictionary')
                    if out.ndim == 3: out = out[:, 0]
                elif isinstance(out, (tuple, list)):
                    out = out[0]
                    if out.ndim == 3: out = out[:, 0]
                if out.ndim != 2:
                    raise RuntimeError(f'Unexpected feature shape: {tuple(out.shape)}')
            feats[np.asarray(indices)] = out.float().cpu().numpy()
            if step % 50 == 0 or step == len(loader):
                print(f'  extracted {min(step*batch_size,len(rows))}/{len(rows)} images', flush=True)
    print(f'  feature extraction took {time.time()-started:.1f}s', flush=True)
    return feats, labels


def metrics(rows, y, probs):
    pred = (probs >= 0.5).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel().tolist()
    by_source = {}
    for src in sorted(set(r[2] for r in rows)):
        idx = np.asarray([r[2] == src for r in rows])
        if idx.any():
            by_source[src] = {'n': int(idx.sum()), 'accuracy': float(accuracy_score(y[idx], pred[idx]))}
    return {
        'n': int(len(y)), 'n_ai': int((y == 1).sum()), 'n_real': int((y == 0).sum()),
        'accuracy': float(accuracy_score(y, pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y, pred)),
        'precision_ai': float(precision_score(y, pred, zero_division=0)),
        'recall_ai': float(recall_score(y, pred, zero_division=0)),
        'specificity_real': float(tn / (tn + fp)) if tn + fp else 0.0,
        'f1_ai': float(f1_score(y, pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y, probs)) if len(set(y)) == 2 else None,
        'average_precision': float(average_precision_score(y, probs)) if len(set(y)) == 2 else None,
        'brier_score': float(brier_score_loss(y, probs)),
        'log_loss': float(log_loss(y, np.clip(probs, 1e-7, 1-1e-7), labels=[0,1])),
        'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
        'source_accuracy': by_source,
        'threshold': 0.5,
        'warning': 'Experimental frozen DINOv2 baseline. Evaluate on genuinely independent data; file-format shortcuts and dataset overlap may inflate results.'
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-manifest', required=True)
    ap.add_argument('--test-manifest', required=True)
    ap.add_argument('--out', default='models/dinov2_vits14_v1')
    ap.add_argument('--batch-size', type=int, default=2)
    ap.add_argument('--device', choices=['cuda','cpu'], default='cuda')
    args = ap.parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable. Re-run with --device cpu')
    device = torch.device(args.device)
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    print('Reading manifests...', flush=True)
    train_rows = read_manifest(args.train_manifest)
    test_rows = read_manifest(args.test_manifest)
    print(f'Train rows: {len(train_rows)} (AI={sum(r[1] for r in train_rows)}, real={sum(1-r[1] for r in train_rows)})')
    print(f'Test rows: {len(test_rows)} (AI={sum(r[1] for r in test_rows)}, real={sum(1-r[1] for r in test_rows)})')
    print('Loading pretrained DINOv2 ViT-S/14. First run downloads weights from Facebook Research.', flush=True)
    encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', trust_repo=True, verbose=True)
    encoder = encoder.to(device).eval()
    for p in encoder.parameters(): p.requires_grad_(False)
    # Explicitly cache features so the encoder is only run once per manifest.
    train_x, train_y = extract_features(train_rows, encoder, device, args.batch_size)
    test_x, test_y = extract_features(test_rows, encoder, device, args.batch_size)
    del encoder
    if device.type == 'cuda': torch.cuda.empty_cache()
    x_fit, x_val, y_fit, y_val = train_test_split(
        train_x, train_y, test_size=0.15, random_state=42, stratify=train_y
    )
    print('Training linear classifier on frozen embeddings...', flush=True)
    clf = LogisticRegression(max_iter=1000, class_weight='balanced', C=1.0, solver='liblinear', random_state=42)
    clf.fit(x_fit, y_fit)
    val_prob = clf.predict_proba(x_val)[:, 1]
    val_metrics = metrics([train_rows[i] for i in []], np.array([], dtype=int), np.array([])) if False else None
    val_pred = (val_prob >= 0.5).astype(np.int64)
    val_summary = {
        'n': int(len(y_val)), 'accuracy': float(accuracy_score(y_val, val_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_val, val_pred)),
        'roc_auc': float(roc_auc_score(y_val, val_prob)),
        'warning': 'Random image-level validation; may share source/generator characteristics with training and is not a generalization guarantee.'
    }
    # Refit using all training data after the validation sanity check.
    clf.fit(train_x, train_y)
    test_prob = clf.predict_proba(test_x)[:, 1]
    test_metrics = metrics(test_rows, test_y, test_prob)
    import joblib
    joblib.dump(clf, outdir / 'classifier.joblib')
    np.savez_compressed(outdir / 'features.npz', train_features=train_x, train_labels=train_y,
                        test_features=test_x, test_labels=test_y)
    report = {
        'architecture': 'DINOv2 ViT-S/14 frozen encoder + LogisticRegression classifier',
        'device': str(device), 'batch_size': args.batch_size,
        'train_manifest': str(Path(args.train_manifest).resolve()),
        'test_manifest': str(Path(args.test_manifest).resolve()),
        'train_rows': len(train_rows), 'test_rows': len(test_rows),
        'random_validation': val_summary, 'held_out_test': test_metrics,
        'inference_note': 'Standalone experiment; not integrated into the existing CLI or web app.'
    }
    (outdir / 'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('\n=== RANDOM VALIDATION ===')
    print(json.dumps(val_summary, indent=2))
    print('\n=== HELD-OUT TEST ===')
    print(json.dumps(test_metrics, indent=2))
    print(f'\nSaved experiment files to: {outdir.resolve()}')

if __name__ == '__main__':
    main()
