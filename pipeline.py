"""OASIS-1 research pipeline. Run python pipeline.py --help.

Raw volumes never become independent slice samples in the classifier.
Optional MRI/PyTorch imports are local so tabular tests run without a GPU.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import platform
import re
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

VERSION = '1.0.0'
SESSION = re.compile(r'(OAS1_\d{4}_MR\d+)')
WEIGHTS = {'cnn': 'ResNet18_Weights.IMAGENET1K_V1',
           'vit': 'ViT_B_16_Weights.IMAGENET1K_V1'}


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def load_config(path):
    path = Path(path).resolve()
    c = yaml.safe_load(path.read_text(encoding='utf-8'))
    for key in ('data_root', 'metadata_csv', 'output_dir'):
        p = Path(c[key]).expanduser()
        c[key] = str(p if p.is_absolute() else path.parent / p)
    return c


def out(c, name=''):
    p = Path(c['output_dir']) / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def digest(value):
    # JSON turns integer dictionary keys into strings; canonicalise before sorting.
    canonical = json.loads(json.dumps(value))
    return hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def image_signature(path):
    p = Path(path)
    files = [p, p.with_suffix('.img')] if p.suffix.lower() == '.hdr' else [p]
    return digest([file_hash(f) for f in files])


def versions():
    result = {'python': platform.python_version(), 'pipeline': VERSION}
    for name in ('numpy', 'pandas', 'nibabel', 'torch', 'torchvision', 'scikit-learn'):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = 'not installed'
    return result


def feature_spec(c):
    return {'version': VERSION, 'geometry': c['geometry'],
            'slice_indices': c['slice_indices'], 'weights': WEIGHTS,
            'normalisation': 'brain_nonzero_p1_p99_pad_square_resize224_imagenet_v1'}


def validate_geometry(c):
    g = c['geometry']
    if not g['verified'] or not str(g.get('notes', '')).strip():
        raise ValueError('Inspect development anatomy first; fill geometry.notes and verified=true.')
    if g['axial_axis'] not in (0, 1, 2):
        raise ValueError('axial_axis must be 0, 1 or 2.')
    indices = c['slice_indices']
    if len(indices) != 16 or len(set(indices)) != 16:
        raise ValueError('Enter exactly 16 unique, verified slice_indices.')
    if any(not isinstance(i, int) or i < 0 for i in indices):
        raise ValueError('Slice indices must be nonnegative integers.')
    if g.get('expected_affine') is None:
        raise ValueError('Copy the verified effective_affine from inspect JSON into config.')
    if np.asarray(g['expected_affine']).shape != (4, 4):
        raise ValueError('expected_affine must be 4 by 4.')
    for n in (4, 8, 16):
        subset = c['subsets'][n]
        if len(subset) != n or len(set(subset)) != n or not set(subset) <= set(range(16)):
            raise ValueError(f'Invalid subset for {n} slices.')
    if not set(c['subsets'][4]) <= set(c['subsets'][8]):
        raise ValueError('4-slice subset must be nested within 8-slice subset.')


def manifest(c):
    p = out(c, 'manifests/participants.csv')
    df = pd.read_csv(p)
    if df.subject_id.duplicated().any():
        raise ValueError('Duplicate participants in manifest.')
    return df


def prepare(c):
    target = out(c, 'manifests/participants.csv')
    if target.exists():
        raise ValueError('Manifest already exists. Keep it immutable; use a new output_dir for a new study.')
    root = Path(c['data_root'])
    if not root.is_dir():
        raise ValueError(f'Dataset root does not exist: {root}')
    meta = pd.read_csv(c['metadata_csv'])
    meta.columns = meta.columns.str.strip()
    print('Metadata columns:', list(meta.columns))
    for name in ('id_column', 'age_column', 'cdr_column'):
        if c[name] not in meta:
            raise ValueError(f'Column {c[name]!r} absent. Update {name} in config.')
    ids = meta[c['id_column']].astype(str).str.strip()
    if not ids.str.fullmatch(r'OAS1_\d{4}_MR\d+').all():
        raise ValueError('Expected original OASIS session IDs in metadata; inspect the table before adapting.')
    if ids.duplicated().any():
        raise ValueError('Duplicate metadata session IDs; resolve conflicts before proceeding.')
    meta = meta.assign(session_id=ids).set_index('session_id', drop=False)
    inventory, images = [], {}
    for p in sorted(root.glob(c['image_glob'])):
        match = SESSION.search(p.name)
        if match is None:
            continue
        sid = match.group(1)
        valid = p.suffix != '.hdr' or p.with_suffix('.img').is_file()
        inventory.append({'session_id': sid, 'image_path': str(p.resolve()), 'pair_ok': valid})
        if sid in images:
            raise ValueError(f'Multiple intended images for {sid}; resolve duplicate archive copies.')
        images[sid] = (str(p.resolve()), valid)
    if not images:
        raise ValueError('No intended MRI files found. Check extracted archives and image_glob.')
    pd.DataFrame(inventory).to_csv(out(c, 'manifests/inventory.csv'), index=False)
    rows, excluded = [], []
    for sid, r in meta.iterrows():
        age = pd.to_numeric(r[c['age_column']], errors='coerce')
        cdr = pd.to_numeric(r[c['cdr_column']], errors='coerce')
        reason = None
        if not np.isfinite(age) or age < c['min_age']:
            reason = 'missing_age_or_below_minimum'
        elif not np.isfinite(cdr):
            reason = 'missing_or_invalid_cdr'
        elif cdr not in (0, 0.5, 1, 2):
            raise ValueError(f'Unexpected CDR {cdr} for {sid}; inspect documentation.')
        elif sid not in images or not images[sid][1]:
            reason = 'missing_image_or_pair'
        if reason:
            excluded.append({'session_id': sid, 'reason': reason})
            continue
        rows.append({'subject_id': sid.rsplit('_MR', 1)[0], 'session_id': sid,
                     'session_number': int(sid.rsplit('_MR', 1)[1]),
                     'image_path': images[sid][0], 'age': float(age),
                     'cdr': float(cdr), 'label': int(cdr > 0)})
    for sid in set(images) - set(meta.index):
        excluded.append({'session_id': sid, 'reason': 'image_without_metadata'})
    if not rows:
        raise ValueError('No eligible matched participants.')
    df = pd.DataFrame(rows).sort_values(['subject_id', 'session_number'])
    for sid in df.loc[df.subject_id.duplicated(), 'session_id']:
        excluded.append({'session_id': sid, 'reason': 'later_eligible_session'})
    df = df.drop_duplicates('subject_id').reset_index(drop=True)
    # Exact repeated image data under different IDs would invalidate independence.
    df['image_sha256'] = [image_signature(p) for p in df.image_path]
    if df.image_sha256.duplicated().any():
        raise ValueError('Identical MRI bytes under different participant IDs; audit before splitting.')
    if df.label.nunique() != 2 or df.label.value_counts().min() < 8:
        raise ValueError('Too few participants per group for this default protocol. Audit cohort first.')
    dev_i, test_i = train_test_split(np.arange(len(df)), test_size=c['test_fraction'],
                                    stratify=df.label, random_state=c['seed'])
    df['split'] = 'development'
    df.loc[test_i, 'split'] = 'test'
    df['fold'] = -1
    dev = df.loc[dev_i].sort_index()
    folds = min(int(c['cv_folds']), int(dev.label.value_counts().min()))
    if folds < 2:
        raise ValueError('Insufficient development data for cross-validation.')
    cv = StratifiedKFold(folds, shuffle=True, random_state=c['seed'])
    for fold, (_, val) in enumerate(cv.split(dev, dev.label)):
        df.loc[dev.index[val], 'fold'] = fold
    df.to_csv(target, index=False)
    pd.DataFrame(excluded, columns=['session_id', 'reason']).to_csv(
        out(c, 'manifests/exclusions.csv'), index=False)
    dump(out(c, 'manifests/preparation.json'),
         {'config': c, 'metadata_sha256': file_hash(c['metadata_csv']), 'versions': versions()})
    print(df.groupby(['split', 'label']).size().to_string())
    print('Saved manifest. Keep test participants out of interactive slice selection.')


def load_volume(path, canonical=False):
    import nibabel as nib
    image = nib.load(str(path))
    original = {'shape': list(image.shape), 'axis_codes': list(nib.aff2axcodes(image.affine))}
    if canonical:
        image = nib.as_closest_canonical(image)
    volume = image.get_fdata(dtype=np.float32)
    if volume.ndim == 4 and volume.shape[-1] == 1:
        volume = volume[..., 0]
    if volume.ndim != 3 or not np.isfinite(volume).all():
        raise ValueError('Expected finite 3D MRI (or singleton fourth dimension).')
    info = {'original': original, 'effective_shape': list(volume.shape),
            'effective_zooms': [float(x) for x in image.header.get_zooms()[:3]],
            'effective_affine': image.affine.tolist(),
            'effective_axis_codes': list(nib.aff2axcodes(image.affine)),
            'minimum': float(volume.min()), 'maximum': float(volume.max())}
    return volume, info


def inspect(c, subject=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    df = manifest(c).query("split == 'development'")
    row = df.iloc[0] if subject is None else df.set_index('subject_id').loc[subject]
    sid = row.subject_id if subject is None else subject
    v, info = load_volume(row.image_path, c['geometry']['canonical'])
    print(json.dumps(info, indent=2))
    dump(out(c, f'qc/{sid}_geometry.json'), info)
    fig, axes = plt.subplots(3, 8, figsize=(16, 6))
    for axis in range(3):
        for ax, index in zip(axes[axis], np.linspace(0, v.shape[axis]-1, 10, dtype=int)[1:-1]):
            ax.imshow(np.take(v, index, axis=axis).T, cmap='gray', origin='lower')
            ax.set_title(f'array axis {axis}: {index}', fontsize=8)
            ax.axis('off')
    fig.suptitle('Array-axis preview; verify anatomical orientation independently')
    fig.tight_layout()
    fig.savefig(out(c, f'qc/{sid}_planes.png'), dpi=150)
    plt.close(fig)
    print('Preview saved in outputs/qc. This does not automatically verify orientation.')


def slices_array(path, c, positions=None):
    validate_geometry(c)
    v, info = load_volume(path, c['geometry']['canonical'])
    g = c['geometry']
    if list(v.shape) != g['expected_shape']:
        raise ValueError(f'Unexpected shape {v.shape}; expected {g["expected_shape"]}.')
    if not np.allclose(info['effective_zooms'], g['expected_zooms'], atol=1e-4):
        raise ValueError('Unexpected voxel sizes; resolve geometry before extraction.')
    if not np.allclose(info['effective_affine'], g['expected_affine'], atol=1e-4):
        raise ValueError('Different effective affine; resolve common grid before extraction.')
    mask = v != 0  # Valid ONLY for the documented zero-background brain-masked product.
    if mask.sum() < 1000:
        raise ValueError('Empty or implausibly small brain mask.')
    lo, hi = np.percentile(v[mask], [1, 99])
    if hi <= lo:
        raise ValueError('Degenerate MRI intensity range.')
    norm = np.zeros(v.shape, dtype=np.float32)
    norm[mask] = np.clip((v[mask]-lo)/(hi-lo), 0, 1)
    positions = list(range(16)) if positions is None else positions
    indices = [c['slice_indices'][i] for i in positions]
    if max(indices) >= v.shape[g['axial_axis']]:
        raise ValueError('Slice index outside the volume.')
    result = np.stack([np.take(norm, i, axis=g['axial_axis']) for i in indices])
    if np.any(np.count_nonzero(result, axis=(1, 2)) < 100):
        raise ValueError('Selected slice has almost no brain. Check range and input quality.')
    return result


def tensor_inputs(slices):
    import torch
    import torch.nn.functional as F
    x = torch.from_numpy(np.ascontiguousarray(slices)).float().unsqueeze(1)
    h, w = x.shape[-2:]
    side = max(h, w)
    x = F.pad(x, ((side-w)//2, side-w-(side-w)//2,
                  (side-h)//2, side-h-(side-h)//2))
    x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False, antialias=True)
    x = x.repeat(1, 3, 1, 1)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (x-mean)/std


def device_for(c):
    import torch
    torch.set_num_threads(max(1, int(c.get('cpu_threads', 4))))
    choice = c.get('device', 'auto')
    return ('cuda' if torch.cuda.is_available() else 'cpu') if choice == 'auto' else choice


def encoder(kind, device):
    from torch import nn
    from torchvision.models import resnet18, ResNet18_Weights, vit_b_16, ViT_B_16_Weights
    if kind == 'cnn':
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Identity()
    else:
        model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        model.heads = nn.Identity()
    return model.eval().requires_grad_(False).to(device)


def encode(model, x, batch_size, device):
    import torch
    result = []
    with torch.inference_mode():
        for b in x.split(batch_size):
            result.append(model(b.to(device)).cpu().numpy())
    a = np.concatenate(result).astype(np.float32)
    if not np.isfinite(a).all():
        raise ValueError('Non-finite encoder features.')
    return a


def cache_file(c, kind, sid):
    return out(c, f'features/{kind}/{sid}.npz')


def lock_config(c):
    return {**feature_spec(c), 'subsets': c['subsets'], 'threshold': c['threshold'],
            'c_grid': c['c_grid'], 'manifest_sha256': file_hash(out(c, 'manifests/participants.csv')),
            'versions': versions()}


def check_lock(c):
    saved = json.loads(out(c, 'models/lock.json').read_text())
    if digest(saved) != digest(lock_config(c)):
        raise ValueError('Configuration, manifest or versions changed since training. Restore the locked setup.')


def extract(c, kind, split='development', limit=None):
    validate_geometry(c)
    if split == 'test':
        check_lock(c)
    df = manifest(c).query('split == @split')
    if limit:
        df = df.head(limit)
    dev = device_for(c)
    print('Device:', dev, '| encoder:', kind, '| batch size:', c['batch_size'], flush=True)
    load_start = time.perf_counter()
    model = encoder(kind, dev)
    print(f'Model load/download: {time.perf_counter()-load_start:.1f} seconds', flush=True)
    fp = digest(feature_spec(c))
    durations = []
    for row in df.itertuples():
        p = cache_file(c, kind, row.subject_id)
        if p.exists():
            read_cache(c, kind, row)
            print('Already cached:', kind, row.subject_id)
            continue
        if image_signature(row.image_path) != row.image_sha256:
            raise ValueError('MRI bytes changed after manifest creation.')
        started = time.perf_counter()
        x = tensor_inputs(slices_array(row.image_path, c))
        features = encode(model, x, int(c['batch_size']), dev)
        expected = (16, 512 if kind == 'cnn' else 768)
        if features.shape != expected:
            raise ValueError(f'Wrong feature shape: {features.shape}, expected {expected}')
        temp = p.with_suffix('.tmp.npz')
        np.savez_compressed(temp, features=features, subject_id=row.subject_id,
                            fingerprint=fp, image_sha256=row.image_sha256,
                            runtime=digest(versions()))
        temp.replace(p)
        elapsed = time.perf_counter()-started
        durations.append(elapsed)
        print('Saved:', kind, row.subject_id, features.shape, f'{elapsed:.1f}s', flush=True)
    if durations:
        median = float(np.median(durations))
        print(f'Median seconds/person: {median:.2f}; multiply by uncached participants for a rough estimate.')


def read_cache(c, kind, row):
    p = cache_file(c, kind, row.subject_id)
    if not p.exists():
        raise ValueError(f'Missing cache: {p}. Run extract for this split/encoder.')
    with np.load(p, allow_pickle=False) as z:
        valid = (str(z['subject_id']) == row.subject_id
                 and str(z['fingerprint']) == digest(feature_spec(c))
                 and str(z['image_sha256']) == row.image_sha256
                 and str(z['runtime']) == digest(versions()))
        if not valid:
            raise ValueError(f'Stale or mismatched cache: {p}')
        a = z['features']
        if a.shape != (16, 512 if kind == 'cnn' else 768) or not np.isfinite(a).all():
            raise ValueError('Invalid feature array.')
        return a


def matrix(c, df, mode, budget):
    if mode == 'age':
        return df[['age']].to_numpy(dtype=np.float32)
    kinds = ['cnn', 'vit'] if mode == 'fusion' else [mode]
    columns = []
    for kind in kinds:
        a = np.stack([read_cache(c, kind, row) for row in df.itertuples()])
        columns.append(a[:, c['subsets'][budget], :].mean(axis=1))
    return np.concatenate(columns, axis=1)


def configurations():
    return [(mode, n) for mode in ('cnn', 'vit', 'fusion') for n in (4, 8, 16)] + [('age', 0)]


def model_name(mode, budget):
    return f'{mode}_{budget}'


def train(c, full=False):
    validate_geometry(c)
    if out(c, 'results/test_metrics.csv').exists():
        raise ValueError('Test already evaluated. Do not retune this study.')
    if out(c, 'models/lock.json').exists():
        raise ValueError('Training already locked. Keep this run; use a new output_dir for a new protocol.')
    df = manifest(c).query("split == 'development'").reset_index(drop=True)
    y = df.label.to_numpy()
    folds = [(np.flatnonzero(df.fold.to_numpy() != f), np.flatnonzero(df.fold.to_numpy() == f))
             for f in sorted(df.fold.unique())]
    for tr, va in folds:
        if len(np.unique(y[tr])) != 2 or len(np.unique(y[va])) != 2:
            raise ValueError('Both labels required in each training and validation fold.')
    summary = []
    experiments = configurations() if full else [('cnn', 16), ('age', 0)]
    for mode, n in experiments:
        name = model_name(mode, n)
        X = matrix(c, df, mode, n)
        pipe = Pipeline([('scale', StandardScaler()),
                         ('clf', LogisticRegression(class_weight='balanced', solver='lbfgs',
                                                    max_iter=5000, random_state=c['seed']))])
        search = GridSearchCV(pipe, {'clf__C': c['c_grid']}, cv=folds,
                              scoring='roc_auc', n_jobs=1, refit=True, error_score='raise')
        start = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            search.fit(X, y)
        elapsed = time.perf_counter()-start
        pd.DataFrame(search.cv_results_).to_csv(out(c, f'results/cv_{name}.csv'), index=False)
        bundle = {'estimator': search.best_estimator_, 'mode': mode, 'budget': n,
                  'feature_fingerprint': digest(feature_spec(c)), 'threshold': c['threshold']}
        joblib.dump(bundle, out(c, f'models/{name}.joblib'))
        summary.append({'model': name, 'mode': mode, 'slices': n,
                        'cv_auc': float(search.best_score_),
                        'cv_auc_fold_sd': float(search.cv_results_['std_test_score'][search.best_index_]),
                        'best_C': float(search.best_params_['clf__C']),
                        'search_and_refit_seconds': elapsed,
                        'refit_seconds': float(search.refit_time_)})
        print(name, summary[-1], flush=True)
    table = pd.DataFrame(summary)
    table.to_csv(out(c, 'results/development_comparison.csv'), index=False)
    # Predeclared selection: highest development AUC; exact ties prefer simpler mode, fewer slices.
    eligible = table[table['mode'] != 'age'].copy()
    eligible['rank'] = eligible['mode'].map({'cnn': 0, 'vit': 1, 'fusion': 2})
    best = eligible.sort_values(['cv_auc', 'rank', 'slices'], ascending=[False, True, True]).iloc[0]
    dump(out(c, 'models/selected.json'), {'model': best['model'],
         'rule': 'highest development CV AUC; exact ties CNN then ViT then fusion, fewer slices'})
    dump(out(c, 'models/lock.json'), lock_config(c))
    dump(out(c, 'models/experiments.json'), experiments)
    print('Locked. Demo candidate:', best['model'])


def metrics(y, scores, threshold=0.5):
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {'auc': float(roc_auc_score(y, scores)),
            'balanced_accuracy': float(balanced_accuracy_score(y, pred)),
            'sensitivity': float(tp/(tp+fn)), 'specificity': float(tn/(tn+fp)),
            'f1': float(f1_score(y, pred, zero_division=0)),
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}


def bootstrap_indices(y, repetitions, seed):
    rng = np.random.default_rng(seed)
    a, b = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    if not len(a) or not len(b):
        raise ValueError('Both groups required for evaluation.')
    return [np.r_[rng.choice(a, len(a), replace=True), rng.choice(b, len(b), replace=True)]
            for _ in range(repetitions)]


def evaluate(c):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    check_lock(c)
    target = out(c, 'results/test_metrics.csv')
    if target.exists():
        raise ValueError('Test already evaluated; saved results are in results/.')
    df = manifest(c).query("split == 'test'").reset_index(drop=True)
    y = df.label.to_numpy()
    indices = bootstrap_indices(y, int(c['bootstrap_repetitions']), c['seed'])
    predictions = df[['subject_id', 'label']].copy()
    summary, scores_by_name = [], {}
    fig, ax = plt.subplots(figsize=(7, 6))
    experiments = json.loads(out(c, 'models/experiments.json').read_text())
    for mode, n in experiments:
        name = model_name(mode, n)
        bundle = joblib.load(out(c, f'models/{name}.joblib'))
        s = bundle['estimator'].predict_proba(matrix(c, df, mode, n))[:, 1]
        m = metrics(y, s, c['threshold'])
        boot = [metrics(y[ix], s[ix], c['threshold']) for ix in indices]
        for key in ('auc', 'balanced_accuracy', 'sensitivity', 'specificity', 'f1'):
            values = [r[key] for r in boot]
            m[key+'_ci_low'], m[key+'_ci_high'] = map(float, np.percentile(values, [2.5, 97.5]))
        summary.append({'model': name, **m})
        predictions[name] = s
        scores_by_name[name] = s
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, label=f'{name}: {m["auc"]:.3f}')
    ax.plot([0, 1], [0, 1], '--', color='gray')
    ax.set(xlabel='False-positive rate', ylabel='True-positive rate', title='Held-out participants')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out(c, 'results/test_roc.png'), dpi=160)
    plt.close(fig)
    paired = []
    for n in (4, 8, 16):
        for baseline in ('cnn', 'vit'):
            if f'fusion_{n}' not in scores_by_name or f'{baseline}_{n}' not in scores_by_name:
                continue
            s, b = scores_by_name[f'fusion_{n}'], scores_by_name[f'{baseline}_{n}']
            diffs = [roc_auc_score(y[ix], s[ix])-roc_auc_score(y[ix], b[ix]) for ix in indices]
            low, high = np.percentile(diffs, [2.5, 97.5])
            paired.append({'comparison': f'fusion_{n} minus {baseline}_{n}',
                           'auc_difference': float(roc_auc_score(y, s)-roc_auc_score(y, b)),
                           'ci_low': float(low), 'ci_high': float(high)})
    predictions.to_csv(out(c, 'results/test_predictions.csv'), index=False)
    pd.DataFrame(paired, columns=['comparison', 'auc_difference', 'ci_low', 'ci_high']).to_csv(
        out(c, 'results/paired_auc_differences.csv'), index=False)
    pd.DataFrame(summary).to_csv(target, index=False)
    dump(out(c, 'results/evaluation_notes.json'), {
        'bootstrap': 'stratified participant percentile bootstrap; conditional on fixed fitted models',
        'comparisons': 'predefined descriptive comparisons; intervals not multiplicity-adjusted',
        'selection': json.loads(out(c, 'models/selected.json').read_text()),
        'n_test': len(df)})
    print(pd.DataFrame(summary)[['model', 'auc', 'balanced_accuracy']].to_string(index=False))


def sync(device):
    import torch
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()


def predict_image(c, path, model_name_override=None):
    """Same preprocessing as extraction. No clinical label is read here."""
    import torch
    check_lock(c)
    name = model_name_override or json.loads(out(c, 'models/selected.json').read_text())['model']
    allowed = {model_name(m, n) for m, n in configurations() if m != 'age'}
    if name not in allowed:
        raise ValueError('Choose one of the nine MRI models.')
    b = joblib.load(out(c, f'models/{name}.joblib'))
    dev = device_for(c)
    sync(dev)
    start = time.perf_counter()
    slices = slices_array(path, c, c['subsets'][b['budget']])
    x = tensor_inputs(slices)
    parts = []
    for kind in (['cnn', 'vit'] if b['mode'] == 'fusion' else [b['mode']]):
        net = encoder(kind, dev)
        parts.append(encode(net, x, c['batch_size'], dev).mean(axis=0))
        del net
        gc.collect()
        if str(dev).startswith('cuda'):
            torch.cuda.empty_cache()
    X = np.concatenate(parts)[None, :]
    score = float(b['estimator'].predict_proba(X)[0, 1])
    sync(dev)
    return {'model': name, 'score': score, 'label': int(score >= b['threshold']),
            'seconds_including_model_loading': time.perf_counter()-start}, slices


def benchmark(c, subjects=5, repeats=3):
    """Component benchmark: warm loaded encoders, actual slice inputs, no feature cache."""
    import torch
    import psutil
    check_lock(c)
    df = manifest(c).query("split == 'development'").head(subjects)
    dev, results = device_for(c), []
    proc = psutil.Process()
    experiments = json.loads(out(c, 'models/experiments.json').read_text())
    kinds = ['cnn', 'vit'] if any(m in ('vit', 'fusion') for m, _ in experiments) else ['cnn']
    budgets = sorted({n for m, n in experiments if m != 'age'})
    for kind in kinds:
        sync(dev)
        start = time.perf_counter()
        net = encoder(kind, dev)
        sync(dev)
        load_s = time.perf_counter()-start
        for n in budgets:
            for row in df.itertuples():
                start = time.perf_counter()
                x = tensor_inputs(slices_array(row.image_path, c, c['subsets'][n]))
                prep_s = time.perf_counter()-start
                encode(net, x, c['batch_size'], dev)  # warm-up
                for rep in range(repeats):
                    if str(dev).startswith('cuda'):
                        torch.cuda.reset_peak_memory_stats()
                    sync(dev)
                    start = time.perf_counter()
                    encode(net, x, c['batch_size'], dev)
                    sync(dev)
                    elapsed = time.perf_counter()-start
                    results.append({'encoder': kind, 'slices': n, 'subject_id': row.subject_id,
                                    'repeat': rep, 'extraction_s': elapsed, 'preprocessing_s': prep_s,
                                    'model_load_s': load_s, 'device': dev, 'batch_size': c['batch_size'],
                                    'gpu_peak_allocated_mb': float(torch.cuda.max_memory_allocated()/2**20)
                                    if str(dev).startswith('cuda') else 0.0,
                                    'cpu_rss_after_mb': proc.memory_info().rss/2**20})
        del net
        gc.collect()
        if str(dev).startswith('cuda'):
            torch.cuda.empty_cache()
    pd.DataFrame(results).to_csv(out(c, 'results/encoder_timings.csv'), index=False)
    # Measure the real sequential-loading inference implementation separately.
    inference = []
    sample = df.iloc[0]
    for mode, n in experiments:
        if mode == 'age':
            continue
        for rep in range(repeats):
            result, _ = predict_image(c, sample.image_path, model_name(mode, n))
            inference.append({**result, 'repeat': rep, 'subject_id': sample.subject_id})
    pd.DataFrame(inference).to_csv(out(c, 'results/end_to_end_timings.csv'), index=False)
    dump(out(c, 'results/benchmark_environment.json'), {
        'versions': versions(), 'device': dev, 'platform': platform.platform(),
        'gpu': torch.cuda.get_device_name() if str(dev).startswith('cuda') else None,
        'precision': 'float32', 'component_cpu_memory': 'RSS after run, not peak RAM',
        'encoder_timing': 'warm loaded model; transfers included; disk model load excluded',
        'app_timing': 'actual sequential encoder loading; weights already downloaded; includes loading'})
    print('Saved actual component and end-to-end timings; CPU RSS is not peak memory.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='config.yaml', help='Put this option BEFORE the subcommand.')
    subs = p.add_subparsers(dest='command', required=True)
    subs.add_parser('prepare')
    s = subs.add_parser('inspect')
    s.add_argument('--subject')
    s = subs.add_parser('extract')
    s.add_argument('--encoder', choices=['cnn', 'vit'], required=True)
    s.add_argument('--split', choices=['development', 'test'], default='development')
    s.add_argument('--limit', type=int)
    s = subs.add_parser('train')
    s.add_argument('--full', action='store_true', help='All nine MRI models; default: CNN-16 + age baseline.')
    subs.add_parser('evaluate')
    s = subs.add_parser('benchmark')
    s.add_argument('--subjects', type=int, default=5)
    s.add_argument('--repeats', type=int, default=3)
    s = subs.add_parser('predict')
    s.add_argument('--image', required=True)
    s.add_argument('--model')
    a = p.parse_args()
    c = load_config(a.config)
    if a.command == 'prepare': prepare(c)
    elif a.command == 'inspect': inspect(c, a.subject)
    elif a.command == 'extract': extract(c, a.encoder, a.split, a.limit)
    elif a.command == 'train': train(c, a.full)
    elif a.command == 'evaluate': evaluate(c)
    elif a.command == 'benchmark': benchmark(c, a.subjects, a.repeats)
    else:
        result, _ = predict_image(c, a.image, a.model)
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
