"""Read-only checkpoint loading and validated post-experiment inputs."""
import os
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
import pandas as pd
import torch
import yaml
from transformers import AutoConfig, AutoModel, AutoTokenizer
from src.models import BaselineSTModel, HALPSTModel

SEEDS = [13, 21, 42, 55, 87]
CONFIGS = dict(zip(['baseline.yaml', 'baseline_max.yaml', 'baseline_attention.yaml', 'ablation_p1.yaml', 'ablation_p2.yaml', 'halpst.yaml'], ['B0', 'B1', 'B2', 'P1', 'P2', 'P3']))
METRICS = ['spearman', 'pearson', 'mse', 'mae', 'rmse']
OUT = ROOT / 'results'

def runs():
    df = pd.read_csv(OUT / 'aggregate_metrics.csv')
    df['architecture'] = df.config.map(lambda x: CONFIGS.get(x.replace('\\', '/').split('/')[-1]))
    if df.architecture.isna().any() or len(df) != 30:
        raise ValueError('Expected exactly the 30 controlled runs, no smoke runs')
    for name, group in df.groupby('architecture'):
        if sorted(group.seed.tolist()) != SEEDS:
            raise ValueError(f'Invalid seeds: {name}')
    for row in df.itertuples():
        path = ROOT / row.result_dir.replace('\\', '/') / 'best.pt'
        if 'smoke' in str(path) or not path.exists():
            raise ValueError(f'Invalid checkpoint {path}')
        cfg = yaml.safe_load((path.parent / 'config.yaml').read_text())
        if cfg['training']['seed'] != row.seed:
            raise ValueError(f'Seed mismatch: {path}')
        recorded = json.loads((path.parent / 'metrics.json').read_text())
        for metric in METRICS:
            if metric not in recorded or not __import__('math').isfinite(getattr(row, metric)) or abs(recorded[metric] - getattr(row, metric)) > 1e-10:
                raise ValueError(f'Aggregate metric mismatch: {path}, {metric}')
        expected = yaml.safe_load((ROOT / row.config.replace('\\', '/')).read_text())
        if cfg['model'] != expected['model'] or cfg['dataset'] != expected['dataset']:
            raise ValueError(f'Architecture/dataset mismatch: {path}')
        for key, value in expected['training'].items():
            if key != 'seed' and cfg['training'].get(key) != value:
                raise ValueError(f'Training control mismatch: {path}, {key}')
    return df.sort_values(['architecture', 'seed'])

def checkpoint(row):
    return ROOT / row.result_dir.replace('\\', '/') / 'best.pt'

def load_checkpoint(row):
    payload = torch.load(checkpoint(row), map_location='cpu', weights_only=True, mmap=True)
    cfg = payload['config']['model']
    encoder = AutoModel.from_config(AutoConfig.from_pretrained(cfg['backbone'], local_files_only=True))
    tokenizer = AutoTokenizer.from_pretrained(cfg['backbone'], local_files_only=True)
    common = dict(backbone_name=cfg['backbone'], max_length=cfg['max_sequence_length'], encoder=encoder, tokenizer=tokenizer)
    if cfg['variant'] == 'baseline':
        model = BaselineSTModel(pooling=cfg.get('pooling', 'mean'), **common)
    else:
        opts = {k: cfg[k] for k in ['adaptive_layer_fusion', 'semantic_token_gate', 'residual_mean', 'projection', 'residual_mode'] if k in cfg}
        model = HALPSTModel(**common, **opts)
    model.load_state_dict(payload['state_dict'], strict=True)
    return model.cpu().eval()

def markdown(df):
    def cell(value):
        return (f'{value:.6f}' if isinstance(value, float) else str(value)).replace('|', '\\|').replace('\n', ' ')
    return '\n'.join(['| ' + ' | '.join(map(str, df.columns)) + ' |', '| ' + ' | '.join(['---'] * len(df.columns)) + ' |'] + ['| ' + ' | '.join(cell(v) for v in row) + ' |' for row in df.itertuples(index=False, name=None)])

def write_table(df, stem, note=''):
    df.to_csv(OUT / f'{stem}.csv', index=False)
    (OUT / f'{stem}.md').write_text(note + '\n\n' + markdown(df) + '\n', encoding='utf-8')

def answers(path):
    path = Path(path)
    df = pd.read_json(path) if path.suffix.lower() == '.json' else pd.read_csv(path)
    df = df.rename(columns={'reference_answer': 'expected_answer', 'label': 'true_label'})
    for col in ['expected_answer', 'student_answer', 'true_label']:
        if col not in df or df[col].isna().any() or df[col].astype(str).str.strip().eq('').any():
            raise ValueError(f'Missing/empty {col}')
    df['true_label'] = df.true_label.str.strip().str.lower()
    if not set(df.true_label) <= {'correct', 'partial', 'incorrect'} or df.empty:
        raise ValueError('Labels must be correct, partial, incorrect; data must not be empty')
    return df
