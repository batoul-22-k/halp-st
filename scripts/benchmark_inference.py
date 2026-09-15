"""Resumable controlled CPU pair inference; loading/tokenization excluded."""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
import platform
import time
import numpy as np
import pandas as pd
import torch
import transformers
from evaluation_common import OUT, ROOT, checkpoint, load_checkpoint, runs, write_table

CACHE = OUT / 'inference_runs'
ORDER = ['B0', 'B1', 'B2', 'P1', 'P2', 'P3']

def atomic_json(path, payload):
    temp = path.with_suffix('.json.tmp')
    with temp.open('w', encoding='utf-8') as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)

def read_complete(path, signature):
    if not path.exists():
        return None
    record = json.loads(path.read_text(encoding='utf-8'))
    if record.get('status') != 'complete':
        return None
    if record.get('signature') != signature:
        raise ValueError(f'Saved benchmark settings/checkpoint differ: {path}; do not mix workloads')
    samples = np.asarray(record.get('seconds', []), dtype=float)
    if len(samples) != signature['repeats'] or not np.isfinite(samples).all() or (samples <= 0).any():
        return None
    return record

def publish(records):
    rows, raw = [], []
    for record in records:
        sig, seconds = record['signature'], np.asarray(record['seconds'])
        row = {key: sig[key] for key in ['architecture', 'seed', 'batch_size', 'max_length', 'threads', 'warmup', 'repeats', 'checkpoint']}
        row.update(mean_latency_ms=seconds.mean()*1000, median_latency_ms=np.median(seconds)*1000, std_latency_ms=seconds.std(ddof=1)*1000, throughput_pairs_per_second=sig['batch_size']/seconds.mean(), session_id=record['session_id'])
        rows.append(row)
        raw += [dict(architecture=sig['architecture'], batch_size=sig['batch_size'], iteration=i, seconds=value, session_id=record['session_id']) for i, value in enumerate(seconds)]
    if rows:
        write_table(pd.DataFrame(rows).sort_values(['batch_size', 'architecture']), 'inference_efficiency', 'CPU float32, eval(), no_grad(); two embedding forwards plus cosine per pair batch. Shared tokenizer and fixed STS-B inputs; padding/truncation to 128. Loading and tokenization excluded. SD uses ddof=1; throughput = batch size / mean batch seconds. Five warmups and thirty repetitions by default. Completed runs are atomically saved and resumed; incomplete runs are repeated cleanly. Architecture order B0, B1, B2, P1, P2, P3. Fixed order, CPU power/thermal state and background activity can affect timings; these are workload-specific descriptive measurements, not a latency significance test. The table includes completed runs only. Exact settings, sessions and raw timings are retained in inference_runs/.')
        pd.DataFrame(raw).to_csv(OUT/'inference_timing_samples.csv', index=False)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch-sizes', nargs='+', type=int, default=[1, 16])
    p.add_argument('--architectures', nargs='+', choices=ORDER, default=ORDER)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--repeats', type=int, default=30)
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args()
    if args.warmup < 1 or args.repeats < 2 or min(args.batch_sizes) < 1 or args.threads < 1:
        p.error('Need positive batch sizes/threads/warmup and at least two repetitions')
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    df = runs()
    input_path = OUT/'inference_benchmark_inputs.csv'
    if input_path.exists():
        pairs = pd.read_csv(input_path)
    else:
        source = checkpoint(df.iloc[0]).parent/'predictions.csv'
        pairs = pd.read_csv(source)[['sentence1','sentence2']].head(max(args.batch_sizes))
        pairs.to_csv(input_path, index=False)
    if len(pairs) < max(args.batch_sizes):
        raise ValueError('Existing fixed input file has insufficient pairs; refusing to change it')
    input_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()
    environment = dict(platform=platform.platform(), processor=platform.processor(), python=platform.python_version(), torch=torch.__version__, transformers=transformers.__version__, interop_threads=1, input_sha256=input_hash)
    CACHE.mkdir(exist_ok=True)
    environment_path = OUT/'inference_environment.json'
    if environment_path.exists() and json.loads(environment_path.read_text()) != environment:
        raise ValueError('Benchmark environment changed; refusing to combine unlike environments')
    if not environment_path.exists():
        atomic_json(environment_path, environment)
    session = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    signatures, existing = {}, {}
    for row in df[df.seed == 13].itertuples():
        stat = checkpoint(row).stat()
        for batch in sorted(set(args.batch_sizes)):
            sig = dict(architecture=row.architecture, seed=13, batch_size=batch, max_length=128, threads=args.threads, warmup=args.warmup, repeats=args.repeats, checkpoint=str(checkpoint(row).relative_to(ROOT)), checkpoint_bytes=stat.st_size, checkpoint_mtime_ns=stat.st_mtime_ns, input_sha256=input_hash, environment=environment)
            key = (row.architecture, batch)
            signatures[key] = sig
            record = read_complete(CACHE/f'{row.architecture}_batch{batch}.json', sig)
            if record:
                existing[key] = record
    publish(list(existing.values()))
    tokenizer = None
    for arch in dict.fromkeys(args.architectures):
        pending = [b for b in sorted(set(args.batch_sizes)) if (arch,b) not in existing]
        if not pending:
            print(f'Skip {arch}: all requested batches durably complete', flush=True)
            continue
        row = df[(df.seed == 13) & (df.architecture == arch)].iloc[0]
        print(f'Load {arch}; missing batches {pending}', flush=True)
        model = load_checkpoint(row)
        tokenizer = tokenizer or model.tokenizer
        for batch in pending:
            print(f'Time {arch} batch {batch}: {args.warmup} warmups + {args.repeats} repetitions', flush=True)
            tokens = [tokenizer(pairs[col].head(batch).tolist(), padding='max_length', truncation=True, max_length=128, return_tensors='pt') for col in ['sentence1','sentence2']]
            def infer():
                a, b = [model(**t) for t in tokens]
                return torch.nn.functional.cosine_similarity(a,b)
            with torch.no_grad():
                for _ in range(args.warmup):
                    infer()
                elapsed = []
                for _ in range(args.repeats):
                    start = time.perf_counter()
                    infer()
                    elapsed.append(time.perf_counter()-start)
            record = dict(status='complete', session_id=session, completed_at=datetime.now(timezone.utc).isoformat(), signature=signatures[(arch,batch)], seconds=elapsed)
            atomic_json(CACHE/f'{arch}_batch{batch}.json', record)
            existing[(arch,batch)] = record
            publish(list(existing.values()))
            print(f'Saved {arch} batch {batch}: median {np.median(elapsed)*1000:.3f} ms', flush=True)
        del model
        gc.collect()
    print(f'Benchmark finished: {len(existing)} complete model/batch runs saved.', flush=True)

if __name__ == '__main__':
    main()
