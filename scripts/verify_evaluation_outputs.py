"""Validate generated artifacts against source data and scoring invariants."""
import hashlib
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from evaluation_common import ROOT, OUT, answers
from src.evaluation.application_scoring import application_score

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--short-answer-data', default=str(ROOT/'data/sample_short_answers.csv'))
    args = parser.parse_args()
    dataset = Path(args.short_answer_data)
    data = answers(dataset)
    pred = pd.read_csv(OUT/'short_answer_predictions.csv')
    assert len(pred) == len(data)*11
    for row in pred.itertuples():
        original = data.iloc[row.example_id]
        assert (row.expected_answer, row.student_answer, row.true_label) == (original.expected_answer, original.student_answer, original.true_label)
        score = application_score(row.semantic_score, row.expected_answer, row.student_answer)
        assert np.isclose(score['final_score'], row.final_score, atol=1e-12)
        assert score['label'] == row.prediction
    for item in json.loads((OUT/'short_answer_confusion_matrices.json').read_text()):
        assert np.asarray(item['matrix']).sum() == len(data)
    timing = pd.read_csv(OUT/'inference_efficiency.csv')
    assert len(timing) == 12
    assert set(timing.architecture) == {'B0','B1','B2','P1','P2','P3'}
    assert not timing.duplicated(['architecture','batch_size']).any()
    assert set(timing.batch_size) == {1,16}
    raw = pd.read_csv(OUT/'inference_timing_samples.csv')
    assert len(raw) == int(timing.repeats.sum())
    for row in timing.itertuples():
        samples = raw[(raw.architecture == row.architecture) & (raw.batch_size == row.batch_size)]
        assert sorted(samples.iteration.tolist()) == list(range(row.repeats))
        assert np.isfinite(samples.seconds).all() and (samples.seconds > 0).all()
        assert np.isclose(row.mean_latency_ms, samples.seconds.mean()*1000)
        assert np.isclose(row.median_latency_ms, samples.seconds.median()*1000)
        assert np.isclose(row.std_latency_ms, samples.seconds.std(ddof=1)*1000)
        record = json.loads((OUT/'inference_runs'/f'{row.architecture}_batch{row.batch_size}.json').read_text())
        assert record['status'] == 'complete'
        assert np.allclose(record['seconds'], samples.seconds)
    assert np.allclose(timing.throughput_pairs_per_second, timing.batch_size*1000/timing.mean_latency_ms)
    assert len(list((OUT/'figures').glob('*.png'))) >= 8
    assert len(list((OUT/'figures').glob('*.pdf'))) >= 8
    # Record provenance after checking each prediction against the input dataset.
    manifest = dict(path=str(dataset.resolve()), sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(), rows=len(data), label_counts=data.true_label.value_counts().to_dict(), synthetic_diagnostic=dataset.name=='sample_short_answers.csv', max_length=128, semantic_weight=.70, lexical_weight=.30, thresholds=[.50,.70])
    manifest_path = OUT/'short_answer_dataset_manifest.json'
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text()) == manifest
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f'Validated {len(pred)} predictions, 11 confusion matrices, 12 benchmarks and eight PNG/PDF figure pairs.')

if __name__ == '__main__':
    main()
