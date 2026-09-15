"""Reproduce final analyses without training or writing any checkpoint."""
import argparse
import gc
import hashlib
import json
import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from evaluation_common import ROOT, OUT, METRICS, answers, checkpoint, load_checkpoint, runs, write_table, markdown
from src.evaluation.application_scoring import application_score

FIG = OUT / 'figures'
LABELS = ['correct', 'partial', 'incorrect']
NAMES = {'B0': 'Final-layer mean', 'B1': 'Max pooling', 'B2': 'Token attention', 'P1': 'Layer fusion + mean', 'P2': 'Layer fusion + token gate', 'P3': 'Full HALP-ST'}

def savefig(name):
    plt.tight_layout()
    plt.savefig(FIG / f'{name}.png', dpi=300, bbox_inches='tight')
    plt.savefig(FIG / f'{name}.pdf', bbox_inches='tight')
    plt.close()

def paired_statistics(deltas):
    rows = []
    for metric in METRICS:
        x = deltas[metric].to_numpy(dtype=float)
        mean, sd, n = x.mean(), x.std(ddof=1), len(x)
        t = stats.ttest_1samp(x, 0)
        # Exhaustive sign permutations handle zeros/ties without asymptotic n=5 p-values.
        if np.any(x != 0):
            w = stats.wilcoxon(x, alternative='two-sided', method=stats.PermutationMethod(n_resamples=np.inf), zero_method='wilcox')
            wstat, wp, status = w.statistic, w.pvalue, 'exhaustive sign permutation'
        else:
            wstat, wp, status = np.nan, np.nan, 'undefined: all differences zero'
        half = stats.t.ppf(.975, n-1)*sd/np.sqrt(n)
        rows.append(dict(metric=metric, n=n, mean_paired_difference=mean, std_paired_difference=sd, t_statistic=t.statistic, t_pvalue=t.pvalue, wilcoxon_statistic=wstat, wilcoxon_pvalue=wp, wilcoxon_status=status, cohens_dz=mean/sd if sd else np.nan, ci95_low=mean-half, ci95_high=mean+half))
    result = pd.DataFrame(rows)
    order = np.argsort(result.t_pvalue.to_numpy())
    adjusted = np.minimum(1, np.maximum.accumulate(result.t_pvalue.to_numpy()[order] * np.arange(len(result), 0, -1)))
    result['t_pvalue_holm'] = np.nan
    result.loc[order, 't_pvalue_holm'] = adjusted
    result['significant_holm_005'] = result.t_pvalue_holm < .05
    return result

def sts_analysis(df):
    rows = []
    for arch, g in df.groupby('architecture'):
        row = dict(architecture=arch, description=NAMES[arch], n=len(g))
        for m in METRICS:
            for stat in ['mean', 'std', 'min', 'max']:
                row[f'{m}_{stat}'] = getattr(g[m], stat)()
        rows.append(row)
    summary = pd.DataFrame(rows)
    for m in METRICS:
        best = summary[f'{m}_mean'].max() if m in METRICS[:2] else summary[f'{m}_mean'].min()
        summary[f'{m}_best_mean'] = summary[f'{m}_mean'] == best
    write_table(summary, 'final_model_comparison', 'Source: aggregate_metrics.csv only; five seeds per architecture. SD is sample SD (ddof=1). Best flags identify best mean for every metric: Spearman/Pearson higher, MSE/MAE/RMSE lower.')
    display = pd.DataFrame({'Architecture': summary.architecture})
    for m in METRICS:
        display[m + (' ↑' if m in METRICS[:2] else ' ↓')] = [f"{'**' if r[f'{m}_best_mean'] else ''}{r[f'{m}_mean']:.6f} ± {r[f'{m}_std']:.6f} [{r[f'{m}_min']:.6f}, {r[f'{m}_max']:.6f}]{'**' if r[f'{m}_best_mean'] else ''}" for _, r in summary.iterrows()]
    (OUT/'final_model_comparison.md').write_text('Five-seed mean ± sample SD [min, max]. **Bold = best mean**. Source: aggregate_metrics.csv; no smoke metrics.\n\n'+markdown(display)+'\n', encoding='utf-8')
    b = df[df.architecture=='B0'].set_index('seed')
    p = df[df.architecture=='P3'].set_index('seed')
    delta = p[METRICS] - b[METRICS]
    per = delta.add_suffix('_delta_P3_minus_B0')
    for m in METRICS:
        per[f'B0_{m}'], per[f'P3_{m}'] = b[m], p[m]
    per.reset_index().to_csv(OUT/'b0_vs_halpst_per_seed.csv', index=False)
    statistics = paired_statistics(delta)
    write_table(statistics, 'b0_vs_halpst_statistics', 'Differences = P3 − B0. Positive favors P3 for correlations; negative favors P3 for errors. Two-sided paired t-tests, t-based 95% CIs (df=4), Cohen dz = mean difference / sample SD. Wilcoxon uses exhaustive sign permutations. Holm adjustment covers the five t-tests. n=5 gives limited statistical power; normality of differences cannot be reliably assessed. These tests measure training-seed variation on one shared test set, not population/dataset uncertainty. With five nonzero pairs the minimum two-sided exact Wilcoxon p-value is 0.0625. CIs are pointwise, not multiplicity-adjusted.')
    for cols, name, title in [(METRICS[:2], 'semantic_metrics', 'STS-B correlations: mean ± sample SD'), (METRICS[2:], 'error_metrics', 'STS-B errors: mean ± sample SD')]:
        fig, axes = plt.subplots(1, len(cols), figsize=(4*len(cols), 4), squeeze=False)
        for ax, m in zip(axes[0], cols):
            ax.bar(summary.architecture, summary[f'{m}_mean'], yerr=summary[f'{m}_std'], capsize=4, color='#4477AA')
            ax.set(title=m.capitalize(), ylim=(0, 1 if m in METRICS[:2] else max(summary[f'{m}_max'])*1.15))
        fig.suptitle(title)
        savefig(name)
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.5))
    for ax, m in zip(axes, METRICS):
        ax.bar(delta.index.astype(str), delta[m], color='#4477AA')
        ax.axhline(0, color='black', linewidth=.8)
        limit = max(abs(delta[m]).max()*1.2, 1e-6)
        ax.set(title=m, ylim=(-limit, limit), xlabel='Seed')
    fig.suptitle('Paired differences: P3 − B0 (zero-centered axes)')
    savefig('paired_seed_differences')
    return summary, statistics

def complexity(df):
    rows, audit = [], []
    signatures = {}
    for row in df.itertuples():
        print(f'Parameters: {row.architecture} seed {row.seed}', flush=True)
        path = checkpoint(row)
        before = path.stat()
        payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        # Model parameter names distinguish parameters from persistent buffers.
        if row.architecture not in signatures:
            model = load_checkpoint(row)
            signatures[row.architecture] = ({name: tuple(value.shape) for name, value in model.state_dict().items()}, [name for name, _ in model.named_parameters()], sum(p.numel() for p in model.parameters()))
            del model
        shapes, parameter_names, actual = signatures[row.architecture]
        if {name: tuple(value.shape) for name, value in payload['state_dict'].items()} != shapes:
            raise ValueError(f'Checkpoint tensor schema mismatch: {path}')
        count = sum(payload['state_dict'][name].numel() for name in parameter_names)
        recorded = json.loads((path.parent/'model_summary.json').read_text())['total_parameters']
        if count != actual or actual != recorded:
            raise ValueError(f'Parameter mismatch {path}')
        audit.append(dict(architecture=row.architecture, seed=row.seed, checkpoint=str(path.relative_to(ROOT)), parameters=count, bytes=before.st_size, mtime_ns=before.st_mtime_ns))
        del payload
        gc.collect()
    audit = pd.DataFrame(audit)
    audit.to_csv(OUT/'checkpoint_audit.csv', index=False)
    for arch, group in audit.groupby('architecture'):
        if group.parameters.nunique() != 1:
            raise ValueError('Parameter counts vary by seed')
        rows.append(dict(architecture=arch, parameters=int(group.parameters.iloc[0]), verified_checkpoints=len(group)))
    result = pd.DataFrame(rows)
    baseline = result.loc[result.architecture=='B0', 'parameters'].iloc[0]
    result['added_parameters_vs_B0'] = result.parameters-baseline
    result['overhead_percent_vs_B0'] = 100*(result.parameters-baseline)/baseline
    write_table(result, 'model_complexity', 'Verified against named model parameters, checkpoint tensors and experiment summaries for all 30 best.pt checkpoints. Includes registered parameters even if inactive in a particular forward path; excludes buffers.')
    plt.figure(figsize=(7,4))
    plt.bar(result.architecture, result.overhead_percent_vs_B0, color='#4477AA')
    plt.ylabel('Added parameters relative to B0 (%)')
    plt.ylim(bottom=0)
    savefig('parameter_overhead')
    return result

def grading_metrics(y, pred):
    result = {'accuracy': accuracy_score(y, pred)}
    for avg in ['macro', 'weighted']:
        p, r, f, _ = precision_recall_fscore_support(y, pred, labels=LABELS, average=avg, zero_division=0)
        result.update({f'{avg}_precision': p, f'{avg}_recall': r, f'{avg}_f1': f})
    p, r, f, support = precision_recall_fscore_support(y, pred, labels=LABELS, zero_division=0)
    for i, label in enumerate(LABELS):
        result.update({f'{label}_precision': p[i], f'{label}_recall': r[i], f'{label}_f1': f[i], f'{label}_support': int(support[i])})
    return result

def downstream(df, path):
    data = answers(path)
    (OUT/'short_answer_dataset_manifest.json').write_text(json.dumps(dict(path=str(Path(path).resolve()), sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(), rows=len(data), label_counts=data.true_label.value_counts().to_dict(), synthetic_diagnostic=Path(path).name=='sample_short_answers.csv', max_length=128, semantic_weight=.70, lexical_weight=.30, thresholds=[.50,.70]), indent=2))
    all_predictions, metric_rows, matrices = [], [], []
    selected = df[df.architecture.isin(['B0', 'P3'])]
    entries = [(r.architecture, r.seed, r) for r in selected.itertuples()] + [('Pretrained', None, None)]
    for arch, seed, row in entries:
        print(f'Short answers: {arch} seed {seed}', flush=True)
        if row is None:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', device='cpu', local_files_only=True)
            model.max_seq_length = 128
        else:
            model = load_checkpoint(row)
        model.eval()
        with torch.no_grad():
            kwargs = dict(batch_size=16, device='cpu')
            a = torch.as_tensor(model.encode(data.expected_answer.tolist(), **kwargs))
            b = torch.as_tensor(model.encode(data.student_answer.tolist(), **kwargs))
            semantic = torch.nn.functional.cosine_similarity(a, b).numpy()
        rows = []
        for i, (record, score) in enumerate(zip(data.to_dict('records'), semantic)):
            scored = application_score(float(score), record['expected_answer'], record['student_answer'])
            rows.append(dict(example_id=i, architecture=arch, seed=seed, expected_answer=record['expected_answer'], student_answer=record['student_answer'], true_label=record['true_label'], semantic_score=scored['semantic_score'], lexical_score=scored['lexical_score'], final_score=scored['final_score'], prediction=scored['label']))
        preds = pd.DataFrame(rows)
        all_predictions.extend(rows)
        metric_rows.append(dict(architecture=arch, seed=seed, **grading_metrics(data.true_label, preds.prediction)))
        cm = confusion_matrix(data.true_label, preds.prediction, labels=LABELS)
        matrices.append(dict(architecture=arch, seed=seed, labels=LABELS, matrix=cm.tolist()))
        del model
        gc.collect()
    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(all_predictions)
    predictions.to_csv(OUT/'short_answer_predictions.csv', index=False)
    controlled = metrics[metrics.architecture!='Pretrained']
    write_table(controlled, 'short_answer_architecture_per_seed', 'Controlled B0 versus P3; unchanged three-class labels, thresholds and lexical/Jaccard scoring. Undefined precision/recall/F1 set to zero; all three classes included in macro averages.')
    cols = [c for c in metrics.columns if c not in ['architecture', 'seed'] and not c.endswith('_support')]
    summary = controlled.groupby('architecture')[cols].agg(['mean', 'std']).reset_index()
    summary.columns = ['_'.join(c).rstrip('_') for c in summary.columns]
    write_table(summary, 'short_answer_architecture_summary', 'Mean and sample SD across five seeds. Same 20 examples are reused across seeds; this does not increase independent dataset size.' if len(data)==20 else 'Mean and sample SD across five seeds; the same examples are reused across seeds.')
    deployment = metrics[metrics.architecture.isin(['Pretrained','P3'])]
    write_table(deployment, 'short_answer_deployment_comparison', 'Practical deployment comparison, separate from architecture comparison: original pretrained SentenceTransformer versus each of five trained P3 seeds. Pretrained is one fixed model, with no seed SD. P3 aggregate mean/sample SD is in short_answer_architecture_summary.csv. Max sequence length 128 and identical grading formula for both.')
    (OUT/'short_answer_confusion_matrices.json').write_text(json.dumps(matrices, indent=2))
    errors = []
    for seed in selected.seed.unique():
        b = predictions[(predictions.architecture=='B0') & (predictions.seed==seed)].set_index('example_id')
        p = predictions[(predictions.architecture=='P3') & (predictions.seed==seed)].set_index('example_id')
        for i in b.index:
            br, pr = b.loc[i], p.loc[i]
            bc, pc = br.prediction==br.true_label, pr.prediction==pr.true_label
            low = (br.final_score>=.50) != (pr.final_score>=.50)
            high = (br.final_score>=.70) != (pr.final_score>=.70)
            categories = [name for flag, name in [(pc and not bc, 'HALP-ST correct and B0 wrong'), (bc and not pc, 'B0 correct and HALP-ST wrong'), (not bc and not pc, 'both wrong'), (low, 'crosses 0.50'), (high, 'crosses 0.70')] if flag]
            if categories:
                errors.append(dict(seed=seed, example_id=i, categories='; '.join(categories), expected_answer=br.expected_answer, student_answer=br.student_answer, true_label=br.true_label, B0_semantic_score=br.semantic_score, P3_semantic_score=pr.semantic_score, lexical_score=br.lexical_score, B0_final_score=br.final_score, P3_final_score=pr.final_score, B0_prediction=br.prediction, P3_prediction=pr.prediction, crosses_050=low, crosses_070=high))
    error_columns = ['seed','example_id','categories','expected_answer','student_answer','true_label','B0_semantic_score','P3_semantic_score','lexical_score','B0_final_score','P3_final_score','B0_prediction','P3_prediction','crosses_050','crosses_070']
    errors = pd.DataFrame(errors, columns=error_columns)
    errors.to_csv(OUT/'short_answer_error_analysis.csv', index=False)
    for m in ['macro_f1', 'accuracy']:
        plt.figure(figsize=(6,4))
        plt.bar(summary.architecture, summary[f'{m}_mean'], yerr=summary[f'{m}_std'], capsize=5, color='#4477AA')
        plt.ylim(0,1)
        plt.ylabel(m.replace('_',' ').title())
        plt.title('Controlled short-answer diagnostic: five seeds')
        savefig(f'short_answer_{m}')
    for arch in ['B0', 'P3']:
        cm = np.mean([x['matrix'] for x in matrices if x['architecture']==arch], axis=0)
        fig, ax = plt.subplots(figsize=(5,4))
        im = ax.imshow(cm, cmap='Blues', vmin=0, vmax=len(data))
        ax.set(xticks=range(3), yticks=range(3), xticklabels=LABELS, yticklabels=LABELS, xlabel='Predicted label', ylabel='True label', title=f'{arch}: mean counts across five seeds')
        for i in range(3):
            for j in range(3):
                ax.text(j,i,f'{cm[i,j]:.1f}',ha='center',va='center')
        fig.colorbar(im, ax=ax)
        savefig(f'confusion_matrix_{arch}')
    return data, summary, metrics, errors

def report(df, summary, statistics, counts, downstream_result, dataset_path):
    by = summary.set_index('architecture')
    improved = [m for m in METRICS if (by.loc['P3',f'{m}_mean']-by.loc['B0',f'{m}_mean'])*(1 if m in METRICS[:2] else -1)>0]
    better_b = [m for m in METRICS if m not in improved]
    text = ['# Final post-experiment evaluation', '## 1. Experimental setup', 'Thirty completed controlled runs; seeds 13, 21, 42, 55, 87. No model retraining or checkpoint writes. STS-B test metrics read directly from aggregate_metrics.csv. All runs use the multilingual MiniLM backbone, max length 128, CoSENT loss, four epochs, batch size 16 and learning rate 2e-5. Checkpoints are best.pt selected by validation Spearman. Correlations use raw cosine; regression errors use (cosine + 1)/2 against normalized labels, following the original experiment. Smoke comparisons are excluded.', '## 2. Compared architectures', '\n'.join(f'- {k}: {v}' for k,v in NAMES.items()), 'Exact checkpoint paths and parameter verification: [checkpoint audit](checkpoint_audit.csv).', '## 3. Five-seed STS-B comparison', (OUT/'final_model_comparison.md').read_text(encoding='utf-8'), '## 4. B0 vs HALP-ST', f'P3 improves mean {", ".join(improved)}. B0 remains better on mean {", ".join(better_b)}. These are descriptive comparisons, not proof of superiority. [Per-seed values and deltas](b0_vs_halpst_per_seed.csv).', '## 5. Statistical analysis', (OUT/'b0_vs_halpst_statistics.md').read_text(encoding='utf-8'), f'Holm-adjusted significant metrics at 0.05: {", ".join(statistics.loc[statistics.significant_holm_005,"metric"]) or "none"}. Non-significance is not evidence of equivalence. Statistical power is limited with n=5.', '## 6. Ablation analysis']
    for left,right in [('B0','B1'),('B0','B2'),('B0','P1'),('P1','P2'),('P2','P3')]:
        text.append(f'{left} → {right} mean changes: '+', '.join(f'{m} {by.loc[right,f"{m}_mean"]-by.loc[left,f"{m}_mean"]:+.6f}' for m in METRICS)+'.')
    text.insert(3, '**Checkpoint provenance:** train.py evaluates the final in-memory model after training; trainer.py saves best.pt but does not reload it before returning. Therefore the supplied aggregate STS-B results describe final-epoch weights (last.pt), while the new downstream and latency evaluations use validation-selected best.pt consistently. These are distinct evaluation stages; the aggregate metrics are not presented as best.pt metrics.')
    text += ['P2 → P3 adds residual mean and projection/LayerNorm together, so that contrast cannot isolate their individual effects. Ablation differences are descriptive; no additional significance claims are made.', '## 7. Parameter efficiency', markdown(counts), 'Counts verified from all 30 checkpoints against instantiated named parameters and saved summaries. Parameter overhead does not predict latency directly.', '## 8. Inference efficiency']
    latency = OUT/'inference_efficiency.md'
    text += [latency.read_text() if latency.exists() else 'Benchmark not yet run. Run scripts/benchmark_inference.py, then rerun this script with --report-only.', '## 9. Short-answer grading evaluation']
    if latency.exists():
        timing = pd.read_csv(OUT/'inference_efficiency.csv')
        for batch, group in timing.groupby('batch_size'):
            pair = group.set_index('architecture')
            if {'B0','P3'} <= set(pair.index):
                b0, p3 = pair.loc['B0'], pair.loc['P3']
                sentence = f'Batch {batch}: B0 mean/median {b0.mean_latency_ms:.3f}/{b0.median_latency_ms:.3f} ms; P3 mean/median {p3.mean_latency_ms:.3f}/{p3.median_latency_ms:.3f} ms. P3/B0 mean latency ratio {p3.mean_latency_ms/b0.mean_latency_ms:.3f}. Throughput: B0 {b0.throughput_pairs_per_second:.3f}, P3 {p3.throughput_pairs_per_second:.3f} sentence pairs/s. These are descriptive timings for this CPU and fixed padded workload.'
                text.insert(text.index('## 9. Short-answer grading evaluation'), sentence)
                if b0.std_latency_ms / b0.mean_latency_ms > .20:
                    text.insert(text.index('## 9. Short-answer grading evaluation'), f'B0 batch {batch} timing is variable: sample SD {b0.std_latency_ms:.3f} ms, or {100*b0.std_latency_ms/b0.mean_latency_ms:.1f}% of its mean. All repetitions are retained without outlier removal. The observed speed ratio should not be interpreted as an architecture-only speed advantage, especially with fixed execution order.')
    if downstream_result:
        data, short_summary, metrics, errors = downstream_result
        synthetic = Path(dataset_path).name == 'sample_short_answers.csv'
        text += [f'Dataset: `{dataset_path}`; {len(data)} rows. Label counts: {data.true_label.value_counts().to_dict()}. '+('**The repository README identifies this dataset as synthetic and only for pipeline testing, not thesis results.** No real labeled downstream evaluation dataset was found in the repository.' if synthetic else 'Dataset supplied to this evaluation; provenance must be documented before thesis claims.'), 'The single partial-labeled sample in the bundled data says “Prices always drop when supply falls” and has numeric score 0.25; its supplied partial label is retained without relabeling. The sample warrants label review before substantive use.' if synthetic else '', 'Semantic score is raw embedding cosine, lexical score is the existing lexical/Jaccard overlap—not true concept coverage. Final = 0.70 × semantic + 0.30 × lexical; correct ≥0.70, partial ≥0.50 and <0.70, incorrect <0.50. No threshold changes. Labels are authoritative; optional numeric scores are not used to relabel answers.', '### A. Fair architecture comparison', markdown(short_summary), 'Full per-class precision, recall, F1, weighted F1 and seed results: [per-seed table](short_answer_architecture_per_seed.csv). All confusion matrices: [JSON](short_answer_confusion_matrices.json). Figures show mean counts, not an artificial 100-example independent dataset.', '### B. Practical deployment comparison', markdown(metrics[metrics.architecture.isin(['Pretrained','P3'])]), 'This contrast includes the effects of STS-B fine-tuning as well as architecture. It cannot establish an architecture-only benefit. The pretrained model is loaded using its original SentenceTransformer module stack; max length is held at 128.', '## 10. Error analysis', f'[Example-level errors](short_answer_error_analysis.csv) contain {len(errors)} seed/example rows; categories may overlap. Threshold disagreement means the two final scores lie on opposite sides of 0.50 or 0.70, without an arbitrary proximity window.']
        text += [f'{category}: {errors.categories.str.contains(category, regex=False).sum()} seed/example cases.' for category in ['HALP-ST correct and B0 wrong','B0 correct and HALP-ST wrong','both wrong','crosses 0.50','crosses 0.70']]
        text += ['The same examples occur across seeds. The one partial example is insufficient to assess partial-answer behavior; lexical overlap may penalize valid paraphrases under these fixed thresholds.']
    else:
        text += ['No labeled dataset available. Fill data/short_answer_evaluation_template.csv with expected_answer, student_answer, true_label; labels must be correct, partial, incorrect.', '## 10. Error analysis', 'Pending a labeled dataset.']
    text += ['## 11. Conclusions', f'At model level, HALP-ST has better mean {", ".join(improved)}, while B0 has better mean {", ".join(better_b)}. '+('No metric is significant after Holm adjustment.' if not statistics.significant_holm_005.any() else 'See the statistical table for significant metrics and directions.'), 'The results do not support an overall-superiority claim. STS-B model-level results and downstream grading results must remain separate. Bundled synthetic downstream results demonstrate pipeline behavior only; real labeled short answers, especially partial answers, are needed for a defensible downstream conclusion.', '## Reproduction', 'Run from the project root:\n\n```powershell\npython scripts/benchmark_inference.py\npython scripts/post_experiment_evaluation.py\n```\n\nUse `--short-answer-data path/to/data.csv` for a real labeled dataset (JSON record lists also supported). Results and figures are replaced; trained checkpoints are never written. PNG figures are 300 DPI; PDF figures are vector exports. Benchmark raw timings, inputs and environment metadata are retained.']
    if downstream_result:
        short = downstream_result[1].set_index('architecture')
        detail = 'Downstream diagnostic: ' + '; '.join(f'{arch} accuracy {short.loc[arch,"accuracy_mean"]:.4f} ± {short.loc[arch,"accuracy_std"]:.4f}, Macro-F1 {short.loc[arch,"macro_f1_mean"]:.4f} ± {short.loc[arch,"macro_f1_std"]:.4f}' for arch in ['B0','P3']) + '. These sample results do not establish downstream superiority or statistical significance.'
        text.insert(text.index('## Reproduction'), detail)
        partial = 'Partial-answer behavior: ' + '; '.join(f'{arch} mean partial precision {short.loc[arch,"partial_precision_mean"]:.4f}, recall {short.loc[arch,"partial_recall_mean"]:.4f}, F1 {short.loc[arch,"partial_f1_mean"]:.4f}' for arch in ['B0','P3']) + '. High recall on the single bundled partial example is compatible with very low precision: many non-partial answers receive partial predictions. This is not evidence of reliable partial-answer grading.'
        if Path(dataset_path).name == 'sample_short_answers.csv':
            text.insert(text.index('## 11. Conclusions'), partial)
        pretrained = downstream_result[2].query('architecture == "Pretrained"').iloc[0]
        text.insert(text.index('## Reproduction'), f'The separate practical comparison gives the fixed pretrained model accuracy {pretrained.accuracy:.4f} and Macro-F1 {pretrained.macro_f1:.4f}. Differences here conflate fine-tuning and architecture; no downstream statistical significance is asserted.')
    (OUT/'final_evaluation_report.md').write_text('\n\n'.join(text)+'\n', encoding='utf-8')

from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--short-answer-data', default=str(ROOT/'data/sample_short_answers.csv'))
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    FIG.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    df = runs()
    if args.report_only:
        summary = pd.read_csv(OUT/'final_model_comparison.csv')
        statistics = pd.read_csv(OUT/'b0_vs_halpst_statistics.csv')
        if Path(args.short_answer_data).exists():
            manifest = json.loads((OUT/'short_answer_dataset_manifest.json').read_text())
            if manifest['sha256'] != hashlib.sha256(Path(args.short_answer_data).read_bytes()).hexdigest():
                raise ValueError('Dataset changed: rerun full downstream evaluation before regenerating report')
        counts = pd.read_csv(OUT/'model_complexity.csv')
        downstream_result = (answers(args.short_answer_data), pd.read_csv(OUT/'short_answer_architecture_summary.csv'), pd.concat([pd.read_csv(OUT/'short_answer_architecture_per_seed.csv'),pd.read_csv(OUT/'short_answer_deployment_comparison.csv').query('architecture == "Pretrained"')]), pd.read_csv(OUT/'short_answer_error_analysis.csv')) if Path(args.short_answer_data).exists() else None
    else:
        summary, statistics = sts_analysis(df)
        print('Verifying all 30 checkpoint parameter counts', flush=True)
        counts = complexity(df)
        downstream_result = downstream(df, args.short_answer_data) if Path(args.short_answer_data).exists() else None
    if not args.report_only:
        pd.DataFrame(columns=['expected_answer','student_answer','true_label']).to_csv(ROOT/'data/short_answer_evaluation_template.csv', index=False)
        (ROOT/'data/short_answer_evaluation_README.md').write_text('Required columns: expected_answer, student_answer, true_label. All must be nonempty. Labels: correct, partial, incorrect (one per row). Optional question_id and score columns are retained as metadata only. Existing reference_answer and label aliases are accepted. Supply independently labeled real answers; do not use the bundled synthetic sample as thesis evidence.\n')
    report(df, summary, statistics, counts, downstream_result, args.short_answer_data)
    if (OUT/'checkpoint_audit.csv').exists():
        for row in pd.read_csv(OUT/'checkpoint_audit.csv').itertuples():
            stat = (ROOT/row.checkpoint).stat()
            if stat.st_size != row.bytes or stat.st_mtime_ns != row.mtime_ns:
                raise RuntimeError(f'Checkpoint metadata changed: {row.checkpoint}')
    print('Evaluation complete; checkpoint sizes and modification times unchanged.', flush=True)

if __name__ == '__main__':
    main()
