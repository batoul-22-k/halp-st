# Post-experiment evaluation

Run from `D:\halp-st`:

```powershell
python scripts/benchmark_inference.py
python scripts/post_experiment_evaluation.py
python -m pytest tests/test_post_experiment_evaluation.py tests/test_application_scoring.py -q
```

Both scripts only load checkpoints; neither imports or invokes a training entry point. The aggregate is validated as exactly six architectures × five matching seeds. Smoke files are never consumed. Existing trained run directories are not output destinations.

The CPU benchmark defaults to all six architectures, seed 13, batch sizes 1 and 16, four CPU threads, five warmups and thirty timed repetitions. It uses the first 16 existing STS-B prediction pairs, one shared tokenizer, and fixed padding/truncation to 128 tokens. Timed work consists of two embedding forwards and cosine similarity; loading and tokenization are excluded. Use `--repeats`, `--warmup`, `--threads`, or `--batch-sizes` to change the workload. Avoid other CPU workloads while timing.

The analysis generates tables, paired inference, checkpoint parameter verification, downstream evaluation, PNG/PDF figures and the final report. `--report-only` rebuilds only the report from existing evaluation outputs; tables, figures, templates and checkpoints remain unchanged. It requires an unchanged downstream dataset. Run the benchmark before regenerating the report so the new timing table is included.

The benchmark is resumable. Each architecture/batch-size result is committed atomically under `results/inference_runs/` immediately after all repetitions finish, and summary tables are refreshed after each commit. Existing complete runs with matching inputs, environment, checkpoint metadata and settings are skipped. Interrupted batches run again from warmup; uncommitted `.tmp` files are not accepted as completed measurements. The default order is B0, B1, B2, P1, P2, P3; `--architectures B0 B1 B2` restricts work to those models. All six remain required for the default output validator.

To recover timing and rebuild the report while preserving existing analyses:

```powershell
python scripts/benchmark_inference.py
python scripts/verify_evaluation_outputs.py
python scripts/post_experiment_evaluation.py --report-only
python scripts/preserve_evaluation_outputs.py --verify
```

The preservation verifier uses `results/evaluation_preservation_manifest.json`, created before recovery with `python scripts/preserve_evaluation_outputs.py`. It checks original artifact hashes, sizes and timestamps, and checkpoint sizes/timestamps. It excludes the final report, which is intentionally regenerated.

For real short answers:

```powershell
python scripts/post_experiment_evaluation.py --short-answer-data data/real_answers.csv
```

Required nonempty columns: `expected_answer`, `student_answer`, `true_label`. Aliases `reference_answer` and `label` are accepted. Labels must be `correct`, `partial`, or `incorrect`. JSON arrays of records are also accepted. No numeric-score conversion or threshold fitting is performed. The bundled CSV is explicitly synthetic, contains 20 examples and only one partial label, and supports diagnostic results only. All downstream output files are replaced when evaluating another dataset.

The controlled downstream comparison uses B0 and P3 `best.pt` for every matching seed. The separate practical comparison loads the original pretrained model via SentenceTransformer and evaluates it against all five P3 checkpoints. Macro scores include all three classes with zero for undefined class scores. Confusion matrices are supplied per seed; plotted matrices are mean counts across seeds.

Paired deltas are P3 minus B0. Correlations favor positive deltas; errors favor negative deltas. Confidence intervals are pointwise Student-t intervals over seed differences; t-test p-values also receive a five-metric Holm adjustment. Wilcoxon uses exhaustive sign permutations. Five seeds offer limited statistical power, and these tests do not measure uncertainty from sampling a new test dataset.

Provenance distinction: the original training code writes aggregate STS-B metrics from final-epoch weights. New inference/downstream analyses use validation-selected `best.pt`. No existing metric is silently substituted or recomputed.
