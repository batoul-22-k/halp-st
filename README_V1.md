# Running HALP-ST V1

V1 is the preserved implementation used for the thesis experiments. This guide
describes its existing behavior; it does not change any source code, configuration,
checkpoint, or result. For the separate optimization implementation, see
[README_V2.md](README_V2.md).

## 1. Prepare the environment

Run commands from the repository root (`D:\halp-st` on this machine):

```powershell
Set-Location D:\halp-st
```

For thesis reproduction, use the original Python environment, package versions,
model and dataset caches, hardware, configuration, and seed. Existing run folders
record environment information in `model_summary.json` and `model_summary.txt`.
The requirements file does not pin versions, so a fresh installation alone does
not guarantee identical numerical results.

If setting up a new environment, these commands create a separate virtual
environment and install the project's dependencies:

```powershell
python -m venv .venv-v1
.\.venv-v1\Scripts\python.exe -m pip install -r requirements.txt
```

Use `.\.venv-v1\Scripts\python.exe` instead of `python` in the commands below
when using that environment. Otherwise, use the interpreter from your existing
research environment.

The loader uses the Hugging Face dataset `sentence-transformers/stsb` and the
backbone `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
Their first load requires network access unless they are already cached.
The runner automatically selects CUDA when available and otherwise uses CPU.
V1 enables mixed precision only on CUDA, even though its main config sets
`mixed_precision: true`.

## 2. Run the original HALP-ST experiment

```powershell
python train.py --config configs/halpst.yaml
```

This runs the full P3 HALP-ST model on STS-B with the existing configuration:

| Setting | V1 value |
| --- | --- |
| Experiment name | `halpst_p3` |
| Seed | `42` |
| Epochs | `4` |
| Batch size | `16` |
| Maximum sequence length | `128` |
| Loss | `cosent` |
| Learning rate | `2.0e-5` for all trainable parameters |
| Warmup ratio | `0.1` |
| Weight decay | `0.01` |
| Gradient accumulation | `1` |
| Encoder | Unfrozen |

The model uses token-adaptive layer fusion, semantic token weighting, residual
final-layer mean pooling (`s_mean + lambda * s_adaptive`), projection, LayerNorm,
and L2 normalization. Trainable lambda starts at `0.1` in the model code.

To select a seed without editing the YAML:

```powershell
python train.py --config configs/halpst.yaml --seed 13
```

The seed override is saved in the run's configuration and adds `_seed13` to its
experiment name.

### Optional small pipeline check

```powershell
python train.py --config configs/smoke_test.yaml
```

This performs real training for one epoch using 32 training pairs, 16 validation
pairs, and 16 test pairs, with batch size 4. It checks the pipeline and does not
produce a full thesis experiment.

## 3. Locate the outputs

Each training command writes to a timestamped directory relative to the working
directory:

```text
results/halpst_p3_YYYYMMDD_HHMMSS/
```

The console prints `saved: <directory>` when the run finishes. Outputs include:

| File | Contents |
| --- | --- |
| `config.yaml` | Effective run configuration, including a seed override |
| `training_history.csv` | Per-epoch training loss, validation metrics, and LR |
| `best.pt` | Model selected by highest validation Spearman |
| `last.pt` | Model after the final training epoch |
| `metrics.json` | Test Spearman, Pearson, MSE, MAE, and RMSE |
| `predictions.csv` | Test sentence pairs, labels, and cosine predictions |
| `efficiency.json` | Parameter counts, checkpoint size, and inference timing |
| `model_summary.json`, `model_summary.txt` | Model and environment information |
| `loss_curve.png` | Training loss plot |
| `confusion_matrix.png` | Placeholder explaining that STS-B is regression |

**V1 evaluates the final epoch model after training.** Although it saves `best.pt`
using validation Spearman, it does not restore that checkpoint before generating
`metrics.json` and `predictions.csv`. Preserve this distinction when comparing
thesis results with V2 or with a separate best-checkpoint evaluation. V1 has no
early stopping or configurable gradient clipping.

V1 timestamps have one-second resolution and its directory creation permits an
existing directory. Avoid launching identical experiment names in the same second
or deliberately reusing an existing run path. Keep all original thesis outputs.

## 4. Baselines, ablations, and multiple seeds

Run an individual comparison with the original runner and its configuration:

```powershell
python train.py --config configs/baseline.yaml
python train.py --config configs/baseline_max.yaml
python train.py --config configs/baseline_attention.yaml
python train.py --config configs/ablation_p1.yaml
python train.py --config configs/ablation_p2.yaml
```

`configs/full_experiment.yaml` specifies all six configurations, including full
HALP-ST, with seeds `13, 21, 42, 55, 87`: 30 sequential training runs.

**The batch runner overwrites `results/aggregate_metrics.csv` and
`results/aggregate_metrics.json`.** To preserve the existing thesis aggregates,
run the full suite in a separate copy of the repository, using the same research
environment and leaving the original `results/` directory untouched. From that
copy's root, run:

```powershell
python run_experiments.py --config configs/full_experiment.yaml
```

Individual `train.py` commands do not write these shared aggregate files.

## 5. Evaluate a saved V1 checkpoint

Replace the example run directory with the actual directory printed by training.
To evaluate the final epoch checkpoint on the test split:

```powershell
python evaluate.py --model-path results/halpst_p3_YYYYMMDD_HHMMSS/last.pt --dataset stsb --split test
```

To evaluate the validation-selected checkpoint separately:

```powershell
python evaluate.py --model-path results/halpst_p3_YYYYMMDD_HHMMSS/best.pt --dataset stsb --split test
```

Without `--output`, evaluation prints metrics and does not replace the saved
training reports. The evaluator uses batch size 32; the training runner uses the
configured batch size for test evaluation, so small numerical differences are
possible even for the same checkpoint. Do not use test scores to select a
checkpoint or tune hyperparameters.

## V1 and V2 entry points

| Version | Runner | HALP-ST config | Results root |
| --- | --- | --- | --- |
| V1 | `train.py` | `configs/halpst.yaml` | `results/` |
| V2 | `run_experiment_v2.py` | `configs/halpst_tuned_v2.yaml` | `results_v2/` |

Use the V1 runner with V1 configs to reproduce the original training procedure.
Keep optimization changes in V2.
