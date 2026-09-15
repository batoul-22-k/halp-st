# HALP-ST

HALP-ST is a standalone research project for testing whether a single-encoder Sentence Transformer can improve sentence-level semantic similarity by learning token-specific layer fusion and token-level semantic pooling while keeping the same MiniLM backbone as the baseline.

The controlled baseline is `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` with final-layer masked mean pooling and L2 normalization. The proposed model uses the same encoder and adds trainable pooling modules only.

```text
Input
  -> tokenizer
  -> multilingual MiniLM-L12 with output_hidden_states=True
  -> H1 ... H12
  -> Token-Adaptive Layer Fusion
  -> Semantic Token Gate
  -> adaptive sentence representation
       + residual final-layer mean representation
  -> projection -> LayerNorm -> L2 normalization
  -> 384-d sentence embedding
```

## Math

For each token `t` and layer `l`:

```text
e(t,l) = v^T tanh(W_layer h_t^(l) + b_layer)
alpha(t,l) = softmax_l(e(t,l))
z_t = sum_l alpha(t,l) h_t^(l)
```

Semantic token weighting:

```text
q_t = w^T tanh(W_token z_t + b_token)
beta_t = masked_softmax_t(q_t)
s_adaptive = sum_t beta_t z_t
```

Default residual fusion:

```text
s_fused = s_mean + lambda * s_adaptive
```

`lambda` starts at `0.1`. The projection is initialized near identity to preserve a representation close to the pretrained mean path at initialization. A gated residual mode is implemented but is not the default.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On Linux, macOS, or Colab, use `source .venv/bin/activate` instead.

## Smoke Test

```bash
pytest -q
python scripts/debug_forward.py
python train.py --config configs/smoke_test.yaml
python scripts/compare_models.py
```

The smoke test uses a tiny STS-B subset and one epoch. It validates the pipeline, not the hypothesis.

## Training

```bash
python train.py --config configs/baseline.yaml
python train.py --config configs/halpst.yaml
python run_experiments.py --config configs/full_experiment.yaml
```

Supported losses:

```text
cosent             continuous similarity labels in [0, 1]
cosine_mse         continuous similarity labels in [0, 1]
multiple_negatives positive pair batches only
```

## Evaluation

```bash
python evaluate.py --model-path results/<experiment>/best.pt --dataset stsb
```

Continuous evaluation reports Spearman, Pearson, MSE, MAE, and RMSE. Correlations use raw cosine similarity without clipping. Regression errors convert cosine to `[0,1]`.

## Ablations

The configs expose switches for the controlled variants:

```yaml
model:
  adaptive_layer_fusion: true
  semantic_token_gate: true
  residual_mean: true
  projection: true
```

B0 is baseline mean pooling. B1 uses `configs/baseline_max.yaml`. B2 uses `configs/baseline_attention.yaml`. P1 and P2 are included as config files. P3 is full HALP-ST.

## Outputs

Each run creates a timestamped directory under `results/` containing:

```text
config.yaml
metrics.json
predictions.csv
training_history.csv
model_summary.txt
model_summary.json
efficiency.json
best.pt
last.pt
```

`scripts/compare_models.py` writes `results/model_comparison.csv` and `results/model_comparison.md`.

## Short-Answer Data

Use CSV columns:

```csv
question_id,reference_answer,student_answer,score,label
```

`score` is optional continuous `[0,1]`; `label` is optional `correct`, `partial`, or `incorrect`. At least one must exist. `data/sample_short_answers.csv` is synthetic and only for pipeline testing, not thesis results.

The optional application scoring mode is implemented as:

```text
0.70 * semantic cosine + 0.30 * lexical Jaccard
>= 0.70 correct, >= 0.50 partial, else incorrect
```

Do not use this system-level score as evidence that the embedding model improved unless semantic-only metrics also improve.

## Attention Analysis

```bash
python scripts/inspect_attention.py --sentence "A firewall blocks unauthorized network traffic."
```

Plots are saved under `results/attention/`.

## Dataset Audit

```bash
python scripts/dataset_audit.py --split train
```

The audit reports duplicates, reversed duplicates, empty sentences, NaN labels, and label distribution. It warns through printed diagnostics and does not silently remove rows.

## Colab

```python
!git clone <your-repo-url> halp-st
%cd halp-st
!pip install -r requirements.txt
!pytest -q
!python train.py --config configs/smoke_test.yaml
```

For full experiments, run the baseline and HALP-ST configs with the same seed list and hardware budget.

## Interpretation

Use Mode A zero-shot only as a diagnostic because HALP-ST adds untrained attention modules. The primary architectural comparison is Mode B: both models fine-tuned on the same split, seed, optimizer, loss, batch size, and epoch budget. Report mean and standard deviation across seeds before making claims. Do not claim statistical significance unless you run the provided bootstrap utility or another paired test.

## Limitations

HALP-ST adds parameters and latency. It may underperform the baseline, especially without sufficient data or careful hyperparameter search. The code is designed to report that honestly so improvements can be separated from training effects, random-seed variation, and lexical system-level scoring.
