from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from scripts import run_lambda_sensitivity as runner
from src.models.halpst import HALPSTModel
from src.training import trainer
from src.utils.config import load_config
from tests.conftest import TinyEncoder, TinyTokenizer


def tiny_model():
    return HALPSTModel(encoder=TinyEncoder(hidden_size=8), tokenizer=TinyTokenizer(), max_length=8)


class Pairs:
    sentence1 = ["one", "a longer sentence", "three words here"]
    sentence2 = ["two words", "different", "another example"]
    labels = np.array([0.1, 0.8, 0.4], dtype=np.float32)
    raw_labels = labels

    def __len__(self):
        return len(self.labels)


@pytest.mark.parametrize("value", runner.INITIALIZATIONS)
def test_initialization(value):
    model = tiny_model()
    before = {name: p.clone() for name, p in model.state_dict().items()}
    runner.initialize_lambda(model, value, False)
    assert model.lambda_param.item() == pytest.approx(value)
    assert model.lambda_param.requires_grad
    assert all(torch.equal(p, before[n]) for n, p in model.state_dict().items() if n != "lambda_param")


@pytest.mark.parametrize("value", [0.0, 0.2, -0.2])
def test_diagnostic_formula_and_no_side_effects(value, monkeypatch):
    model = tiny_model()

    def encoder_forward(input_ids, **kwargs):
        hidden = torch.ones((*input_ids.shape, 8)) * 2
        return SimpleNamespace(hidden_states=tuple(hidden for _ in range(13)))

    def gate_forward(tokens, mask):
        return torch.ones((tokens.shape[0], 8)) * 3, mask / mask.sum(-1, keepdim=True)

    monkeypatch.setattr(model.encoder, "forward", encoder_forward)
    monkeypatch.setattr(model.token_gate, "forward", gate_forward)
    with torch.no_grad():
        model.lambda_param.fill_(value)
    model.train()
    model.encoder.eval()
    modes = [m.training for m in model.modules()]
    before = {n: p.clone() for n, p in model.state_dict().items()}
    rng = torch.random.get_rng_state().clone()
    ratio, count = runner.contribution_ratio(model, Pairs(), batch_size=2)
    assert ratio == pytest.approx(abs(value) * 3 / 2)
    assert count == 6
    assert modes == [m.training for m in model.modules()]
    assert torch.equal(rng, torch.random.get_rng_state())
    assert not model.token_gate._forward_hooks
    assert all(torch.equal(p, before[n]) for n, p in model.state_dict().items())
    assert all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize("frozen", [False, True])
def test_v1_final_epoch_pipeline_and_optimizer(tmp_path, monkeypatch, frozen):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model = tiny_model()
    config = runner.configuration(load_config(runner.ROOT / "configs/halpst.yaml"), 0.0 if frozen else 0.1, frozen, 42)
    monkeypatch.setattr(runner, "build_model", lambda config: model)
    monkeypatch.setattr(runner, "load_stsb", lambda split: Pairs())
    monkeypatch.setattr(runner, "environment_report", lambda: {"device": "cpu"})
    monkeypatch.setattr(trainer, "predict_pairs", lambda *args: [0.1, 0.8, 0.4])
    val_scores = iter([0.9, 0.8, 0.7, 0.6])
    monkeypatch.setattr(trainer, "regression_metrics", lambda *args: {
        "spearman": next(val_scores), "pearson": 0.5, "mse": 0.1, "mae": 0.2, "rmse": 0.3})
    optimizer_class = torch.optim.AdamW
    steps = []

    class CheckedAdamW(optimizer_class):
        def __init__(self, params, **kwargs):
            params = list(params)
            assert any(p is model.lambda_param for p in params) == (not frozen)
            assert kwargs == {"lr": 2e-5, "weight_decay": 0.01}
            super().__init__(params, **kwargs)

        def step(self, *args, **kwargs):
            result = super().step(*args, **kwargs)
            steps.append(model.lambda_param.item())
            if frozen:
                assert model.lambda_param.item() == 0
                assert model.lambda_param.grad is None
            return result

    monkeypatch.setattr(torch.optim, "AdamW", CheckedAdamW)
    output = tmp_path / "run"

    def test_predictions(current, *args):
        last = torch.load(output / "last.pt", weights_only=True)["state_dict"]
        best = torch.load(output / "best.pt", weights_only=True)["state_dict"]
        assert all(torch.equal(p, last[n]) for n, p in current.state_dict().items())
        assert any(not torch.equal(p, best[n]) for n, p in current.state_dict().items())
        return np.array([0.2, 0.6, 0.4])

    monkeypatch.setattr(runner, "predict_pairs", test_predictions)
    record = runner.run_one(config, output)
    assert len(steps) == 4
    assert record["max_val_epoch"] == 1
    assert record["val_spearman_epoch4"] == 0.6
    assert record["evaluation_epoch"] == 4
    assert runner.verify_run(output, config) == record
    if frozen:
        assert record["adaptive_contribution_ratio"] == 0
    with pytest.raises(FileExistsError):
        runner.run_one(config, output)
    # A last-epoch result cannot be relabeled as a best-checkpoint result.
    record["evaluation_checkpoint"] = "best.pt"
    (output / "run_result.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="final-epoch"):
        runner.verify_run(output, config)


def test_protocol_rejects_drift():
    config = load_config(runner.ROOT / "configs/halpst.yaml")
    runner.validate_protocol(config)
    for field, value in [("epochs", 8), ("encoder_learning_rate", 5e-6), ("early_stopping_patience", 2)]:
        changed = deepcopy(config)
        changed["training"][field] = value
        with pytest.raises(ValueError, match="protocol"):
            runner.validate_protocol(changed)


def test_aggregation_uses_validation_and_sample_std(tmp_path):
    records = []
    for value, frozen in [(0.1, False), (0.5, False), (0.0, True)]:
        for i, seed in enumerate(runner.SEEDS):
            row = {field: float(i) for field in runner.SUMMARY_FIELDS}
            row.update(condition=f"{value}", lambda_init=value, frozen_zero=frozen,
                       seed=seed, matched_baseline=value == 0.1,
                       val_spearman_epoch4=(0.8 if value == 0.1 else 0.7) + i * 0.01,
                       test_spearman=0.1 if value == 0.1 else 0.99)
            records.append(row)
    runner.aggregate(records, tmp_path, runner.SEEDS)
    summary = pd.read_csv(tmp_path / "summary.csv")
    assert (summary.n_seeds == 5).all()
    assert summary.final_lambda_std.tolist() == pytest.approx([np.std(range(5), ddof=1)] * 3)
    comparison = pd.read_csv(tmp_path / "validation_comparison.csv")
    assert comparison.loc[comparison.lambda_init == 0.1, "validation_rank"].item() == 1
    assert comparison.loc[comparison.lambda_init == 0.5, "validation_rank"].item() == 2
    assert comparison.loc[comparison.frozen_zero, "validation_rank"].isna().all()
    assert not any("test" in c for c in comparison.columns)
    assert comparison.loc[comparison.lambda_init == 0.5, "delta_vs_010_mean"].item() == pytest.approx(-0.1)
    with pytest.raises(ValueError, match="Duplicate"):
        runner.aggregate(records + [records[0]], tmp_path, runner.SEEDS)
