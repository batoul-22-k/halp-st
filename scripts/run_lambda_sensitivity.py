"""Isolated Table 5.4 V1 sensitivity sweep, with a seed-42 pilot gate."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from train import build_model
from src.data.stsb import load_stsb
from src.evaluation.semantic_metrics import regression_metrics
from src.models.pooling import masked_mean_pool
from src.training.seed import set_seed
from src.training.trainer import predict_pairs, train_model
from src.utils.config import load_config
from src.utils.logging import environment_report
from src.utils.serialization import save_history, save_json, save_yaml

OUTPUT_ROOT = ROOT / "results_lambda_sensitivity"
INITIALIZATIONS = [0.01, 0.05, 0.10, 0.20, 0.50, 1.00]
SEEDS = [13, 21, 42, 55, 87]
METRICS = ["spearman", "pearson", "mse", "mae", "rmse"]
SUMMARY_FIELDS = ["final_lambda", "val_spearman_epoch4", "max_val_spearman",
                  "max_val_epoch", "adaptive_contribution_ratio", "training_seconds",
                  *[f"test_{m}" for m in METRICS]]
EXPECTED = {
    "mode": "finetune", "dataset": {"name": "stsb"},
    "model": {"variant": "halpst",
              "backbone": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
              "max_sequence_length": 128, "adaptive_layer_fusion": True,
              "semantic_token_gate": True, "residual_mean": True,
              "projection": True, "residual_mode": "lambda"},
    "encoder": {"freeze": False},
    "training": {"loss": "cosent", "epochs": 4, "batch_size": 16,
                 "learning_rate": 2e-5, "warmup_ratio": 0.1, "weight_decay": 0.01,
                 "gradient_accumulation": 1, "mixed_precision": True},
}


def validate_protocol(config):
    actual = deepcopy(config)
    actual.pop("experiment_name", None)
    actual.pop("sensitivity", None)
    actual.get("training", {}).pop("seed", None)
    if actual != EXPECTED:
        raise ValueError("Configuration must exactly match the Table 5.4 V1 protocol")


def initialize_lambda(model, value, frozen):
    if not math.isfinite(value) or value < 0 or (frozen and value != 0):
        raise ValueError("Invalid lambda initialization/frozen-zero condition")
    with torch.no_grad():
        model.lambda_param.fill_(value)
    model.lambda_param.requires_grad_(not frozen)
    return model.lambda_param.detach().item()


@torch.no_grad()
def contribution_ratio(model, validation, batch_size=16):
    """Mean of sentence-wise ratios; repeated sentence occurrences retain weight."""
    modes = [(module, module.training) for module in model.modules()]
    device = next(model.parameters()).device
    captured = {}

    def capture(module, inputs, output):
        captured["adaptive"] = output[0].detach()

    handle = model.token_gate.register_forward_hook(capture)
    total, count = 0.0, 0
    try:
        model.eval()
        devices = [device.index or 0] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            for sentences in (validation.sentence1, validation.sentence2):
                for start in range(0, len(sentences), batch_size):
                    batch = {k: v.to(device) for k, v in
                             model._tokenize(sentences[start:start + batch_size]).items()}
                    analysis = model(**batch, return_analysis=True)
                    mean = masked_mean_pool(analysis["hidden_states"][-1], batch["attention_mask"])
                    denominator = torch.linalg.vector_norm(mean, dim=-1)
                    if torch.any(denominator == 0):
                        raise ValueError("Undefined contribution ratio: zero mean-vector norm")
                    numerator = torch.linalg.vector_norm(model.lambda_param * captured["adaptive"], dim=-1)
                    values = numerator / denominator
                    if not torch.isfinite(values).all():
                        raise ValueError("Nonfinite adaptive contribution ratio")
                    total += values.double().sum().item()
                    count += values.numel()
    finally:
        handle.remove()
        for module, mode in modes:
            module.training = mode
    if count != 2 * len(validation) or count == 0:
        raise ValueError("Validation diagnostic did not cover both sides of all pairs")
    return total / count, count


def source_hashes():
    paths = ["train.py", "src/training/trainer.py", "src/models/halpst.py",
             "src/models/layer_fusion.py", "src/models/token_gate.py", "src/models/pooling.py",
             "src/losses/cosent.py", "src/data/stsb.py", "src/training/seed.py",
             "src/evaluation/semantic_metrics.py", "scripts/run_lambda_sensitivity.py"]
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths}


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run_one(config, output):
    validate_protocol(config)
    output.mkdir(parents=True, exist_ok=False)
    save_yaml(config, output / "config.yaml")
    spec = config["sensitivity"]
    seed = config["training"]["seed"]
    set_seed(seed)
    # Match V1's data-loading order before constructing the model.
    train_data, validation, test = [load_stsb(split) for split in ("train", "validation", "test")]
    model = build_model(config)
    observed = initialize_lambda(model, spec["lambda_init"], spec["frozen_zero"])
    eligible = any(p is model.lambda_param for p in model.parameters() if p.requires_grad)
    if eligible == spec["frozen_zero"]:
        raise ValueError("Incorrect lambda optimizer eligibility")
    save_json({**environment_report(), "torch_threads": torch.get_num_threads(),
               "source_sha256": source_hashes(), "argv": sys.argv,
               "backbone_revision": getattr(model.encoder.config, "_commit_hash", None),
               "train_pairs": len(train_data), "validation_pairs": len(validation),
               "test_pairs": len(test)}, output / "environment.json")
    save_json({**spec, "seed": seed, "observed_initial_lambda": observed,
               "lambda_requires_grad": model.lambda_param.requires_grad,
               "lambda_optimizer_eligible": eligible}, output / "initialization.json")
    print(f"START seed={seed} lambda_init={observed:.17g} frozen_zero={spec['frozen_zero']}", flush=True)
    synchronize()
    started = time.perf_counter()
    history = train_model(model, train_data, validation, config, output)
    synchronize()
    training_seconds = time.perf_counter() - started
    save_history(history, output / "training_history.csv")
    if [r["epoch"] for r in history] != [1, 2, 3, 4]:
        raise ValueError("V1 must execute exactly four epochs")
    last = torch.load(output / "last.pt", map_location="cpu", weights_only=True, mmap=True)
    if last["config"] != config or not all(
        torch.equal(p.detach().cpu(), last["state_dict"][name]) for name, p in model.state_dict().items()
    ):
        raise ValueError("Final in-memory model does not match last.pt")
    final_lambda = last["state_dict"]["lambda_param"].item()
    del last
    # No checkpoint is restored. Test predictions use the unchanged V1 final model.
    scores = predict_pairs(model, test, 16, str(next(model.parameters()).device))
    metrics = regression_metrics(test.labels, scores)
    save_json(metrics, output / "metrics.json")
    pd.DataFrame({"sentence1": test.sentence1, "sentence2": test.sentence2,
                  "label": test.labels, "raw_label": test.raw_labels,
                  "cosine": scores}).to_csv(output / "predictions.csv", index=False)
    ratio, occurrences = contribution_ratio(model, validation)
    maximum = max(history, key=lambda row: row["val_spearman"])
    record = {"condition": spec["condition"], "lambda_init": spec["lambda_init"],
              "frozen_zero": spec["frozen_zero"], "seed": seed,
              "matched_baseline": spec["lambda_init"] == 0.1 and not spec["frozen_zero"],
              "final_lambda": final_lambda, "val_spearman_epoch4": history[-1]["val_spearman"],
              "max_val_spearman": maximum["val_spearman"], "max_val_epoch": maximum["epoch"],
              **{f"test_{m}": metrics[m] for m in METRICS},
              "training_seconds": training_seconds, "adaptive_contribution_ratio": ratio,
              "diagnostic_sentence_occurrences": occurrences, "epochs_executed": len(history),
              "evaluation_epoch": 4, "evaluation_checkpoint": "last.pt",
              "final_state_verified": True, "run_directory": str(output.resolve())}
    if not all(math.isfinite(record[k]) for k in SUMMARY_FIELDS):
        raise ValueError("Nonfinite result")
    if spec["frozen_zero"] and (final_lambda != 0 or ratio != 0):
        raise ValueError("Frozen-zero contribution must remain exactly zero")
    save_json(record, output / "run_result.json")
    print(json.dumps(record), flush=True)
    return record


def verify_run(output, expected):
    config = load_config(output / "config.yaml")
    if config != expected:
        raise ValueError(f"Configuration mismatch: {output}")
    result = json.loads((output / "run_result.json").read_text())
    initial = json.loads((output / "initialization.json").read_text())
    history = pd.read_csv(output / "training_history.csv")
    spec = expected["sensitivity"]
    frozen = spec["frozen_zero"]
    if history.epoch.tolist() != [1, 2, 3, 4] or result["evaluation_checkpoint"] != "last.pt":
        raise ValueError("Incorrect final-epoch protocol")
    if not result["final_state_verified"] or result["evaluation_epoch"] != 4:
        raise ValueError("Missing final-state verification")
    if initial["lambda_requires_grad"] != (not frozen) or initial["lambda_optimizer_eligible"] != (not frozen):
        raise ValueError("Incorrect lambda freezing")
    if not math.isclose(initial["observed_initial_lambda"], spec["lambda_init"], rel_tol=1e-6, abs_tol=0):
        raise ValueError("Incorrect initialization")
    if result["seed"] != expected["training"]["seed"] or result["lambda_init"] != spec["lambda_init"] or result["frozen_zero"] != frozen:
        raise ValueError("Incorrect run identity")
    for checkpoint in ("best.pt", "last.pt"):
        payload = torch.load(output / checkpoint, map_location="cpu", weights_only=True, mmap=True)
        value = payload["state_dict"]["lambda_param"].item()
        if payload["config"] != expected or (frozen and value != 0):
            raise ValueError("Checkpoint configuration or frozen lambda mismatch")
        if checkpoint == "last.pt" and value != result["final_lambda"]:
            raise ValueError("Logged lambda differs from last.pt")
        del payload
    if not frozen and result["final_lambda"] == initial["observed_initial_lambda"]:
        raise ValueError("Trainable lambda did not change; inspect this run")
    if frozen and result["adaptive_contribution_ratio"] != 0:
        raise ValueError("Frozen-zero diagnostic must equal zero")
    environment = json.loads((output / "environment.json").read_text())
    if result["diagnostic_sentence_occurrences"] != 2 * environment["validation_pairs"]:
        raise ValueError("Incomplete validation diagnostic")
    expected_numbers = {"val_spearman_epoch4": history.iloc[-1].val_spearman,
                        "max_val_spearman": history.val_spearman.max(),
                        "max_val_epoch": int(history.loc[history.val_spearman.idxmax(), "epoch"])}
    metrics = json.loads((output / "metrics.json").read_text())
    expected_numbers.update({f"test_{m}": metrics[m] for m in METRICS})
    if not all(math.isclose(result[k], v, rel_tol=1e-12, abs_tol=1e-12) for k, v in expected_numbers.items()):
        raise ValueError("Logged metrics differ from source artifacts")
    if not all(math.isfinite(result[k]) for k in SUMMARY_FIELDS) or result["training_seconds"] <= 0:
        raise ValueError("Invalid numerical result")
    return result


def aggregate(records, output, seeds):
    frame = pd.DataFrame(records).sort_values(["frozen_zero", "lambda_init", "seed"])
    if frame.duplicated(["condition", "seed"]).any():
        raise ValueError("Duplicate condition/seed")
    frame.to_csv(output / "all_runs.csv", index=False)
    summaries, comparisons = [], []
    baseline = frame.loc[frame.matched_baseline].set_index("seed")
    for condition, group in frame.groupby("condition", sort=True):
        first = group.iloc[0]
        complete = sorted(group.seed.tolist()) == sorted(seeds)
        row = {"condition": condition, "lambda_init": first.lambda_init,
               "frozen_zero": bool(first.frozen_zero), "n_seeds": len(group),
               "complete": complete, "matched_baseline": bool(first.matched_baseline)}
        for field in SUMMARY_FIELDS:
            row[f"{field}_mean"] = group[field].mean()
            row[f"{field}_std"] = group[field].std(ddof=1)
        summaries.append(row)
        paired = group.set_index("seed")[["val_spearman_epoch4"]].join(
            baseline[["val_spearman_epoch4"]], lsuffix="_condition", rsuffix="_baseline", how="inner")
        differences = paired.val_spearman_epoch4_condition - paired.val_spearman_epoch4_baseline
        comparisons.append({k: row[k] for k in ["condition", "lambda_init", "frozen_zero", "n_seeds", "complete"]} |
                           {"val_spearman_epoch4_mean": row["val_spearman_epoch4_mean"],
                            "val_spearman_epoch4_std": row["val_spearman_epoch4_std"],
                            "paired_seeds": len(paired), "delta_vs_010_mean": differences.mean(),
                            "delta_vs_010_std": differences.std(ddof=1)})
    pd.DataFrame(summaries).to_csv(output / "summary.csv", index=False)
    comparison = pd.DataFrame(comparisons)
    comparison["validation_rank"] = np.nan
    eligible = comparison.complete & ~comparison.frozen_zero
    comparison.loc[eligible, "validation_rank"] = comparison.loc[eligible, "val_spearman_epoch4_mean"].rank(
        ascending=False, method="min")
    comparison.sort_values(["frozen_zero", "validation_rank", "lambda_init"]).to_csv(
        output / "validation_comparison.csv", index=False)


def configuration(base, value, frozen, seed):
    config = deepcopy(base)
    condition = "frozen_zero" if frozen else f"lambda_{value:.2f}"
    config["experiment_name"] = f"sensitivity_{condition}_seed{seed}"
    config["training"]["seed"] = seed
    config["sensitivity"] = {"condition": condition, "lambda_init": value, "frozen_zero": frozen}
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", default="configs/halpst.yaml")
    parser.add_argument("--lambda-inits", type=float, nargs="+", default=INITIALIZATIONS)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--include-frozen-zero", action="store_true")
    parser.add_argument("--pilot-only", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--worker-config", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_config:
        run_one(load_config(args.worker_config), args.worker_output)
        return
    if args.lambda_inits != INITIALIZATIONS or args.seeds != SEEDS or not args.include_frozen_zero:
        parser.error("This controlled sweep requires all six initializations, five specified seeds, and frozen zero")
    source = (ROOT / args.base_config).resolve()
    base = load_config(source)
    validate_protocol(base)
    manifest = {"protocol": "Table5.4_V1_final_epoch4", "base_config": base,
                "lambda_inits": args.lambda_inits, "seeds": args.seeds, "include_frozen_zero": True,
                "pilot_seed": 42, "source_sha256": source_hashes(),
                "ranking": "mean epoch-4 validation Spearman; frozen zero separate",
                "diagnostic": "mean per-sentence norm(lambda*adaptive)/norm(mean), both validation sides, occurrences weighted",
                "timing": "V1 train_model wall time including validation/checkpoint writes, excluding loading/test/diagnostic"}
    if args.resume:
        output = args.resume.resolve()
        if output.parent != OUTPUT_ROOT.resolve():
            raise ValueError("Resume path must be an existing sensitivity sweep directory")
        if json.loads((output / "manifest.json").read_text()) != manifest:
            raise ValueError("Resume manifest/source mismatch")
    else:
        output = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output.mkdir(parents=True, exist_ok=False)
        (output / "configs").mkdir()
        (output / "logs").mkdir()
        (output / "runs").mkdir()
        save_json(manifest, output / "manifest.json")
        (output / "base_config.yaml").write_bytes(source.read_bytes())
    print(f"SWEEP_DIRECTORY={output}", flush=True)
    conditions = [(value, False) for value in args.lambda_inits] + [(0.0, True)]
    records = []

    def execute(value, frozen, seed):
        config = configuration(base, value, frozen, seed)
        name = config["experiment_name"]
        run_dir = output / "runs" / name
        config_path = output / "configs" / f"{name}.yaml"
        if not (run_dir / "run_result.json").exists():
            if run_dir.exists():
                raise ValueError(f"Incomplete run preserved at {run_dir}; start a new sweep instead of overwriting")
            save_yaml(config, config_path)
            command = [sys.executable, "-B", "-u", str(Path(__file__).resolve()),
                       "--worker-config", str(config_path), "--worker-output", str(run_dir)]
            print(f"RUN {name}", flush=True)
            with (output / "logs" / f"{name}.log").open("x", encoding="utf-8") as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        record = verify_run(run_dir, config)
        records.append(record)
        aggregate(records, output, args.seeds)
        print(f"VERIFIED {name} final_lambda={record['final_lambda']:.9g} ratio={record['adaptive_contribution_ratio']:.9g}", flush=True)

    for value, frozen in conditions:
        execute(value, frozen, 42)
    save_json({"passed": True, "seed": 42, "conditions_verified": 7,
               "checks": ["initialization", "optimizer eligibility", "four epochs", "last.pt state equality",
                          "last.pt lambda", "frozen zero", "validation diagnostic", "artifact consistency"],
               "test_metrics_used_for_selection": False}, output / "pilot_validation.json")
    print("PILOT PASSED: all seven seed-42 runs verified", flush=True)
    if args.pilot_only:
        return
    for seed in args.seeds:
        if seed != 42:
            for value, frozen in conditions:
                execute(value, frozen, seed)
    save_json({"complete": True, "runs": len(records), "evaluation_checkpoint": "last.pt",
               "evaluation_epoch": 4, "selection_metric": "val_spearman_epoch4"}, output / "completion.json")
    print(f"COMPLETE: {len(records)} runs at {output}", flush=True)


if __name__ == "__main__":
    main()
