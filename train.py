from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import matplotlib.pyplot as plt

from src.data.stsb import label_range_report, load_stsb
from src.evaluation.efficiency import checkpoint_size_mb, inference_latency, parameter_counts
from src.evaluation.semantic_metrics import regression_metrics
from src.models import BaselineSTModel, HALPSTModel
from src.training.seed import set_seed
from src.training.trainer import predict_pairs, train_model
from src.utils.config import load_config
from src.utils.logging import environment_report, print_experiment_header
from src.utils.serialization import ensure_dir, save_history, save_json, save_yaml


def build_model(config: dict):
    model_cfg = config["model"]
    enc_cfg = config.get("encoder", {})
    if model_cfg["variant"] == "baseline":
        return BaselineSTModel(
            backbone_name=model_cfg["backbone"],
            pooling=model_cfg.get("pooling", "mean"),
            max_length=model_cfg["max_sequence_length"],
            freeze_encoder=enc_cfg.get("freeze", False),
        )
    return HALPSTModel(
        backbone_name=model_cfg["backbone"],
        max_length=model_cfg["max_sequence_length"],
        adaptive_layer_fusion=model_cfg.get("adaptive_layer_fusion", True),
        semantic_token_gate=model_cfg.get("semantic_token_gate", True),
        residual_mean=model_cfg.get("residual_mean", True),
        projection=model_cfg.get("projection", True),
        residual_mode=model_cfg.get("residual_mode", "lambda"),
        freeze_encoder=enc_cfg.get("freeze", False),
    )


def save_loss_curve(history: list[dict], path: Path) -> None:
    plt.figure(figsize=(6, 4))
    if history:
        epochs = [row["epoch"] for row in history]
        plt.plot(epochs, [row["train_loss"] for row in history], marker="o", label="train loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
    else:
        plt.text(0.5, 0.5, "zero-shot run: no training loss", ha="center", va="center")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_confusion_placeholder(path: Path) -> None:
    plt.figure(figsize=(5, 4))
    plt.text(0.5, 0.5, "Not applicable\ncontinuous STS-B regression", ha="center", va="center")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.seed is not None:
        config["training"]["seed"] = args.seed
        config["experiment_name"] = f"{config['experiment_name']}_seed{args.seed}"
    set_seed(config["training"].get("seed", 42))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = ensure_dir(Path("results") / f"{config['experiment_name']}_{stamp}")
    save_yaml(config, output_dir / "config.yaml")

    subset = config["dataset"].get("subset")
    train_data = load_stsb("train", subset=subset)
    val_data = load_stsb("validation", subset=config["dataset"].get("validation_subset", subset))
    test_data = load_stsb("test", subset=config["dataset"].get("test_subset", subset))
    ranges = label_range_report(train_data.raw_labels, train_data.labels)
    model = build_model(config)
    counts = parameter_counts(model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print_experiment_header(
        {
            "model variant": config["model"]["variant"],
            "backbone": config["model"]["backbone"],
            "dataset": config["dataset"]["name"],
            "train size": len(train_data),
            "validation size": len(val_data),
            "test size": len(test_data),
            **ranges,
            "loss": config["training"]["loss"],
            "seed": config["training"].get("seed", 42),
            "device": device,
            "batch size": config["training"]["batch_size"],
            "learning rate": config["training"]["learning_rate"],
            "epochs": config["training"]["epochs"],
            **counts,
        }
    )
    save_json({"environment": environment_report(), "label_ranges": ranges, **counts}, output_dir / "model_summary.json")

    if config.get("mode", "finetune") == "zero_shot":
        history = []
    else:
        history = train_model(model, train_data, val_data, config, output_dir)
    save_history(history, output_dir / "training_history.csv")
    save_loss_curve(history, output_dir / "loss_curve.png")
    save_confusion_placeholder(output_dir / "confusion_matrix.png")

    scores = predict_pairs(model, test_data, config["training"]["batch_size"], device)
    metrics = regression_metrics(test_data.labels, scores)
    save_json(metrics, output_dir / "metrics.json")
    pd.DataFrame(
        {"sentence1": test_data.sentence1, "sentence2": test_data.sentence2, "label": test_data.labels, "raw_label": test_data.raw_labels, "cosine": scores}
    ).to_csv(output_dir / "predictions.csv", index=False)
    eff = {**parameter_counts(model), "checkpoint_size_mb": checkpoint_size_mb(model)}
    eff.update(inference_latency(model, test_data.sentence1[: min(64, len(test_data))], config["training"]["batch_size"], device))
    save_json(eff, output_dir / "efficiency.json")
    with open(output_dir / "model_summary.txt", "w", encoding="utf-8") as f:
        f.write(str(model))
        f.write("\n\n")
        f.write(str({**counts, **environment_report()}))
    print(f"saved: {output_dir}")
    print(metrics)


if __name__ == "__main__":
    main()
