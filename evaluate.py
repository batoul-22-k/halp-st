from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from src.data.short_answers import load_short_answer_csv
from src.data.stsb import load_stsb
from src.evaluation.application_scoring import application_score
from src.evaluation.grading_metrics import grading_metrics
from src.evaluation.semantic_metrics import regression_metrics
from src.models import BaselineSTModel, HALPSTModel
from src.training.trainer import predict_pairs


def load_model(path: str):
    payload = torch.load(path, map_location="cpu")
    cfg = payload["config"]
    model = BaselineSTModel(cfg["model"]["backbone"], cfg["model"].get("pooling", "mean"), cfg["model"]["max_sequence_length"]) if cfg["model"]["variant"] == "baseline" else HALPSTModel(
        cfg["model"]["backbone"],
        cfg["model"]["max_sequence_length"],
        adaptive_layer_fusion=cfg["model"].get("adaptive_layer_fusion", True),
        semantic_token_gate=cfg["model"].get("semantic_token_gate", True),
        residual_mean=cfg["model"].get("residual_mean", True),
        projection=cfg["model"].get("projection", True),
        residual_mode=cfg["model"].get("residual_mode", "lambda"),
    )
    model.load_state_dict(payload["state_dict"])
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset", default="stsb")
    parser.add_argument("--split", default="test")
    parser.add_argument("--short-answer-csv", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    ckpt = Path(args.model_path)
    if ckpt.is_dir():
        ckpt = ckpt / "best.pt"
    model = load_model(str(ckpt))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.short_answer_csv:
        df = load_short_answer_csv(args.short_answer_csv)
        e1 = model.encode(df["reference_answer"].tolist(), batch_size=32, device=device, return_tensor=True)
        e2 = model.encode(df["student_answer"].tolist(), batch_size=32, device=device, return_tensor=True)
        scores = torch.sum(e1 * e2, dim=-1).numpy()
        semantic_pred = ["correct" if s >= 0.70 else "partial" if s >= 0.50 else "incorrect" for s in scores]
        system_rows = [application_score(float(s), r, a) for s, r, a in zip(scores, df["reference_answer"], df["student_answer"])]
        result = {
            "semantic_only": grading_metrics(df["label"].astype(str).str.lower(), semantic_pred),
            "system_70_semantic_30_lexical": grading_metrics(df["label"].astype(str).str.lower(), [r["label"] for r in system_rows]),
        }
        print(json.dumps(result, indent=2))
        if args.output:
            out = df.copy()
            out["semantic_cosine"] = scores
            out["semantic_pred"] = semantic_pred
            out["system_pred"] = [r["label"] for r in system_rows]
            out["system_score"] = [r["final_score"] for r in system_rows]
            out.to_csv(args.output, index=False)
        return
    data = load_stsb(args.split)
    scores = predict_pairs(model, data, 32, device)
    metrics = regression_metrics(data.labels, scores)
    print(json.dumps(metrics, indent=2))
    if args.output:
        pd.DataFrame({"label": data.labels, "cosine": scores}).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
