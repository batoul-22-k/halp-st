from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from src.evaluation.semantic_metrics import regression_metrics
from src.losses import CoSENTLoss, CosineMSELoss, MultipleNegativesRankingLoss
from src.utils.serialization import save_checkpoint


class PairTorchDataset(Dataset):
    def __init__(self, pair_dataset):
        self.data = pair_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data.sentence1[idx], self.data.sentence2[idx], float(self.data.labels[idx])


def make_collate(tokenizer, max_length: int):
    def collate(batch):
        s1, s2, labels = zip(*batch)
        e1 = tokenizer(list(s1), padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        e2 = tokenizer(list(s2), padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        return e1, e2, torch.tensor(labels, dtype=torch.float32)

    return collate


def build_loss(name: str):
    if name == "cosent":
        return CoSENTLoss()
    if name == "cosine_mse":
        return CosineMSELoss()
    if name == "multiple_negatives":
        return MultipleNegativesRankingLoss()
    raise ValueError(f"Unsupported loss: {name}")


@torch.no_grad()
def predict_pairs(model, pair_dataset, batch_size: int, device: str):
    model.eval()
    scores = []
    for start in range(0, len(pair_dataset), batch_size):
        s1 = pair_dataset.sentence1[start : start + batch_size]
        s2 = pair_dataset.sentence2[start : start + batch_size]
        e1 = model.encode(s1, batch_size=batch_size, device=device, return_tensor=True).to(device)
        e2 = model.encode(s2, batch_size=batch_size, device=device, return_tensor=True).to(device)
        scores.extend(torch.sum(e1 * e2, dim=-1).cpu().numpy().tolist())
    return np.asarray(scores)


def train_model(model, train_data, val_data, config: dict, output_dir: Path):
    training = config["training"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_loader = DataLoader(
        PairTorchDataset(train_data),
        batch_size=training["batch_size"],
        shuffle=True,
        collate_fn=make_collate(model.tokenizer, config["model"]["max_sequence_length"]),
    )
    loss_fn = build_loss(training["loss"])
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=training["learning_rate"],
        weight_decay=training.get("weight_decay", 0.0),
    )
    total_steps = max(1, math.ceil(len(train_loader) / training.get("gradient_accumulation", 1)) * training["epochs"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * training.get("warmup_ratio", 0.0)),
        num_training_steps=total_steps,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=bool(training.get("mixed_precision", False) and device.type == "cuda"))
    best_spearman = -1e9
    history = []
    accum = training.get("gradient_accumulation", 1)

    for epoch in range(1, training["epochs"] + 1):
        model.train()
        losses = []
        optimizer.zero_grad(set_to_none=True)
        for step, (e1, e2, labels) in enumerate(tqdm(train_loader, desc=f"epoch {epoch}"), start=1):
            e1 = {k: v.to(device) for k, v in e1.items()}
            e2 = {k: v.to(device) for k, v in e2.items()}
            labels = labels.to(device)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                a = model(**e1)
                b = model(**e2)
                loss = loss_fn(a, b, labels) / accum
            scaler.scale(loss).backward()
            if step % accum == 0 or step == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach().cpu()) * accum)
        val_scores = predict_pairs(model, val_data, training["batch_size"], str(device))
        metrics = regression_metrics(val_data.labels, val_scores)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **{f"val_{k}": v for k, v in metrics.items()}, "lr": scheduler.get_last_lr()[0]}
        print(f"epoch={epoch} train_loss={row['train_loss']:.4f} val_spearman={row['val_spearman']:.4f} val_pearson={row['val_pearson']:.4f} val_mae={row['val_mae']:.4f} lr={row['lr']:.2e}")
        history.append(row)
        if metrics["spearman"] > best_spearman:
            best_spearman = metrics["spearman"]
            save_checkpoint(model, output_dir / "best.pt", config)
    save_checkpoint(model, output_dir / "last.pt", config)
    return history

