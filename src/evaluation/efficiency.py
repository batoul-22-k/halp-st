from __future__ import annotations

import os
import tempfile
import time

import torch


def parameter_counts(model, baseline_total: int | None = None) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    out = {"total_parameters": total, "trainable_parameters": trainable}
    if baseline_total:
        out["additional_parameters"] = total - baseline_total
        out["parameter_increase_pct"] = (total - baseline_total) / baseline_total * 100.0
    return out


def checkpoint_size_mb(model) -> float:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
        path = tmp.name
    try:
        torch.save(model.state_dict(), path)
        return os.path.getsize(path) / (1024 * 1024)
    finally:
        if os.path.exists(path):
            os.remove(path)


@torch.no_grad()
def inference_latency(model, sentences, batch_size: int = 16, device: str = "cpu") -> dict:
    model.to(device)
    model.eval()
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    _ = model.encode(sentences, batch_size=batch_size, device=device, return_tensor=True)
    elapsed = time.perf_counter() - start
    n = len(sentences)
    out = {
        "mean_latency_per_sentence_ms": elapsed / max(n, 1) * 1000,
        "sentences_per_second": n / elapsed if elapsed > 0 else 0.0,
    }
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        out["peak_gpu_memory_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)
    return out

