from __future__ import annotations

import platform

import torch


def environment_report() -> dict:
    try:
        import transformers
    except Exception:
        transformers = None
    try:
        import sentence_transformers
    except Exception:
        sentence_transformers = None
    return {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "transformers": getattr(transformers, "__version__", "unavailable"),
        "sentence_transformers": getattr(sentence_transformers, "__version__", "unavailable"),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def print_experiment_header(info: dict) -> None:
    for key, value in info.items():
        print(f"{key}: {value}")

