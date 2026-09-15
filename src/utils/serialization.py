from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd
import torch
import yaml


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_yaml(obj, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def save_history(rows: list[dict], path: str | Path) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def save_checkpoint(model, path: str | Path, config: dict | None = None) -> None:
    payload = {"state_dict": model.state_dict(), "config": config or {}}
    torch.save(payload, path)

