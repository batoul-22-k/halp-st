from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    rows = []
    for child_config in cfg["configs"]:
        for seed in cfg.get("seeds", [42]):
            cmd = [sys.executable, "train.py", "--config", child_config, "--seed", str(seed)]
            print("Run:", " ".join(cmd))
            completed = subprocess.run(cmd, check=True, text=True, capture_output=True)
            print(completed.stdout)
            result_dir = None
            for line in completed.stdout.splitlines():
                if line.startswith("saved:"):
                    result_dir = Path(line.split("saved:", 1)[1].strip())
            if result_dir and (result_dir / "metrics.json").exists():
                metrics = json.loads((result_dir / "metrics.json").read_text(encoding="utf-8"))
                rows.append({"config": child_config, "seed": seed, "result_dir": str(result_dir), **metrics})
    if rows:
        df = pd.DataFrame(rows)
        Path("results").mkdir(exist_ok=True)
        df.to_csv("results/aggregate_metrics.csv", index=False)
        metric_cols = [c for c in ["spearman", "pearson", "mse", "mae", "rmse"] if c in df.columns]
        aggregate = {}
        for config_name, group in df.groupby("config"):
            aggregate[config_name] = {
                metric: {"mean": float(group[metric].mean()), "std": float(group[metric].std(ddof=1))}
                for metric in metric_cols
            }
        Path("results/aggregate_metrics.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
