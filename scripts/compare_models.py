from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd


def to_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "No completed experiment metrics found."
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            vals.append(f"{val:.6g}" if isinstance(val, float) else str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main():
    rows = []
    for metrics_path in Path("results").glob("*/*metrics.json"):
        exp_dir = metrics_path.parent
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        eff_path = exp_dir / "efficiency.json"
        cfg_path = exp_dir / "config.yaml"
        eff = json.loads(eff_path.read_text(encoding="utf-8")) if eff_path.exists() else {}
        rows.append(
            {
                "experiment": exp_dir.name,
                "spearman": metrics.get("spearman"),
                "pearson": metrics.get("pearson"),
                "mae": metrics.get("mae"),
                "params": eff.get("total_parameters"),
                "latency_ms": eff.get("mean_latency_per_sentence_ms"),
            }
        )
    df = pd.DataFrame(rows).sort_values("experiment") if rows else pd.DataFrame()
    Path("results").mkdir(exist_ok=True)
    df.to_csv("results/model_comparison.csv", index=False)
    if not df.empty:
        base = df.iloc[0]
        df["delta_spearman_vs_first"] = df["spearman"] - base["spearman"]
        df["delta_pearson_vs_first"] = df["pearson"] - base["pearson"]
        df["delta_mae_vs_first"] = df["mae"] - base["mae"]
    md = to_markdown_table(df)
    Path("results/model_comparison.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
