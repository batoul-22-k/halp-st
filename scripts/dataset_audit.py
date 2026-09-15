from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.stsb import load_stsb, pairs_to_frame


def audit(df: pd.DataFrame) -> dict:
    pairs = df[["sentence1", "sentence2"]].fillna("")
    pair_keys = pairs.apply(lambda r: (r["sentence1"], r["sentence2"]), axis=1)
    reversed_keys = set((b, a) for a, b in pair_keys)
    return {
        "rows": len(df),
        "duplicate_sentence_pairs": int(pair_keys.duplicated().sum()),
        "reversed_duplicate_pairs": int(sum(k in reversed_keys for k in pair_keys)),
        "empty_sentences": int(((pairs["sentence1"].str.len() == 0) | (pairs["sentence2"].str.len() == 0)).sum()),
        "nan_labels": int(df["label"].isna().sum()),
        "label_distribution": df["label"].describe().to_dict(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train")
    args = parser.parse_args()
    df = pairs_to_frame(load_stsb(args.split))
    print(audit(df))


if __name__ == "__main__":
    main()
