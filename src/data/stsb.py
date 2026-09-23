from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from datasets import load_dataset


@dataclass
class PairDataset:
    sentence1: list[str]
    sentence2: list[str]
    labels: np.ndarray
    raw_labels: np.ndarray

    def __len__(self) -> int:
        return len(self.labels)

    def subset(self, n: int | None) -> "PairDataset":
        if not n:
            return self
        n = min(n, len(self))
        return PairDataset(self.sentence1[:n], self.sentence2[:n], self.labels[:n], self.raw_labels[:n])


def _columns(ds):
    names = ds.column_names
    s1 = "sentence1" if "sentence1" in names else "sentence_A"
    s2 = "sentence2" if "sentence2" in names else "sentence_B"
    score = "score" if "score" in names else "label"
    return s1, s2, score


def normalize_stsb_scores(raw_scores) -> np.ndarray:
    raw = np.asarray(raw_scores, dtype=np.float32)
    assert np.isfinite(raw).all(), "STS labels contain NaN or Inf"
    assert raw.min() >= 0, f"STS labels below 0: {raw.min()}"
    expected_max = 5.0 if raw.max() > 1.0 else 1.0
    assert raw.max() <= expected_max + 1e-6, f"STS labels exceed expected max {expected_max}: {raw.max()}"
    return raw / expected_max


def label_range_report(raw_labels, normalized_labels) -> dict:
    return {
        "raw_label_min": float(np.min(raw_labels)),
        "raw_label_max": float(np.max(raw_labels)),
        "normalized_label_min": float(np.min(normalized_labels)),
        "normalized_label_max": float(np.max(normalized_labels)),
    }


def load_stsb(split: str, subset: int | None = None, dataset_name: str = "sentence-transformers/stsb") -> PairDataset:
    ds = load_dataset(dataset_name, split=split)
    s1, s2, score = _columns(ds)
    raw = np.asarray(ds[score], dtype=np.float32)
    labels = normalize_stsb_scores(raw)
    return PairDataset(list(ds[s1]), list(ds[s2]), labels, raw).subset(subset)


def pairs_to_frame(dataset: PairDataset) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sentence1": dataset.sentence1,
            "sentence2": dataset.sentence2,
            "label": dataset.labels,
            "raw_label": dataset.raw_labels,
        }
    )

