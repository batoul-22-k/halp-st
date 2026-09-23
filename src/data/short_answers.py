from __future__ import annotations

import pandas as pd

LABEL_TO_SCORE = {"incorrect": 0.0, "partial": 0.5, "correct": 1.0}


def load_short_answer_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"question_id", "reference_answer", "student_answer"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if "score" not in df.columns and "label" not in df.columns:
        raise ValueError("CSV must include at least one of score or label")
    if "score" in df.columns:
        scores = pd.to_numeric(df["score"], errors="coerce")
        if scores.notna().any():
            valid = scores.dropna()
            assert valid.min() >= 0 and valid.max() <= 1, "score must be in [0, 1]"
        df["score"] = scores
    if "label" in df.columns:
        labels = df["label"].dropna().astype(str).str.lower()
        unknown = set(labels) - set(LABEL_TO_SCORE)
        if unknown:
            raise ValueError(f"Unknown labels: {sorted(unknown)}")
    if "score" not in df.columns:
        df["score"] = df["label"].str.lower().map(LABEL_TO_SCORE)
    if "label" not in df.columns:
        df["label"] = pd.cut(df["score"], bins=[-0.01, 0.49, 0.69, 1.0], labels=["incorrect", "partial", "correct"])
    return df

