import numpy as np

from src.data.short_answers import load_short_answer_csv
from src.data.stsb import normalize_stsb_scores


def test_stsb_normalizes_once():
    raw = np.array([0.0, 2.5, 5.0])
    norm = normalize_stsb_scores(raw)
    assert np.allclose(norm, [0.0, 0.5, 1.0])
    assert np.allclose(normalize_stsb_scores(np.array([0.0, 0.5, 1.0])), [0.0, 0.5, 1.0])


def test_short_answer_loader():
    df = load_short_answer_csv("data/sample_short_answers.csv")
    assert {"reference_answer", "student_answer", "score", "label"}.issubset(df.columns)
    assert df["score"].dropna().between(0, 1).all()

