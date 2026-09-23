from src.evaluation.grading_metrics import grading_metrics
from src.evaluation.semantic_metrics import regression_metrics


def test_semantic_metrics():
    m = regression_metrics([0, 0.5, 1], [-1, 0, 1])
    assert m["spearman"] > 0.99
    assert m["mae"] == 0.0


def test_grading_metrics():
    m = grading_metrics(["correct", "partial", "incorrect"], ["correct", "incorrect", "incorrect"])
    assert m["accuracy"] == 2 / 3
    assert "per_class" in m

