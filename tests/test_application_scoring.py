from src.evaluation.application_scoring import application_score, jaccard_similarity


def test_jaccard_and_thresholds():
    assert jaccard_similarity("A B C", "a c") == 2 / 3
    assert application_score(1.0, "a b", "a b")["label"] == "correct"
    assert application_score(0.1, "a b", "x y")["label"] == "incorrect"

