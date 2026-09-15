from __future__ import annotations

from src.data.preprocessing import normalized_unique_tokens


def jaccard_similarity(a: str, b: str) -> float:
    ta = normalized_unique_tokens(a)
    tb = normalized_unique_tokens(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def application_score(semantic_score: float, reference: str, answer: str) -> dict:
    lexical = jaccard_similarity(reference, answer)
    final = 0.70 * semantic_score + 0.30 * lexical
    if final >= 0.70:
        label = "correct"
    elif final >= 0.50:
        label = "partial"
    else:
        label = "incorrect"
    return {"semantic_score": semantic_score, "lexical_score": lexical, "final_score": final, "label": label}

