import re


def normalize_text(text: str) -> str:
    text = "" if text is None else str(text)
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def normalized_unique_tokens(text: str) -> set[str]:
    return set(normalize_text(text).split())

