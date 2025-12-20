from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple


_STOPWORDS = {
    "the","a","an","and","or","but","if","then","so","to","of","in","on","for","with","as","at","by",
    "i","you","he","she","we","they","me","him","her","us","them","my","your","his","hers","our","their",
    "is","are","was","were","be","been","being","do","does","did","doing",
    "this","that","these","those","it","its","im","i'm","ive","i've","dont","don't","cant","can't",
    "from","into","out","up","down","over","again","really","just","like","kind","sort",
}


_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]{1,}")


def top_keywords(text: str, k: int = 25) -> List[Tuple[str, int]]:
    words = [w.lower() for w in _WORD_RE.findall(text)]
    words = [w for w in words if w not in _STOPWORDS and len(w) >= 3]
    freq = Counter(words)
    return freq.most_common(k)


def naive_entities(text: str, k: int = 25) -> List[Tuple[str, int]]:
    """
    Naive entity extraction: sequences of Capitalized Words (2+ chars).
    Works surprisingly well for transcripts without installing spaCy.
    """
    candidates: List[str] = []
    tokens = re.findall(r"[A-Za-z][A-Za-z'\-]*", text)

    buf: List[str] = []
    for t in tokens:
        if len(t) >= 2 and t[0].isupper():
            buf.append(t)
        else:
            if buf:
                candidates.append(" ".join(buf))
                buf = []
    if buf:
        candidates.append(" ".join(buf))

    # Normalize + count
    cleaned = [c.strip() for c in candidates if c.strip()]
    freq = Counter(cleaned)
    return freq.most_common(k)


def extractive_summary(text: str, max_sentences: int = 8) -> str:
    """
    Cheap extractive summary:
      - split into sentences
      - score by keyword frequency
      - take top N in original order
    """
    # sentence split
    sents = re.split(r"(?<=[.!?])\s+", text.replace("\n", " ").strip())
    sents = [s.strip() for s in sents if len(s.strip()) > 0]

    if not sents:
        return ""

    kw = dict(top_keywords(text, k=60))

    def score(sent: str) -> int:
        words = [w.lower() for w in _WORD_RE.findall(sent)]
        return sum(kw.get(w, 0) for w in words)

    scored = [(i, s, score(s)) for i, s in enumerate(sents)]
    scored.sort(key=lambda x: x[2], reverse=True)
    keep = sorted(scored[: max(1, max_sentences)], key=lambda x: x[0])
    return "\n".join(f"- {s}" for _, s, _ in keep)
