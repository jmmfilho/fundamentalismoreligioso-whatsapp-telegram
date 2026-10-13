import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

SMALL_STOPWORDS = frozenset({"de", "a", "o", "que", "e", "do", "da", "em", "kk"})


@pytest.fixture(autouse=True)
def _offline_stopwords(monkeypatch):
    """Avoid downloading NLTK data during tests."""
    from fundrel import preprocessing
    monkeypatch.setattr(preprocessing, "stopwords", lambda: SMALL_STOPWORDS)
    import re
    import fundrel.liwc as liwc
    monkeypatch.setattr(liwc, "nltk_stopwords", lambda: SMALL_STOPWORDS)
    # NLTK's punkt model is not available offline; split words and punctuation
    monkeypatch.setattr(liwc, "tokenize", lambda t: re.findall(r"\w+|[^\w\s]", t.lower()))
