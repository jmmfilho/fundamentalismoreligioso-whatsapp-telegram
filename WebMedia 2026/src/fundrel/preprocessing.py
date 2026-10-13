"""Text preprocessing shared by every analysis.

This is the single source of truth for the `preprocess_text` function that was
duplicated in each original notebook. Its behaviour is kept identical to the
paper run (URL -> domain, lowercase, accent removal, punctuation removal,
emoji/laughter normalisation, stopword removal).
"""
from __future__ import annotations

import re
import string
from functools import lru_cache

from unidecode import unidecode

from . import config

_URL = re.compile(r"https?://(?:www\.)?\S+|www\.\S+")
_PUNCT = str.maketrans("", "", string.punctuation)
_EMOJI_REPEAT = re.compile(r"([\U00010000-\U0010FFFF])\1+")
_EMOJI_BLOCK_REPEAT = re.compile(
    r"([\U0001F600-\U0001F64F]|[\U0001F300-\U0001F5FF]|[\U0001F680-\U0001F6FF]"
    r"|[\U0001F700-\U0001F77F]|[\U0001F780-\U0001F7FF]|[\U0001F800-\U0001F8FF]"
    r"|[\U0001F900-\U0001F9FF]|[\U0001FA00-\U0001FA6F]|[\U0001FA70-\U0001FAFF])\1+"
)


@lru_cache(maxsize=1)
def nltk_stopwords() -> frozenset[str]:
    """NLTK's Portuguese stopword list (accented)."""
    try:
        from nltk.corpus import stopwords as sw
        return frozenset(sw.words("portuguese"))
    except LookupError:  # pragma: no cover - first run
        import nltk
        nltk.download("stopwords", quiet=True)
        from nltk.corpus import stopwords as sw
        return frozenset(sw.words("portuguese"))


@lru_cache(maxsize=1)
def stopwords() -> frozenset[str]:
    """NLTK list + the project's custom stopwords (used by preprocess_text)."""
    return nltk_stopwords() | frozenset(config.load()["preprocessing"]["custom_stopwords"])


def _url_to_domain(match: re.Match) -> str:
    url = re.sub(r"^https?://(?:www\.)?|www\.", "", match.group(0))
    url = re.split(r"[/?#]", url)[0]
    return url.split(".")[0]


def preprocess_text(text: str, stop_words: frozenset[str] | None = None) -> str:
    if not isinstance(text, str):
        return ""
    stop_words = stopwords() if stop_words is None else stop_words
    text = _URL.sub(_url_to_domain, text)
    text = unidecode(text.lower())
    text = text.translate(_PUNCT)
    text = _EMOJI_REPEAT.sub(r"\1", text)
    text = _EMOJI_BLOCK_REPEAT.sub(r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"k{2,}|K{2,}", "kk", text)
    text = re.sub(r"(ha){2,}", "haha", text, flags=re.IGNORECASE)
    text = re.sub(r"(ka){2,}", "kaka", text, flags=re.IGNORECASE)
    return " ".join(w for w in text.split() if w not in stop_words)
