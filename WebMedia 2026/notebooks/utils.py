"""Shared code for the notebooks in this folder.

Every parameter used in the paper is defined at the top of this file, so it
only needs to be changed in one place. The notebooks import it with
`from utils import *`.
"""
import re
import string
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from unidecode import unidecode

# =============================================================================
# Paths
# =============================================================================
ROOT = Path(__file__).resolve().parent.parent
DATASETS = ROOT / "datasets"
RESULTS = ROOT / "results"
RAW_FILES = {
    "whatsapp": DATASETS / "fakeWhatsApp.BR_2022.csv",
    "telegram": DATASETS / "fakeTelegram.BR_2022.csv",
}
LIWC_DICTIONARY = DATASETS / "LIWC2007_Portugues_win.dic"
NAMES = {"whatsapp": "FW.Br", "telegram": "FT.Br"}
SEED = 42

# =============================================================================
# Parameters of the paper (Section 3.1)
# =============================================================================
# Hashed member IDs removed before any analysis (not described in the paper;
# see docs/REPRODUCIBILITY_NOTES.md, section 2).
EXCLUDED_MEMBERS = {
    "whatsapp": [
        "6ef561ec0f448afcd7b3751124bb0712", "d3e678a0ba0e1485548260a7c4599152",
        "ac5703154484de05336af617455ca55e", "a39edbd64d378226ffa60433649a0acf",
        "c66d0d4ae5a4b281bff67e1fa4fbd6ba", "819bbc872ed6d81f44d746b710eecf06",
        "ec94da4d54f9a5693e88fa582926be53", "6a38c72316d87c028dfd66c10442476b",
        "94099e1e46f129856541e2b3640896d1", "8f367d1693fff47218603fa47ded525c",
        "c09caffee0d1bd30926dea9df25dc88f",
    ],
    "telegram": [
        "b73c0b674b28d87375d78dcae2f1a21f", "62b3aa83e4df122e48cb1f97583e403f",
        "893874da7344daaa00b8b5b3dc295b59", "38737f3c7a07586c4efbfa39aa345705",
        "da4cbbc1b7b1192ad83bcdae613ac2c3", "bf66a467459264e0d33e4c8518a82827",
        "46e009035e6150d656d0c194db88fb07", "811541f6ce49347d1798e8dd2d3cbf83",
        "1d2a0a63333d12ad188e9e6f7f2e0419", "7c379613d4da713fe959d8c7ddc11ce2",
        "debe17fd10d504076d2df8682f63bfc6", "e9713ae04a02a810d6f33dd956f42794",
        "d6c77928db26721ce46aca2d549780f0", "8e53464d3b01eea3e39d07e51ecbb1b4",
        "bacd40da04dd7d13f646993bdcf8e79d", "2273d1167a6212812d95dc8fadbae78e",
        "cd3bb1bdf75be7595e6373171a5c2225", "add8c1ba533c5e5450d92c061a5ee7bf",
        "e198f90df1995528531dd43db0c935ea", "06dc9ac55ed64caab2bd97e9ab717302",
        "c0110feb539d212836605b66192722dd", "f71912700ac5331415408ce229681359",
        "2ee692357a9c948351c43a9540e859ae", "25b66278176dabe814dfc25a405a2470",
        "08b21cb1e7de74ef5fe1085230075523",
    ],
}

# Inclusion filter: intersection of the lists from three Theology professionals.
RELIGIOUS_KEYWORDS = [
    "deus", "jesus", "misericordia", "davi", "salomao", "reino", "templo",
    "conservador", "pentecostal", "rcc", "renovacao", "carismatic",
    "paulo ricardo", "bernardo kuster", "herege", "ateu", "jerico", "heresia",
]
# Exclusion filter: election-campaign terms (defined empirically).
POLITICAL_KEYWORDS = [
    "lula", "bolsonaro", "pt", "pl", "stf", "patria", "55", "22", "13",
    "senadores", "lulaladrao", "urnas", "alexandre", "moraes", "comunismo",
    "eleicao", "eleicoes", "esquerda", "direita", "presidente", "tse", "fraude",
    "voto", "turno", "ministro",
]
# The paper run matches keywords as SUBSTRINGS of the preprocessed text
# ("pt" also matches "adapto"). Set to True for whole-word matching
# (gives different counts than the paper).
WHOLE_WORD_MATCH = False

CUSTOM_STOPWORDS = {
    "t", "be", "nao", "youtu", "vai", "av", "ja", "to", "the", "this", "i", "and",
    "you", "y", "www", "sao", "pois", "contra", "user", "ai", "so", "gente", "voce",
    "of", "ola", "gift", "card", "kwaivideo", "r", "bom", "q", "vc", "vcs", "pra",
    "ta", "phone", "ok", "la", "sera", "ha", "aqui", "ate", "dia", "mc", "im",
    "tmj", "pix", "g", "diz", "ti", "etc", "tudo", "todo", "toda", "youtube", "g1",
    "m", "instagram", "fb", "in", "link", "was", "blocked", "kk",
}


# =============================================================================
# Preprocessing (identical to the original notebooks)
# =============================================================================
def nltk_stopwords():
    import nltk
    try:
        from nltk.corpus import stopwords
        return set(stopwords.words("portuguese"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        from nltk.corpus import stopwords
        return set(stopwords.words("portuguese"))


STOP_WORDS = None


def preprocess_text(text):
    """URL -> domain, lowercase, remove accents, punctuation and stopwords,
    normalise repeated emojis and laughter."""
    global STOP_WORDS
    if STOP_WORDS is None:
        STOP_WORDS = nltk_stopwords() | CUSTOM_STOPWORDS
    if not isinstance(text, str):
        return ""

    def domain(match):
        url = re.sub(r"^https?://(?:www\.)?|www\.", "", match.group(0))
        return re.split(r"[/?#]", url)[0].split(".")[0]

    text = re.sub(r"https?://(?:www\.)?\S+|www\.\S+", domain, text)
    text = unidecode(text.lower())
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"([\U00010000-\U0010FFFF])\1+", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"k{2,}|K{2,}", "kk", text)
    text = re.sub(r"(ha){2,}", "haha", text, flags=re.IGNORECASE)
    text = re.sub(r"(ka){2,}", "kaka", text, flags=re.IGNORECASE)
    return " ".join(w for w in text.split() if w not in STOP_WORDS)


def contains_any(text, keywords):
    text = text.lower()
    if WHOLE_WORD_MATCH:
        return any(re.search(rf"(?<!\w){re.escape(k)}(?!\w)", text) for k in keywords)
    return any(k in text for k in keywords)


# =============================================================================
# Datasets
# =============================================================================
def load_raw(platform):
    path = RAW_FILES[platform]
    with open(path, "rb") as fh:
        if fh.read(40).startswith(b"version https://git-lfs"):
            raise RuntimeError(
                f"{path.name} is only a Git LFS pointer. Run `git lfs install` "
                "and `git lfs pull` in the repository folder to download the data.")
    return pd.read_csv(path, low_memory=False)


def build_dataset(platform):
    """Return (filtered dataset, dict with the size after each step)."""
    raw = load_raw(platform)
    steps = {"raw messages": len(raw),
             "raw users": raw["id_member_anonymous"].nunique(),
             "raw groups": raw["id_group_anonymous"].nunique()}
    df = raw[~raw["id_member_anonymous"].isin(EXCLUDED_MEMBERS[platform])]
    df = df.dropna(subset=["text_content_anonymous"])
    df = df[df["trava_zap"] == False].copy()  # noqa: E712
    steps["after removing excluded IDs, empty and trava-zap"] = len(df)
    df["text_processed"] = df["text_content_anonymous"].apply(preprocess_text)
    df = df[df["text_processed"].apply(contains_any, args=(RELIGIOUS_KEYWORDS,))]
    steps["after inclusion filter (religious)"] = len(df)
    df = df[~df["text_processed"].apply(contains_any, args=(POLITICAL_KEYWORDS,))]
    steps["after exclusion filter (political)"] = len(df)
    return df, steps


def processed_file(platform):
    return DATASETS / "processed" / f"{NAMES[platform]}.csv.gz"


def load_dataset(platform):
    """Load FW.Br / FT.Br; builds and saves it first if needed."""
    path = processed_file(platform)
    if not path.exists():
        df, _ = build_dataset(platform)
        save_dataset(df, platform)
    df = pd.read_csv(path, low_memory=False)
    df["text_processed"] = df["text_processed"].fillna("")
    return df


def save_dataset(df, platform):
    path = processed_file(platform)
    path.parent.mkdir(exist_ok=True)
    df.to_csv(path, index=False)
    # list of selected messages without text (safe to publish)
    df[["id_message", "id_group_anonymous", "date_message"]].to_csv(
        path.parent / f"{NAMES[platform]}_message_ids.csv", index=False)


def results_folder(name):
    folder = RESULTS / name
    folder.mkdir(parents=True, exist_ok=True)
    return folder


# =============================================================================
# Comparison with the paper
# =============================================================================
def compare_with_paper(rows, tolerance=0.005):
    """rows: list of (item, value in the paper, value obtained)."""
    table = pd.DataFrame(rows, columns=["item", "paper", "obtained"])
    diff = (table["obtained"].astype(float) - table["paper"].astype(float)).abs()
    table["matches"] = np.where(diff <= tolerance, "yes", "NO")
    print(f"{(table['matches'] == 'yes').sum()} of {len(table)} values match the paper")
    return table


# =============================================================================
# Statistics (Section 4.3)
# =============================================================================
def cliffs_delta(x, y):
    """Cliff's delta = P(X > Y) - P(X < Y) with the 95% CI from Cliff's (1993)
    consistent variance estimator. Returns (delta, (low, high))."""
    z = 1.959963985
    x, y = np.sort(np.asarray(x, float)), np.sort(np.asarray(y, float))
    n, m = len(x), len(y)
    less, lesseq = np.searchsorted(y, x, "left"), np.searchsorted(y, x, "right")
    di = (less - (m - lesseq)) / m
    delta = di.mean()
    xless, xlesseq = np.searchsorted(x, y, "left"), np.searchsorted(x, y, "right")
    dj = ((n - xlesseq) - xless) / n
    ties = (lesseq - less).sum()
    var = (m * m * di.var(ddof=1) * (n - 1) + n * n * dj.var(ddof=1) * (m - 1)
           + ((n * m - ties) - n * m * delta ** 2)) / (n * m * (n - 1) * (m - 1))
    sd = np.sqrt(max(var, 0))
    den = 1 - delta ** 2 + z ** 2 * var
    root = z * sd * np.sqrt((1 - delta ** 2) ** 2 + z ** 2 * var)
    return delta, ((delta - delta ** 3 - root) / den, (delta - delta ** 3 + root) / den)


def holm(pvalues):
    p = np.asarray(pvalues, float)
    order, m = np.argsort(p), len(p)
    adjusted, running = np.empty(m), 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adjusted[i] = min(1.0, running)
    return adjusted


# =============================================================================
# Topic modeling metrics (Section 4.4)
# =============================================================================
def tokenize_for_topics(texts):
    from nltk import tokenize
    ws, wp = tokenize.WhitespaceTokenizer(), tokenize.WordPunctTokenizer()
    return [wp.tokenize(" ".join(ws.tokenize(t.lower()))) for t in texts]


def topic_diversity(topics, top_n=10):
    words = [w for t in topics for w in t[:top_n]]
    return len(set(words)) / (len(topics) * top_n)


def irbo(topics, top_n=10):
    """Mean (1 - RBO) between topic pairs, `rbo` package (used for LDA and PTM)."""
    import rbo
    from itertools import combinations
    pairs = list(combinations([t[:top_n] for t in topics], 2))
    return float(np.mean([1 - rbo.RankingSimilarity(a, b).rbo() for a, b in pairs]))


def irbo_truncated(topics, top_n=10, p=0.9):
    """Truncated RBO written in the original BERTopic notebooks."""
    from itertools import combinations

    def rbo(a, b):
        overlap, score = 0.0, 0.0
        for d in range(1, min(len(a), len(b)) + 1):
            if a[d - 1] in b[:d] and b[d - 1] in a[:d]:
                overlap += 1
            score += overlap / d * (p ** d)
        return (1 - p) * score
    pairs = list(combinations([t[:top_n] for t in topics], 2))
    return float(np.mean([1 - rbo(a, b) for a, b in pairs]))


# =============================================================================
# LIWC (Section 4.5)
# =============================================================================
def load_liwc_dictionary(path=LIWC_DICTIONARY):
    """Returns (exact entries, wildcard prefixes longest first, category names)."""
    with open(path, encoding="latin1") as fh:
        lines = [l.rstrip("\n\r") for l in fh]
    marks = [i for i, l in enumerate(lines) if l.strip() == "%"]
    categories = {}
    for l in lines[marks[0] + 1:marks[1]]:
        parts = l.strip().split("\t")
        if len(parts) >= 2 and parts[0].isdigit():
            categories[parts[0]] = parts[1]
    exact, prefixes = {}, []
    for l in lines[marks[1] + 1:]:
        if not l.strip():
            continue
        word, *codes = l.strip().split("\t")
        names = [categories.get(c, "?") for c in codes if c.isdigit()]
        word = word.lower()
        if word.endswith("*"):
            prefixes.append((word[:-1], names))
        else:
            exact[word] = names
    prefixes.sort(key=lambda x: -len(x[0]))
    return exact, prefixes, categories


def liwc_count(texts, dictionary):
    """Count LIWC categories over NLTK word tokens (lowercase, NLTK stopwords
    and punctuation removed). Returns (Counter of categories, total tokens)."""
    from nltk.tokenize import word_tokenize
    exact, prefixes, _ = dictionary
    stop = nltk_stopwords()
    counts, total = Counter(), 0
    for text in texts:
        if not isinstance(text, str):
            continue
        tokens = [t for t in word_tokenize(text.lower(), language="portuguese")
                  if t not in stop and t not in string.punctuation]
        total += len(tokens)
        for tok in tokens:
            if tok in exact:
                counts.update(exact[tok])
                continue
            for prefix, names in prefixes:
                if tok.startswith(prefix):
                    counts.update(names)
                    break
    return counts, total
