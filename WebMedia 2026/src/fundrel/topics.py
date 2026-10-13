"""Topic modeling with LDA, PTM and BERTopic (Section 4.4, Tables 4-7).

For LDA and PTM the number of topics k is swept over range(k_min, k_max, k_step)
and the model with the highest C_V coherence is selected. All reported metrics
(C_V, C_NPMI, diversity, IRBO) are computed on that *selected* model. The full
sweep is saved too, so the choice of k can be audited.
"""
from __future__ import annotations

from itertools import combinations

import pandas as pd
from gensim import corpora
from gensim.models import CoherenceModel

from . import PLATFORMS, config, data


# ----------------------------------------------------------------- utilities
def tokenize(texts) -> list[list[str]]:
    """Whitespace + WordPunct tokenisation, as in the paper notebooks."""
    from nltk import tokenize as tk
    ws, wp = tk.WhitespaceTokenizer(), tk.WordPunctTokenizer()
    return [wp.tokenize(" ".join(ws.tokenize(t.lower()))) for t in texts]


def coherence(topics, texts, dictionary, measure: str, topn: int | None = None, model=None) -> float:
    kw = dict(texts=texts, dictionary=dictionary, coherence=measure)
    if model is not None:
        kw.update(model=model, **({"topn": topn} if topn else {}))
    else:
        kw.update(topics=topics)
    return float(CoherenceModel(**kw).get_coherence())


def diversity(topics: list[list[str]], top_n: int) -> float:
    words = [w for t in topics for w in t[:top_n]]
    return len(set(words)) / (len(topics) * top_n)


def irbo_rbo_package(topics: list[list[str]], top_n: int) -> float:
    """Mean (1 - RBO) over topic pairs using the `rbo` package (LDA/PTM in the paper)."""
    import rbo
    pairs = list(combinations([t[:top_n] for t in topics], 2))
    return sum(1 - rbo.RankingSimilarity(a, b).rbo() for a, b in pairs) / len(pairs) if pairs else 1.0


def irbo_truncated(topics: list[list[str]], top_n: int, p: float = 0.9) -> float:
    """Truncated RBO implementation used for BERTopic in the paper notebooks."""
    def rbo(a, b):
        overlap, score = 0.0, 0.0
        for d in range(1, min(len(a), len(b)) + 1):
            if a[d - 1] in b[:d] and b[d - 1] in a[:d]:
                overlap += 1
            score += overlap / d * (p ** d)
        return (1 - p) * score
    pairs = list(combinations([t[:top_n] for t in topics], 2))
    return sum(1 - rbo(a, b) for a, b in pairs) / len(pairs) if pairs else 1.0


def _metrics(topics, texts, dictionary, top_n, model=None, topn=None) -> dict:
    return {
        "c_v": coherence(topics, texts, dictionary, "c_v", topn, model),
        "c_npmi": coherence(topics, texts, dictionary, "c_npmi", topn, model),
        "diversity": diversity(topics, top_n),
        "irbo_rbo_package": irbo_rbo_package(topics, top_n),
        "irbo_truncated": irbo_truncated(topics, top_n),
    }


# -------------------------------------------------------------------- models
def run_lda(texts, cfg, seed):
    from gensim.models import LdaMulticore
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(t) for t in texts]
    p, top_n = cfg["lda"], cfg["top_n_words"]
    sweep, models = [], {}
    for k in range(cfg["k_min"], cfg["k_max"], cfg["k_step"]):
        m = LdaMulticore(corpus=corpus, id2word=dictionary, num_topics=k, passes=p["passes"],
                         workers=p["workers"], random_state=seed)
        topics = [[w for w, _ in m.show_topic(i, topn=top_n)] for i in range(k)]
        sweep.append({"k": k, **{s: coherence(None, texts, dictionary, s, p["coherence_topn"], m)
                                 for s in ("c_v", "c_npmi")}})
        models[k] = (m, topics)
    return dictionary, sweep, models, dict(topn=p["coherence_topn"], use_model=True)


def run_ptm(texts, cfg, seed):
    import tomotopy as tp
    dictionary = corpora.Dictionary(texts)
    top_n = cfg["top_n_words"]
    sweep, models = [], {}
    for k in range(cfg["k_min"], cfg["k_max"], cfg["k_step"]):
        m = tp.PTModel(k=k, seed=seed)
        for doc in texts:
            if doc:
                m.add_doc(doc)
        m.train(cfg["ptm"]["iterations"], workers=1)  # workers=1 -> deterministic
        topics = [[w for w, _ in m.get_topic_words(i, top_n=top_n)] for i in range(m.k)]
        sweep.append({"k": k, **{s: coherence(topics, texts, dictionary, s) for s in ("c_v", "c_npmi")}})
        models[k] = (m, topics)
    return dictionary, sweep, models, dict(topn=None, use_model=False)


def run_bertopic(docs, cfg, seed):
    import hdbscan
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    from umap import UMAP
    b = cfg["bertopic"]
    emb_model = SentenceTransformer(b["embedding_model"])
    model = BERTopic(embedding_model=emb_model,
                     umap_model=UMAP(**b["umap"], random_state=seed),
                     hdbscan_model=hdbscan.HDBSCAN(**b["hdbscan"], prediction_data=True),
                     language="portuguese")
    model.fit_transform(docs, emb_model.encode(docs, show_progress_bar=True))
    topics = [[w for w, _ in model.get_topic(t) if isinstance(w, str) and w.strip()]
              for t in model.get_topics() if t != -1]
    texts = [d.split() for d in docs]
    return model, topics, texts


# ---------------------------------------------------------------------- main
def run(models=("lda", "ptm", "bertopic")) -> pd.DataFrame:
    cfg, seed = config.load()["topics"], config.load()["seed"]
    out = config.results_dir("05_topics")
    summary = []
    for p in PLATFORMS:
        name = config.dataset_name(p)
        docs = data.load(p)[data.TEXT].tolist()
        for kind in models:
            if kind == "bertopic":
                m, topics, texts = run_bertopic(docs, cfg, seed)
                dictionary = corpora.Dictionary(texts)
                metrics = _metrics(topics, texts, dictionary, cfg["top_n_words"])
                k = len(topics)
            else:
                texts = tokenize(docs)
                runner = run_lda if kind == "lda" else run_ptm
                dictionary, sweep, fitted, how = runner(texts, cfg, seed)
                sweep = pd.DataFrame(sweep)
                sweep.to_csv(out / f"{kind}_{name}_k_sweep.csv", index=False)
                k = int(sweep.loc[sweep[cfg["selection_metric"]].idxmax(), "k"])
                m, topics = fitted[k]
                metrics = _metrics(topics, texts, dictionary, cfg["top_n_words"],
                                   model=m if how["use_model"] else None, topn=how["topn"])
            pd.DataFrame({"topic": range(len(topics)),
                          "words": ["; ".join(t[:cfg["top_n_words"]]) for t in topics]}
                         ).to_csv(out / f"{kind}_{name}_topics.csv", index=False)
            summary.append({"model": kind, "dataset": name, "k": k, **metrics})
    summary = pd.DataFrame(summary)
    summary.to_csv(out / "topic_quality.csv", index=False)
    return summary
