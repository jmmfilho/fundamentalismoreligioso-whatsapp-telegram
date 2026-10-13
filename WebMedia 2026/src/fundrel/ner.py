"""Named Entity Recognition (Section 4.1, Table 2).

As in the paper run, NER is applied to the *preprocessed* text (lowercase,
no accents, no stopwords). This explains fragments such as "je"/"sus" produced
by XLM-RoBERTa's sub-word aggregation.
"""
from __future__ import annotations

from collections import Counter

import pandas as pd

from . import PLATFORMS, config, data


def spacy_entities(texts) -> Counter:
    import spacy
    nlp = spacy.load(config.load()["ner"]["spacy_model"])
    counter = Counter()
    for doc in nlp.pipe(texts, batch_size=256):
        counter.update(ent.text for ent in doc.ents)
    return counter


def transformer_entities(texts, max_chars: int | None) -> Counter:
    from transformers import pipeline
    import torch
    ner = pipeline("ner", model=config.load()["ner"]["transformer_model"],
                   aggregation_strategy="simple", device=0 if torch.cuda.is_available() else -1)
    counter = Counter()
    for text in texts:
        if not isinstance(text, str) or not text.strip():
            continue
        for ent in ner(text[:max_chars] if max_chars else text):
            counter[ent["word"]] += 1
    return counter


def run() -> pd.DataFrame:
    cfg = config.load()["ner"]
    rows = []
    for p in PLATFORMS:
        texts = data.load(p)[data.TEXT].tolist()
        for model, counter in (("spaCy", spacy_entities(texts)),
                               ("XLM-RoBERTa", transformer_entities(texts, cfg["max_chars"][p]))):
            for rank, (ent, freq) in enumerate(counter.most_common(cfg["top_n"]), 1):
                rows.append({"dataset": config.dataset_name(p), "model": model,
                             "rank": rank, "entity": ent, "frequency": freq})
    out = pd.DataFrame(rows)
    out.to_csv(config.results_dir("02_ner") / "top_entities.csv", index=False)
    return out
