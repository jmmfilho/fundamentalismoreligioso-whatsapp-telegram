"""Message retrieval for manual discourse analysis (Section 4.2).

The paper indexed all messages in Elasticsearch and used the top named
entities as queries. `retrieve` reproduces this with a plain pandas
backend (no server needed); `index_elasticsearch` creates the same index for
interactive exploration if an Elasticsearch instance is available.

The share count of a message is the number of times its exact raw text
occurs in the dataset (e.g. the 53 / 48 / 8 counts reported in the paper).
"""
from __future__ import annotations

import pandas as pd

from . import PLATFORMS, config, data


def share_counts(df: pd.DataFrame) -> pd.DataFrame:
    counts = (df.groupby(data.TEXT_RAW)
                .agg(shares=("id_message", "size"),
                     groups=("id_group_anonymous", "nunique"),
                     first_seen=("date_message", "min"))
                .sort_values("shares", ascending=False)
                .reset_index())
    return counts


def retrieve(df: pd.DataFrame, entity: str) -> pd.DataFrame:
    hits = df[df[data.TEXT].str.contains(entity.lower(), regex=False, na=False)]
    return share_counts(hits)


def index_elasticsearch(df: pd.DataFrame, index: str, host: str = "http://localhost:9200") -> None:
    from elasticsearch import Elasticsearch, helpers
    es = Elasticsearch(host)
    es.options(ignore_status=[400, 404]).indices.delete(index=index)
    actions = ({"_index": index, "_id": str(r.id_message),
                "_source": {"text": r.text_content_anonymous, "text_processed": r.text_processed,
                            "group": r.id_group_anonymous, "date": str(r.date_message)}}
               for r in df.itertuples())
    helpers.bulk(es, actions)


def run(entities_csv=None, top_messages: int = 20) -> None:
    """Export, per dataset, the most shared messages overall and per top entity.
    Output files contain message text: keep them out of public commits."""
    out = config.results_dir("03_retrieval")
    ents = pd.read_csv(entities_csv or config.results_dir("02_ner") / "top_entities.csv")
    for p in PLATFORMS:
        df = data.load(p)
        name = config.dataset_name(p)
        share_counts(df).head(top_messages).to_csv(out / f"{name}_most_shared.csv", index=False)
        frames = []
        for ent in ents.loc[ents["dataset"] == name, "entity"].dropna().unique():
            if len(str(ent).strip()) < 3:
                continue
            hits = retrieve(df, str(ent)).head(top_messages)
            hits.insert(0, "query", ent)
            frames.append(hits)
        if frames:
            pd.concat(frames).to_csv(out / f"{name}_retrieved_by_entity.csv", index=False)
