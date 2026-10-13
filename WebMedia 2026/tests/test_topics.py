from fundrel import topics


def test_diversity_and_irbo_bounds():
    t = [["deus", "jesus", "fe"], ["brasil", "povo", "nacao"], ["deus", "brasil", "povo"]]
    assert topics.diversity(t, 3) == 6 / 9
    assert 0 <= topics.irbo_truncated(t, 3) <= 1
    assert topics.irbo_truncated([["a", "b"], ["a", "b"]], 2) < topics.irbo_truncated([["a", "b"], ["c", "d"]], 2)


def test_tokenize_matches_notebook_tokenisation():
    assert topics.tokenize(["Deus  acima-de todos"]) == [["deus", "acima", "-", "de", "todos"]]


def test_lda_sweep_selects_k_by_cv():
    docs = ["deus jesus senhor oracao amem"] * 20 + ["brasil povo nacao patria bandeira"] * 20
    texts = topics.tokenize(docs)
    cfg = {"k_min": 2, "k_max": 5, "k_step": 2, "top_n_words": 5,
           "lda": {"passes": 2, "workers": 1, "coherence_topn": 5}}
    dictionary, sweep, models, how = topics.run_lda(texts, cfg, seed=42)
    assert [r["k"] for r in sweep] == [2, 4]
    assert all(set(r) == {"k", "c_v", "c_npmi"} for r in sweep)
