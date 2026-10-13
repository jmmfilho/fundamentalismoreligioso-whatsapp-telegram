from fundrel.preprocessing import preprocess_text


def test_url_is_replaced_by_domain():
    assert preprocess_text("veja https://www.youtube.com/watch?v=1") == "veja youtube"


def test_lowercase_accents_punctuation_and_stopwords():
    assert preprocess_text("A Misericórdia de DEUS, é infinita!") == "misericordia deus infinita"


def test_laughter_is_normalised_and_removed():
    assert preprocess_text("kkkkkk que hahahaha") == "haha"


def test_non_string_returns_empty():
    assert preprocess_text(None) == ""
