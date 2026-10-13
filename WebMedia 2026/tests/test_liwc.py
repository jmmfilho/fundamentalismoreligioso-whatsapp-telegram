from collections import Counter

from fundrel import liwc

DIC = "%\n1\tfunct\n125\taffect\n359\trelig\n%\ndeus\t359\nfé\t359\t125\nabafad*\t125\nabaf*\t1\n"


def test_dictionary_header_wildcards_and_accents(tmp_path):
    f = tmp_path / "d.dic"
    f.write_bytes(DIC.encode("latin1"))
    d = liwc.load_dictionary(f)
    assert d.categories == {"1": "funct", "125": "affect", "359": "relig"}
    assert d.lookup("fé") == ["relig", "affect"]          # accented entry
    assert d.lookup("abafado") == ["affect"]               # longest prefix wins
    assert d.lookup("abafo") == ["funct"]
    assert d.lookup("1") == []                             # header codes are not words
    counts, total = liwc.count(["Deus é fé!", "abafado de"], d)
    # "de" is a stopword and "!" is punctuation; "é" is a token without category
    assert total == 4 and counts == Counter({"relig": 2, "affect": 2})


def test_compare_reports_phi_and_holm():
    df = liwc.compare(Counter({"relig": 120, "time": 50}), 1000, Counter({"relig": 100, "time": 80}), 1000, 0.05)
    assert set(df["category"]) == {"relig", "time"}
    assert (df["p_holm"] >= df["p"]).all()
    assert df["phi"].between(0, 1).all()


def test_category_missing_in_one_dataset_is_still_compared():
    df = liwc.compare(Counter({"relig": 10}), 100, Counter({"relig": 12, "money": 3}), 100, 0.05)
    assert set(df["category"]) == {"relig", "money"}
