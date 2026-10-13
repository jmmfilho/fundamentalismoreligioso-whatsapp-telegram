import pandas as pd
import pytest

from fundrel import data


def test_substring_matching_reproduces_paper_behaviour():
    # short keywords such as "pt" match inside other words under substring matching
    assert data.contains_any("aceito jesus", ["pt"], "substring") is False
    assert data.contains_any("adapto", ["pt"], "substring") is True
    assert data.contains_any("treino", ["reino"], "substring") is True


def test_word_matching_is_stricter():
    assert data.contains_any("treino hoje", ["reino"], "word") is False
    assert data.contains_any("o reino de deus", ["reino"], "word") is True
    assert data.contains_any("paulo ricardo falou", ["paulo ricardo"], "word") is True


def test_clean_drops_excluded_members_missing_text_and_trava_zap(monkeypatch):
    monkeypatch.setattr(data.config, "load", lambda: {"datasets": {"whatsapp": {"excluded_member_ids": ["bot"]}}})
    df = pd.DataFrame({
        "id_member_anonymous": ["u1", "bot", "u2", "u3"],
        "text_content_anonymous": ["ok", "spam", None, "x"],
        "trava_zap": [False, False, False, True],
    })
    assert data.clean(df, "whatsapp")["id_member_anonymous"].tolist() == ["u1"]


def test_build_applies_inclusion_then_exclusion(monkeypatch):
    raw = pd.DataFrame({
        "date_message": ["2022-10-01"] * 4,
        "id_member_anonymous": ["a", "b", "c", "d"],
        "id_group_anonymous": ["g1", "g1", "g2", "g2"],
        "id_message": [1, 2, 3, 4],
        "trava_zap": [False] * 4,
        "text_content_anonymous": ["Deus é fiel", "Bom dia", "Deus e Lula", "Jesus salva"],
    })
    cfg = {"filters": {"match": "substring", "religious_keywords": ["deus", "jesus"],
                       "political_keywords": ["lula"]},
           "datasets": {"whatsapp": {"name": "FW.Br", "excluded_member_ids": []}}}
    monkeypatch.setattr(data.config, "load", lambda: cfg)
    monkeypatch.setattr(data, "load_raw", lambda p: raw)
    out, steps = data.build("whatsapp", save=False)
    assert out["id_message"].tolist() == [1, 4]
    assert steps["after_inclusion_filter"] == 3 and steps["after_exclusion_filter"] == 2


def test_lfs_pointer_is_detected(tmp_path):
    f = tmp_path / "x.csv"
    f.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 1\n")
    with pytest.raises(data.LFSPointerError):
        data._check_not_lfs_pointer(f)
