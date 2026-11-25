"""Uncapped writer context (GR_MAX_CONTEXT_WORDS=0), the record of where
collected words were cut, the writer-input guard, the opt-in length rewrite
and the off-by-default GR_BUDGET_FORCE_FULL."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from gpt_researcher.actions import report_generation as rg
from gpt_researcher.skills import deep_research as dr
from gpt_researcher.skills.deep_research import DeepResearchSkill, build_final_context


def _item(i, words, rnd=1):
    return SimpleNamespace(
        source_url=f"https://example.com/{i}",
        content=" ".join(["w"] * words),
        embedding=None,
        retrieval_round=rnd,
        tree_depth=1,
    )


def test_writer_cap_off_values(monkeypatch):
    monkeypatch.delenv("GR_MAX_CONTEXT_WORDS", raising=False)
    assert dr._max_context_words() == dr.MAX_CONTEXT_WORDS and not dr._writer_cap_off()
    for raw in ("0", "none", "off", "unlimited"):
        monkeypatch.setenv("GR_MAX_CONTEXT_WORDS", raw)
        assert dr._writer_cap_off()
    monkeypatch.setenv("GR_MAX_CONTEXT_WORDS", "60000")
    assert dr._max_context_words() == 60000 and not dr._writer_cap_off()


def test_item_cap_follows_the_writer_cap_unless_set(monkeypatch):
    monkeypatch.delenv("GR_FINAL_ITEM_MAX_WORDS", raising=False)
    monkeypatch.delenv("GR_MAX_CONTEXT_WORDS", raising=False)
    assert dr._item_max_words() == dr.DEFAULT_ITEM_MAX_WORDS
    monkeypatch.setenv("GR_MAX_CONTEXT_WORDS", "0")
    assert dr._item_max_words() == 0
    monkeypatch.setenv("GR_FINAL_ITEM_MAX_WORDS", "800")
    assert dr._item_max_words() == 800


def test_uncapped_context_keeps_every_word():
    ev = [_item(i, 3000, rnd=1 + i % 3) for i in range(40)]  # 120K words
    learnings = ["fact " * 50 for _ in range(400)]  # 20K words
    ctx, stats = build_final_context(learnings, ev, max_words=dr.NO_WORD_CAP, item_max_words=0)
    assert stats["evidence_used"] == 40 and stats["learnings_used"] == 400
    assert stats["evidence_words_cut_by_item_cap"] == 0
    assert stats["evidence_words_cut_by_window"] == 0
    assert stats["learnings_words_cut_by_share"] == 0
    assert stats["word_cap"] is None
    assert stats["words_collected"] == 120_000 + 20_000


def test_capped_context_reports_where_words_were_cut():
    ev = [_item(i, 3000) for i in range(20)]  # 60K words
    ctx, stats = build_final_context([], ev, max_words=25_000, item_max_words=1500)
    assert stats["evidence_items_cut_by_item_cap"] == 20
    assert stats["evidence_words_cut_by_item_cap"] == 20 * 1500
    assert stats["evidence_items_dropped_by_window"] > 0
    assert stats["evidence_words_cut_by_window"] > 0
    assert stats["word_cap"] == 25_000


def _recorder(monkeypatch, limit=None):
    if limit is None:
        monkeypatch.delenv("GR_WRITER_MAX_INPUT_TOKENS", raising=False)
    else:
        monkeypatch.setenv("GR_WRITER_MAX_INPUT_TOKENS", str(limit))
    monkeypatch.setattr(dr, "_count_tokens", lambda text: len(text.split()))
    return SimpleNamespace(token_stop_info={}, writer_context_info={})


def test_writer_guard_records_and_raises(monkeypatch):
    s = _recorder(monkeypatch)
    DeepResearchSkill._record_writer_context(s, ["a b c", "d e"], {"words_collected": 5})
    assert s.writer_context_info["context_tokens"] == 5
    assert s.writer_context_info["words_collected"] == 5
    s = _recorder(monkeypatch, limit=4)
    with pytest.raises(RuntimeError):
        DeepResearchSkill._record_writer_context(s, ["a b c", "d e"], {})


def test_force_full_is_off_by_default(monkeypatch):
    monkeypatch.delenv("GR_BUDGET_FORCE_FULL", raising=False)
    assert dr._env_flag(dr.ENV_BUDGET_FORCE_FULL, False) is False
    monkeypatch.setenv("GR_BUDGET_FORCE_FULL", "1")
    assert dr._env_flag(dr.ENV_BUDGET_FORCE_FULL, False) is True


def test_length_rewrite_is_opt_in(monkeypatch):
    monkeypatch.delenv("GR_REPORT_REWRITE", raising=False)
    assert rg._report_rewrite_on() is False
    monkeypatch.setenv("GR_REPORT_REWRITE", "1")
    assert rg._report_rewrite_on() is True
