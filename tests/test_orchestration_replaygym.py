"""GR_ORCHESTRATOR=replaygym evidence handling.

The harness's ReplayGym scores every expansion plan from the legacy writer
input (per-node filtered contexts), so the live policy must default to that
evidence path; other rows are opt-in through GR_REPLAYGYM_EVIDENCE.
"""

from __future__ import annotations

import pytest

from gpt_researcher.orchestration import NoPruningPolicy, TopKPolicy, build_policy


def test_replaygym_defaults_to_the_legacy_evidence_path(monkeypatch):
    monkeypatch.delenv("GR_REPLAYGYM_EVIDENCE", raising=False)
    assert build_policy("replaygym") is None


@pytest.mark.parametrize("evidence", ["legacy", "replaygym", " LEGACY "])
def test_replaygym_legacy_aliases(monkeypatch, evidence):
    monkeypatch.setenv("GR_REPLAYGYM_EVIDENCE", evidence)
    assert build_policy("replaygym") is None


@pytest.mark.parametrize("evidence,cls", [("none", NoPruningPolicy), ("topk", TopKPolicy)])
def test_replaygym_composes_with_another_evidence_row(monkeypatch, evidence, cls):
    monkeypatch.setenv("GR_REPLAYGYM_EVIDENCE", evidence)
    assert isinstance(build_policy("replaygym"), cls)
