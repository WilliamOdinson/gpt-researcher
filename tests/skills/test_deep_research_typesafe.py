"""GR_ORCHESTRATOR=typesafe through the real deep-research checkpoint, with
the sub-researchers mocked (see test_deep_research_orchestration for the
fixtures) and a scripted fake TypeSafe client injected into the policy."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from gpt_researcher.skills.deep_research import DeepResearchSkill
from gpt_researcher.utils.trajectory_logger import make_item_id
from gpt_researcher.orchestration.typesafe import TypeSafePolicy, TypeSafeThresholds

from tests.skills.test_deep_research_orchestration import _clean_cache, _researcher, _run  # noqa: F401


@dataclass
class _Answer:
    type: str
    noul: float


@dataclass
class _Usage:
    input_tokens: int = 50
    output_tokens: int = 2


@dataclass
class _Response:
    model: str
    answers: dict
    usage: _Usage


class ScriptedClient:
    """Page C is judged useful, page B off-topic; every sub-question and
    branch is reported answered (so termination fires once allowed)."""

    def __init__(self):
        self.requests = []

    async def system_one(self, *, state, questions, model=None):
        self.requests.append((state, questions))
        answers = {}
        for qid in questions:
            kind = qid.split("_")[0]
            if kind in ("useful", "boilerplate", "sub") and "evidence" in state:
                idx = int(qid.split("_")[1])
                domain = state["evidence"][idx]["source"]
                if kind == "useful":
                    answers[qid] = _Answer("noul", 0.1 if domain == "b" else 0.9)
                elif kind == "boilerplate":
                    answers[qid] = _Answer("noul", 0.05)
                else:
                    answers[qid] = _Answer("noul", 0.8)
            else:  # stop request: sub_k / branch_j
                answers[qid] = _Answer("noul", 0.9)
        return _Response("jev-1.13.0", answers, _Usage())


def _install(skill: DeepResearchSkill, **thresholds) -> ScriptedClient:
    """Swap in a TypeSafe policy backed by the scripted client. The skill is
    built under another policy name so the test needs neither the SDK nor a
    key; the recorded policy name is set to match the injected orchestrator."""
    client = ScriptedClient()
    skill.orchestrator = TypeSafePolicy(
        client=client, meter=lambda *_: None, thresholds=TypeSafeThresholds(**thresholds)
    )
    skill.policy_name = "typesafe"
    return client


async def test_typesafe_keep_mask_and_context_follow_the_verdicts(monkeypatch):
    monkeypatch.setenv("GR_ORCHESTRATOR", "topk")  # placeholder; replaced by _install
    skill = DeepResearchSkill(_researcher(depth=1, breadth=2))
    skill.query_embedding = [1.0, 0.0, 0.0]
    skill.subq_embeddings = [[1.0, 0.0, 0.0]]
    skill.trajectory_logger.trajectory.subquestions = ["what is x exactly"]
    # The fixture's page C sits at cosine 0.9 to page A; raise the duplicate
    # threshold so this test exercises the verdicts rather than the dedup.
    client = _install(skill, dedup_tau=0.95)

    out, _ = await _run(skill, depth=1, breadth=2)
    snap = skill.trajectory_logger.trajectory.rounds[0]
    ids = {e.source_url: iid for iid, e in skill.trajectory_logger.get_all_evidence().items()}

    assert snap.decision.policy == "typesafe"
    # B is pruned by the model's verdict even though the EmbeddingsFilter kept
    # it; C is kept even though the filter had pruned it.
    assert set(snap.decision.kept_item_ids) == {ids["https://a"], ids["https://c"]}
    assert snap.decision.meta["pruned_reasons"] == {ids["https://b"]: "low_usefulness"}
    joined = "\n".join(out["context"])
    assert "Source: https://a" in joined and "Source: https://c" in joined
    assert "https://b" not in joined
    # One judging request (3 pages) plus one stop request.
    assert len(client.requests) == 2
    judging, stop = client.requests
    assert {e["source"] for e in judging[0]["evidence"]} == {"a", "b", "c"}
    assert [s["question"] for s in stop[0]["subquestions"]] == ["what is x exactly"]
    assert snap.decision.meta["usage"]["requests"] == 2
    # Round 1 < min_rounds (2): continue despite every sub-question answered.
    assert snap.decision.type == "continue"


async def test_typesafe_terminates_and_allocates_from_branch_answers(monkeypatch):
    monkeypatch.setenv("GR_ORCHESTRATOR", "topk")
    skill = DeepResearchSkill(_researcher(depth=2, breadth=4))
    skill.query_embedding = [1.0, 0.0, 0.0]
    skill.subq_embeddings = [[1.0, 0.0, 0.0]]
    skill.trajectory_logger.trajectory.subquestions = ["what is x exactly"]
    _install(skill, min_rounds=1)

    out, MockR = await _run(skill, depth=2, breadth=4)
    snap = skill.trajectory_logger.trajectory.rounds[0]
    assert snap.decision.type == "terminate"
    assert snap.decision.meta["terminate_reason"] == "all_subquestions_answered"
    assert skill.stop_requested is True
    assert MockR.call_count == 2
    assert len(skill.trajectory_logger.trajectory.rounds) == 1
    # Both branches reported answered with the same probability: uniform split.
    n1, n2 = make_item_id("goal 1", ""), make_item_id("goal 2", "")
    assert snap.decision.branch_allocation == pytest.approx({n1: 0.5, n2: 0.5})


async def test_typesafe_env_selection_builds_the_policy(monkeypatch):
    monkeypatch.setenv("GR_ORCHESTRATOR", "typesafe")
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    skill = DeepResearchSkill(_researcher(depth=1, breadth=2))
    assert isinstance(skill.orchestrator, TypeSafePolicy)
    assert skill.policy_name == "typesafe"
