"""LearnedPolicy: the behavior-cloned orchestrator.

Same 3-d synthetic pool as test_orchestration_random.py. The key property is
that the prompt the policy sends online is byte-identical to the one the
harness rebuilds offline from ``decision.meta`` (that is what the model was
trained on), and that bad model output never crashes a run.
"""

from __future__ import annotations

import math

import pytest

from gpt_researcher.orchestration import (
    FEATURE_NAMES,
    FrontierInfo,
    LearnedPolicy,
    OrchestrationInput,
    PoolItem,
    StateFrontier,
    StateItem,
    SYSTEM_PROMPT,
    build_policy,
    serialize_state,
)


def _vec(rho: float) -> list[float]:
    return [rho, math.sqrt(max(0.0, 1 - rho * rho)), 0.0]


def _item(iid: str, rho: float, tokens: int = 100, *, new: bool = True, rnd: int = 2, url: str | None = None) -> PoolItem:
    return PoolItem(
        item_id=iid,
        source_url=url or f"https://example.org/{iid}",
        text=f"text of {iid} " * 20,
        embedding=_vec(rho),
        tree_depth=1,
        retrieval_round=rnd,
        is_new=new,
        source_subquery="branch A" if new else "",
    )


NEW = [_item("P1", 1.0), _item("P2", 0.8), _item("P3", 0.6), _item("P4", 0.0)]
PREV = [_item("R1", 0.9, new=False, rnd=1, url="https://data.gov/r1")]
FRONTIER = [FrontierInfo("nA", "branch A", _vec(0.3)), FrontierInfo("nB", "branch B", _vec(0.95))]


def _inp(round_id: int = 2, budget: int | None = None, frontier=FRONTIER) -> OrchestrationInput:
    return OrchestrationInput(
        root_query="what is x",
        query_embedding=_vec(1.0),
        subquestions=["sub 1", "sub 2"],
        new_items=list(NEW),
        retained_prev=list(PREV),
        frontier=list(frontier),
        round_id=round_id,
        tree_depth=1,
        tokens_used=1234,
        token_budget=budget,
        subquestion_embeddings=[_vec(1.0), _vec(0.4)],
    )


class _FakeLLM:
    def __init__(self, reply: str | Exception) -> None:
        self.reply = reply
        self.calls: list[list[dict[str, str]]] = []

    async def __call__(self, messages):
        self.calls.append(messages)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


async def test_learned_parses_model_answer_and_logs_bc_meta():
    llm = _FakeLLM("KEEP: P1 P2 R1\nALLOC: nA=0.75 nB=0.25\nDECISION: TERMINATE")
    d = await LearnedPolicy(llm_call=llm).decide(_inp())
    assert d.kept_ids == {"P1", "P2", "R1"}
    assert d.terminate is True
    assert d.branch_allocation["nA"] == pytest.approx(0.75) and d.branch_allocation["nB"] == pytest.approx(0.25)
    m = d.meta
    assert m["policy"] == "learned" and m["parse_failure"] is False
    assert m["terminate_reason"] == "model"
    # Everything the harness bc_pairs builder reads is there.
    assert set(m["features"]) == {"P1", "P2", "P3", "P4", "R1"}
    assert m["feature_names"] == list(FEATURE_NAMES)
    assert set(m["frontier_stats"]) == {"nA", "nB"}
    for key in ("phi_prev", "phi_pool", "phi_after", "tokens_used", "token_budget", "n_pool", "n_new", "sources", "raw"):
        assert key in m
    assert "_rows" not in m
    # One call with the shared system prompt.
    assert len(llm.calls) == 1
    assert llm.calls[0][0] == {"role": "system", "content": SYSTEM_PROMPT}


async def test_learned_prompt_matches_offline_rebuild():
    """The prompt built online equals serialize_state() over the meta the
    policy logs — the same path the harness uses to build BC pairs."""
    llm = _FakeLLM("KEEP: ALL\nALLOC: nA=0.5 nB=0.5\nDECISION: CONTINUE")
    inp = _inp(budget=5000)
    d = await LearnedPolicy(llm_call=llm).decide(inp)
    online_prompt = llm.calls[0][1]["content"]

    m = d.meta
    items = [
        StateItem(it.item_id, m["features"][it.item_id], it.is_new, it.text, m["sources"][it.item_id])
        for it in inp.pool
    ]
    frontier = [StateFrontier(f.node_id, f.subquery, m["frontier_stats"][f.node_id]["gap"]) for f in inp.frontier]
    offline_prompt = serialize_state(
        root_query=inp.root_query,
        subquestions=inp.subquestions,
        round_id=inp.round_id,
        tree_depth=inp.tree_depth,
        tokens_used=m["tokens_used"],
        token_budget=m["token_budget"],
        items=items,
        frontier=frontier,
        feature_names=m["feature_names"],
    )
    assert online_prompt == offline_prompt
    assert "context budget 5000" in online_prompt


async def test_learned_degrades_on_garbage_output():
    d = await LearnedPolicy(llm_call=_FakeLLM("I am not sure what to do here.")).decide(_inp())
    assert d.kept_ids == {"P1", "P2", "P3", "P4", "R1"}
    assert d.terminate is False
    assert d.branch_allocation == {"nA": 0.5, "nB": 0.5}
    assert d.meta["parse_failure"] is True


async def test_learned_degrades_on_transport_error():
    d = await LearnedPolicy(llm_call=_FakeLLM(RuntimeError("endpoint down"))).decide(_inp())
    assert d.kept_ids == {"P1", "P2", "P3", "P4", "R1"}
    assert d.terminate is False
    assert d.meta["parse_failure"] is True and "endpoint down" in d.meta["error"]
    assert "phi_after" in d.meta


def test_build_policy_learned_needs_llm_call(monkeypatch):
    monkeypatch.setenv("GR_ORCHESTRATOR", "learned")
    with pytest.raises(ValueError):
        build_policy()

    async def _noop(messages):
        return ""

    pol = build_policy(llm_call=_noop)
    assert isinstance(pol, LearnedPolicy)
