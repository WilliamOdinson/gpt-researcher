"""TypeSafe (Jev) orchestration on a fixed synthetic pool.

A fake client answers every Noul from a script keyed by question id, records
each request's state and questions, and reports usage, so the tests check the
request shape the docs prescribe and the code-side routing without a network.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pytest

from gpt_researcher.orchestration import (
    FrontierInfo,
    OrchestrationInput,
    PoolItem,
    TypeSafePolicy,
    TypeSafeThresholds,
    build_policy,
    policy_name_from_env,
)
from gpt_researcher.orchestration.typesafe import ItemVerdict, item_questions

S1 = [1.0, 0.0, 0.0]
S2 = [0.0, 1.0, 0.0]


@dataclass
class _Usage:
    input_tokens: int = 100
    output_tokens: int = 5


@dataclass
class _Answer:
    type: str
    noul: float


@dataclass
class _Response:
    model: str
    answers: dict[str, _Answer]
    usage: _Usage


class FakeClient:
    """Scripted System One endpoint. ``script(question_id, state) -> float``."""

    def __init__(self, script: Callable[[str, dict], float], fail: Callable[[dict], bool] | None = None):
        self.script = script
        self.fail = fail or (lambda state: False)
        self.requests: list[tuple[dict, dict]] = []

    async def system_one(self, *, state: dict, questions: dict, model: str | None = None):
        self.requests.append((state, questions))
        if self.fail(state):
            raise RuntimeError("simulated 529 overloaded")
        answers = {qid: _Answer("noul", self.script(qid, state)) for qid in questions}
        return _Response(model="jev-1.13.0", answers=answers, usage=_Usage())


def _item(iid, emb, *, text=None, new=True, rnd=2, tokens=100, url=None):
    return PoolItem(
        item_id=iid,
        source_url=url or f"https://example.org/{iid}",
        text=text or f"{iid} " * (tokens * 4 // (len(iid) + 1)),
        embedding=emb,
        tree_depth=1,
        retrieval_round=rnd,
        is_new=new,
    )


FRONTIER = [FrontierInfo("nA", "branch A", S1), FrontierInfo("nB", "branch B", S2)]


def _inp(new, prev=(), *, budget=None, subq=("sub one", "sub two"), round_id=2, frontier=FRONTIER):
    return OrchestrationInput(
        root_query="what is x",
        query_embedding=S1,
        subquestions=list(subq),
        new_items=list(new),
        retained_prev=list(prev),
        frontier=list(frontier),
        round_id=round_id,
        tree_depth=1,
        tokens_used=1234,
        token_budget=budget,
        subquestion_embeddings=[S1, S2][: len(subq)],
    )


# Scripted verdicts, keyed by the evidence source domain so the script does
# not depend on batch positions.
VERDICTS = {
    "a.org": {"useful": 0.9, "boilerplate": 0.1, "sub": [0.9, 0.2]},   # supports sub one
    "b.org": {"useful": 0.8, "boilerplate": 0.1, "sub": [0.1, 0.9]},   # supports sub two
    "dup.org": {"useful": 0.85, "boilerplate": 0.0, "sub": [0.9, 0.1]},  # near-duplicate of a
    "nav.org": {"useful": 0.9, "boilerplate": 0.95, "sub": [0.5, 0.5]},  # boilerplate
    "off.org": {"useful": 0.1, "boilerplate": 0.0, "sub": [0.0, 0.0]},   # off topic
}
STOP = {"sub_0": 0.9, "sub_1": 0.3, "branch_0": 0.9, "branch_1": 0.2}


def script(qid: str, state: dict) -> float:
    if qid in STOP:
        return STOP[qid]
    kind, rest = qid.split("_", 1)
    idx = int(rest.split("_")[0])
    verdict = VERDICTS[state["evidence"][idx]["source"]]
    if kind == "sub":
        return verdict["sub"][int(rest.split("_")[1])]
    return verdict[kind]


A = _item("A", S1, url="https://a.org/1")
B = _item("B", S2, url="https://b.org/1")
DUP = _item("DUP", [0.99, 0.14, 0.0], url="https://dup.org/1")
NAV = _item("NAV", [0.0, 0.0, 1.0], url="https://nav.org/1")
OFF = _item("OFF", [0.0, 0.6, 0.8], url="https://off.org/1")
POOL = [A, B, DUP, NAV, OFF]


def _policy(client, **kw) -> TypeSafePolicy:
    meter_calls: list[tuple[str, int, int]] = []
    pol = TypeSafePolicy(client=client, meter=lambda m, i, o: meter_calls.append((m, i, o)), **kw)
    pol.meter_calls = meter_calls  # type: ignore[attr-defined]
    return pol


# ---------------------------------------------------------------- request shape


def test_item_questions_follow_the_documented_shape():
    q = item_questions(3, 2)
    assert set(q) == {"useful_3", "boilerplate_3", "sub_3_0", "sub_3_1"}
    assert all(v["type"] == "noul" for v in q.values())
    assert "`evidence[3].text`" in q["useful_3"]["instructions"]
    assert "`subquestions[1]`" in q["sub_3_1"]["instructions"]
    assert set(q["useful_3"]["criteria"]) == {"true", "false"}


async def test_judging_request_batches_items_with_compact_state():
    client = FakeClient(script)
    pol = _policy(client, thresholds=TypeSafeThresholds(batch_size=2, snippet_chars=20))
    await pol.decide(_inp(POOL))
    judging = [r for r in client.requests if "evidence" in r[0]]
    assert len(judging) == 3  # 5 items in batches of 2
    state, questions = judging[0]
    assert state["question"] == "what is x"
    assert state["subquestions"] == ["sub one", "sub two"]
    assert [e["source"] for e in state["evidence"]] == ["a.org", "b.org"]
    assert all(len(e["text"]) <= 20 for e in state["evidence"])
    assert len(questions) == 2 * (2 + 2)  # two items x (useful, boilerplate, two sub-questions)


# ---------------------------------------------------------------- keep mask


async def test_keep_mask_filters_boilerplate_offtopic_and_duplicates():
    client = FakeClient(script)
    dec = await _policy(client).decide(_inp(POOL))
    assert dec.kept_ids == {"A", "B"}
    assert dec.meta["pruned_reasons"] == {"NAV": "boilerplate", "OFF": "low_usefulness", "DUP": "duplicate"}
    assert dec.meta["verdicts"]["A"]["support"] == [0.9, 0.2]
    assert dec.rewritten == {}
    assert dec.meta["round_usage"]["errors"] == 0


async def test_keep_mask_respects_budget_after_coverage_first_ordering():
    client = FakeClient(script)
    dec = await _policy(client).decide(_inp(POOL, budget=150))
    # Coverage first: A (best for sub one) then B (best for sub two); B no
    # longer fits in 150 tokens, so only A survives and B is a budget prune.
    assert dec.kept_ids == {"A"}
    assert dec.meta["pruned_reasons"]["B"] == "budget"


async def test_verdicts_are_cached_and_retained_items_are_not_resent():
    client = FakeClient(script)
    pol = _policy(client)
    await pol.decide(_inp([A, B]))
    n_first = len(client.requests)
    dec = await pol.decide(_inp([DUP], prev=[A, B], round_id=3))
    judging = [r for r in client.requests[n_first:] if "evidence" in r[0]]
    assert len(judging) == 1 and [e["source"] for e in judging[0][0]["evidence"]] == ["dup.org"]
    assert dec.kept_ids == {"A", "B"}
    assert dec.meta["n_judged"] == 3


async def test_always_keeps_one_item():
    client = FakeClient(script)
    dec = await _policy(client).decide(_inp([OFF, NAV]))
    assert len(dec.kept_ids) == 1


# ---------------------------------------------------------------- stop and allocation


async def test_stop_request_carries_supporting_passages_and_drives_u_and_w():
    client = FakeClient(script)
    dec = await _policy(client, thresholds=TypeSafeThresholds(min_rounds=1)).decide(_inp(POOL))
    state, questions = [r for r in client.requests if "branches" in r[0]][0]
    assert [s["question"] for s in state["subquestions"]] == ["sub one", "sub two"]
    assert state["subquestions"][0]["evidence"][0].startswith("A ")  # A is sub one's top supporter
    assert state["subquestions"][1]["evidence"][0].startswith("B ")
    assert [b["subquery"] for b in state["branches"]] == ["branch A", "branch B"]
    assert state["branches"][0]["evidence"][0].startswith("A ")  # closest retained passage by cosine
    assert set(questions) == {"sub_0", "sub_1", "branch_0", "branch_1"}

    # sub two is answered with p=0.3 < 0.75: continue.
    assert dec.terminate is False
    assert dec.meta["subquestion_answered"] == {"sub one": 0.9, "sub two": 0.3}
    # Effort goes to the branch that is not answered yet: (1-0.9+0.05) : (1-0.2+0.05).
    assert dec.branch_allocation["nB"] > dec.branch_allocation["nA"]
    assert dec.branch_allocation == pytest.approx({"nA": 0.15 / 1.0, "nB": 0.85 / 1.0})


async def test_terminates_when_every_subquestion_is_answered_after_min_rounds():
    stop = dict(STOP, sub_1=0.8)
    client = FakeClient(lambda q, s: stop[q] if q in stop else script(q, s))
    pol = _policy(client, thresholds=TypeSafeThresholds(min_rounds=2))
    first = await pol.decide(_inp(POOL, round_id=1))
    assert first.terminate is False
    second = await pol.decide(_inp([], prev=POOL, round_id=2))
    assert second.terminate is True
    assert second.meta["terminate_reason"] == "all_subquestions_answered"


async def test_no_subquestions_never_terminates_but_still_allocates():
    client = FakeClient(script)
    dec = await _policy(client, thresholds=TypeSafeThresholds(min_rounds=1)).decide(_inp(POOL, subq=()))
    assert dec.terminate is False
    assert dec.branch_allocation["nB"] > dec.branch_allocation["nA"]


# ---------------------------------------------------------------- failure and metering


async def test_failed_judging_request_keeps_its_items_and_flags_them():
    client = FakeClient(script, fail=lambda state: "evidence" in state)
    dec = await _policy(client).decide(_inp(POOL))
    # Unjudged items are kept unconditionally; embedding dedup still runs.
    assert dec.kept_ids == {"A", "B", "NAV", "OFF"}
    assert dec.meta["pruned_reasons"] == {"DUP": "duplicate"}
    assert all(v["fallback"] for v in dec.meta["verdicts"].values())
    assert dec.meta["round_usage"]["errors"] == 1  # the single judging batch
    assert dec.terminate is False


async def test_failed_stop_request_falls_back_to_uniform_and_continue():
    client = FakeClient(script, fail=lambda state: "branches" in state)
    dec = await _policy(client, thresholds=TypeSafeThresholds(min_rounds=1)).decide(_inp(POOL))
    assert dec.kept_ids == {"A", "B"}
    assert dec.terminate is False
    assert dec.meta["stop_error"] is True
    assert dec.branch_allocation == pytest.approx({"nA": 0.5, "nB": 0.5})


async def test_usage_is_metered_per_request_under_the_orchestrator_tag():
    client = FakeClient(script)
    pol = _policy(client)
    dec = await pol.decide(_inp(POOL))
    assert pol.meter_calls == [("jev-1.13.0", 100, 5)] * 2  # one judging batch + one stop request
    assert dec.meta["usage"]["requests"] == 2
    assert dec.meta["usage"]["input_tokens"] == 200
    assert dec.meta["usage"]["usd"] == pytest.approx(200 * 0.042 / 1e6)
    assert dec.meta["usage"]["models"] == ["jev-1.13.0"]


def test_thresholds_validate_ranges():
    with pytest.raises(ValueError):
        TypeSafeThresholds(keep_min=1.5)
    with pytest.raises(ValueError):
        TypeSafeThresholds(batch_size=0)


def test_item_verdict_defaults():
    assert ItemVerdict(0.5, 0.1, [0.2]).fallback is False


# ---------------------------------------------------------------- selection


def test_build_from_env_requires_key_and_reads_thresholds(monkeypatch):
    monkeypatch.setenv("GR_ORCHESTRATOR", "typesafe")
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    assert policy_name_from_env() == "typesafe"
    with pytest.raises(RuntimeError):
        build_policy()

    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    monkeypatch.setenv("GR_CONTEXT_BUDGET_TOKENS", "5000")
    monkeypatch.setenv("GR_TYPESAFE_MODEL", "jev-latest")
    monkeypatch.setenv("GR_TYPESAFE_KEEP_MIN", "0.6")
    monkeypatch.setenv("GR_TYPESAFE_STOP_MIN", "0.8")
    monkeypatch.setenv("GR_TYPESAFE_MIN_ROUNDS", "3")
    monkeypatch.setenv("GR_TYPESAFE_BATCH", "10")
    monkeypatch.setenv("GR_FEAT_TAU", "0.9")
    pol = build_policy()
    assert isinstance(pol, TypeSafePolicy)
    assert pol.budget == 5000 and pol.model == "jev-latest"
    assert (pol.t.keep_min, pol.t.stop_min, pol.t.min_rounds, pol.t.batch_size, pol.t.dedup_tau) == (
        0.6, 0.8, 3, 10, 0.9,
    )
    assert pol.needs_embeddings and pol.needs_frontier_embeddings

    monkeypatch.delenv("GR_TYPESAFE_MODEL")
    assert build_policy().model == "jev-1.13.0"
