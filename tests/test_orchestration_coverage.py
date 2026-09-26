"""Greedy marginal utility and heuristic stopping on a fixed synthetic pool.

Embeddings are 3-d unit vectors so every cosine is exact:
  sub-questions  s1 = (1, 0, 0)      s2 = (0, 1, 0)
  root query     q  = (1, 0, 0)
  A  = s1 (covers s1 fully)      B  = s2 (covers s2 fully)
  A2 = s1 (exact duplicate of A) C  = (0.6, 0.8, 0) (covers both partly)
  D  = (0, 0, 1)                  (covers nothing)
Token estimate is len(text) // 4; every item below is 100 tokens.
"""

from __future__ import annotations

import pytest

from gpt_researcher.orchestration import (
    FrontierInfo,
    GreedyMarginalUtilityPolicy,
    HeuristicStopPolicy,
    OrchestrationInput,
    PoolItem,
    build_policy,
    policy_name_from_env,
)

S1 = [1.0, 0.0, 0.0]
S2 = [0.0, 1.0, 0.0]


def _item(iid, emb, *, new=True, rnd=2, tokens=100, chunks=None, chunk_kept=None, filter_kept=True):
    text = "\n\n".join(chunks) if chunks else "x" * (tokens * 4)
    return PoolItem(
        item_id=iid,
        source_url=f"https://example.org/{iid}",
        text=text,
        embedding=emb,
        tree_depth=1,
        retrieval_round=rnd,
        is_new=new,
        chunks=chunks or [],
        chunk_kept=chunk_kept or [],
        filter_kept=filter_kept,
    )


FRONTIER = [FrontierInfo("nA", "branch A"), FrontierInfo("nB", "branch B")]


def _inp(new, prev=(), *, budget=None, subq=(S1, S2), round_id=2, query=S1):
    return OrchestrationInput(
        root_query="what is x",
        query_embedding=query,
        subquestions=[f"sub {i}" for i in range(len(subq))],
        new_items=list(new),
        retained_prev=list(prev),
        frontier=FRONTIER,
        round_id=round_id,
        tree_depth=1,
        tokens_used=1234,
        token_budget=budget,
        subquestion_embeddings=[list(v) for v in subq],
    )


A = _item("A", S1)
B = _item("B", S2)
A2 = _item("A2", S1)
C = _item("C", [0.6, 0.8, 0.0])
D = _item("D", [0.0, 0.0, 1.0])
POOL = [A, B, A2, C, D]


# ---------------------------------------------------------------- greedy


async def test_greedy_covers_first_then_novel_items_and_drops_duplicates():
    dec = await GreedyMarginalUtilityPolicy().decide(_inp(POOL))
    # C has the largest first gain (0.7); then A (residual 0.4 on s1, novelty
    # 0.4 vs C) beats B (residual 0.2, novelty 0.2); then B. A2 is an exact
    # duplicate of A (novelty 0) and D covers nothing, so both have zero
    # marginal value and are never added even with no budget.
    assert dec.meta["order"] == ["C", "A", "B"]
    assert dec.kept_ids == {"C", "A", "B"}
    assert dec.meta["stop_reason"] == "no_positive_gain"
    assert dec.terminate is False
    assert dec.rewritten == {}
    assert dec.branch_allocation == pytest.approx({"nA": 0.5, "nB": 0.5})
    assert dec.meta["values"]["C"] == pytest.approx(0.7)
    assert dec.meta["values"]["A"] == pytest.approx(0.2 * 0.4)
    assert dec.meta["values"]["B"] == pytest.approx(0.1 * 0.2)
    # Paper's Phi over the kept set: both sub-questions covered with cos 1.
    assert dec.meta["phi_after"] == pytest.approx(1.0)


async def test_greedy_min_gain_stops_early():
    dec = await GreedyMarginalUtilityPolicy(min_gain=0.05).decide(_inp(POOL))
    assert dec.meta["order"] == ["C", "A"]
    assert dec.meta["stop_reason"] == "min_gain"


async def test_greedy_respects_token_budget():
    dec = await GreedyMarginalUtilityPolicy().decide(_inp(POOL, budget=200))
    assert dec.kept_ids == {"C", "A"}
    assert dec.meta["tokens_kept"] == 200
    assert dec.meta["stop_reason"] == "budget"


async def test_greedy_reads_budget_from_input_when_not_configured():
    dec = await GreedyMarginalUtilityPolicy(budget=None).decide(_inp(POOL, budget=100))
    assert dec.kept_ids == {"C"}


async def test_greedy_always_keeps_one_item_even_over_budget():
    dec = await GreedyMarginalUtilityPolicy().decide(_inp(POOL, budget=10))
    assert dec.kept_ids == {"C"}
    assert dec.meta["stop_reason"] == "forced_min_keep"


async def test_greedy_lambda_zero_disables_novelty_discount():
    dec = await GreedyMarginalUtilityPolicy(redundancy=0.0).decide(_inp(POOL))
    # Without the discount A and A2 tie on gain after C; the tie breaks on
    # pool order, and A2 then has zero residual gain, so it is still dropped.
    assert dec.meta["order"] == ["C", "A", "B"]


async def test_greedy_supersedes_previously_retained_items():
    prev = [_item("R", [0.0, 0.0, 1.0], new=False, rnd=1)]
    dec = await GreedyMarginalUtilityPolicy().decide(_inp([A, B], prev))
    assert dec.kept_ids == {"A", "B"}
    assert "R" not in dec.kept_ids


async def test_greedy_falls_back_to_relevance_without_subquestions():
    dec = await GreedyMarginalUtilityPolicy().decide(_inp([A, B, D], subq=()))
    assert dec.meta["coverage_source"] == "root_query"
    # A is the only item with positive cosine to the root query.
    assert dec.kept_ids == {"A"}


async def test_greedy_fills_budget_without_any_embeddings():
    items = [_item("P1", None), _item("P2", None), _item("P3", None)]
    dec = await GreedyMarginalUtilityPolicy().decide(_inp(items, subq=(), query=None, budget=200))
    assert dec.meta["coverage_source"] == "none"
    assert len(dec.kept_ids) == 2


def test_greedy_rejects_bad_hyperparameters():
    with pytest.raises(ValueError):
        GreedyMarginalUtilityPolicy(min_gain=-0.1)
    with pytest.raises(ValueError):
        GreedyMarginalUtilityPolicy(redundancy=-1.0)


# ---------------------------------------------------------------- heuristic_stop


async def test_heuristic_stop_reproduces_filter_verdict_at_chunk_level():
    prev = [_item("R1", [0.0, 0.0, 1.0], new=False, rnd=1, filter_kept=False)]
    new = [
        _item("K", S1, chunks=["k1", "p1", "k2"], chunk_kept=[True, False, True]),
        _item("F", S2, chunks=["f1", "f2"], chunk_kept=[True, True]),
        _item("P", S2, filter_kept=False, chunks=["z1"], chunk_kept=[False]),
        _item("M", S2, chunks=["m1", "m2"], chunk_kept=[True]),  # misaligned flags
    ]
    dec = await HeuristicStopPolicy(threshold=0.0).decide(_inp(new, prev))
    # Previously retained items always survive; new items follow the filter.
    assert dec.kept_ids == {"R1", "K", "F", "M"}
    assert dec.rewritten == {"K": "k1\n\nk2"}
    assert dec.meta["n_filter_pruned"] == 1
    assert dec.meta["compressed_chars"] == {"K": (len("k1\n\np1\n\nk2"), len("k1\n\nk2"))}
    assert dec.terminate is False
    assert dec.branch_allocation == pytest.approx({"nA": 0.5, "nB": 0.5})


async def test_heuristic_stop_retain_all_keeps_everything():
    new = [_item("P", S2, filter_kept=False, chunks=["z1", "z2"], chunk_kept=[False, True])]
    dec = await HeuristicStopPolicy(retain="all").decide(_inp(new))
    assert dec.kept_ids == {"P"}
    assert dec.rewritten == {}


async def test_heuristic_stop_gain_and_threshold():
    # K_{t-1} = {R1}: cos(R1, s1) = 0.5, so Phi(K_{t-1}) = 0.5 with one
    # sub-question; the new page A covers s1 fully, so Phi(K_t) = 1.0.
    prev = [_item("R1", [0.5, 0.8660254, 0.0], new=False, rnd=1)]
    inp = _inp([A], prev, subq=(S1,))

    pol = HeuristicStopPolicy(threshold=0.6, min_rounds=1)
    dec = await pol.decide(inp)
    assert dec.meta["phi_prev"] == pytest.approx(0.5, abs=1e-4)
    assert dec.meta["phi_after"] == pytest.approx(1.0, abs=1e-4)
    assert dec.meta["gain"] == pytest.approx(0.5, abs=1e-4)
    assert dec.terminate is True
    assert dec.meta["terminate_reason"] == "coverage_gain_below_threshold"

    dec = await HeuristicStopPolicy(threshold=0.4, min_rounds=1).decide(inp)
    assert dec.terminate is False


async def test_heuristic_stop_honours_min_rounds_and_patience():
    inp = _inp([A], subq=(S1,), round_id=2)

    # Round 2 < min_rounds 3: the rule is not evaluated at all.
    pol = HeuristicStopPolicy(threshold=10.0, min_rounds=3)
    dec = await pol.decide(inp)
    assert dec.terminate is False and dec.meta["low_streak"] == 0

    # Patience 2: the first low-gain round only starts the streak.
    pol = HeuristicStopPolicy(threshold=10.0, min_rounds=1, patience=2)
    first = await pol.decide(inp)
    assert first.terminate is False and first.meta["low_streak"] == 1
    second = await pol.decide(_inp([A], subq=(S1,), round_id=3))
    assert second.terminate is True and second.meta["low_streak"] == 2
    assert len(pol.history) == 2

    # A round at or above the threshold resets the streak.
    pol = HeuristicStopPolicy(threshold=0.9, min_rounds=1, patience=2)
    # R1 already covers s1 fully, so A adds nothing: gain 0.
    low = await pol.decide(_inp([A], [_item("R1", S1, new=False, rnd=1)], subq=(S1,)))
    assert low.meta["gain"] == pytest.approx(0.0, abs=1e-6) and low.meta["low_streak"] == 1
    # Empty K_{t-1}: gain = Phi({A}) = 1.0 >= 0.9.
    reset = await pol.decide(_inp([A], subq=(S1,), round_id=3))
    assert reset.meta["low_streak"] == 0 and reset.terminate is False


async def test_heuristic_stop_never_terminates_without_embeddings():
    dec = await HeuristicStopPolicy(threshold=10.0, min_rounds=1).decide(_inp([A], subq=()))
    assert dec.terminate is False
    assert dec.meta["stop_enabled"] is False
    assert dec.meta["terminate_reason"] == "stop_disabled_no_embeddings"

    dec = await HeuristicStopPolicy(threshold=10.0, min_rounds=1).decide(_inp([_item("N", None)], subq=(S1,)))
    assert dec.terminate is False and dec.meta["stop_enabled"] is False


def test_heuristic_stop_rejects_bad_retain_mode():
    with pytest.raises(ValueError):
        HeuristicStopPolicy(retain="some")


# ---------------------------------------------------------------- selection


def test_new_policies_build_from_env(monkeypatch):
    monkeypatch.setenv("GR_ORCHESTRATOR", "greedy")
    monkeypatch.setenv("GR_CONTEXT_BUDGET_TOKENS", "5000")
    monkeypatch.setenv("GR_GREEDY_MIN_GAIN", "0.02")
    monkeypatch.setenv("GR_GREEDY_LAMBDA", "0.5")
    assert policy_name_from_env() == "greedy"
    pol = build_policy()
    assert isinstance(pol, GreedyMarginalUtilityPolicy)
    assert (pol.budget, pol.min_gain, pol.redundancy) == (5000, 0.02, 0.5)

    monkeypatch.setenv("GR_ORCHESTRATOR", "heuristic_stop")
    monkeypatch.setenv("GR_STOP_GAIN_THRESHOLD", "0.03")
    monkeypatch.setenv("GR_STOP_MIN_ROUNDS", "4")
    monkeypatch.setenv("GR_STOP_PATIENCE", "2")
    monkeypatch.setenv("GR_STOP_RETAIN", "ALL")
    pol = build_policy()
    assert isinstance(pol, HeuristicStopPolicy)
    assert (pol.threshold, pol.min_rounds, pol.patience, pol.retain) == (0.03, 4, 2, "all")

    for name in ("GR_STOP_GAIN_THRESHOLD", "GR_STOP_MIN_ROUNDS", "GR_STOP_PATIENCE", "GR_STOP_RETAIN"):
        monkeypatch.delenv(name)
    pol = build_policy()
    assert (pol.threshold, pol.min_rounds, pol.patience, pol.retain) == (0.01, 2, 1, "filter")
