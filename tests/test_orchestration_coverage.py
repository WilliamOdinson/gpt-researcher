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

# Frontier branches for the pruning tests; items are attributed to a branch
# through PoolItem.source_node_id (the checkpoint fills it from the research
# scope of the sub-researcher that retrieved the page).
BRANCH_A = FrontierInfo("nA", "branch A")
BRANCH_B = FrontierInfo("nB", "branch B")


def _binp(new, prev=(), *, subq=(S1, S2), round_id=2, frontier=(BRANCH_A, BRANCH_B)):
    inp = _inp(new, prev, subq=subq, round_id=round_id)
    inp.frontier = list(frontier)
    return inp


def _bitem(iid, emb, branch, **kw):
    it = _item(iid, emb, **kw)
    it.source_node_id = branch
    return it


async def test_heuristic_stop_reproduces_filter_verdict_at_chunk_level():
    prev = [_item("R1", [0.0, 0.0, 1.0], new=False, rnd=1, filter_kept=False)]
    new = [
        _bitem("K", S1, "nA", chunks=["k1", "p1", "k2"], chunk_kept=[True, False, True]),
        _bitem("F", S2, "nB", chunks=["f1", "f2"], chunk_kept=[True, True]),
        _bitem("P", S2, "nB", filter_kept=False, chunks=["z1"], chunk_kept=[False]),
        _bitem("M", S2, "nB", chunks=["m1", "m2"], chunk_kept=[True]),  # misaligned flags
    ]
    dec = await HeuristicStopPolicy(prune_gain=0.0).decide(_binp(new, prev))
    # Previously retained items always survive; new items follow the filter.
    assert dec.kept_ids == {"R1", "K", "F", "M"}
    assert dec.rewritten == {"K": "k1\n\nk2"}
    assert dec.meta["n_filter_pruned"] == 1
    assert dec.meta["compressed_chars"] == {"K": (len("k1\n\np1\n\nk2"), len("k1\n\nk2"))}
    assert dec.terminate is False
    assert dec.pruned_branches == set()
    assert dec.branch_allocation == pytest.approx({"nA": 0.5, "nB": 0.5})


async def test_heuristic_stop_retain_all_keeps_everything():
    new = [_bitem("P", S2, "nA", filter_kept=False, chunks=["z1", "z2"], chunk_kept=[False, True])]
    dec = await HeuristicStopPolicy(retain="all").decide(_binp(new, frontier=(BRANCH_A,)))
    assert dec.kept_ids == {"P"}
    assert dec.rewritten == {}


async def test_heuristic_stop_prunes_only_the_unproductive_branch():
    # K_{t-1} = {R1} already covers s1 fully. Branch A returns another page
    # on s1 (gain 0); branch B returns B, which covers s2 (gain 0.5).
    prev = [_item("R1", S1, new=False, rnd=1)]
    new = [_bitem("A", S1, "nA"), _bitem("B", S2, "nB")]
    dec = await HeuristicStopPolicy(prune_gain=0.01).decide(_binp(new, prev))
    assert dec.meta["branch_gain"] == pytest.approx({"nA": 0.0, "nB": 0.5}, abs=1e-4)
    assert dec.pruned_branches == {"nA"}
    # Pruned evidence stays in the pool; only the expansion stops.
    assert dec.kept_ids == {"R1", "A", "B"}
    assert dec.branch_allocation == pytest.approx({"nA": 0.0, "nB": 1.0})
    assert dec.terminate is False
    assert dec.meta["phi_after"] == pytest.approx(1.0)


async def test_heuristic_stop_branch_gain_is_against_the_previous_pool_not_siblings():
    # Both branches return the same page: each still gets the full gain
    # over K_{t-1} = {}, so neither is pruned for duplicating the other.
    new = [_bitem("A", S1, "nA"), _bitem("A2", S1, "nB")]
    dec = await HeuristicStopPolicy(prune_gain=0.01).decide(_binp(new, subq=(S1,)))
    assert dec.meta["branch_gain"] == pytest.approx({"nA": 1.0, "nB": 1.0})
    assert dec.pruned_branches == set()


async def test_heuristic_stop_never_prunes_top_level_branches_at_the_root():
    # Round 1, empty pool: every branch that returned anything relevant has
    # gain Phi(N_v) > 0, so a run never loses a top-level branch unsearched.
    new = [_bitem("A", S1, "nA"), _bitem("C", [0.6, 0.8, 0.0], "nB")]
    dec = await HeuristicStopPolicy(prune_gain=0.01).decide(_binp(new, round_id=1))
    assert dec.pruned_branches == set()


async def test_heuristic_stop_prunes_a_branch_whose_items_the_filter_dropped():
    # Branch B's only page was pruned by the EmbeddingsFilter: with filter
    # retention it contributes nothing, so the branch is not expanded.
    new = [_bitem("A", S1, "nA"), _bitem("P", S2, "nB", filter_kept=False)]
    dec = await HeuristicStopPolicy(prune_gain=0.01).decide(_binp(new))
    assert dec.pruned_branches == {"nB"}
    dec = await HeuristicStopPolicy(prune_gain=0.01, retain="all").decide(_binp(new))
    assert dec.pruned_branches == set()


async def test_heuristic_stop_pruning_needs_branch_attribution():
    # Kept pages with no branch attribution at all: pruning would discard
    # every path on zero gain, so the round skips it and says why.
    new = [_item("A", S1), _item("B", S2)]
    dec = await HeuristicStopPolicy(prune_gain=0.01).decide(_binp(new))
    assert dec.pruned_branches == set()
    assert dec.meta["prune_disabled_reason"] == "no_branch_attribution"
    # Attribution by sub-query string still works for older checkpoints.
    a = _item("A", S1); a.source_subquery = "branch A"
    b = _item("B", S2); b.source_subquery = "branch B"
    prev = [_item("R1", S1, new=False, rnd=1)]
    dec = await HeuristicStopPolicy(prune_gain=0.01).decide(_binp([a, b], prev))
    assert dec.pruned_branches == {"nA"}


async def test_heuristic_stop_global_stop_only_on_satisfaction():
    # s1 is covered with cos 1 by A, s2 only with 0.8 by C.
    new = [_bitem("A", S1, "nA"), _bitem("C", [0.6, 0.8, 0.0], "nB")]
    dec = await HeuristicStopPolicy(satisfaction=0.9, min_rounds=1).decide(_binp(new))
    assert dec.meta["min_coverage"] == pytest.approx(0.8, abs=1e-4)
    assert dec.terminate is False
    dec = await HeuristicStopPolicy(satisfaction=0.75, min_rounds=1).decide(_binp(new))
    assert dec.terminate is True
    assert dec.meta["terminate_reason"] == "all_subquestions_covered"
    # Before min_rounds the run keeps going even when satisfied.
    dec = await HeuristicStopPolicy(satisfaction=0.75, min_rounds=3).decide(_binp(new, round_id=2))
    assert dec.terminate is False
    # The default (1.0) never terminates.
    pol = HeuristicStopPolicy(min_rounds=1)
    dec = await pol.decide(_binp(new))
    assert dec.terminate is False and dec.meta["stop_enabled"] is False
    assert len(pol.history) == 1


async def test_heuristic_stop_disables_both_rules_without_embeddings():
    new = [_bitem("A", S1, "nA")]
    dec = await HeuristicStopPolicy(prune_gain=10.0, satisfaction=0.1, min_rounds=1).decide(
        _binp(new, subq=(), frontier=(BRANCH_A,))
    )
    assert dec.terminate is False and dec.pruned_branches == set()
    assert dec.meta["terminate_reason"] == "stop_disabled_no_embeddings"
    assert dec.meta["prune_disabled_reason"] == "no_embeddings"


def test_heuristic_stop_rejects_bad_hyperparameters():
    with pytest.raises(ValueError):
        HeuristicStopPolicy(retain="some")
    with pytest.raises(ValueError):
        HeuristicStopPolicy(prune_gain=-1.0)
    with pytest.raises(ValueError):
        HeuristicStopPolicy(satisfaction=0.0)


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
    monkeypatch.setenv("GR_STOP_PRUNE_GAIN", "0.03")
    monkeypatch.setenv("GR_STOP_SATISFACTION", "0.8")
    monkeypatch.setenv("GR_STOP_MIN_ROUNDS", "4")
    monkeypatch.setenv("GR_STOP_RETAIN", "ALL")
    pol = build_policy()
    assert isinstance(pol, HeuristicStopPolicy)
    assert (pol.prune_gain, pol.satisfaction, pol.min_rounds, pol.retain) == (0.03, 0.8, 4, "all")

    for name in ("GR_STOP_PRUNE_GAIN", "GR_STOP_SATISFACTION", "GR_STOP_MIN_ROUNDS", "GR_STOP_RETAIN"):
        monkeypatch.delenv(name)
    pol = build_policy()
    assert (pol.prune_gain, pol.satisfaction, pol.min_rounds, pol.retain) == (0.01, 1.0, 2, "filter")
