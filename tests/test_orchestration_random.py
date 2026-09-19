"""RandomizedPolicy, feature computation, and the shared state/action format.

Same 3-d synthetic pool as test_orchestration_policies.py: q = (1, 0, 0),
P1 rho=1.0, P2 0.8, P3 0.6, P4 0.0 (new); R1 rho=0.9 retained from round 1.
"""

from __future__ import annotations

import json
import math

import pytest

from gpt_researcher.orchestration import (
    FEATURE_NAMES,
    PARAM_SPACE,
    FrontierInfo,
    FrontierRow,
    ItemRow,
    OrchestrationInput,
    PoolItem,
    RandomizedPolicy,
    StateFrontier,
    StateItem,
    build_policy,
    compute_features,
    coverage_potential_for,
    derive_seed,
    parse_action,
    serialize_action,
    serialize_state,
)


def _vec(rho: float) -> list[float]:
    return [rho, math.sqrt(max(0.0, 1 - rho * rho)), 0.0]


def _item(iid: str, rho: float, tokens: int = 100, *, new: bool = True, rnd: int = 2, url: str | None = None) -> PoolItem:
    return PoolItem(
        item_id=iid,
        source_url=url or f"https://example.org/{iid}",
        text="x" * (tokens * 4),
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


# ------------------------------------------------------------------ features


def test_features_match_definitions():
    rows = [
        ItemRow(it.item_id, it.source_url, it.text, it.embedding, it.tree_depth, it.retrieval_round, it.is_new, it.source_subquery)
        for it in [*PREV, *NEW]
    ]
    fr = [FrontierRow(f.node_id, f.subquery, f.embedding) for f in FRONTIER]
    b = compute_features(
        query_embedding=_vec(1.0),
        subquestion_embeddings=[_vec(1.0), _vec(0.4)],
        items=rows,
        frontier=fr,
        round_id=2,
        tau=0.85,
        tau_c=0.5,
    )
    assert b.names == FEATURE_NAMES
    f = {k: dict(zip(FEATURE_NAMES, v)) for k, v in b.features.items()}

    # rho is the exact cosine to the query.
    assert f["P1"]["rho"] == pytest.approx(1.0, abs=1e-5)
    assert f["P2"]["rho"] == pytest.approx(0.8, abs=1e-5)
    assert f["P4"]["rho"] == pytest.approx(0.0, abs=1e-5)
    # nov = 1 - max cos to K_{t-1} = {R1 (0.9)}.
    assert f["P1"]["nov"] == pytest.approx(1 - 0.9, abs=1e-4)
    assert f["P4"]["nov"] == pytest.approx(1 - math.sqrt(1 - 0.81), abs=1e-4)
    # R1 is the only retained item: nothing else in K_{t-1} to compare with.
    assert f["R1"]["nov"] == 1.0
    # red counts neighbours above tau over n_t = 5. P4 is far from everything.
    assert f["P4"]["red"] == 0.0
    assert f["P1"]["red"] > 0.0
    # cov: subq 1 == query direction, subq 2 at rho 0.4. tau_c = 0.5.
    assert f["P1"]["cov"] == pytest.approx(0.5)   # only sub 1
    assert f["P2"]["cov"] == pytest.approx(1.0)   # cos to sub 2 = .8*.4+.6*.917 > .5
    assert f["P4"]["cov"] == pytest.approx(0.5)   # cos to sub 2 = 0.917 > .5
    # tok / depth / age / src
    assert f["P1"]["tok"] == 100 and f["P1"]["depth"] == 1 and f["P1"]["age"] == 0
    assert f["R1"]["age"] == 1
    assert f["R1"]["src"] == 2 and f["P1"]["src"] == 1
    assert b.sources["R1"] == "data.gov"

    # Frontier gap uses the retained pool K_{t-1} = {R1}.
    assert b.frontier["nA"]["gap"] == pytest.approx(1 - (0.9 * 0.3 + math.sqrt(0.19) * math.sqrt(0.91)), abs=1e-4)
    assert b.frontier["nA"]["n_items"] == 4  # all new items came from "branch A"
    assert b.frontier["nB"]["n_items"] == 0
    # Phi over K_{t-1} and over the whole pool.
    assert b.phi_prev == pytest.approx(0.5 * (0.9 + (0.9 * 0.4 + math.sqrt(0.19) * math.sqrt(0.84))), abs=1e-4)
    assert b.phi_pool >= b.phi_prev


def test_features_without_embeddings_degrade_gracefully():
    rows = [ItemRow("a", "https://x", "text", None, 1, 1, True), ItemRow("b", "https://y", "t", None, 1, 1, False)]
    b = compute_features(query_embedding=None, subquestion_embeddings=[], items=rows, frontier=[FrontierRow("n", "q")], round_id=1)
    assert b.features["a"][:4] == [0.0, 1.0, 0.0, 0.0]
    assert b.frontier["n"]["gap"] == 1.0
    assert b.phi_prev == 0.0 and b.phi_pool == 0.0


def test_coverage_potential_for_subset():
    rows = [ItemRow(it.item_id, it.source_url, it.text, it.embedding, 1, 2, True) for it in NEW]
    phi_all = coverage_potential_for({"P1", "P2", "P3", "P4"}, rows, [_vec(1.0)])
    phi_p4 = coverage_potential_for({"P4"}, rows, [_vec(1.0)])
    assert phi_all == pytest.approx(1.0, abs=1e-5)
    assert phi_p4 == pytest.approx(0.0, abs=1e-5)
    assert coverage_potential_for(set(), rows, [_vec(1.0)]) == 0.0


# ------------------------------------------------------------------ policy


async def test_random_is_deterministic_for_seed_and_query():
    d1 = await RandomizedPolicy(seed=11).decide(_inp())
    d2 = await RandomizedPolicy(seed=11).decide(_inp())
    assert d1.kept_ids == d2.kept_ids
    assert d1.branch_allocation == d2.branch_allocation
    assert d1.terminate == d2.terminate
    assert d1.meta["params"] == d2.meta["params"]
    assert d1.meta["query_seed"] == derive_seed(11, "what is x")


async def test_random_explores_across_seeds():
    params = set()
    for s in range(12):
        d = await RandomizedPolicy(seed=s).decide(_inp())
        params.add(round(d.meta["params"]["keep_ratio"], 3))
    assert len(params) > 6  # different seeds -> different operating points


async def test_random_keep_count_follows_keep_ratio(monkeypatch):
    pol = RandomizedPolicy(seed=1, fixed_params={"keep_ratio": 0.5, "temperature": 0.001})
    d = await pol.decide(_inp())
    assert len(d.kept_ids) == math.ceil(0.5 * 5) == 3
    # With near-zero noise and default-ish weights the best items win.
    assert "P1" in d.kept_ids
    assert "P4" not in d.kept_ids


async def test_random_respects_token_budget():
    pol = RandomizedPolicy(seed=3, budget=200, fixed_params={"keep_ratio": 1.0, "budget_frac": 1.0})
    d = await pol.decide(_inp())
    assert d.meta["tokens_kept"] <= 200
    assert len(d.kept_ids) == 2  # two 100-token items fit


async def test_random_allocation_is_a_distribution_over_the_frontier():
    d = await RandomizedPolicy(seed=5).decide(_inp())
    assert set(d.branch_allocation) == {"nA", "nB"}
    assert sum(d.branch_allocation.values()) == pytest.approx(1.0)
    assert all(v >= 0 for v in d.branch_allocation.values())

    empty = await RandomizedPolicy(seed=5).decide(_inp(frontier=[]))
    assert empty.branch_allocation == {}


async def test_random_terminate_waits_for_min_rounds():
    pol = RandomizedPolicy(seed=2, fixed_params={"min_rounds": 3, "p_terminate": 1.0})
    assert (await pol.decide(_inp(round_id=1))).terminate is False
    assert (await pol.decide(_inp(round_id=2))).terminate is False
    d3 = await pol.decide(_inp(round_id=3))
    assert d3.terminate is True and d3.meta["terminate_reason"] == "bernoulli"


async def test_random_terminates_on_low_coverage_gain():
    pol = RandomizedPolicy(seed=2, fixed_params={"min_rounds": 1, "p_terminate": 0.0, "phi_gain_stop": 10.0})
    assert (await pol.decide(_inp(round_id=1))).terminate is False  # never on round 1
    d = await pol.decide(_inp(round_id=2))
    assert d.terminate is True and d.meta["terminate_reason"] == "phi_gain_below_threshold"


async def test_random_meta_has_everything_bc_needs():
    d = await RandomizedPolicy(seed=9).decide(_inp())
    m = d.meta
    assert m["policy"] == "random"
    assert set(m["features"]) == {"P1", "P2", "P3", "P4", "R1"}
    assert all(len(v) == len(FEATURE_NAMES) for v in m["features"].values())
    assert m["feature_names"] == list(FEATURE_NAMES)
    assert set(m["frontier_stats"]) == {"nA", "nB"}
    assert set(m["params"]) == set(PARAM_SPACE)
    for key in ("phi_prev", "phi_pool", "phi_after", "phi_gain", "tokens_used", "n_pool", "n_new", "scores", "sources"):
        assert key in m


def test_build_policy_random_reads_env(monkeypatch):
    monkeypatch.setenv("GR_ORCHESTRATOR", "random")
    monkeypatch.setenv("GR_ORCH_SEED", "42")
    monkeypatch.setenv("GR_RANDOM_PARAMS", json.dumps({"keep_ratio": 0.33}))
    pol = build_policy()
    assert isinstance(pol, RandomizedPolicy)
    assert pol.base_seed == 42 and pol.fixed_params == {"keep_ratio": 0.33}


def test_fixed_params_rejects_unknown_key():
    with pytest.raises(ValueError):
        RandomizedPolicy(seed=0, fixed_params={"nope": 1})._ensure_rng("q")


# ------------------------------------------------------------------ serialization


async def test_action_round_trips_through_parser():
    d = await RandomizedPolicy(seed=4, fixed_params={"keep_ratio": 0.6}).decide(_inp())
    pool_ids = [it.item_id for it in _inp().pool]
    text = serialize_action(kept_ids=d.kept_ids, pool_ids=pool_ids, branch_allocation=d.branch_allocation, terminate=d.terminate)
    parsed = parse_action(text, pool_ids, [f.node_id for f in FRONTIER])
    assert parsed.kept_ids == d.kept_ids
    assert parsed.terminate == d.terminate
    for nid, w in d.branch_allocation.items():
        assert parsed.branch_allocation[nid] == pytest.approx(w, abs=0.011)
    assert parsed.meta["parse_failure"] is False


def test_action_keep_all_and_empty_alloc():
    text = serialize_action(kept_ids={"a", "b"}, pool_ids=["a", "b"], branch_allocation={}, terminate=True)
    assert text == "KEEP: ALL\nALLOC: -\nDECISION: TERMINATE"
    parsed = parse_action(text, ["a", "b"], [])
    assert parsed.kept_ids == {"a", "b"} and parsed.terminate and parsed.branch_allocation == {}


def test_state_serialization_is_deterministic_and_compact():
    items = [StateItem("P1", [1.0, 0.1, 0.2, 0.5, 100, 1, 0, 1], True, "long text " * 100, "example.org")]
    fr = [StateFrontier("nA", "branch A", 0.314)]
    kwargs = dict(root_query="what is x", subquestions=["s1"], round_id=2, tree_depth=1, tokens_used=10,
                  token_budget=5000, items=items, frontier=fr, feature_names=FEATURE_NAMES)
    a = serialize_state(**kwargs)
    b = serialize_state(**kwargs)
    assert a == b
    line = next(l for l in a.splitlines() if l.strip().startswith("P1 |"))
    # features at 2 dp for floats, ints for tok/depth/age/src, snippet truncated
    assert "1.00 0.10 0.20 0.50 100 1 0 1 | new | example.org |" in line
    assert len(line) < 320
    assert "context budget 5000" in a and "nA | 0.31 | branch A" in a
