"""Compact per-item and frontier features for the orchestrator (PILOT §4.2).

Every function here is pure and works on plain lists/dicts so the same code
computes features online (inside a policy, from ``OrchestrationInput``) and
offline (from a saved trajectory JSON + ``_emb.npz``). The online path dumps
the result into ``RoundDecision.meta["features"]`` so behavior-cloning data is
built from exactly what the policy saw.

Per-item vector, in this order (``FEATURE_NAMES``):

  rho    cosine(e(q), e(c_i))                              relevance to root query
  nov    1 - max_{c_j in K_{t-1}} cosine(e(c_i), e(c_j))   novelty vs. retained set
  red    |{j != i : cosine(e(c_i), e(c_j)) > tau}| / n_t   near-duplicate density
  cov    (1/M) sum_k 1[cosine(e(c_i), e(q^(k))) > tau_c]   sub-question coverage
  tok    estimated token length
  depth  tree depth of the node that retrieved the item
  age    rounds since retrieval (round_id - retrieval_round)
  src    source tier: 2 = .gov/.edu, 1 = .org/wikipedia, 0 = other

``tau`` / ``tau_c`` default to 0.85 / 0.50 and can be set with
``GR_FEAT_TAU`` / ``GR_FEAT_TAU_C`` (embedding-model dependent).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Sequence
from urllib.parse import urlparse

import numpy as np

FEATURE_NAMES = ("rho", "nov", "red", "cov", "tok", "depth", "age", "src")

ENV_TAU = "GR_FEAT_TAU"
ENV_TAU_C = "GR_FEAT_TAU_C"
DEFAULT_TAU = 0.85
DEFAULT_TAU_C = 0.50


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


def thresholds() -> tuple[float, float]:
    return _env_float(ENV_TAU, DEFAULT_TAU), _env_float(ENV_TAU_C, DEFAULT_TAU_C)


def estimate_tokens(text: str) -> int:
    return max(1, len(text or "") // 4)


def source_tier(url: str) -> int:
    host = (urlparse(url or "").hostname or "").lower()
    if host.endswith(".gov") or host.endswith(".edu") or ".gov." in host or ".edu." in host:
        return 2
    if host.endswith(".org") or "wikipedia." in host:
        return 1
    return 0


def source_domain(url: str) -> str:
    host = (urlparse(url or "").hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def _unit_matrix(vectors: Sequence[Sequence[float] | None], dim: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Stack vectors into a unit-normalised matrix; rows with no vector are
    zero and flagged in the returned boolean mask."""
    rows = [np.asarray(v, dtype=np.float32) for v in vectors if v is not None and len(v) > 0]
    if dim is None:
        dim = int(rows[0].shape[0]) if rows else 1
    mat = np.zeros((len(vectors), dim), dtype=np.float32)
    have = np.zeros(len(vectors), dtype=bool)
    for i, v in enumerate(vectors):
        if v is None or len(v) == 0:
            continue
        arr = np.asarray(v, dtype=np.float32)
        if arr.shape[0] != dim:
            continue
        norm = float(np.linalg.norm(arr))
        if norm > 0:
            mat[i] = arr / norm
            have[i] = True
    return mat, have


@dataclass
class ItemRow:
    """Minimal per-item record, independent of PoolItem so trajectory JSON can
    be replayed through the same code."""

    item_id: str
    source_url: str
    text: str
    embedding: list[float] | None
    tree_depth: int
    retrieval_round: int
    is_new: bool
    source_subquery: str = ""


@dataclass
class FrontierRow:
    node_id: str
    subquery: str
    embedding: list[float] | None = None


@dataclass
class FeatureBundle:
    names: tuple[str, ...]
    features: dict[str, list[float]]          # item_id -> vector in FEATURE_NAMES order
    sources: dict[str, str]                   # item_id -> domain
    frontier: dict[str, dict[str, float]]     # node_id -> {"gap": ..., "n_items": ...}
    phi_prev: float                           # coverage potential of K_{t-1}
    phi_pool: float                           # coverage potential of C_t (whole pool)
    tau: float
    tau_c: float
    extra: dict[str, Any] = field(default_factory=dict)

    def as_meta(self) -> dict[str, Any]:
        return {
            "feature_names": list(self.names),
            "features": {k: [round(float(x), 4) for x in v] for k, v in self.features.items()},
            "sources": dict(self.sources),
            "frontier_stats": {k: {kk: round(float(vv), 4) for kk, vv in v.items()} for k, v in self.frontier.items()},
            "phi_prev": round(float(self.phi_prev), 4),
            "phi_pool": round(float(self.phi_pool), 4),
            "tau": self.tau,
            "tau_c": self.tau_c,
        }


def coverage_potential(item_vectors: np.ndarray, have: np.ndarray, subq: np.ndarray, subq_have: np.ndarray) -> float:
    """Phi(K) = (1/M) sum_k max_{c in K} cos(e(c), e(q^(k))) (Eq. 15)."""
    if item_vectors.shape[0] == 0 or not have.any() or subq.shape[0] == 0 or not subq_have.any():
        return 0.0
    sims = item_vectors[have] @ subq[subq_have].T  # (n_items, M)
    return float(np.mean(np.max(sims, axis=0)))


def coverage_potential_for(
    kept_ids: set[str],
    items: Sequence[ItemRow],
    subquestion_embeddings: Sequence[Sequence[float] | None],
) -> float:
    """Phi over the subset of ``items`` whose id is in ``kept_ids``."""
    chosen = [it for it in items if it.item_id in kept_ids]
    if not chosen:
        return 0.0
    dim = _dim_of(items, subquestion_embeddings)
    mat, have = _unit_matrix([it.embedding for it in chosen], dim)
    sq, sq_have = _unit_matrix(list(subquestion_embeddings), dim)
    return coverage_potential(mat, have, sq, sq_have)


def _dim_of(items: Sequence[ItemRow], *others: Sequence[Sequence[float] | None]) -> int:
    for it in items:
        if it.embedding:
            return len(it.embedding)
    for seq in others:
        for v in seq:
            if v:
                return len(v)
    return 1


def compute_features(
    *,
    query_embedding: Sequence[float] | None,
    subquestion_embeddings: Sequence[Sequence[float] | None],
    items: Sequence[ItemRow],
    frontier: Sequence[FrontierRow],
    round_id: int,
    tau: float | None = None,
    tau_c: float | None = None,
) -> FeatureBundle:
    """Compute x_i for every item in the pool plus frontier coverage gaps.

    ``items`` is the pool the policy sees this round: previously retained
    items (``is_new=False``, together they are K_{t-1}) and this round's new
    items (``is_new=True``).
    """
    env_tau, env_tau_c = thresholds()
    tau = env_tau if tau is None else tau
    tau_c = env_tau_c if tau_c is None else tau_c

    n = len(items)
    dim = _dim_of(items, subquestion_embeddings, [query_embedding])
    mat, have = _unit_matrix([it.embedding for it in items], dim)
    q, q_have = _unit_matrix([query_embedding], dim)
    sq, sq_have = _unit_matrix(list(subquestion_embeddings), dim)
    fr, fr_have = _unit_matrix([f.embedding for f in frontier], dim)

    # Pairwise item similarity; diagonal excluded where used.
    sim = mat @ mat.T if n else np.zeros((0, 0), dtype=np.float32)
    rho = (mat @ q.T)[:, 0] if q_have.any() else np.zeros(n, dtype=np.float32)

    prev_mask = np.asarray([not it.is_new for it in items], dtype=bool) & have
    nov = np.ones(n, dtype=np.float32)
    if prev_mask.any():
        sim_prev = sim[:, prev_mask].copy()  # (n, n_prev)
        # An item retained earlier must not count itself as its own neighbour.
        prev_idx = np.flatnonzero(prev_mask)
        for col, i in enumerate(prev_idx):
            sim_prev[i, col] = -1.0
        best = np.max(sim_prev, axis=1)
        best = np.where(np.isfinite(best), best, 0.0)
        nov = 1.0 - np.clip(best, -1.0, 1.0)
        # Items without an embedding: no information, treat as novel.
        nov = np.where(have, nov, 1.0)
        # If an item is the only previously retained one, its "best" is -1 -> nov 2; clamp.
        nov = np.clip(nov, 0.0, 1.0)

    red = np.zeros(n, dtype=np.float32)
    if n > 1:
        over = (sim > tau) & have[:, None] & have[None, :]
        np.fill_diagonal(over, False)
        red = over.sum(axis=1).astype(np.float32) / float(n)

    cov = np.zeros(n, dtype=np.float32)
    if sq_have.any():
        cs = mat @ sq[sq_have].T  # (n, M_have)
        m_total = max(1, sq.shape[0])
        cov = (cs > tau_c).sum(axis=1).astype(np.float32) / float(m_total)
        cov = np.where(have, cov, 0.0)

    features: dict[str, list[float]] = {}
    sources: dict[str, str] = {}
    for i, it in enumerate(items):
        features[it.item_id] = [
            float(rho[i]) if have[i] else 0.0,
            float(nov[i]),
            float(red[i]),
            float(cov[i]),
            float(estimate_tokens(it.text)),
            float(it.tree_depth),
            float(max(0, round_id - it.retrieval_round)),
            float(source_tier(it.source_url)),
        ]
        sources[it.item_id] = source_domain(it.source_url)

    # Frontier: coverage gap of each open branch given the *retained* pool,
    # and how many pool items were retrieved under that branch's sub-query.
    frontier_stats: dict[str, dict[str, float]] = {}
    retained_mask = prev_mask  # K_{t-1}
    for j, f in enumerate(frontier):
        gap = 1.0
        if fr_have[j] and retained_mask.any():
            gap = 1.0 - float(np.max(mat[retained_mask] @ fr[j]))
        elif fr_have[j] and have.any():
            gap = 1.0 - float(np.max(mat[have] @ fr[j]))
        n_items = sum(1 for it in items if it.source_subquery and it.source_subquery == f.subquery)
        frontier_stats[f.node_id] = {"gap": float(np.clip(gap, 0.0, 1.0)), "n_items": float(n_items)}

    phi_prev = coverage_potential(mat[prev_mask], have[prev_mask], sq, sq_have) if prev_mask.any() else 0.0
    phi_pool = coverage_potential(mat, have, sq, sq_have)

    return FeatureBundle(
        names=FEATURE_NAMES,
        features=features,
        sources=sources,
        frontier=frontier_stats,
        phi_prev=phi_prev,
        phi_pool=phi_pool,
        tau=tau,
        tau_c=tau_c,
    )
