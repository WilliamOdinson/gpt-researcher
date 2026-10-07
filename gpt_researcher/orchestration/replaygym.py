"""ReplayGym expansion policy (GR_ORCHESTRATOR=replaygym).

A small MLP, trained with GRPO in the exact-replay environment (harness:
adr.replay.rl), decides at each checkpoint whether a branch is worth
researching deeper. Evidence retention is untouched (keep everything, uniform
allocation, like ``none``); only the expansion decision changes:

  * first-round checkpoint (tree depth 1): one "go deeper?" per top-level branch;
  * a branch's children checkpoint (tree depth 2): one "go to full depth?" for
    that branch's children as a group.

Features must match adr.replay.rl.EpisodeState.features exactly (same inputs,
same embedding model). Env:
  GR_REPLAYGYM_POLICY      JSON weights exported by scripts/replaygym_export.py
  GR_REPLAYGYM_EMBED_URL   OpenAI-style /v1/embeddings for bge-small (default :8323)
  GR_REPLAYGYM_THRESHOLD   go deeper if p >= threshold (default 0.5)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Callable, Sequence

import numpy as np

logger = logging.getLogger(__name__)

N_FEAT = 10


def _http_embed(texts: Sequence[str]) -> np.ndarray:
    import requests

    url = os.environ.get("GR_REPLAYGYM_EMBED_URL", "http://127.0.0.1:8323/v1/embeddings")
    r = requests.post(url, json={"input": [t[:4000] for t in texts]}, timeout=300)
    r.raise_for_status()
    m = np.array([d["embedding"] for d in r.json()["data"]], dtype=np.float32)
    return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)


def full_tree_size(breadth: int, depth: int) -> int:
    """Sub-research count of the fork's fixed tree (child breadth max(2, b//2))."""
    total, width, b = 0, 1, breadth
    for _ in range(depth):
        width *= b
        total += width
        b = max(2, b // 2)
    return total


class ExpansionPolicy:
    def __init__(self, weights: dict, threshold: float = 0.5, embed: Callable[[Sequence[str]], np.ndarray] = _http_embed):
        self.w1 = np.asarray(weights["w1"], np.float32)  # (hidden, N_FEAT)
        self.b1 = np.asarray(weights["b1"], np.float32)
        self.w2 = np.asarray(weights["w2"], np.float32)  # (1, hidden)
        self.b2 = np.asarray(weights["b2"], np.float32)
        self.threshold = threshold
        self.embed = embed
        self.query_vec: np.ndarray | None = None
        self.nodes: dict[str, dict] = {}  # researched node -> {"goal": vec, "learn": vec}

    @classmethod
    def from_env(cls) -> "ExpansionPolicy":
        path = os.environ.get("GR_REPLAYGYM_POLICY")
        if not path:
            raise ValueError("GR_ORCHESTRATOR=replaygym needs GR_REPLAYGYM_POLICY (weights JSON)")
        with open(path, encoding="utf-8") as f:
            w = json.load(f)
        return cls(w, float(os.environ.get("GR_REPLAYGYM_THRESHOLD", "0.5")))

    # -- state ------------------------------------------------------------------
    def set_query(self, query: str) -> None:
        self.query_vec = self.embed([query])[0]

    def observe(self, node_ids: Sequence[str], goals: Sequence[str], learnings: Sequence[Sequence[str]]) -> None:
        """Record a round's researched nodes (goal text + joined findings)."""
        texts = list(goals) + [" ".join(l) or "(none)" for l in learnings]
        vecs = self.embed(texts)
        k = len(node_ids)
        for i, nid in enumerate(node_ids):
            self.nodes[nid] = {"goal": vecs[i], "learn": vecs[k + i]}

    # -- decisions --------------------------------------------------------------
    def features(self, parents: Sequence[str], level: int, n_full: int) -> np.ndarray:
        q = self.query_vec
        pg = np.stack([self.nodes[p]["goal"] for p in parents])
        pv = np.stack([self.nodes[p]["learn"] for p in parents])
        rest_ids = [n for n in self.nodes if n not in set(parents)]
        rest = np.stack([self.nodes[n]["learn"] for n in rest_ids]) if rest_ids else np.zeros((0, len(q)), np.float32)
        f = np.zeros(N_FEAT, np.float32)
        f[level - 1] = 1.0
        f[3] = float((pg @ q).mean())
        f[4] = float((pv @ rest.T).max()) if len(rest) else 0.0
        f[5] = float(np.clip(1.0 - f[4], 0.0, 1.0)) if len(rest) else 1.0
        f[6] = float((pv @ q).mean())
        f[7] = len(self.nodes) / max(1, n_full)
        f[9] = 1.0
        return f

    def p_deeper(self, f: np.ndarray) -> float:
        h = np.tanh(self.w1 @ f + self.b1)
        z = float((self.w2 @ h + self.b2).reshape(-1)[0])
        return 1.0 / (1.0 + np.exp(-z))

    def decide(self, node_ids: Sequence[str], tree_depth: int, n_full: int) -> tuple[set[str], list[dict]]:
        """Return (node ids NOT to expand, decision log) for this checkpoint."""
        if tree_depth == 1:
            groups = [[n] for n in node_ids]      # one decision per top-level branch
        else:
            groups = [list(node_ids)]              # one decision for this branch's children
        skip, log = set(), []
        for g in groups:
            f = self.features(g, tree_depth + 1, n_full)
            p = self.p_deeper(f)
            go = p >= self.threshold
            if not go:
                skip.update(g)
            log.append({"nodes": list(g), "p_deeper": round(p, 4), "expand": bool(go)})
        return skip, log
