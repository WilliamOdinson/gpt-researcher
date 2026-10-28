"""Learnings reach the writer in discovery order, the same in every process.

Branch replay (harness adr.replay.tree) rebuilds the writer input of any
sub-tree from the cached nodes in run order; that only reproduces the live
input if the live run orders learnings the same way.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from gpt_researcher.skills.deep_research import unique_in_order


def test_unique_in_order_keeps_first_occurrence():
    assert unique_in_order(["b", "a", "b", "c", "a"]) == ["b", "a", "c"]
    assert unique_in_order([]) == []


def test_order_does_not_depend_on_the_hash_seed():
    root = Path(__file__).resolve().parents[1]
    code = (
        "from gpt_researcher.skills.deep_research import unique_in_order;"
        "print(unique_in_order([f'learning {i}' for i in range(40)] * 2))"
    )
    outs = set()
    for seed in ("1", "2", "3"):
        env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": str(root)}
        outs.add(subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True).stdout)
    assert len(outs) == 1
