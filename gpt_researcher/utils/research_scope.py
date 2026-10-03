"""Context-local attribution of retrieved evidence to a frontier branch.

``DeepResearchSkill.deep_research`` runs one sub-researcher per frontier
branch; the embedding-cache records that feed the checkpoint are produced
deep inside that sub-researcher (``ContextCompressor``), and sub-researchers
of one node run concurrently, so a record cannot be tied to its branch by
timing. Each sub-researcher therefore runs under a ``ContextVar`` holding its
frontier node id. asyncio tasks copy the context they are created in, so the
id follows the work through ``asyncio.gather`` and ``asyncio.to_thread``
without any plumbing through GPTResearcher, and the checkpoint reads it back
from the record as ``source_node_id``.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

_branch: ContextVar[str | None] = ContextVar("gr_research_branch", default=None)


def current_branch() -> str | None:
    return _branch.get()


@contextmanager
def branch_scope(branch: str) -> Iterator[str]:
    """Attribute everything recorded inside the block to ``branch``."""
    token = _branch.set(branch)
    try:
        yield branch
    finally:
        _branch.reset(token)
