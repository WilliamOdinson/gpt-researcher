from types import SimpleNamespace

from gpt_researcher.skills.deep_research import (
    build_final_context,
    trim_context_to_word_limit,
)


def item(url, words, emb, rnd, depth=0):
    return SimpleNamespace(
        source_url=url,
        content="w " * words,
        embedding=emb,
        retrieval_round=rnd,
        tree_depth=depth,
    )


def test_learnings_survive_when_evidence_exceeds_cap():
    learnings = ["insight one [Source: https://a]", "insight two [Source: https://b]"]
    evidence = [item(f"https://e{i}", 400, [1.0, 0.0], i) for i in range(10)]

    # The old path drops the learnings: they come first and the trim keeps the tail.
    old = trim_context_to_word_limit(
        learnings + [f"Source: {e.source_url}\nContent: {e.content}\n" for e in evidence],
        max_words=1000,
    )
    assert not any("insight" in c for c in old)

    ctx, stats = build_final_context(learnings, evidence, max_words=1000, item_max_words=0)
    assert ctx[:2] == learnings
    assert stats["learnings_used"] == 2
    assert sum(len(c.split()) for c in ctx) <= 1000


def test_relevance_beats_recency():
    q = [1.0, 0.0]
    evidence = [
        item("https://relevant-early", 300, [1.0, 0.0], 1),
        item("https://offtopic-late", 300, [0.0, 1.0], 5),
    ]
    ctx, _ = build_final_context([], evidence, query_embedding=q, max_words=400, item_max_words=0)
    assert len(ctx) == 1 and "relevant-early" in ctx[0]


def test_round_robin_across_subquestions():
    q = [1.0, 1.0]
    subqs = [[1.0, 0.0], [0.0, 1.0]]
    evidence = [
        item("https://a1", 100, [1.0, 0.05], 1),
        item("https://a2", 100, [1.0, 0.04], 1),
        item("https://a3", 100, [1.0, 0.03], 1),
        item("https://b1", 100, [0.05, 1.0], 2),
    ]
    ctx, _ = build_final_context(
        [], evidence, query_embedding=q, subquestion_embeddings=subqs, max_words=250, item_max_words=0
    )
    urls = " ".join(ctx)
    assert "https://b1" in urls and "https://a1" in urls


def test_long_items_are_cut_and_output_in_retrieval_order():
    evidence = [
        item("https://late", 5000, [1.0, 0.0], 3),
        item("https://early", 5000, [0.9, 0.1], 1),
    ]
    ctx, stats = build_final_context(
        [], evidence, query_embedding=[1.0, 0.0], max_words=4000, item_max_words=1500
    )
    assert stats["evidence_used"] == 2
    assert "https://early" in ctx[0] and "https://late" in ctx[1]
    assert all(len(c.split()) <= 1500 + 4 for c in ctx)


def test_missing_embeddings_fall_back_to_retrieval_order():
    evidence = [item(f"https://e{i}", 100, None, i) for i in range(3)]
    ctx, stats = build_final_context([], evidence, max_words=10_000)
    assert stats["evidence_used"] == 3
    assert [c.split()[1] for c in ctx] == ["https://e0", "https://e1", "https://e2"]
