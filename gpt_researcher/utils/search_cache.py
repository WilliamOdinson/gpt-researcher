import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _cache_key(subquery: str, retriever_class_name: str, max_results: int | None = None) -> str:
    raw = f"{retriever_class_name}:{subquery}:max={max_results}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


class SearchResultCache:
    """Caches retriever search results keyed by (subquery, retriever_class, max_results).

    Stores the complete result dicts (url, title, snippet, raw_content)
    so they can be replayed for counterfactual trajectory forks.
    """

    def __init__(self, cache_dir: str | None = None):
        self._memory: dict[str, list[dict[str, Any]]] = {}
        self._cache_dir = cache_dir or os.environ.get(
            "SEARCH_CACHE_DIR",
            os.path.join(os.getcwd(), "trajectory_logs", "search_cache"),
        )

    def get(self, subquery: str, retriever_class_name: str, max_results: int | None = None) -> list[dict[str, Any]] | None:
        key = _cache_key(subquery, retriever_class_name, max_results)
        return self._memory.get(key)

    def put(self, subquery: str, retriever_class_name: str, results: list[dict[str, Any]], max_results: int | None = None):
        key = _cache_key(subquery, retriever_class_name, max_results)
        self._memory[key] = results

    def save(self) -> str:
        out_dir = Path(self._cache_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "search_results.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._memory, f, ensure_ascii=False, indent=2)
        logger.info(f"Search cache saved: {len(self._memory)} entries to {path}")
        return str(path)

    def load(self, path: str | None = None):
        p = Path(path or os.path.join(self._cache_dir, "search_results.json"))
        if p.exists():
            with open(p, encoding="utf-8") as f:
                self._memory = json.load(f)
            logger.info(f"Search cache loaded: {len(self._memory)} entries from {p}")


_global_search_cache = SearchResultCache()
