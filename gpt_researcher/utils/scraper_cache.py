import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _url_key(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]


class ScraperCache:
    """Caches scraped page content keyed by URL.

    Stores raw_content so counterfactual replay can skip re-scraping.
    """

    def __init__(self, cache_dir: str | None = None):
        self._memory: dict[str, dict[str, Any]] = {}
        self._cache_dir = cache_dir or os.environ.get(
            "SCRAPER_CACHE_DIR",
            os.path.join(os.getcwd(), "trajectory_logs", "scraper_cache"),
        )

    def get(self, url: str) -> dict[str, Any] | None:
        return self._memory.get(_url_key(url))

    def put(self, url: str, content: dict[str, Any]):
        self._memory[_url_key(url)] = content

    def save(self) -> str:
        out_dir = Path(self._cache_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "scraped_content.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._memory, f, ensure_ascii=False, indent=2)
        logger.info(f"Scraper cache saved: {len(self._memory)} entries to {path}")
        return str(path)

    def load(self, path: str | None = None):
        p = Path(path or os.path.join(self._cache_dir, "scraped_content.json"))
        if p.exists():
            with open(p, encoding="utf-8") as f:
                self._memory = json.load(f)
            logger.info(f"Scraper cache loaded: {len(self._memory)} entries from {p}")


_global_scraper_cache = ScraperCache()
