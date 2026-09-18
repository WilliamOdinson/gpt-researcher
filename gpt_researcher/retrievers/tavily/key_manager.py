"""Rotating API key manager for Tavily.

Loads keys from env/tavily.csv and rotates to the next key when the
current one is exhausted (HTTP 429 / 402 / 403).  Thread-safe singleton
so every TavilySearch instance shares the same key state.
"""

import csv
import logging
import os
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

# Status codes that signal the current key is drained / rate-limited.
# 432 is Tavily's custom code for "exceeds your plan's usage limit".
_EXHAUSTED_STATUS_CODES = {429, 402, 403, 432}


def _find_csv() -> Path:
    """Walk upward from this file to locate env/tavily.csv."""
    # Repo root is 3 levels up: retrievers/tavily/key_manager.py
    repo_root = Path(__file__).resolve().parents[3]
    csv_path = repo_root / "env" / "tavily.csv"
    if csv_path.exists():
        return csv_path
    # Fallback: check env var
    env_path = os.environ.get("TAVILY_KEYS_CSV")
    if env_path and Path(env_path).exists():
        return Path(env_path)
    raise FileNotFoundError(
        f"Tavily key CSV not found at {csv_path}. "
        "Set TAVILY_KEYS_CSV env var or place the file at env/tavily.csv."
    )


class TavilyKeyManager:
    """Thread-safe rotating key pool for Tavily API keys."""

    _instance: "TavilyKeyManager | None" = None
    _init_lock = threading.Lock()

    def __new__(cls) -> "TavilyKeyManager":
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    inst = super().__new__(cls)
                    inst._keys: list[str] = []
                    inst._index: int = 0
                    inst._lock = threading.Lock()
                    inst._loaded = False
                    cls._instance = inst
        return cls._instance

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            csv_path = _find_csv()
            with open(csv_path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    key = (row.get("api_key") or "").strip()
                    if key:
                        self._keys.append(key)
            if not self._keys:
                raise RuntimeError(f"No valid API keys found in {csv_path}")
            logger.info("Loaded %d Tavily API keys from %s", len(self._keys), csv_path)
            self._loaded = True

    @property
    def current_key(self) -> str:
        self._ensure_loaded()
        with self._lock:
            return self._keys[self._index]

    def rotate(self) -> str | None:
        """Advance to the next key. Returns the new key, or None if all keys exhausted."""
        self._ensure_loaded()
        with self._lock:
            old_idx = self._index
            new_idx = old_idx + 1
            if new_idx >= len(self._keys):
                logger.error("All %d Tavily API keys exhausted.", len(self._keys))
                return None
            self._index = new_idx
            logger.warning(
                "Tavily key #%d exhausted, rotating to key #%d (%d remaining).",
                old_idx,
                new_idx,
                len(self._keys) - new_idx,
            )
            return self._keys[new_idx]

    @property
    def remaining(self) -> int:
        self._ensure_loaded()
        with self._lock:
            return len(self._keys) - self._index

    @staticmethod
    def is_exhausted_status(status_code: int) -> bool:
        return status_code in _EXHAUSTED_STATUS_CODES
