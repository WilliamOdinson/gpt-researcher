"""DeepResearchGym retriever for GPT Researcher.

Searches the ClueWeb22 / FineWeb corpora via the DeepResearchGym API
(https://clueweb22.us). Results include full document text, so no
downstream scraping is needed.

Set DEEPRESEARCHGYM_API_KEY in the environment. Optionally set
DEEPRESEARCHGYM_DATASET to one of 'fineweb' (default), 'clueweb_b',
or 'clueweb_a'.
"""

import base64
import json
import logging
import os
from typing import Any, Dict, List

import requests

from ..base import BaseRetriever

logger = logging.getLogger(__name__)

_VALID_DATASETS = ("fineweb", "clueweb_b", "clueweb_a")

_BASE_URLS = {
    "clueweb_b": "https://clueweb22.us/search",
    "clueweb_a": "https://clueweb22.us/search",
    "fineweb": "https://clueweb22.us/fineweb/search",
}


class DeepResearchGymSearch(BaseRetriever):
    """Retriever backed by the DeepResearchGym corpus API.

    Returns full document text; pages do not need to be scraped afterward.
    """

    requires_scraping = False

    def __init__(self, query: str, query_domains=None):
        self.query = query
        self.query_domains = query_domains  # unused but required by contract
        self.api_key = os.environ.get("DEEPRESEARCHGYM_API_KEY", "")
        self.dataset = os.environ.get("DEEPRESEARCHGYM_DATASET", "fineweb")
        if self.dataset not in _VALID_DATASETS:
            logger.warning(
                "DEEPRESEARCHGYM_DATASET=%r is invalid, falling back to 'fineweb'",
                self.dataset,
            )
            self.dataset = "fineweb"

    def search(self, max_results: int = 10) -> List[Dict[str, Any]]:
        """Query the Gym corpus and return results with full text."""
        base_url = _BASE_URLS[self.dataset]
        params = {"query": self.query, "k": max_results}
        if self.dataset == "clueweb_a":
            params["cw22_a"] = "True"

        headers = {}
        if self.api_key:
            headers["X-API-Key"] = self.api_key

        try:
            resp = requests.get(base_url, params=params, headers=headers, timeout=60)
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.error("DeepResearchGym search failed: %s", e)
            return []

        raw_results = resp.json().get("results", [])
        search_response = []
        for doc_b64 in raw_results:
            try:
                decoded = base64.b64decode(doc_b64).decode("utf-8")
                parsed = json.loads(decoded)
            except Exception as e:
                logger.warning("Failed to decode Gym result: %s", e)
                continue

            url = parsed.get("URL") or parsed.get("url") or ""
            text = parsed.get("Clean-Text") or parsed.get("text") or ""
            if not text:
                continue

            search_response.append({
                "href": url or f"urn:deepresearchgym:{self.dataset}",
                "url": url or f"urn:deepresearchgym:{self.dataset}",
                "raw_content": text,
                "body": text[:200],
            })

        logger.info(
            "[DeepResearchGym] dataset=%s query=%r results=%d",
            self.dataset,
            self.query,
            len(search_response),
        )
        return search_response
