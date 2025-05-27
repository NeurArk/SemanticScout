"""Simple performance test stub."""
from __future__ import annotations

import time

from core.vector_store import VectorStore
from core.models.search import SearchQuery


def main() -> None:
    store = VectorStore()
    query = SearchQuery(query_text="test")
    start = time.time()
    store.search([0.1] * 3072, query)
    duration = time.time() - start
    print({"search_time_s": duration})


if __name__ == "__main__":
    main()
