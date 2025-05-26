from __future__ import annotations

from typing import Optional, List, Dict
import hashlib
import time
import pickle
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """LRU cache for embeddings with optional disk persistence."""

    def __init__(self, cache_dir: Optional[str] = None, max_size: int = 1000) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_size = max_size
        self._memory_cache: Dict[str, Dict] = {}
        self._access_times: Dict[str, float] = {}

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_disk_cache()

    def _generate_key(self, text: str, model: str) -> str:
        content = f"{model}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()

    def get(self, text: str, model: str) -> Optional[List[float]]:
        key = self._generate_key(text, model)
        if key in self._memory_cache:
            self._access_times[key] = time.time()
            logger.debug("Cache hit (memory): %s", key[:8])
            return self._memory_cache[key]["embedding"]

        if self.cache_dir:
            disk_path = self.cache_dir / f"{key}.pkl"
            if disk_path.exists():
                try:
                    with disk_path.open("rb") as f:
                        data = pickle.load(f)
                    self._add_to_memory(key, data)
                    logger.debug("Cache hit (disk): %s", key[:8])
                    return data["embedding"]
                except Exception as exc:
                    logger.warning("Failed to load from disk cache: %s", exc)
        logger.debug("Cache miss: %s", key[:8])
        return None

    def set(self, text: str, model: str, embedding: List[float]) -> None:
        key = self._generate_key(text, model)
        data = {
            "text": text[:100],
            "model": model,
            "embedding": embedding,
            "timestamp": time.time(),
        }
        self._add_to_memory(key, data)
        if self.cache_dir:
            try:
                with (self.cache_dir / f"{key}.pkl").open("wb") as f:
                    pickle.dump(data, f)
            except Exception as exc:
                logger.warning("Failed to save to disk cache: %s", exc)

    def _add_to_memory(self, key: str, data: Dict) -> None:
        if len(self._memory_cache) >= self.max_size:
            oldest_key = min(self._access_times, key=self._access_times.get)
            self._memory_cache.pop(oldest_key, None)
            self._access_times.pop(oldest_key, None)
        self._memory_cache[key] = data
        self._access_times[key] = time.time()

    def _load_disk_cache(self) -> None:
        if not self.cache_dir:
            return
        cache_files = list(self.cache_dir.glob("*.pkl"))
        cache_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        for cache_file in cache_files[: self.max_size]:
            try:
                with cache_file.open("rb") as f:
                    data = pickle.load(f)
                key = cache_file.stem
                self._memory_cache[key] = data
                self._access_times[key] = cache_file.stat().st_mtime
            except Exception as exc:
                logger.warning("Failed to load cache file %s: %s", cache_file, exc)
        logger.info("Loaded %s items from disk cache", len(self._memory_cache))

    def get_stats(self) -> Dict[str, int]:
        return {
            "memory_items": len(self._memory_cache),
            "disk_items": len(list(self.cache_dir.glob("*.pkl"))) if self.cache_dir else 0,
            "max_size": self.max_size,
        }
