from __future__ import annotations

from typing import Optional, List, Dict
import hashlib
import time
import pickle
from pathlib import Path
import logging
from config.settings import get_settings

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """LRU cache for embeddings with optional disk persistence."""

    def __init__(self, cache_dir: Optional[str] = None, max_size: int = 1000) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_size = max_size
        self.settings = get_settings()
        self._memory_cache: Dict[str, Dict] = {}
        self._access_times: Dict[str, float] = {}
        self._hits = 0
        self._misses = 0

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_disk_cache()

    def _generate_key(self, text: str, model: str) -> str:
        content = f"{model}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()

    def get(self, text: str, model: str) -> Optional[List[float]]:
        key = self._generate_key(text, model)
        if key in self._memory_cache:
            if self._is_expired(key):
                self._remove_key(key)
            else:
                self._access_times[key] = time.time()
                self._hits += 1
                logger.debug("Cache hit (memory): %s", key[:8])
                return self._memory_cache[key]["embedding"]

        if self.cache_dir:
            disk_path = self.cache_dir / f"{key}.pkl"
            if disk_path.exists():
                try:
                    with disk_path.open("rb") as f:
                        data = pickle.load(f)
                    if not self._is_expired(key, data):
                        self._add_to_memory(key, data)
                        self._hits += 1
                        logger.debug("Cache hit (disk): %s", key[:8])
                        return data["embedding"]
                    disk_path.unlink(missing_ok=True)
                except Exception as exc:
                    logger.warning("Failed to load from disk cache: %s", exc)
        self._misses += 1
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
            oldest_key = min(self._access_times, key=self._access_times.get)  # type: ignore[arg-type]
            self._memory_cache.pop(oldest_key, None)
            self._access_times.pop(oldest_key, None)
        self._memory_cache[key] = data
        self._access_times[key] = time.time()

    def _is_expired(self, key: str, data: Optional[Dict] = None) -> bool:
        ttl = getattr(self.settings, "cache_ttl", 0)
        if ttl <= 0:
            return False
        info = data or self._memory_cache.get(key)
        if not info:
            return False
        return time.time() - info.get("timestamp", 0) > ttl

    def _remove_key(self, key: str) -> None:
        self._memory_cache.pop(key, None)
        self._access_times.pop(key, None)
        if self.cache_dir:
            path = self.cache_dir / f"{key}.pkl"
            path.unlink(missing_ok=True)

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
                if self._is_expired(key, data):
                    cache_file.unlink(missing_ok=True)
                    continue
                self._memory_cache[key] = data
                self._access_times[key] = cache_file.stat().st_mtime
            except Exception as exc:
                logger.warning("Failed to load cache file %s: %s", cache_file, exc)
        logger.info("Loaded %s items from disk cache", len(self._memory_cache))

    def get_stats(self) -> Dict[str, float | int]:
        return {
            "memory_items": len(self._memory_cache),
            "disk_items": (
                len(list(self.cache_dir.glob("*.pkl"))) if self.cache_dir else 0
            ),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.get_hit_rate(),
        }

    def get_hit_rate(self) -> float:
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._hits / total

    def clear_expired(self) -> None:
        keys = list(self._memory_cache.keys())
        for key in keys:
            if self._is_expired(key):
                self._remove_key(key)

    def clear(self) -> None:
        self._memory_cache.clear()
        self._access_times.clear()
        if self.cache_dir:
            for f in self.cache_dir.glob("*.pkl"):
                f.unlink(missing_ok=True)
