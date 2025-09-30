"""Data caching module for storing and managing fetched data."""

import time
from typing import Dict, Any, Optional


class DataCache:
    """Cache for storing and managing fetched PDU data with TTL."""

    def __init__(self):
        self.cache: Dict[str, Any] = {}
        self.last_refresh_time = 0
        self.refresh_interval = 300  # 5 minutes

    def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache by key."""
        return self.cache.get(key)

    def set(self, key: str, value: Any):
        """Set a value in the cache."""
        self.cache[key] = value

    def update_cache(self, data: Dict[str, Any]):
        """Update the cache with multiple key-value pairs."""
        self.cache.update(data)
        self.last_refresh_time = time.time()

    def should_refresh(self) -> bool:
        """Check if the cache should be refreshed based on TTL."""
        return time.time() - self.last_refresh_time > self.refresh_interval

    def get_cache_summary(self):
        """Get a summary of cached keys and their availability."""
        return [(key, bool(value)) for key, value in self.cache.items()]

    def clear(self):
        """Clear all cached data."""
        self.cache.clear()
        self.last_refresh_time = 0
