import time
from typing import Dict, Any, Optional

class DataCache:
    def __init__(self):
        self.cache: Dict[str, Any] = {}
        self.last_refresh_time = 0
        self.refresh_interval = 300  # 5 minutes

    def get(self, key: str) -> Optional[Any]:
        return self.cache.get(key)

    def set(self, key: str, value: Any):
        self.cache[key] = value

    def update_cache(self, data: Dict[str, Any]):
        self.cache.update(data)
        self.last_refresh_time = time.time()

    def should_refresh(self) -> bool:
        return time.time() - self.last_refresh_time > self.refresh_interval

    def get_cache_summary(self):
        return [(key, bool(value)) for key, value in self.cache.items()]

    def clear(self):
        self.cache.clear()
        self.last_refresh_time = 0