"""Layer 3f -- cache replacement policies (LRU / LFU / ARC).

The semantic cache evicts FIFO today. ARC (Adaptive Replacement Cache) balances recency and
frequency adaptively and is scan-resistant: a burst of one-shot keys cannot evict the
frequently-reused working set. T1 (recent, seen once) + T2 (frequent, seen >=2) hold the
cached values; B1/B2 are ghost lists of recently-evicted keys that steer the adaptive target
p between recency and frequency. Pure stdlib.
"""
from __future__ import annotations

from collections import OrderedDict


class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self._d: OrderedDict = OrderedDict()

    def get(self, key):
        if key not in self._d:
            return None
        self._d.move_to_end(key)
        return self._d[key]

    def put(self, key, value):
        self._d[key] = value
        self._d.move_to_end(key)
        if len(self._d) > self.capacity:
            self._d.popitem(last=False)


class LFUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self._d: dict = {}
        self._freq: dict = {}

    def get(self, key):
        if key not in self._d:
            return None
        self._freq[key] += 1
        return self._d[key]

    def put(self, key, value):
        if key not in self._d and len(self._d) >= self.capacity:
            victim = min(self._freq, key=self._freq.get)
            del self._d[victim]
            del self._freq[victim]
        self._d[key] = value
        self._freq[key] = self._freq.get(key, 0) + 1


class ARCCache:
    """Adaptive Replacement Cache (Megiddo & Modha). c = capacity; p = adaptive target T1 size."""

    def __init__(self, capacity: int):
        self.c = capacity
        self.p = 0
        self.t1: OrderedDict = OrderedDict()   # recent, seen once  (key -> value)
        self.t2: OrderedDict = OrderedDict()   # frequent, seen >=2 (key -> value)
        self.b1: OrderedDict = OrderedDict()   # ghosts evicted from t1
        self.b2: OrderedDict = OrderedDict()   # ghosts evicted from t2

    def get(self, key):
        if key in self.t1:                      # promote to frequent
            val = self.t1.pop(key)
            self.t2[key] = val
            return val
        if key in self.t2:
            self.t2.move_to_end(key)
            return self.t2[key]
        return None

    def _replace(self, key):
        if self.t1 and (len(self.t1) > self.p or (key in self.b2 and len(self.t1) == self.p)):
            old, val = self.t1.popitem(last=False)
            self.b1[old] = True
        elif self.t2:
            old, val = self.t2.popitem(last=False)
            self.b2[old] = True

    def put(self, key, value):
        if key in self.t1:
            self.t1.pop(key)
            self.t2[key] = value
            return
        if key in self.t2:
            self.t2[key] = value
            self.t2.move_to_end(key)
            return
        if key in self.b1:                       # recency ghost hit -> favor recency
            self.p = min(self.c, self.p + max(1, len(self.b2) // max(1, len(self.b1))))
            self._replace(key)
            self.b1.pop(key, None)
            self.t2[key] = value
            return
        if key in self.b2:                       # frequency ghost hit -> favor frequency
            self.p = max(0, self.p - max(1, len(self.b1) // max(1, len(self.b2))))
            self._replace(key)
            self.b2.pop(key, None)
            self.t2[key] = value
            return
        # brand-new key
        if len(self.t1) + len(self.t2) >= self.c:
            self._replace(key)
        # bound the ghost lists
        while len(self.b1) > self.c:
            self.b1.popitem(last=False)
        while len(self.b2) > self.c:
            self.b2.popitem(last=False)
        self.t1[key] = value

    def __contains__(self, key):
        return key in self.t1 or key in self.t2
