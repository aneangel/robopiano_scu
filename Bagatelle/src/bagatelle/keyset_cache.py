"""Per-song keyset fingerprint cache for the Bagatelle IK solver.

Two operating modes:
- exact_only: on exact keyset match, return cached qpos and skip IK entirely
- exact_and_warm_start: also use near-miss matches as scipy LM warm starts

The cache is per-planner-instance (per-song). It does not persist across
runs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

CACHE_MODES = ("off", "exact_only", "exact_and_warm_start")


@dataclass(frozen=True)
class KeysetCacheKey:
    """Canonical cache key for a waypoint's keypress goal.

    All three tuples are co-indexed: ``active_keys[i]`` is the MIDI key
    index, ``hand_assignment[i]`` is the hand label ('L' or 'R') that will
    press it, and ``finger_assignment[i]`` is the dense finger index
    (0..9) chosen by the assignment stage.
    """

    active_keys: tuple  # sorted tuple of active key indices (0..87)
    hand_assignment: tuple  # per-key hand label ('L' or 'R'), aligned with active_keys
    finger_assignment: tuple  # per-key finger index (0..9), aligned with active_keys


@dataclass
class KeysetCacheEntry:
    qpos: np.ndarray
    residual_norm: float


@dataclass
class KeysetCacheStats:
    exact_hits: int = 0
    warm_start_hits: int = 0
    misses: int = 0
    inserts: int = 0
    rejected_low_quality: int = 0

    def total_lookups(self) -> int:
        return int(self.exact_hits + self.warm_start_hits + self.misses)

    def hit_rate(self) -> float:
        total = self.total_lookups()
        if total <= 0:
            return 0.0
        return float(self.exact_hits + self.warm_start_hits) / float(total)


def _jaccard_similarity(a: frozenset, b: frozenset) -> float:
    """Jaccard similarity between two sets of active key indices.

    Defined as |A intersect B| / |A union B|. Two empty sets are treated
    as perfectly similar (1.0) so that two rest waypoints match.
    """
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    intersection = a & b
    return float(len(intersection)) / float(len(union))


class KeysetCache:
    """In-memory keyset -> qpos cache for one song's IK."""

    def __init__(
        self,
        mode: str = "off",
        jaccard_threshold: float = 0.8,
        max_residual_for_insert: float = 0.02,
        estimated_seconds_per_ik: float = 0.4,
    ):
        if mode not in CACHE_MODES:
            raise ValueError(
                f"mode must be one of {CACHE_MODES!r}, got {mode!r}"
            )
        self.mode = mode
        self.jaccard_threshold = float(jaccard_threshold)
        self.max_residual_for_insert = float(max_residual_for_insert)
        self.estimated_seconds_per_ik = float(estimated_seconds_per_ik)
        self._entries: dict[KeysetCacheKey, KeysetCacheEntry] = {}
        self.stats = KeysetCacheStats()

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    @property
    def warm_start_enabled(self) -> bool:
        return self.mode == "exact_and_warm_start"

    def make_key(
        self,
        active_keys: np.ndarray,
        hand_assignment,
        finger_assignment,
    ) -> KeysetCacheKey:
        """Construct a canonical cache key.

        The three arrays are zipped, sorted by active key index, and
        converted to tuples. Sorting makes the key invariant to the order
        the assignment stage happens to emit keys in.
        """
        keys = np.asarray(active_keys, dtype=np.int64).reshape(-1)
        hands = list(hand_assignment) if hand_assignment is not None else []
        fingers = (
            np.asarray(finger_assignment, dtype=np.int64).reshape(-1).tolist()
            if finger_assignment is not None
            else []
        )
        if len(hands) != keys.size or len(fingers) != keys.size:
            raise ValueError(
                "active_keys, hand_assignment, finger_assignment must have the same length"
            )
        order = np.argsort(keys, kind="stable")
        sorted_keys = tuple(int(keys[i]) for i in order)
        sorted_hands = tuple(str(hands[i]) for i in order)
        sorted_fingers = tuple(int(fingers[i]) for i in order)
        return KeysetCacheKey(
            active_keys=sorted_keys,
            hand_assignment=sorted_hands,
            finger_assignment=sorted_fingers,
        )

    def lookup(self, key: KeysetCacheKey) -> tuple[Optional[np.ndarray], str]:
        """Return ``(cached_qpos_or_None, hit_kind)``.

        ``hit_kind`` is one of:
        - ``'exact'``: identical key found, cached qpos is the final pose.
        - ``'warm_start'``: a similar key (Jaccard > threshold) was found;
          cached qpos should be used to seed scipy LM, not as the answer.
        - ``'miss'``: nothing useful in the cache; caller should run IK
          normally.
        """
        if not self.enabled:
            self.stats.misses += 1
            return None, "miss"

        exact = self._entries.get(key)
        if exact is not None:
            self.stats.exact_hits += 1
            return exact.qpos.copy(), "exact"

        if not self.warm_start_enabled or not self._entries:
            self.stats.misses += 1
            return None, "miss"

        query_keys = frozenset(key.active_keys)
        best_qpos: Optional[np.ndarray] = None
        best_sim = self.jaccard_threshold
        for cached_key, entry in self._entries.items():
            cached_keys = frozenset(cached_key.active_keys)
            sim = _jaccard_similarity(query_keys, cached_keys)
            if sim > best_sim:
                best_sim = sim
                best_qpos = entry.qpos
        if best_qpos is not None:
            self.stats.warm_start_hits += 1
            return best_qpos.copy(), "warm_start"

        self.stats.misses += 1
        return None, "miss"

    def insert(
        self,
        key: KeysetCacheKey,
        qpos: np.ndarray,
        residual_norm: float,
    ) -> bool:
        """Insert if ``residual_norm <= max_residual_for_insert``.

        If an entry already exists for ``key``, keep whichever has the
        lower residual. Returns True if the cache was updated.
        """
        if not self.enabled:
            return False
        if float(residual_norm) > self.max_residual_for_insert:
            self.stats.rejected_low_quality += 1
            return False
        qpos_arr = np.asarray(qpos, dtype=np.float32).copy()
        existing = self._entries.get(key)
        if existing is not None and existing.residual_norm <= float(residual_norm):
            return False
        self._entries[key] = KeysetCacheEntry(
            qpos=qpos_arr,
            residual_norm=float(residual_norm),
        )
        self.stats.inserts += 1
        return True

    def __len__(self) -> int:
        return len(self._entries)

    def report(self) -> dict:
        """Return summary stats for inclusion in ``metadata.json``."""
        ik_calls_saved = int(self.stats.exact_hits)
        seconds_saved = float(ik_calls_saved) * self.estimated_seconds_per_ik
        return {
            "mode": self.mode,
            "jaccard_threshold": self.jaccard_threshold,
            "max_residual_for_insert": self.max_residual_for_insert,
            "estimated_seconds_per_ik": self.estimated_seconds_per_ik,
            "cache_size": int(len(self._entries)),
            "exact_hits": int(self.stats.exact_hits),
            "warm_start_hits": int(self.stats.warm_start_hits),
            "misses": int(self.stats.misses),
            "inserts": int(self.stats.inserts),
            "rejected_low_quality": int(self.stats.rejected_low_quality),
            "total_lookups": int(self.stats.total_lookups()),
            "hit_rate": float(self.stats.hit_rate()),
            "ik_calls_saved": ik_calls_saved,
            "estimated_seconds_saved": seconds_saved,
        }
