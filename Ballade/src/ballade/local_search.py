from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from ballade.config import CostWeights, LocalSearchConfig
from ballade.costs import total_tracking_cost
from ballade.features import as_float_array, get_field, target_fingertip_error, target_q_error


@dataclass(frozen=True, slots=True)
class LocalSearchResult:
    action: np.ndarray
    best_cost: float
    base_cost: float
    used_search: bool
    candidate_count: int


def should_run_search(
    obs: Any,
    target: Any,
    config: LocalSearchConfig,
) -> bool:
    goal = as_float_array(get_field(target, "goal_key_mask", "goals"))
    piano = as_float_array(get_field(obs, "piano_activation", "piano_state"))
    q_error = target_q_error(obs, target)
    tip_error = target_fingertip_error(obs, target)
    if goal.size and bool((goal > 0.5).any()):
        return True
    if q_error.size and float(np.linalg.norm(q_error)) >= float(config.q_error_trigger):
        return True
    if tip_error.size and float(np.linalg.norm(tip_error)) >= float(config.fingertip_error_trigger):
        return True
    if bool(config.wrong_key_trigger) and goal.size and piano.size:
        width = min(goal.size, piano.size)
        wrong = (goal[:width] <= 0.5) & (piano[:width] > 0.5)
        if bool(wrong.any()):
            return True
    return False


class CandidateActionSearch:
    def __init__(
        self,
        config: LocalSearchConfig | None = None,
        weights: CostWeights | dict[str, float] | None = None,
    ) -> None:
        self.config = config or LocalSearchConfig()
        self.weights = weights or CostWeights()
        self.rng = np.random.default_rng(int(self.config.seed))

    def search(
        self,
        env: Any,
        obs: Any,
        target: Any,
        base_action: np.ndarray,
        previous_action: np.ndarray | None,
        *,
        active_mask: np.ndarray | None = None,
        cost_fn: Callable[[Any, Any, np.ndarray, np.ndarray | None, Any], float] = total_tracking_cost,
    ) -> LocalSearchResult:
        base = np.asarray(base_action, dtype=np.float32).reshape(-1)
        if not self.config.enabled or not should_run_search(obs, target, self.config):
            base_cost = cost_fn(obs, target, base, previous_action, self.weights)
            return LocalSearchResult(base.copy(), float(base_cost), float(base_cost), False, 0)
        mask = np.ones_like(base, dtype=bool) if active_mask is None else np.asarray(active_mask, dtype=bool).reshape(-1)
        if mask.size != base.size:
            raise ValueError(f"active_mask size {mask.size} does not match action size {base.size}")
        snapshot = env.snapshot()
        candidate_count = max(int(self.config.candidate_count), 1)
        candidates = [base]
        for _idx in range(candidate_count - 1):
            noise = self.rng.normal(0.0, float(self.config.action_sigma), size=base.shape).astype(np.float32)
            noise = np.where(mask, noise, 0.0)
            candidates.append(np.clip(base + noise, -1.0, 1.0).astype(np.float32))

        best_action = base.copy()
        best_cost = float("inf")
        base_cost = float("inf")
        try:
            for idx, candidate in enumerate(candidates):
                env.restore(snapshot)
                next_obs = obs
                total = 0.0
                for _step in range(max(int(self.config.horizon_microsteps), 1)):
                    next_obs, _reward, done, _info = env.step_normalized(candidate)
                    total += cost_fn(next_obs, target, candidate, previous_action, self.weights)
                    if done:
                        total += 1e3
                        break
                if idx == 0:
                    base_cost = float(total)
                if float(total) < best_cost:
                    best_cost = float(total)
                    best_action = candidate.copy()
        finally:
            env.restore(snapshot)
        return LocalSearchResult(
            action=best_action.astype(np.float32),
            best_cost=float(best_cost),
            base_cost=float(base_cost),
            used_search=True,
            candidate_count=int(candidate_count),
        )
