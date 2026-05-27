#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_SRC = Path(__file__).resolve().parents[1] / "src"
for path in (MODULE_SRC, REPO_ROOT):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from pid_controller.action_space import (  # noqa: E402
    expected_action_space_summary,
)


def _selected_spaces(compare: str) -> tuple[bool, ...]:
    if compare == "reduced":
        return (True,)
    if compare == "full":
        return (False,)
    return (True, False)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")
    return out


def _live_probe(
    *,
    reduced_action_space: bool,
    environment_name: str,
    seed: int,
    dataset_timestep: float,
    simulation_timestep: float,
) -> dict[str, Any]:
    name = "reduced" if reduced_action_space else "full"
    try:
        from rp1m_simulator.simulator import (  # noqa: PLC0415
            RolloutConfig,
            _hand_actuator_names,
            _hand_joint_names,
            _load_env,
            _locate_task_physics_piano,
            write_goals_proto,
        )

        with tempfile.TemporaryDirectory(prefix=f"pid_{name}_action_space_") as temp:
            midi_proto = Path(temp) / "silent.proto"
            write_goals_proto(
                np.zeros((4, 88), dtype=np.float32),
                midi_proto,
                dt=float(simulation_timestep),
                title=f"PID {name} action-space dimension probe",
            )
            config = RolloutConfig(
                mode="action",
                dataset_timestep=float(dataset_timestep),
                simulation_timestep=float(simulation_timestep),
                reduced_action_space=bool(reduced_action_space),
                auto_hand_anchor_y_offset=False,
                seed=int(seed),
            )
            env, load_info = _load_env(
                config,
                midi_proto,
                str(environment_name),
                float(simulation_timestep),
            )
            try:
                env.reset()
                task, _physics, _piano = _locate_task_physics_piano(env)
                action_spec = env.action_spec()
                action_spec_dim = int(action_spec.shape[0])
                hand_joint_names = _hand_joint_names(task)
                actuator_names = _hand_actuator_names(task)
                sustain_count = 1 if actuator_names and actuator_names[-1] == "sustain" else 0
                hand_action_dim = action_spec_dim - sustain_count
                return {
                    "name": name,
                    "live": True,
                    "load_info": load_info,
                    "action_spec_dim": action_spec_dim,
                    "hand_action_dim": int(hand_action_dim),
                    "hand_joint_dim": int(len(hand_joint_names)),
                    "hand_actuator_name_count": int(max(len(actuator_names) - sustain_count, 0)),
                    "sustain_action_count": int(sustain_count),
                    "has_46_hand_actions_for_46_hand_states": bool(
                        hand_action_dim == 46 and len(hand_joint_names) == 46
                    ),
                    "has_46_env_actions_for_46_hand_states": bool(
                        action_spec_dim == 46 and len(hand_joint_names) == 46
                    ),
                    "hand_action_state_dims_match": bool(hand_action_dim == len(hand_joint_names)),
                    "hand_joint_names": hand_joint_names,
                    "actuator_names": actuator_names,
                }
            finally:
                close = getattr(env, "close", None)
                if callable(close):
                    close()
    except Exception as exc:  # pragma: no cover - depends on live RoboPianist stack.
        return {
            "name": name,
            "live": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "note": (
                "Live probing requires the RoboPianist/MuJoCo environment used for "
                "rp1m_simulator rollouts."
            ),
        }


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    static = [
        expected_action_space_summary(reduced_action_space=value).to_dict()
        for value in _selected_spaces(args.compare)
    ]
    payload: dict[str, Any] = {
        "source_paths": [
            "robopianist/models/hands/shadow_hand_constants.py",
            "robopianist/models/hands/shadow_hand.py",
            "robopianist/models/hands/third_party/shadow_hand/right_hand.xml",
            "robopianist/suite/tasks/piano_with_shadow_hands.py",
            "rp1m_simulator/simulator.py",
        ],
        "static": static,
        "conclusion": {
            "full_action_space_is_direct_46_to_46": False,
            "recommended_for_46d_impromptu_hand_rollouts": "reduced_39d_controller_mapping",
            "reason": (
                "Reduced space has 38 hand actions plus sustain for 46 hand qpos; "
                "full space has 44 hand actions plus sustain for 52 hand qpos. "
                "Both spaces keep tendon actions that drive FFJ2+FFJ1, MFJ2+MFJ1, "
                "RFJ2+RFJ1, and LFJ2+LFJ1."
            ),
        },
    }
    if args.live:
        payload["live"] = [
            _live_probe(
                reduced_action_space=value,
                environment_name=args.environment_name,
                seed=args.seed,
                dataset_timestep=args.dataset_timestep,
                simulation_timestep=args.simulation_timestep,
            )
            for value in _selected_spaces(args.compare)
        ]
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Check whether RoboPianist full action space gives a direct 46 action "
            "to 46 hand-state map."
        )
    )
    parser.add_argument("--compare", choices=["both", "reduced", "full"], default="both")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Also load rp1m_simulator/RoboPianist and count live env specs.",
    )
    parser.add_argument(
        "--environment-name",
        default="RoboPianist-debug-TwinkleTwinkleLittleStar-v0",
    )
    parser.add_argument("--dataset-timestep", type=float, default=0.05)
    parser.add_argument("--simulation-timestep", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    payload = build_payload(args)
    if args.output_json:
        _write_json(args.output_json, payload)
    print(json.dumps(_jsonable(payload), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
