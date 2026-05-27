#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_SRC = Path(__file__).resolve().parents[1] / "src"
for path in (MODULE_SRC, REPO_ROOT):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from pid_controller.mapping import (  # noqa: E402
    action_signals_from_hand_state,
    build_reduced_action_mapping,
    mapping_to_jsonable,
)
from pid_controller.rollout import save_json  # noqa: E402


def _names(elements) -> list[str]:
    out = []
    for element in elements:
        for attr in ("full_identifier", "identifier", "name"):
            value = getattr(element, attr, None)
            if callable(value):
                try:
                    value = value()
                except TypeError:
                    pass
            if value:
                out.append(str(value))
                break
        else:
            out.append(str(element))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Empirically probe each reduced action input against hand qpos deltas.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--environment-name", default="RoboPianist-debug-TwinkleTwinkleLittleStar-v0")
    parser.add_argument("--settle-steps", type=int, default=8)
    parser.add_argument("--pulse-fraction", type=float, default=0.35)
    parser.add_argument("--threshold", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from rp1m_simulator import RolloutConfig
    from rp1m_simulator.simulator import (
        _capture_hand_qpos,
        _hand_actuator_names,
        _hand_joint_names,
        _load_env,
        _locate_task_physics_piano,
        write_goals_proto,
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pid_mapping_probe_") as temp:
        midi_proto = Path(temp) / "silent.proto"
        write_goals_proto(
            np.zeros((max(int(args.settle_steps) + 2, 4), 88), dtype=np.float32),
            midi_proto,
            dt=0.005,
            title="PID action mapping probe",
        )
        env, load_info = _load_env(
            RolloutConfig(
                mode="action",
                dataset_timestep=0.05,
                simulation_timestep=0.005,
                auto_hand_anchor_y_offset=False,
                seed=int(args.seed),
            ),
            midi_proto,
            str(args.environment_name),
            0.005,
        )
        rows: list[dict[str, object]] = []
        try:
            env.reset()
            task, physics, _piano = _locate_task_physics_piano(env)
            action_spec = env.action_spec()
            joint_names = _hand_joint_names(task)
            actuator_names = _hand_actuator_names(task)
            mapping = build_reduced_action_mapping(
                action_dim=int(action_spec.shape[0]),
                hand_joint_names=joint_names,
                actuator_names=actuator_names,
            )
            minimum = np.asarray(action_spec.minimum, dtype=np.float32).reshape(-1)
            maximum = np.asarray(action_spec.maximum, dtype=np.float32).reshape(-1)
            settle_steps = max(int(args.settle_steps), 1)
            for entry in mapping:
                if entry.kind == "sustain":
                    continue
                env.reset()
                task, physics, _piano = _locate_task_physics_piano(env)
                baseline_qpos = _capture_hand_qpos(task, physics)
                if baseline_qpos is None:
                    raise RuntimeError("Could not capture baseline hand qpos")
                baseline_control = action_signals_from_hand_state(baseline_qpos, mapping)
                baseline_control = np.clip(baseline_control, minimum, maximum).astype(np.float32)
                span = float(maximum[entry.action_index] - minimum[entry.action_index])
                pulse = float(args.pulse_fraction) * span
                if not np.isfinite(pulse) or pulse <= 0.0:
                    pulse = 0.1
                control = baseline_control.copy()
                high = baseline_control[entry.action_index] + pulse
                low = baseline_control[entry.action_index] - pulse
                control[entry.action_index] = (
                    high if high <= float(maximum[entry.action_index]) else max(float(minimum[entry.action_index]), low)
                )
                for _ in range(settle_steps):
                    env.step(control)
                after_qpos = _capture_hand_qpos(task, physics)
                if after_qpos is None:
                    raise RuntimeError("Could not capture post-pulse hand qpos")
                delta = np.asarray(after_qpos - baseline_qpos, dtype=np.float32)
                changed = np.flatnonzero(np.abs(delta) >= float(args.threshold))
                ranked = sorted(
                    (
                        {
                            "joint_index": int(index),
                            "joint_name": str(joint_names[int(index)]),
                            "delta": float(delta[int(index)]),
                        }
                        for index in changed
                    ),
                    key=lambda item: abs(float(item["delta"])),
                    reverse=True,
                )
                rows.append(
                    {
                        "action_index": int(entry.action_index),
                        "actuator_name": str(entry.actuator_name),
                        "expected_kind": entry.kind,
                        "expected_joint_indices": list(entry.joint_indices),
                        "expected_joint_names": list(entry.joint_names),
                        "changed_joint_count": int(len(changed)),
                        "top_changed_joints": ranked[:6],
                    }
                )
        finally:
            env.close()

    save_json(
        out / "action_mapping_probe.json",
        {
            "environment_name": str(args.environment_name),
            "load_info": load_info,
            "settle_steps": int(args.settle_steps),
            "pulse_fraction": float(args.pulse_fraction),
            "mapping": mapping_to_jsonable(mapping),
            "rows": rows,
        },
    )
    with (out / "action_mapping_probe.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "action_index",
                "actuator_name",
                "expected_kind",
                "expected_joint_indices",
                "expected_joint_names",
                "changed_joint_count",
                "top_changed_joints",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value for key, value in row.items()})
    print(json.dumps({"rows": len(rows), "json": str(out / "action_mapping_probe.json")}, sort_keys=True))


if __name__ == "__main__":
    main()
