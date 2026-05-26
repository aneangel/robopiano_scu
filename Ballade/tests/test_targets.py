from __future__ import annotations

import numpy as np

from ballade.replay_buffer import OnlineTeacherReplayBuffer, TeacherTransition
from ballade.targets import build_micro_targets


def test_target_sequence_fields_and_phase() -> None:
    q = np.asarray([[0.0, 0.0], [1.0, 1.0], [2.0, 4.0]], dtype=np.float32)
    goals = np.zeros((3, 88), dtype=np.float32)
    goals[1, 40] = 1.0
    targets = build_micro_targets(q, goals_20hz=goals, source_dt=0.05, control_dt=0.005)
    assert len(targets) == 20
    assert targets.substeps == 10
    assert targets[0].source_frame_index == 0
    assert targets[0].microstep_index == 0
    assert targets[0].microstep_phase == 0.10000000149011612
    assert targets[9].microstep_phase == 1.0
    assert targets[0].goal_key_mask[40] == 1.0
    assert targets.target_q_micro.shape == (20, 2)


def test_replay_buffer_jsonl_roundtrip(tmp_path) -> None:
    buffer = OnlineTeacherReplayBuffer()
    buffer.append(
        TeacherTransition(
            obs={"q": [0.0], "qvel": [0.0], "piano_activation": [0.0] * 88},
            target={"target_q_micro": [1.0], "target_qvel_micro": [0.0], "goal_key_mask": [0.0] * 88},
            base_action=np.zeros((2,), dtype=np.float32),
            selected_action=np.ones((2,), dtype=np.float32),
            next_obs={"q": [0.1], "qvel": [0.0], "piano_activation": [0.0] * 88},
            cost_before=1.0,
            cost_after=0.5,
            source_frame_index=0,
            microstep_index=0,
            song_key="song",
            demo_id=0,
        )
    )
    path = buffer.save_shard(tmp_path)
    assert path.exists()
    loaded = OnlineTeacherReplayBuffer.load_shards(tmp_path)
    assert len(loaded) == 1
    base, selected = loaded.action_arrays()
    assert base.shape == (1, 2)
    assert selected.shape == (1, 2)
