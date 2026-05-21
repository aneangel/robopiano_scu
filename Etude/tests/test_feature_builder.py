from __future__ import annotations

import numpy as np

from etude.data.feature_builder import FeatureSpec, build_tracking_features


def test_feature_builder_includes_lookahead_and_optional_features() -> None:
    q = np.zeros(46, dtype=np.float32)
    qdot = np.zeros(46, dtype=np.float32)
    q_ref = np.ones((12, 46), dtype=np.float32)
    qdot_ref = np.ones((12, 46), dtype=np.float32) * 0.5
    prev = np.zeros(10, dtype=np.float32)
    target_keys = np.zeros((12, 88), dtype=np.float32)
    fingertips = np.zeros((12, 30), dtype=np.float32)
    features = build_tracking_features(
        q=q,
        qdot=qdot,
        q_ref=q_ref,
        qdot_ref=qdot_ref,
        t=0,
        previous_action=prev,
        target_keys=target_keys,
        fingertips=fingertips,
        spec=FeatureSpec(lookahead_steps=(1, 5, 10)),
    )
    assert features.dtype == np.float32
    assert features.shape == (46 * 7 + 10 + 88 + 30,)



def test_feature_builder_can_include_goal_schema_channels() -> None:
    q = np.zeros(46, dtype=np.float32)
    qdot = np.zeros(46, dtype=np.float32)
    q_ref = np.ones((6, 46), dtype=np.float32)
    qdot_ref = np.zeros((6, 46), dtype=np.float32)
    prev = np.zeros(4, dtype=np.float32)
    target_keys = np.zeros((6, 88), dtype=np.float32)
    target_keys[2, 10] = 1.0
    metadata = {"target_keys": target_keys, "dt": 0.01}

    features = build_tracking_features(
        q=q,
        qdot=qdot,
        q_ref=q_ref,
        qdot_ref=qdot_ref,
        t=1,
        previous_action=prev,
        metadata=metadata,
        spec=FeatureSpec(
            lookahead_steps=(1,),
            include_target_keys=True,
            include_fingertips=False,
            target_key_lookahead_steps=(1, 2),
            include_time_to_next_press=True,
            include_phase=True,
        ),
    )

    expected = (46 * 5) + 4 + (88 * 3) + 1 + 6
    assert features.shape == (expected,)
    assert np.isclose(features[-7], 0.01)
