from __future__ import annotations

import numpy as np

from allegro.alignment import (
    AllegroAligner,
    HighFrequencyAlignmentConfig,
    OnlineAffineActionModel,
    build_alignment_feature,
    build_shadow_hand_action_jacobian,
    clip_residual_norm,
    clip_residual_per_dim,
    interpolate_source_target,
)


def test_config_derives_200hz_substeps() -> None:
    cfg = HighFrequencyAlignmentConfig(source_hz=20.0, control_hz=200.0)
    assert np.isclose(cfg.source_dt, 0.05)
    assert np.isclose(cfg.control_dt, 0.005)
    assert cfg.substeps_per_source_step == 10


def test_alignment_feature_shape_and_phase_terms() -> None:
    feature = build_alignment_feature(
        q_error=np.ones(3, dtype=np.float32),
        qvel_error=np.ones(3, dtype=np.float32) * 2.0,
        base_action=np.ones(2, dtype=np.float32) * 3.0,
        previous_base_action=None,
        previous_residual=None,
        source_phase=0.25,
    )
    assert feature.shape == (3 + 3 + 2 + 2 + 2 + 3,)
    np.testing.assert_allclose(feature[:3], 1.0)
    np.testing.assert_allclose(feature[3:6], 2.0)
    np.testing.assert_allclose(feature[6:8], 3.0)
    np.testing.assert_allclose(feature[-3], 0.25)
    np.testing.assert_allclose(feature[-2], 1.0, atol=1e-6)
    np.testing.assert_allclose(feature[-1], 0.0, atol=1e-6)


def test_residual_clipping_limits_norm_and_per_dim() -> None:
    residual = np.array([3.0, 4.0], dtype=np.float32)
    clipped_norm = clip_residual_norm(residual, 2.5)
    assert np.isclose(np.linalg.norm(clipped_norm), 2.5)
    clipped_dim = clip_residual_per_dim(residual, 1.0)
    np.testing.assert_allclose(clipped_dim, np.array([1.0, 1.0], dtype=np.float32))


def test_interpolation_uses_dense_substep_phase() -> None:
    values = np.array([[0.0, 0.0], [10.0, 20.0]], dtype=np.float32)
    first = interpolate_source_target(values, source_step=0, substep=0, substeps_per_source_step=10)
    last = interpolate_source_target(values, source_step=0, substep=9, substeps_per_source_step=10)
    np.testing.assert_allclose(first, np.array([1.0, 2.0], dtype=np.float32))
    np.testing.assert_allclose(last, np.array([10.0, 20.0], dtype=np.float32))
    early = interpolate_source_target(values, source_step=0, substep=0, substeps_per_source_step=10, phase_power=0.5)
    assert early[0] > first[0]
    assert early[0] < last[0]


def test_shadow_hand_action_jacobian_matches_reduced_action_order() -> None:
    jac = build_shadow_hand_action_jacobian(action_dim=39, q_dim=46)
    assert jac is not None
    assert jac.shape == (39, 46)
    assert jac[0, 0] == 1.0
    assert jac[2, 18] == 1.0
    assert jac[5, 2] == 1.0
    assert jac[7, 4] == 0.5
    assert jac[7, 5] == 0.5
    assert jac[17, 21] == 1.0
    assert jac[19, 23] == 1.0
    assert jac[21, 41] == 1.0
    assert jac[37, 45] == 1.0
    assert not jac[38].any()


def test_aligner_outputs_action_and_online_update_diagnostics() -> None:
    cfg = HighFrequencyAlignmentConfig(
        source_hz=20.0,
        control_hz=200.0,
        kp=0.2,
        kd=0.0,
        residual_clip_norm=10.0,
        residual_clip_per_dim=1.0,
        smoothing_alpha=0.0,
        replay_capacity=64,
        warmup_samples=2,
        update_every_substeps=1,
        batch_size=2,
        train_steps_per_update=1,
        hidden_dim=16,
        hidden_layers=1,
        device="cpu",
        seed=3,
    )
    aligner = AllegroAligner(config=cfg, q_dim=4, action_dim=4)
    result0 = aligner.act(
        base_action=np.zeros(4, dtype=np.float32),
        current_q=np.zeros(4, dtype=np.float32),
        current_qvel=np.zeros(4, dtype=np.float32),
        target_q=np.ones(4, dtype=np.float32),
        target_qvel=np.zeros(4, dtype=np.float32),
        source_step=0,
        substep=0,
    )
    result1 = aligner.act(
        base_action=np.zeros(4, dtype=np.float32),
        current_q=np.zeros(4, dtype=np.float32),
        current_qvel=np.zeros(4, dtype=np.float32),
        target_q=np.ones(4, dtype=np.float32),
        target_qvel=np.zeros(4, dtype=np.float32),
        source_step=0,
        substep=1,
    )
    assert result0.action.shape == (4,)
    assert result0.feedback_residual.shape == (4,)
    assert len(aligner.trainer.replay) == 2
    assert result1.diagnostics["online_update"] is True
    assert aligner.trainer.updates == 1


def test_online_affine_action_model_solves_inverse_action() -> None:
    model = OnlineAffineActionModel(action_dim=2, q_dim=2, capacity=16, ridge=1e-6)
    for action in (
        np.array([0.0, 0.0], dtype=np.float32),
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0], dtype=np.float32),
        np.array([1.0, 1.0], dtype=np.float32),
    ):
        before = np.zeros(2, dtype=np.float32)
        after = np.array([2.0 * action[0], -3.0 * action[1]], dtype=np.float32)
        model.observe(action, before, after)

    desired = np.array([1.0, -1.5], dtype=np.float32)
    action = model.inverse_action(
        desired_delta=desired,
        base_action=np.zeros(2, dtype=np.float32),
        damping=1e-6,
    )

    assert model.ready
    np.testing.assert_allclose(action, np.array([0.5, 0.5], dtype=np.float32), atol=1e-3)


def test_aligner_inverse_model_observes_transition_and_contributes_residual() -> None:
    cfg = HighFrequencyAlignmentConfig(
        source_hz=20.0,
        control_hz=200.0,
        feedback_residual_scale=0.0,
        learned_residual_scale=0.0,
        inverse_model_enabled=True,
        inverse_model_warmup=1,
        inverse_model_damping=1e-4,
        inverse_model_ridge=1e-6,
        inverse_residual_scale=1.0,
        residual_clip_norm=10.0,
        residual_clip_per_dim=10.0,
        smoothing_alpha=0.0,
        device="cpu",
    )
    aligner = AllegroAligner(config=cfg, q_dim=2, action_dim=2)
    for action in (
        np.array([0.0, 0.0], dtype=np.float32),
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0], dtype=np.float32),
        np.array([1.0, 1.0], dtype=np.float32),
    ):
        aligner.observe_transition(
            action=action,
            q_before=np.zeros(2, dtype=np.float32),
            q_after=action.copy(),
        )

    result = aligner.act(
        base_action=np.zeros(2, dtype=np.float32),
        current_q=np.zeros(2, dtype=np.float32),
        current_qvel=np.zeros(2, dtype=np.float32),
        target_q=np.array([0.5, 0.25], dtype=np.float32),
        target_qvel=np.zeros(2, dtype=np.float32),
        source_step=0,
        substep=0,
    )

    assert result.diagnostics["alignment/inverse_ready"] is True
    assert result.diagnostics["alignment/inverse_residual_norm"] > 0.0
    np.testing.assert_allclose(result.action, np.array([0.5, 0.25], dtype=np.float32), atol=1e-2)
