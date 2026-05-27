from __future__ import annotations

import numpy as np

from pid_controller.controller import HandPIDController, make_controller_config


class FakeActionSpec:
    shape = (39,)
    minimum = np.full((39,), -10.0, dtype=np.float32)
    maximum = np.full((39,), 10.0, dtype=np.float32)


def test_p_controller_moves_to_next_target_in_actuator_units() -> None:
    controller = HandPIDController(make_controller_config("p", kp=1.0, setpoint_policy="next"))
    controller.reset(action_spec=FakeActionSpec())
    current = np.zeros((46,), dtype=np.float32)
    source = np.zeros((46,), dtype=np.float32)
    target = np.zeros((46,), dtype=np.float32)
    target[0] = 0.2
    target[4] = 0.4
    target[5] = 0.6
    target[22] = 0.03

    control = controller.compute_control(
        current_hand_qpos=current,
        current_hand_qvel=np.zeros((46,), dtype=np.float32),
        source_hand_qpos=source,
        target_hand_qpos=target,
        substep=0,
        substeps=10,
        simulation_timestep=0.005,
        dataset_timestep=0.05,
    )

    assert control[0] == np.float32(0.2)
    assert control[7] == np.float32(0.5)
    assert control[18] == np.float32(0.03)
    assert control[38] == np.float32(0.0)


def test_pd_controller_adds_velocity_lead() -> None:
    controller = HandPIDController(
        make_controller_config(
            "pd",
            kp=1.0,
            kd=0.01,
            setpoint_policy="minimum_jerk",
            use_target_velocity=True,
            target_velocity_scale=1.0,
        )
    )
    controller.reset(action_spec=FakeActionSpec())
    current = np.zeros((46,), dtype=np.float32)
    source = np.zeros((46,), dtype=np.float32)
    target = np.zeros((46,), dtype=np.float32)
    target[0] = 0.5

    control = controller.compute_control(
        current_hand_qpos=current,
        current_hand_qvel=np.zeros((46,), dtype=np.float32),
        source_hand_qpos=source,
        target_hand_qpos=target,
        substep=4,
        substeps=10,
        simulation_timestep=0.005,
        dataset_timestep=0.05,
    )

    alpha = np.float32(10 * (4 / 9) ** 3 - 15 * (4 / 9) ** 4 + 6 * (4 / 9) ** 5)
    assert control[0] > np.float32(0.5) * alpha


def test_target_velocity_scale_controls_derivative_lead() -> None:
    current = np.zeros((46,), dtype=np.float32)
    source = np.zeros((46,), dtype=np.float32)
    target = np.zeros((46,), dtype=np.float32)
    target[0] = 0.5

    no_lead = HandPIDController(
        make_controller_config(
            "pd",
            kp=1.0,
            kd=0.01,
            setpoint_policy="minimum_jerk",
            target_velocity_scale=0.0,
        )
    )
    no_lead.reset(action_spec=FakeActionSpec())
    lead = HandPIDController(
        make_controller_config(
            "pd",
            kp=1.0,
            kd=0.01,
            setpoint_policy="minimum_jerk",
            use_target_velocity=True,
            target_velocity_scale=1.0,
        )
    )
    lead.reset(action_spec=FakeActionSpec())

    no_lead_control = no_lead.compute_control(
        current_hand_qpos=current,
        current_hand_qvel=np.zeros((46,), dtype=np.float32),
        source_hand_qpos=source,
        target_hand_qpos=target,
        substep=4,
        substeps=10,
        simulation_timestep=0.005,
        dataset_timestep=0.05,
    )
    lead_control = lead.compute_control(
        current_hand_qpos=current,
        current_hand_qvel=np.zeros((46,), dtype=np.float32),
        source_hand_qpos=source,
        target_hand_qpos=target,
        substep=4,
        substeps=10,
        simulation_timestep=0.005,
        dataset_timestep=0.05,
    )

    alpha = np.float32(10 * (4 / 9) ** 3 - 15 * (4 / 9) ** 4 + 6 * (4 / 9) ** 5)
    assert np.isclose(no_lead_control[0], np.float32(0.5) * alpha)
    assert lead_control[0] > no_lead_control[0]


def test_linear_setpoint_uses_substep_fraction() -> None:
    controller = HandPIDController(make_controller_config("p", kp=1.0, setpoint_policy="linear"))
    controller.reset(action_spec=FakeActionSpec())
    current = np.zeros((46,), dtype=np.float32)
    source = np.zeros((46,), dtype=np.float32)
    target = np.zeros((46,), dtype=np.float32)
    target[0] = 1.0

    control = controller.compute_control(
        current_hand_qpos=current,
        current_hand_qvel=np.zeros((46,), dtype=np.float32),
        source_hand_qpos=source,
        target_hand_qpos=target,
        substep=4,
        substeps=10,
        simulation_timestep=0.005,
        dataset_timestep=0.05,
    )

    assert np.isclose(control[0], np.float32(4 / 9))


def test_minimum_jerk_setpoint_uses_smoothstep_fraction_in_actuator_space() -> None:
    controller = HandPIDController(
        make_controller_config("p", kp=1.0, setpoint_policy="minimum_jerk")
    )
    controller.reset(action_spec=FakeActionSpec())
    current = np.zeros((46,), dtype=np.float32)
    source = np.zeros((46,), dtype=np.float32)
    target = np.zeros((46,), dtype=np.float32)
    target[0] = 1.0
    target[4] = 0.2
    target[5] = 0.8

    control = controller.compute_control(
        current_hand_qpos=current,
        current_hand_qvel=np.zeros((46,), dtype=np.float32),
        source_hand_qpos=source,
        target_hand_qpos=target,
        substep=1,
        substeps=10,
        simulation_timestep=0.005,
        dataset_timestep=0.05,
    )

    alpha = np.float32(10 * (1 / 9) ** 3 - 15 * (1 / 9) ** 4 + 6 * (1 / 9) ** 5)
    assert np.isclose(control[0], alpha)
    assert np.isclose(control[7], np.float32(0.5) * alpha)


def test_minimum_jerk_reference_is_anchored_to_window_start_not_live_state() -> None:
    controller = HandPIDController(
        make_controller_config("p", kp=1.0, setpoint_policy="minimum_jerk")
    )
    controller.reset(action_spec=FakeActionSpec())
    current = np.zeros((46,), dtype=np.float32)
    current[0] = 0.8
    source = np.zeros((46,), dtype=np.float32)
    target = np.zeros((46,), dtype=np.float32)
    target[0] = 1.0

    control = controller.compute_control(
        current_hand_qpos=current,
        current_hand_qvel=np.zeros((46,), dtype=np.float32),
        source_hand_qpos=source,
        target_hand_qpos=target,
        substep=4,
        substeps=10,
        simulation_timestep=0.005,
        dataset_timestep=0.05,
    )

    alpha = np.float32(10 * (4 / 9) ** 3 - 15 * (4 / 9) ** 4 + 6 * (4 / 9) ** 5)
    assert np.isclose(control[0], alpha)


def test_feedforward_scale_tracks_reference_with_zero_residual_gain() -> None:
    controller = HandPIDController(
        make_controller_config(
            "p",
            kp=0.0,
            setpoint_policy="linear",
            feedforward_scale=1.0,
        )
    )
    controller.reset(action_spec=FakeActionSpec())
    current = np.zeros((46,), dtype=np.float32)
    source = np.zeros((46,), dtype=np.float32)
    target = np.zeros((46,), dtype=np.float32)
    target[0] = 1.0

    control = controller.compute_control(
        current_hand_qpos=current,
        current_hand_qvel=np.zeros((46,), dtype=np.float32),
        source_hand_qpos=source,
        target_hand_qpos=target,
        substep=4,
        substeps=10,
        simulation_timestep=0.005,
        dataset_timestep=0.05,
    )

    assert np.isclose(control[0], np.float32(4 / 9))
