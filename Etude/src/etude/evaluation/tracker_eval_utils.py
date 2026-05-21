from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import torch

from etude.controllers import build_controller, normalize_controller_family
from etude.controllers.bagatelle_residual import BagatelleResidualController
from etude.controllers.contact_policy import IdentityContactPolicy
from etude.controllers.pd import PDController
from etude.controllers.pd_scheduled import ScheduledPDController
from etude.controllers.residual_safety import PhaseGatingConfig, ResidualSafetyConfig
from etude.data.feature_builder import FeatureSpec
from etude.features.fingertip_phase_blocks import DEFAULT_PHASE_NAMES, FingertipFeatureSpec, PhaseFeatureSpec
from etude.features.inverse_dynamics_blocks import InverseDynamicsFeatureSpec
from etude.utils.import_utils import load_symbol


def build_tracker_controller(mapping: Any, config: dict[str, Any], *, checkpoint_path: str | None) -> Any:
    controller_cfg = _as_dict(config.get("controller"))
    family = normalize_controller_family(str(controller_cfg.get("family") or controller_cfg.get("type") or "pd"))
    device = resolve_device(controller_cfg, config)
    pd_controller = build_pd_controller(mapping, config)

    if family == "pd":
        return pd_controller

    if family == "hierarchical":
        return build_controller(
            family,
            IdentityContactPolicy(),
            build_hierarchical_low_level(mapping, config),
        )

    checkpoint = load_checkpoint(checkpoint_path or controller_cfg.get("model_path"))
    if checkpoint.get("kind") == "bagatelle_refinement":
        return build_tracker_controller_from_checkpoint_payload(mapping, checkpoint)
    model = instantiate_model(controller_cfg, checkpoint, device=device)

    if family == "pd_residual":
        controller_type = str(controller_cfg.get("type") or controller_cfg.get("family") or family)
        kwargs: dict[str, Any] = {
            "mapping": mapping,
            "residual_model": model,
            "pd": pd_controller,
            "feature_spec": build_tracking_feature_spec(config),
            "device": device,
        }
        if "safe" in controller_type.lower():
            kwargs["safety"] = build_residual_safety(config)
        return build_controller(controller_type, **kwargs)

    if family == "key_aware_residual":
        return build_controller(
            family,
            mapping,
            residual_model=model,
            pd=pd_controller,
            feature_block_paths=list(controller_cfg.get("feature_block_paths", [])) or None,
            feature_block_kwargs=_feature_block_kwargs(config, controller_cfg),
            device=device,
            residual_scale=float(controller_cfg.get("residual_scale", 1.0)),
            residual_clip=float(controller_cfg.get("residual_clip", 1.0)),
        )

    if family == "fingertip_residual":
        fingertip_cfg = _as_dict(config.get("fingertip_control"))
        fingertip_cfg.update(_as_dict(controller_cfg.get("fingertip_control")))
        phase_cfg = _as_dict(config.get("phase"))
        phase_cfg.update(_as_dict(controller_cfg.get("phase")))
        return build_controller(
            family,
            mapping,
            residual_model=model,
            pd=pd_controller,
            fingertip_spec=FingertipFeatureSpec(
                include_current=bool(fingertip_cfg.get("enabled", True)),
                include_desired=bool(fingertip_cfg.get("enabled", True)),
                include_error=bool(fingertip_cfg.get("use_fingertip_error_features", True)),
                include_weights=bool(fingertip_cfg.get("use_fingertip_weights", True)),
                include_active_mask=bool(fingertip_cfg.get("use_active_finger_mask", True)),
                include_inactive_mask=bool(fingertip_cfg.get("use_inactive_finger_mask", True)),
                allow_missing=bool(fingertip_cfg.get("allow_missing", True)),
            ),
            phase_spec=PhaseFeatureSpec(
                source=str(phase_cfg.get("source", "metadata")),
                encode_as=str(phase_cfg.get("encode_as", "both")),
                include_mask=bool(phase_cfg.get("include_mask", True)),
                allow_missing=bool(phase_cfg.get("allow_missing", True)),
                phase_names=tuple(phase_cfg.get("phase_names", DEFAULT_PHASE_NAMES)),
            ),
            phase_gain=float(controller_cfg.get("phase_gain", 1.0)),
            device=device,
            residual_limit=controller_cfg.get("residual_limit"),
        )

    if family == "temporal_residual":
        temporal_cfg = _as_dict(config.get("temporal"))
        return build_controller(
            family,
            mapping,
            temporal_model=model,
            pd=pd_controller,
            feature_spec=build_tracking_feature_spec(config),
            history_steps=int(temporal_cfg.get("history_steps", 16)),
            include_previous_residual=bool(temporal_cfg.get("include_previous_residual", True)),
            reset_hidden_on_episode=bool(temporal_cfg.get("reset_hidden_on_episode", True)),
            residual_scale=float(controller_cfg.get("residual_scale", 1.0)),
            residual_clip=float(controller_cfg.get("residual_clip", 1.0)),
            device=device,
            use_hidden_state=bool(controller_cfg.get("use_hidden_state", True)),
        )

    if family == "inverse_dynamics":
        return build_controller(
            family,
            mapping,
            model=model,
            output_mode=str(controller_cfg.get("output_mode", "full_action")),
            pd=pd_controller,
            feature_spec=InverseDynamicsFeatureSpec(),
            device=device,
        )

    raise ValueError(f"Unsupported controller family for evaluation: {family}")


def build_tracker_controller_from_checkpoint_payload(mapping: Any, checkpoint: dict[str, Any]) -> Any:
    if checkpoint.get("kind") == "bagatelle_refinement":
        base_checkpoint = checkpoint.get("base_checkpoint")
        if not isinstance(base_checkpoint, dict):
            raise ValueError("Refinement checkpoint is missing base_checkpoint")
        base_controller = build_tracker_controller_from_checkpoint_payload(mapping, base_checkpoint)
        config = _as_dict(checkpoint.get("config"))
        controller_cfg = _as_dict(config.get("controller"))
        device = resolve_device(controller_cfg, config)
        model = instantiate_refinement_model(checkpoint, device=device)
        return BagatelleResidualController(
            mapping,
            base_controller,
            model,
            fingertip_spec=build_fingertip_feature_spec(config),
            phase_spec=build_phase_feature_spec(config),
            device=device,
            residual_limit=controller_cfg.get("residual_limit"),
            residual_scale=float(controller_cfg.get("residual_scale", 1.0)),
        )

    config = _as_dict(checkpoint.get("config"))
    if not config:
        raise ValueError("Checkpoint does not contain a usable config")
    controller_cfg = _as_dict(config.get("controller"))
    family = normalize_controller_family(str(controller_cfg.get("family") or controller_cfg.get("type") or "pd"))
    device = resolve_device(controller_cfg, config)
    pd_controller = build_pd_controller(mapping, config)

    if family == "pd":
        return pd_controller
    if family == "hierarchical":
        return build_controller(
            family,
            IdentityContactPolicy(),
            build_hierarchical_low_level(mapping, config),
        )

    model = instantiate_model(controller_cfg, checkpoint, device=device)
    if family == "pd_residual":
        controller_type = str(controller_cfg.get("type") or controller_cfg.get("family") or family)
        kwargs: dict[str, Any] = {
            "mapping": mapping,
            "residual_model": model,
            "pd": pd_controller,
            "feature_spec": build_tracking_feature_spec(config),
            "device": device,
        }
        if "safe" in controller_type.lower():
            kwargs["safety"] = build_residual_safety(config)
        return build_controller(controller_type, **kwargs)
    if family == "key_aware_residual":
        return build_controller(
            family,
            mapping,
            residual_model=model,
            pd=pd_controller,
            feature_block_paths=list(controller_cfg.get("feature_block_paths", [])) or None,
            feature_block_kwargs=_feature_block_kwargs(config, controller_cfg),
            device=device,
            residual_scale=float(controller_cfg.get("residual_scale", 1.0)),
            residual_clip=float(controller_cfg.get("residual_clip", 1.0)),
        )
    if family == "fingertip_residual":
        return build_controller(
            family,
            mapping,
            residual_model=model,
            pd=pd_controller,
            fingertip_spec=build_fingertip_feature_spec(config),
            phase_spec=build_phase_feature_spec(config),
            phase_gain=float(controller_cfg.get("phase_gain", 1.0)),
            device=device,
            residual_limit=controller_cfg.get("residual_limit"),
        )
    if family == "temporal_residual":
        temporal_cfg = _as_dict(config.get("temporal"))
        return build_controller(
            family,
            mapping,
            temporal_model=model,
            pd=pd_controller,
            feature_spec=build_tracking_feature_spec(config),
            history_steps=int(temporal_cfg.get("history_steps", 16)),
            include_previous_residual=bool(temporal_cfg.get("include_previous_residual", True)),
            reset_hidden_on_episode=bool(temporal_cfg.get("reset_hidden_on_episode", True)),
            residual_scale=float(controller_cfg.get("residual_scale", 1.0)),
            residual_clip=float(controller_cfg.get("residual_clip", 1.0)),
            device=device,
            use_hidden_state=bool(controller_cfg.get("use_hidden_state", True)),
        )
    if family == "inverse_dynamics":
        return build_controller(
            family,
            mapping,
            model=model,
            output_mode=str(controller_cfg.get("output_mode", "full_action")),
            pd=pd_controller,
            feature_spec=InverseDynamicsFeatureSpec(),
            device=device,
        )
    raise ValueError(f"Unsupported controller family in checkpoint: {family}")


def build_hierarchical_low_level(mapping: Any, config: dict[str, Any]) -> Any:
    low_level_type = str(_as_dict(_as_dict(config.get("hierarchical")).get("lowlevel")).get("type", "scheduled_pd"))
    if low_level_type != "scheduled_pd":
        raise ValueError(f"Unsupported hierarchical low-level controller: {low_level_type}")
    return build_pd_controller(mapping, config)


def build_pd_controller(mapping: Any, config: dict[str, Any]) -> Any:
    controller_cfg = _as_dict(config.get("controller"))
    pd_cfg = _as_dict(config.get("pd"))
    lookahead = first_lookahead(pd_cfg.get("lookahead_steps"), _as_dict(config.get("trajectory")).get("lookahead_steps"))
    controller_type = str(controller_cfg.get("type") or controller_cfg.get("family") or "pd")

    if "scheduled_pd" in controller_type or controller_cfg.get("family") == "pd" or "mode" in pd_cfg:
        return ScheduledPDController(
            mapping,
            kp=pd_cfg.get("kp", pd_cfg.get("kp_init", 12.0)),
            kd=pd_cfg.get("kd", pd_cfg.get("kd_init", 0.6)),
            mode=str(pd_cfg.get("mode", "scalar")),
            lookahead_steps=lookahead,
            action_clip=bool(pd_cfg.get("action_clip", True)),
            joint_groups=_maybe_mapping(pd_cfg.get("joint_groups")),
            kp_groups=_maybe_mapping(pd_cfg.get("kp_groups")),
            kd_groups=_maybe_mapping(pd_cfg.get("kd_groups")),
            phase_kp_scales=_maybe_mapping(pd_cfg.get("phase_kp_scales")),
            phase_kd_scales=_maybe_mapping(pd_cfg.get("phase_kd_scales")),
            action_smoothing=_maybe_mapping(pd_cfg.get("action_smoothing")),
        )

    return PDController(
        mapping,
        kp=pd_cfg.get("kp_init", 12.0),
        kd=pd_cfg.get("kd_init", 0.6),
        lookahead=lookahead,
        clip=bool(pd_cfg.get("action_clip", True)),
    )


def build_tracking_feature_spec(config: dict[str, Any]) -> FeatureSpec:
    controller_cfg = _as_dict(config.get("controller"))
    temporal_cfg = _as_dict(config.get("temporal"))
    lookahead = controller_cfg.get("lookahead_steps", temporal_cfg.get("lookahead_steps"))
    return FeatureSpec(
        lookahead_steps=coerce_steps(lookahead, default=(1, 5, 10)),
        include_target_keys=True,
        include_fingertips=True,
    )


def build_fingertip_feature_spec(config: dict[str, Any]) -> FingertipFeatureSpec:
    controller_cfg = _as_dict(config.get("controller"))
    fingertip_cfg = _as_dict(config.get("fingertip_control"))
    fingertip_cfg.update(_as_dict(controller_cfg.get("fingertip_control")))
    return FingertipFeatureSpec(
        include_current=bool(fingertip_cfg.get("enabled", True)),
        include_desired=bool(fingertip_cfg.get("enabled", True)),
        include_error=bool(fingertip_cfg.get("use_fingertip_error_features", True)),
        include_weights=bool(fingertip_cfg.get("use_fingertip_weights", True)),
        include_active_mask=bool(fingertip_cfg.get("use_active_finger_mask", True)),
        include_inactive_mask=bool(fingertip_cfg.get("use_inactive_finger_mask", True)),
        allow_missing=bool(fingertip_cfg.get("allow_missing", True)),
    )


def build_phase_feature_spec(config: dict[str, Any]) -> PhaseFeatureSpec:
    controller_cfg = _as_dict(config.get("controller"))
    phase_cfg = _as_dict(config.get("phase"))
    phase_cfg.update(_as_dict(controller_cfg.get("phase")))
    return PhaseFeatureSpec(
        source=str(phase_cfg.get("source", "metadata")),
        encode_as=str(phase_cfg.get("encode_as", "both")),
        include_mask=bool(phase_cfg.get("include_mask", True)),
        allow_missing=bool(phase_cfg.get("allow_missing", True)),
        phase_names=tuple(phase_cfg.get("phase_names", DEFAULT_PHASE_NAMES)),
    )


def build_residual_safety(config: dict[str, Any]) -> ResidualSafetyConfig:
    residual_cfg = _as_dict(config.get("residual"))
    phase_cfg = _as_dict(residual_cfg.get("phase_gating"))
    return ResidualSafetyConfig(
        scale=float(residual_cfg.get("scale", 1.0)),
        clip_norm=residual_cfg.get("clip_norm"),
        clip_per_dim=residual_cfg.get("clip_per_dim"),
        smoothing_alpha=float(residual_cfg.get("smoothing_alpha", 0.0)),
        phase_gating=PhaseGatingConfig(
            enabled=bool(phase_cfg.get("enabled", False)),
            approach=float(phase_cfg.get("approach", 1.0)),
            pre_contact=float(phase_cfg.get("pre_contact", 1.0)),
            contact=float(phase_cfg.get("contact", 1.0)),
            hold=float(phase_cfg.get("hold", 1.0)),
            release=float(phase_cfg.get("release", 1.0)),
        ),
    )


def _feature_block_kwargs(config: dict[str, Any], controller_cfg: dict[str, Any]) -> dict[str, Any]:
    kwargs = dict(_as_dict(controller_cfg.get("feature_block_kwargs")))
    feature_cfg = _as_dict(config.get("features"))
    kwargs.update(_as_dict(feature_cfg.get("block_kwargs")))
    return kwargs


def instantiate_model(controller_cfg: dict[str, Any], checkpoint: dict[str, Any], *, device: str) -> torch.nn.Module:
    merged_cfg = dict(_as_dict(checkpoint.get("config", {})).get("controller", {}))
    merged_cfg.update(controller_cfg)

    model_module = str(merged_cfg.get("model_module") or default_model_module(merged_cfg))
    model_cls = load_symbol(model_module)
    if not isinstance(model_cls, type):
        raise TypeError(f"Resolved model module is not a class: {model_module}")

    input_dim = int(checkpoint.get("input_dim") or merged_cfg.get("input_dim") or 0)
    action_dim = int(checkpoint.get("action_dim") or merged_cfg.get("action_dim") or 0)
    if input_dim <= 0 or action_dim <= 0:
        raise ValueError("Checkpoint or controller config must define positive input_dim and action_dim")

    signature = inspect.signature(model_cls)
    kwargs: dict[str, Any] = {}
    for name in signature.parameters:
        if name == "self":
            continue
        if name == "input_dim":
            kwargs[name] = input_dim
        elif name == "action_dim":
            kwargs[name] = action_dim
        elif name in merged_cfg:
            kwargs[name] = merged_cfg[name]
    model = model_cls(**kwargs)
    state_dict = checkpoint.get("model")
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint does not contain a model state_dict")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_checkpoint(path: str | None) -> dict[str, Any]:
    if path is None:
        raise ValueError("This controller family requires a checkpoint path")
    checkpoint_path = Path(path).expanduser()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Checkpoint must deserialize to a dict, got {type(payload)!r}")
    return payload


def instantiate_refinement_model(checkpoint: dict[str, Any], *, device: str) -> torch.nn.Module:
    module_path = str(checkpoint.get("correction_model_module") or "etude.controllers.residual_mlp:ResidualMLP")
    model_cls = load_symbol(module_path)
    if not isinstance(model_cls, type):
        raise TypeError(f"Resolved correction model is not a class: {module_path}")
    input_dim = int(checkpoint.get("input_dim") or 0)
    action_dim = int(checkpoint.get("action_dim") or 0)
    if input_dim <= 0 or action_dim <= 0:
        raise ValueError("Refinement checkpoint must define positive input_dim and action_dim")
    model = model_cls(input_dim=input_dim, action_dim=action_dim)
    state_dict = checkpoint.get("correction_model")
    if not isinstance(state_dict, dict):
        raise ValueError("Refinement checkpoint does not contain a correction_model state_dict")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def default_model_module(controller_cfg: dict[str, Any]) -> str:
    model_type = str(controller_cfg.get("model_type") or controller_cfg.get("residual_model") or "mlp").lower()
    if model_type == "gru":
        return "etude.controllers.residual_gru:ResidualGRU"
    if model_type == "mlp":
        return "etude.controllers.residual_mlp:ResidualMLP"
    raise ValueError(f"Unsupported model_type without explicit model_module: {model_type}")


def resolve_device(controller_cfg: dict[str, Any], config: dict[str, Any]) -> str:
    requested = str(controller_cfg.get("device") or _as_dict(config.get("training")).get("device") or "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return requested


def first_lookahead(*values: Any) -> int:
    for value in values:
        if isinstance(value, (list, tuple)) and value:
            return int(value[0])
        if value is not None:
            return int(value)
    return 1


def coerce_steps(value: Any, *, default: tuple[int, ...]) -> tuple[int, ...]:
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        return tuple(int(step) for step in value) or default
    return (int(value),)


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _maybe_mapping(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return dict(value)
    return None
