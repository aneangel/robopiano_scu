from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn
from torch.utils.data import DataLoader

from rhapsody.config import RhapsodyConfig
from rhapsody.data import RPIKArrays, RPIKDataset
from rhapsody.evaluation import EvaluationMetrics, evaluate_forward_model, evaluate_policy
from rhapsody.models import ForwardKinematicsSurrogate, ResidualIKPolicy
from rhapsody.normalization import RhapsodyNormalizer
from rhapsody.reward import fingertip_reward


@dataclass(slots=True)
class RhapsodyTrainingResult:
    policy: ResidualIKPolicy
    fk_model: ForwardKinematicsSurrogate
    normalizer: RhapsodyNormalizer
    train_metrics: EvaluationMetrics
    validation_metrics: EvaluationMetrics
    history: dict[str, list[float]] = field(default_factory=dict)
    train_song_names: tuple[str, ...] = ()
    validation_song_names: tuple[str, ...] = ()


def train_rhapsody(
    arrays: RPIKArrays,
    config: RhapsodyConfig | None = None,
    *,
    fk_epochs: int = 20,
    bc_epochs: int = 5,
    policy_epochs: int = 20,
    validation_song_count: int = 0,
    device: torch.device | str = "cpu",
) -> RhapsodyTrainingResult:
    cfg = config or RhapsodyConfig()
    torch.manual_seed(int(cfg.seed))
    device = torch.device(device)
    if int(validation_song_count) > 0:
        train_arrays, val_arrays = arrays.split_by_song(
            int(validation_song_count),
            seed=int(cfg.seed),
        )
    else:
        train_arrays, val_arrays = arrays.split(cfg.validation_fraction, seed=int(cfg.seed))
    normalizer = RhapsodyNormalizer.fit(train_arrays).to(device)
    fk_model = ForwardKinematicsSurrogate(
        qpos_dim=cfg.qpos_dim,
        num_fingers=cfg.num_fingers,
        hidden_dims=cfg.fk_hidden_dims,
    ).to(device)
    policy = ResidualIKPolicy(
        qpos_dim=cfg.qpos_dim,
        num_fingers=cfg.num_fingers,
        hidden_dims=cfg.policy_hidden_dims,
        action_scale=cfg.policy_action_scale,
    ).to(device)

    history: dict[str, list[float]] = {
        "fk_loss": [],
        "bc_loss": [],
        "policy_loss": [],
        "policy_reward": [],
    }
    fit_forward_linear_warmstart(fk_model, normalizer, train_arrays, device=device)
    train_forward_model(fk_model, normalizer, train_arrays, cfg, epochs=fk_epochs, device=device, history=history)
    fit_policy_linear_warmstart(policy, normalizer, train_arrays, device=device)
    pretrain_policy_imitation(
        policy,
        normalizer,
        train_arrays,
        cfg,
        epochs=bc_epochs,
        device=device,
        history=history,
    )
    train_policy(
        policy,
        fk_model,
        normalizer,
        train_arrays,
        cfg,
        epochs=policy_epochs,
        device=device,
        history=history,
    )
    train_metrics = evaluate_policy(policy, fk_model, normalizer, train_arrays, cfg, device=device)
    validation_metrics = evaluate_policy(policy, fk_model, normalizer, val_arrays, cfg, device=device)
    return RhapsodyTrainingResult(
        policy=policy,
        fk_model=fk_model,
        normalizer=normalizer.to("cpu"),
        train_metrics=train_metrics,
        validation_metrics=validation_metrics,
        history=history,
        train_song_names=train_arrays.active_song_names(),
        validation_song_names=val_arrays.active_song_names(),
    )


def fit_forward_linear_warmstart(
    fk_model: ForwardKinematicsSurrogate,
    normalizer: RhapsodyNormalizer,
    arrays: RPIKArrays,
    *,
    device: torch.device,
    ridge: float = 1.0e-4,
) -> None:
    """Fit the FK affine term by ridge regression before residual training."""

    qpos = torch.from_numpy(arrays.expert_qpos).to(device)
    tips = torch.from_numpy(arrays.target_fingertips).to(device)
    x = normalizer.normalize_qpos(qpos)
    y = normalizer.normalize_fingertips(tips)
    ones = torch.ones((x.shape[0], 1), dtype=x.dtype, device=device)
    design = torch.cat([x, ones], dim=1)
    eye = torch.eye(design.shape[1], dtype=x.dtype, device=device)
    eye[-1, -1] = 0.0
    lhs = design.T @ design + float(ridge) * eye
    rhs = design.T @ y
    coeff = torch.linalg.solve(lhs, rhs)
    with torch.no_grad():
        fk_model.linear.weight.copy_(coeff[:-1].T)
        fk_model.linear.bias.copy_(coeff[-1])


def fit_policy_linear_warmstart(
    policy: ResidualIKPolicy,
    normalizer: RhapsodyNormalizer,
    arrays: RPIKArrays,
    *,
    device: torch.device,
    ridge: float = 1.0e-4,
) -> None:
    """Fit the policy affine term to RP1M residual hand states."""

    target = torch.from_numpy(arrays.target_fingertips).to(device)
    mask = torch.from_numpy(arrays.active_mask).to(device)
    previous = torch.from_numpy(arrays.previous_qpos).to(device)
    expert = torch.from_numpy(arrays.expert_qpos).to(device)
    target_norm = normalizer.normalize_fingertips(target).reshape(target.shape)
    previous_norm = normalizer.normalize_qpos(previous)
    expert_norm = normalizer.normalize_qpos(expert)
    features = policy.features(target_norm, mask, previous_norm)
    scaled_delta = ((expert_norm - previous_norm) / policy.action_scale).clamp(-0.95, 0.95)
    logits = torch.atanh(scaled_delta)
    ones = torch.ones((features.shape[0], 1), dtype=features.dtype, device=device)
    design = torch.cat([features, ones], dim=1)
    eye = torch.eye(design.shape[1], dtype=features.dtype, device=device)
    eye[-1, -1] = 0.0
    lhs = design.T @ design + float(ridge) * eye
    rhs = design.T @ logits
    coeff = torch.linalg.solve(lhs, rhs)
    with torch.no_grad():
        policy.linear.weight.copy_(coeff[:-1].T)
        policy.linear.bias.copy_(coeff[-1])


def train_forward_model(
    fk_model: ForwardKinematicsSurrogate,
    normalizer: RhapsodyNormalizer,
    arrays: RPIKArrays,
    config: RhapsodyConfig,
    *,
    epochs: int,
    device: torch.device,
    history: dict[str, list[float]] | None = None,
) -> None:
    fk_model.train()
    loader = DataLoader(
        RPIKDataset(arrays),
        batch_size=max(1, int(config.batch_size)),
        shuffle=True,
        drop_last=False,
    )
    optimizer = torch.optim.AdamW(
        fk_model.parameters(),
        lr=float(config.fk_learning_rate),
        weight_decay=float(config.weight_decay),
    )
    loss_fn = nn.MSELoss()
    for _epoch in range(max(0, int(epochs))):
        losses = []
        for batch in loader:
            qpos = batch["expert_qpos"].to(device)
            target = batch["target_fingertips"].to(device)
            pred = fk_model(normalizer.normalize_qpos(qpos))
            loss = loss_fn(pred.reshape(pred.shape[0], -1), normalizer.normalize_fingertips(target))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fk_model.parameters(), float(config.gradient_clip_norm))
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if history is not None:
            history.setdefault("fk_loss", []).append(sum(losses) / max(1, len(losses)))


def pretrain_policy_imitation(
    policy: ResidualIKPolicy,
    normalizer: RhapsodyNormalizer,
    arrays: RPIKArrays,
    config: RhapsodyConfig,
    *,
    epochs: int,
    device: torch.device,
    history: dict[str, list[float]] | None = None,
) -> None:
    policy.train()
    loader = DataLoader(
        RPIKDataset(arrays),
        batch_size=max(1, int(config.batch_size)),
        shuffle=True,
        drop_last=False,
    )
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=float(config.policy_learning_rate),
        weight_decay=float(config.weight_decay),
    )
    for _epoch in range(max(0, int(epochs))):
        losses = []
        for batch in loader:
            target = batch["target_fingertips"].to(device)
            mask = batch["active_mask"].to(device)
            previous = batch["previous_qpos"].to(device)
            expert = batch["expert_qpos"].to(device)
            target_norm = normalizer.normalize_fingertips(target).reshape(target.shape)
            previous_norm = normalizer.normalize_qpos(previous)
            expert_norm = normalizer.normalize_qpos(expert)
            pred = policy(target_norm, mask, previous_norm)
            loss = torch.mean((pred - expert_norm) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), float(config.gradient_clip_norm))
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if history is not None:
            history.setdefault("bc_loss", []).append(sum(losses) / max(1, len(losses)))


def train_policy(
    policy: ResidualIKPolicy,
    fk_model: ForwardKinematicsSurrogate,
    normalizer: RhapsodyNormalizer,
    arrays: RPIKArrays,
    config: RhapsodyConfig,
    *,
    epochs: int,
    device: torch.device,
    history: dict[str, list[float]] | None = None,
) -> None:
    fk_model.eval()
    policy.train()
    loader = DataLoader(
        RPIKDataset(arrays),
        batch_size=max(1, int(config.batch_size)),
        shuffle=True,
        drop_last=False,
    )
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=float(config.policy_learning_rate),
        weight_decay=float(config.weight_decay),
    )
    samples_per_state = max(1, int(config.policy_samples_per_state))
    std = float(config.policy_exploration_std)
    for _epoch in range(max(0, int(epochs))):
        losses = []
        rewards_seen = []
        for batch in loader:
            target = batch["target_fingertips"].to(device)
            mask = batch["active_mask"].to(device)
            previous = batch["previous_qpos"].to(device)
            expert = batch["expert_qpos"].to(device)
            target_norm_flat = normalizer.normalize_fingertips(target)
            target_norm = target_norm_flat.reshape(target.shape)
            previous_norm = normalizer.normalize_qpos(previous)
            expert_norm = normalizer.normalize_qpos(expert)
            mean = policy(target_norm, mask, previous_norm)

            if samples_per_state == 1:
                sampled = mean
                log_prob = torch.zeros(mean.shape[0], device=device)
            else:
                repeated_mean = mean[:, None, :].expand(-1, samples_per_state, -1)
                noise = torch.randn_like(repeated_mean) * std
                sampled = repeated_mean + noise
                sampled = sampled.reshape(-1, mean.shape[-1])
                mean_for_log_prob = repeated_mean.reshape(-1, mean.shape[-1])
                log_prob = -0.5 * torch.sum(
                    ((sampled.detach() - mean_for_log_prob) / std) ** 2,
                    dim=-1,
                )

            expanded_target = target[:, None, :, :].expand(-1, samples_per_state, -1, -1)
            expanded_target = expanded_target.reshape(-1, target.shape[1], target.shape[2])
            expanded_mask = mask[:, None, :].expand(-1, samples_per_state, -1).reshape(-1, mask.shape[1])
            expanded_previous = previous_norm[:, None, :].expand(-1, samples_per_state, -1)
            expanded_previous = expanded_previous.reshape(-1, previous_norm.shape[-1])

            with torch.no_grad():
                predicted_tips = normalizer.denormalize_fingertips(fk_model(sampled))
                rewards = fingertip_reward(
                    predicted_fingertips=predicted_tips,
                    target_fingertips=expanded_target,
                    active_mask=expanded_mask,
                    predicted_qpos_norm=sampled,
                    previous_qpos_norm=expanded_previous,
                    active_weight=config.reward_active_weight,
                    max_error_weight=config.reward_max_error_weight,
                    smoothness_weight=config.reward_smoothness_weight,
                )
                advantages = rewards - rewards.mean()
                reward_std = rewards.std(unbiased=False).clamp_min(1.0e-6)
                advantages = advantages / reward_std

            if samples_per_state == 1:
                policy_loss = torch.zeros((), device=device)
            else:
                policy_loss = -(advantages.detach() * log_prob).mean()
            imitation_loss = torch.mean((mean - expert_norm) ** 2)
            loss = policy_loss + float(config.imitation_weight) * imitation_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), float(config.gradient_clip_norm))
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            rewards_seen.append(float(rewards.mean().detach().cpu()))
        if history is not None:
            history.setdefault("policy_loss", []).append(sum(losses) / max(1, len(losses)))
            history.setdefault("policy_reward", []).append(sum(rewards_seen) / max(1, len(rewards_seen)))


def forward_model_metrics(
    fk_model: ForwardKinematicsSurrogate,
    normalizer: RhapsodyNormalizer,
    arrays: RPIKArrays,
    config: RhapsodyConfig,
    *,
    device: torch.device | str = "cpu",
) -> EvaluationMetrics:
    return evaluate_forward_model(fk_model, normalizer, arrays, config, device=device)
