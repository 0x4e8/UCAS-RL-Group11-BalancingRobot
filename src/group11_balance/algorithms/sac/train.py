"""Train a Soft Actor-Critic controller with curriculum learning."""

from __future__ import annotations

import argparse
import csv
import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from tqdm.auto import tqdm

from group11_balance.algorithms.ppo.train import (
    LEVEL_ORDER,
    curriculum_levels,
    evaluate_policy,
    level_index,
    sample_teacher_states,
)
from group11_balance.sim.control import lqr_common_affine_policy, lqr_common_normalized_action
from group11_balance.sim.env import LEVELS, TwoStageBalanceEnv
from group11_balance.sim.task import TASK_BALANCE, TASKS, validate_task, validate_target_wheel_velocity
from group11_balance.visualization.rollout_html import write_policy_rollout_html


@dataclass
class TrainConfig:
    seed: int = 11
    total_steps: int = 200_000
    learning_rate: float = 3e-4
    lr_schedule: str = "constant"
    gamma: float = 0.995
    tau: float = 0.005
    buffer_size: int = 200_000
    learning_starts: int = 2048
    batch_size: int = 256
    train_freq: int = 1
    gradient_steps: int = 1
    ent_coef: str | float = "auto_0.05"
    target_entropy: str | float = "auto"
    net_arch: tuple[int, ...] = ()
    log_std_bias: float = -4.0
    start_level: str = "easy"
    max_level: str = "hard"
    curriculum: bool = True
    promotion_success_rate: float = 0.75
    promotion_patience: int = 1
    promotion_check_freq: int = 20_000
    promotion_eval_episodes: int = 20
    best_success_gate: float = 0.75
    eval_episodes: int = 20
    model_path: str = "outputs/models/group11_sac.zip"
    eval_csv: str = "outputs/logs/group11_sac_eval.csv"
    train_log: str = "outputs/logs/group11_sac_train.log"
    lqr_warm_start: bool = True
    lqr_exact_linear_init: bool = True
    lqr_warm_start_steps: int = 2000
    lqr_warm_start_samples: int = 12_288
    lqr_warm_start_batch: int = 512
    lqr_warm_start_lr: float = 1e-3
    lqr_trajectory_fraction: float = 0.15
    lqr_rollout_max_steps: int = 500
    task: str = TASK_BALANCE
    target_wheel_velocity: float = 0.0
    action_limit: float = 8000.0
    save_rollout_html: bool = True
    rollout_html: str | None = "outputs/visualizations/group11_sac.html"
    rollout_steps: int = 1000
    rollout_fps: int = 50
    device: str = "auto"


def configure_logger(path: str) -> logging.Logger:
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("group11_sac")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def read_config(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return {} if data is None else dict(data)


def _coerce_auto_or_float(value: Any) -> str | float:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "auto" or stripped.startswith("auto_"):
            return stripped
        return float(stripped)
    return float(value)


def _coerce_target_entropy(value: Any) -> str | float:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "auto":
            return stripped
        return float(stripped)
    return float(value)


def parse_net_arch(value: Any) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, int):
        items = [value]
    else:
        items = list(value)
    if len(items) == 0:
        return ()
    if len(items) == 1 and str(items[0]).lower() in {"none", "linear"}:
        return ()
    return tuple(int(v) for v in items)


def format_net_arch(net_arch: tuple[int, ...]) -> str:
    return "linear" if len(net_arch) == 0 else "-".join(str(v) for v in net_arch)


def merge_config(config: dict[str, Any], args: argparse.Namespace) -> TrainConfig:
    values = asdict(TrainConfig())
    values.update(config)
    for key in values:
        override = getattr(args, key, None)
        if override is not None:
            values[key] = override
    values["net_arch"] = parse_net_arch(values["net_arch"])
    values["ent_coef"] = _coerce_auto_or_float(values["ent_coef"])
    values["target_entropy"] = _coerce_target_entropy(values["target_entropy"])
    values["task"] = validate_task(values["task"])
    values["target_wheel_velocity"] = validate_target_wheel_velocity(values["target_wheel_velocity"])
    values["action_limit"] = float(values["action_limit"])
    if values["action_limit"] <= 0.0:
        raise ValueError("action_limit must be positive")
    return TrainConfig(**values)


def make_env(
    level: str,
    seed: int | None = None,
    *,
    task: str = TASK_BALANCE,
    target_wheel_velocity: float = 0.0,
    action_limit: float = 8000.0,
):
    env = TwoStageBalanceEnv(
        init_level=level,
        action_limit=action_limit,
        task=task,
        target_wheel_velocity=target_wheel_velocity,
    )
    if seed is not None:
        env.reset(seed=seed)
    return Monitor(env)


def make_schedule(value: float, schedule: str):
    if schedule == "constant":
        return value
    if schedule == "linear":
        return lambda progress_remaining: progress_remaining * value
    raise ValueError(f"unknown learning-rate schedule: {schedule}")


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def actor_mean_action(model: SAC, obs_tensor: torch.Tensor) -> torch.Tensor:
    return model.policy.actor(obs_tensor, deterministic=True)


def assign_linear_lqr_actor(
    model: SAC,
    action_limit: float = 8000.0,
    *,
    task: str = TASK_BALANCE,
    target_wheel_velocity: float = 0.0,
    log_std_bias: float = -4.0,
) -> bool:
    """Initialize a no-hidden-layer SAC actor mean with the common-mode LQR law.

    SAC still applies a final tanh squash in deterministic inference, so this is
    a stabilizing mean initialization rather than an exact linear policy.
    """
    if len(model.policy.actor.latent_pi) != 0:
        return False
    weights_env_order, bias = lqr_common_affine_policy(
        action_limit=action_limit,
        task=task,
        target_wheel_velocity=target_wheel_velocity,
    )
    layer = model.policy.actor.mu
    if layer.weight.shape != (1, len(weights_env_order)):
        return False
    with torch.no_grad():
        layer.weight.copy_(torch.as_tensor(weights_env_order[None, :], dtype=layer.weight.dtype, device=layer.weight.device))
        layer.bias.fill_(bias)
        model.policy.actor.log_std.weight.zero_()
        model.policy.actor.log_std.bias.fill_(float(log_std_bias))
    return True


def clone_actor_from_lqr(
    model: SAC,
    levels: list[str],
    *,
    n_samples: int,
    steps: int,
    batch_size: int,
    lr: float,
    seed: int,
    trajectory_fraction: float = 0.0,
    rollout_max_steps: int = 500,
    task: str = TASK_BALANCE,
    target_wheel_velocity: float = 0.0,
    action_limit: float = 8000.0,
    log_std_bias: float = -4.0,
) -> tuple[float, int]:
    if steps <= 0 or n_samples <= 0:
        return 0.0, 0
    states = sample_teacher_states(
        levels,
        n_samples=n_samples,
        seed=seed,
        trajectory_fraction=trajectory_fraction,
        rollout_max_steps=rollout_max_steps,
        task=task,
        target_wheel_velocity=target_wheel_velocity,
        action_limit=action_limit,
    )
    targets = np.asarray(
        [
            lqr_common_normalized_action(
                state,
                action_limit=action_limit,
                task=task,
                target_wheel_velocity=target_wheel_velocity,
            )
            for state in states
        ],
        dtype=np.float32,
    )
    device = model.policy.device
    obs_tensor = torch.as_tensor(states, dtype=torch.float32, device=device)
    target_tensor = torch.as_tensor(targets, dtype=torch.float32, device=device)

    params = list(model.policy.actor.latent_pi.parameters()) + list(model.policy.actor.mu.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    batch = min(batch_size, len(states))
    last_loss = 0.0
    for _ in range(steps):
        idx = torch.randint(0, len(states), (batch,), device=device)
        pred = actor_mean_action(model, obs_tensor[idx])
        loss = torch.mean((pred - target_tensor[idx]) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().cpu())
    with torch.no_grad():
        model.policy.actor.log_std.weight.zero_()
        model.policy.actor.log_std.bias.fill_(float(log_std_bias))
    return last_loss, len(states)


def warm_start_actor_from_lqr(model: SAC, cfg: TrainConfig, logger: logging.Logger) -> None:
    if not cfg.lqr_warm_start:
        logger.info("LQR warm start disabled")
        return
    if cfg.lqr_exact_linear_init and assign_linear_lqr_actor(
        model,
        action_limit=cfg.action_limit,
        task=cfg.task,
        target_wheel_velocity=cfg.target_wheel_velocity,
        log_std_bias=cfg.log_std_bias,
    ):
        logger.info("LQR linear SAC actor mean initialization finished")

    if cfg.lqr_warm_start_steps <= 0:
        return

    teacher_levels = curriculum_levels(cfg.start_level, cfg.max_level) if cfg.curriculum else [cfg.start_level]
    last_loss, samples = clone_actor_from_lqr(
        model,
        teacher_levels,
        n_samples=cfg.lqr_warm_start_samples,
        steps=cfg.lqr_warm_start_steps,
        batch_size=cfg.lqr_warm_start_batch,
        lr=cfg.lqr_warm_start_lr,
        seed=cfg.seed + 700_000,
        trajectory_fraction=cfg.lqr_trajectory_fraction,
        rollout_max_steps=cfg.lqr_rollout_max_steps,
        task=cfg.task,
        target_wheel_velocity=cfg.target_wheel_velocity,
        action_limit=cfg.action_limit,
        log_std_bias=cfg.log_std_bias,
    )
    logger.info(
        "LQR SAC actor warm start finished levels=%s samples=%d steps=%d batch=%d "
        "trajectory_fraction=%.3f rollout_max_steps=%d final_mse=%.8f",
        ",".join(teacher_levels),
        samples,
        cfg.lqr_warm_start_steps,
        min(cfg.lqr_warm_start_batch, samples),
        cfg.lqr_trajectory_fraction,
        cfg.lqr_rollout_max_steps,
        last_loss,
    )


class TqdmProgressCallback(BaseCallback):
    def __init__(self, total_steps: int):
        super().__init__()
        self.total_steps = int(total_steps)
        self.bar: tqdm | None = None
        self.last_seen_steps = 0

    def _on_training_start(self) -> None:
        self.bar = tqdm(total=self.total_steps, desc="SAC training", unit="step", dynamic_ncols=True)

    def _on_step(self) -> bool:
        if self.bar is not None:
            delta = self.num_timesteps - self.last_seen_steps
            if delta > 0:
                self.bar.update(delta)
                self.last_seen_steps = self.num_timesteps
        return True

    def _on_training_end(self) -> None:
        if self.bar is not None:
            remaining = max(0, self.total_steps - self.last_seen_steps)
            if remaining:
                self.bar.update(remaining)
            self.bar.close()


class CurriculumCallback(BaseCallback):
    def __init__(self, cfg: TrainConfig, logger: logging.Logger):
        super().__init__()
        self.cfg = cfg
        self.file_logger = logger
        self.current_level = cfg.start_level
        self.reached_level = cfg.start_level
        self.max_level_idx = level_index(cfg.max_level)
        self.last_check = 0
        self.success_streak = 0
        model_path = Path(cfg.model_path)
        self.best_model_path = model_path.with_name(f"{model_path.stem}_best.zip")
        self.best_level = cfg.start_level
        self.best_score: tuple[int, int, float, float] = (-1, -1, -1.0, -1.0)

    def _on_training_start(self) -> None:
        self.file_logger.info(
            "curriculum start level=%s max=%s threshold=%.3f patience=%d check_freq=%d "
            "eval_episodes=%d best_success_gate=%.3f",
            self.cfg.start_level,
            self.cfg.max_level,
            self.cfg.promotion_success_rate,
            self.cfg.promotion_patience,
            self.cfg.promotion_check_freq,
            self.cfg.promotion_eval_episodes,
            self.cfg.best_success_gate,
        )
        for level in curriculum_levels(self.cfg.start_level, self.cfg.max_level):
            metrics = evaluate_policy(
                self.model,
                level=level,
                episodes=self.cfg.promotion_eval_episodes,
                seed=self.cfg.seed + 90_000 + 1000 * level_index(level),
                logger=self.file_logger,
                tag=f"warm_start_eval_{level}",
                task=self.cfg.task,
                target_wheel_velocity=self.cfg.target_wheel_velocity,
                action_limit=self.cfg.action_limit,
            )
            solved = int(metrics["success_rate"] >= self.cfg.best_success_gate)
            score = (
                solved,
                level_index(level) if solved else -1,
                metrics["success_rate"],
                metrics["length_mean"],
            )
            self.file_logger.info(
                "warm-start candidate level=%s success=%.3f return=%.3f length=%.3f",
                level,
                metrics["success_rate"],
                metrics["return_mean"],
                metrics["length_mean"],
            )
            if score > self.best_score:
                self.best_score = score
                self.best_level = level
                self.best_model_path.parent.mkdir(parents=True, exist_ok=True)
                self.model.save(self.best_model_path)
                self.file_logger.info(
                    "saved warm-start checkpoint path=%s level=%s success=%.3f length=%.3f",
                    self.best_model_path,
                    self.best_level,
                    metrics["success_rate"],
                    metrics["length_mean"],
                )

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_check < self.cfg.promotion_check_freq:
            return True
        self.last_check = self.num_timesteps
        metrics = evaluate_policy(
            self.model,
            level=self.current_level,
            episodes=self.cfg.promotion_eval_episodes,
            seed=self.cfg.seed + 100_000 + self.num_timesteps,
            logger=self.file_logger,
            tag=f"curriculum_check_step_{self.num_timesteps}",
            task=self.cfg.task,
            target_wheel_velocity=self.cfg.target_wheel_velocity,
            action_limit=self.cfg.action_limit,
        )
        self.file_logger.info(
            "curriculum check step=%d level=%s success=%.3f return=%.3f length=%.3f",
            self.num_timesteps,
            self.current_level,
            metrics["success_rate"],
            metrics["return_mean"],
            metrics["length_mean"],
        )
        solved = int(metrics["success_rate"] >= self.cfg.best_success_gate)
        score = (
            solved,
            level_index(self.current_level) if solved else -1,
            metrics["success_rate"],
            metrics["length_mean"],
        )
        if score > self.best_score:
            self.best_score = score
            self.best_level = self.current_level
            self.best_model_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(self.best_model_path)
            self.file_logger.info(
                "saved best checkpoint path=%s level=%s success=%.3f length=%.3f",
                self.best_model_path,
                self.best_level,
                metrics["success_rate"],
                metrics["length_mean"],
            )
        if metrics["success_rate"] < self.cfg.promotion_success_rate:
            self.success_streak = 0
            return True
        self.success_streak += 1
        self.file_logger.info(
            "curriculum promotion streak=%d/%d level=%s",
            self.success_streak,
            self.cfg.promotion_patience,
            self.current_level,
        )
        if self.success_streak < self.cfg.promotion_patience:
            return True

        next_idx = level_index(self.current_level) + 1
        if next_idx > self.max_level_idx:
            self.file_logger.info("curriculum already at maximum level=%s", self.current_level)
            return True
        self.current_level = LEVEL_ORDER[next_idx]
        self.reached_level = self.current_level
        self.success_streak = 0
        self.training_env.env_method("set_level", self.current_level)
        self.file_logger.info("curriculum promoted to level=%s at step=%d", self.current_level, self.num_timesteps)
        return True


def write_eval_csv(path: str, row: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def train(cfg: TrainConfig) -> None:
    if cfg.start_level not in LEVELS:
        raise ValueError(f"start_level must be one of {sorted(LEVELS)}")
    if cfg.max_level not in LEVELS:
        raise ValueError(f"max_level must be one of {sorted(LEVELS)}")

    set_global_seed(cfg.seed)
    logger = configure_logger(cfg.train_log)
    logger.info("training config=%s", cfg)

    env = make_env(
        cfg.start_level,
        seed=cfg.seed,
        task=cfg.task,
        target_wheel_velocity=cfg.target_wheel_velocity,
        action_limit=cfg.action_limit,
    )
    model = SAC(
        "MlpPolicy",
        env,
        seed=cfg.seed,
        learning_rate=make_schedule(cfg.learning_rate, cfg.lr_schedule),
        policy_kwargs={"net_arch": list(cfg.net_arch)},
        buffer_size=cfg.buffer_size,
        learning_starts=cfg.learning_starts,
        batch_size=cfg.batch_size,
        tau=cfg.tau,
        gamma=cfg.gamma,
        train_freq=cfg.train_freq,
        gradient_steps=cfg.gradient_steps,
        ent_coef=cfg.ent_coef,
        target_entropy=cfg.target_entropy,
        verbose=0,
        device=cfg.device,
    )
    warm_start_actor_from_lqr(model, cfg, logger)

    callbacks: list[BaseCallback] = [TqdmProgressCallback(cfg.total_steps)]
    curriculum_callback = None
    if cfg.curriculum:
        curriculum_callback = CurriculumCallback(cfg, logger)
        callbacks.append(curriculum_callback)

    if cfg.total_steps > 0:
        model.learn(total_timesteps=cfg.total_steps, callback=CallbackList(callbacks), progress_bar=False)
    else:
        logger.info("skipped SAC updates because total_steps=0")

    final_level = cfg.start_level
    if curriculum_callback is not None and curriculum_callback.best_model_path.exists():
        model = SAC.load(curriculum_callback.best_model_path, env=env, device=cfg.device)
        final_level = curriculum_callback.best_level
        logger.info(
            "restored best checkpoint path=%s level=%s score=%s",
            curriculum_callback.best_model_path,
            final_level,
            curriculum_callback.best_score,
        )
    elif curriculum_callback is not None:
        final_level = curriculum_callback.reached_level

    model_path = Path(cfg.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    logger.info("saved final model path=%s eval_level=%s", model_path, final_level)

    metrics = evaluate_policy(
        model,
        level=final_level,
        episodes=cfg.eval_episodes,
        seed=cfg.seed + 900_000,
        logger=logger,
        tag="final_eval",
        task=cfg.task,
        target_wheel_velocity=cfg.target_wheel_velocity,
        action_limit=cfg.action_limit,
    )
    row = {
        "algorithm": "SAC",
        "total_steps": cfg.total_steps,
        "net_arch": format_net_arch(cfg.net_arch),
        "task": cfg.task,
        "target_wheel_velocity": cfg.target_wheel_velocity,
        "action_limit": cfg.action_limit,
        "start_level": cfg.start_level,
        "final_eval_level": final_level,
        "curriculum": cfg.curriculum,
        **metrics,
    }
    write_eval_csv(cfg.eval_csv, row)
    logger.info("saved evaluation csv=%s row=%s", cfg.eval_csv, row)

    if cfg.save_rollout_html and cfg.rollout_html:
        html_path = write_policy_rollout_html(
            model,
            output=cfg.rollout_html,
            algorithm_name="SAC",
            level=final_level,
            seed=cfg.seed + 910_000,
            task=cfg.task,
            target_wheel_velocity=cfg.target_wheel_velocity,
            action_limit=cfg.action_limit,
            steps=cfg.rollout_steps,
            fps=cfg.rollout_fps,
        )
        logger.info("saved rollout html=%s", html_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--total-steps", dest="total_steps", type=int, default=None)
    parser.add_argument("--learning-rate", dest="learning_rate", type=float, default=None)
    parser.add_argument("--lr-schedule", dest="lr_schedule", choices=["constant", "linear"], default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--buffer-size", dest="buffer_size", type=int, default=None)
    parser.add_argument("--learning-starts", dest="learning_starts", type=int, default=None)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=None)
    parser.add_argument("--train-freq", dest="train_freq", type=int, default=None)
    parser.add_argument("--gradient-steps", dest="gradient_steps", type=int, default=None)
    parser.add_argument("--ent-coef", dest="ent_coef", default=None)
    parser.add_argument("--target-entropy", dest="target_entropy", default=None)
    parser.add_argument("--net-arch", dest="net_arch", nargs="*", default=None)
    parser.add_argument("--log-std-bias", dest="log_std_bias", type=float, default=None)
    parser.add_argument("--start-level", dest="start_level", choices=LEVEL_ORDER, default=None)
    parser.add_argument("--max-level", dest="max_level", choices=LEVEL_ORDER, default=None)
    parser.add_argument("--task", choices=TASKS, default=None)
    parser.add_argument("--target-wheel-velocity", dest="target_wheel_velocity", type=float, default=None)
    parser.add_argument("--action-limit", dest="action_limit", type=float, default=None)
    parser.add_argument("--curriculum", dest="curriculum", action="store_true", default=None)
    parser.add_argument("--no-curriculum", dest="curriculum", action="store_false")
    parser.add_argument("--device", default=None)
    parser.add_argument("--model-path", dest="model_path", default=None)
    parser.add_argument("--eval-csv", dest="eval_csv", default=None)
    parser.add_argument("--train-log", dest="train_log", default=None)
    parser.add_argument("--lqr-warm-start", dest="lqr_warm_start", action="store_true", default=None)
    parser.add_argument("--no-lqr-warm-start", dest="lqr_warm_start", action="store_false")
    parser.add_argument("--lqr-warm-start-steps", dest="lqr_warm_start_steps", type=int, default=None)
    parser.add_argument("--lqr-warm-start-samples", dest="lqr_warm_start_samples", type=int, default=None)
    parser.add_argument("--lqr-warm-start-batch", dest="lqr_warm_start_batch", type=int, default=None)
    parser.add_argument("--lqr-warm-start-lr", dest="lqr_warm_start_lr", type=float, default=None)
    parser.add_argument("--lqr-exact-linear-init", dest="lqr_exact_linear_init", action="store_true", default=None)
    parser.add_argument("--no-lqr-exact-linear-init", dest="lqr_exact_linear_init", action="store_false")
    parser.add_argument("--lqr-trajectory-fraction", dest="lqr_trajectory_fraction", type=float, default=None)
    parser.add_argument("--lqr-rollout-max-steps", dest="lqr_rollout_max_steps", type=int, default=None)
    parser.add_argument("--promotion-success-rate", dest="promotion_success_rate", type=float, default=None)
    parser.add_argument("--promotion-patience", dest="promotion_patience", type=int, default=None)
    parser.add_argument("--promotion-check-freq", dest="promotion_check_freq", type=int, default=None)
    parser.add_argument("--promotion-eval-episodes", dest="promotion_eval_episodes", type=int, default=None)
    parser.add_argument("--best-success-gate", dest="best_success_gate", type=float, default=None)
    parser.add_argument("--eval-episodes", dest="eval_episodes", type=int, default=None)
    parser.add_argument("--rollout-html", dest="rollout_html", default=None)
    parser.add_argument("--no-rollout-html", dest="save_rollout_html", action="store_false", default=None)
    parser.add_argument("--rollout-steps", dest="rollout_steps", type=int, default=None)
    parser.add_argument("--rollout-fps", dest="rollout_fps", type=int, default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cfg = merge_config(read_config(args.config), args)
    train(cfg)


if __name__ == "__main__":
    main()
