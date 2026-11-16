"""Aim correction around the frozen policy, without changing the original env."""
from dataclasses import dataclass
import hashlib
import inspect
from pathlib import Path

from takehome import runtime  # noqa: F401
import jax
import jax.numpy as jp
import numpy as np
from peg_insertion import rollout
from peg_insertion.env import PegInsertion, default_config


@dataclass(frozen=True)
class AimConfig:
    step_cap: float = 0.0001
    guess_cap: float = 0.003

    def __post_init__(self):
        if not (np.isfinite(self.step_cap) and np.isfinite(self.guess_cap)
                and 0 < self.step_cap <= self.guess_cap):
            raise ValueError("Require finite 0 < step_cap <= guess_cap")


def advance_guess(guess, action, config):
    """Actions are dimensionless in [-1, 1]; guesses are in metres."""
    return jp.clip(guess + config.step_cap * jp.clip(action, -1.0, 1.0),
                   -config.guess_cap, config.guess_cap)


def corrected_observation(observation, guess):
    return {"state": observation.at[..., :2].add(-guess)}


def split_controls(action, outputs):
    if outputs == 2:
        return action, jp.zeros(3, dtype=action.dtype)
    if outputs == 3:
        return jp.zeros(2, dtype=action.dtype), action
    return action[:2], action[2:]


def add_movement_correction(robot_action, movement, movement_cap, robot_scale):
    return jp.clip(robot_action + jp.clip(movement, -1, 1) * movement_cap / robot_scale, -1, 1)


def learner_outcome(was_ended, was_success, success, env_failure, at_limit):
    """First-success reward, true termination and separate time truncation."""
    newly = (~was_ended) & (~was_success) & success
    terminated = (~was_ended) & (success | env_failure)
    truncated = (~was_ended) & at_limit & (~terminated)
    return newly.astype(jp.float32), terminated, truncated


def policy_digest(policy):
    """Hash actual in-memory frozen parameters, not just the checkpoint file."""
    params = inspect.getclosurevars(policy).nonlocals["params"]
    digest = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(params):
        value = np.asarray(leaf)
        digest.update(str(value.shape).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


class AimCorrectedEnv(PegInsertion):
    """Subclass compatible with the existing domain randomization and rollout.

    Observations contain the original sensor vector followed by the two-value
    running guess. Simulator info's true bias is never given to a learner.
    Physics continues after learner termination for full-length test rollouts;
    collection must discard transitions whose incoming state is already ended.
    """

    def __init__(self, policy_dir: Path, aim: AimConfig = AimConfig(), *,
                 outputs=2, movement_cap=0.0002, **overrides):
        self.frozen_policy, net_cfg = rollout.load_policy(Path(policy_dir))
        self.aim = aim
        self.outputs = outputs
        self.movement_cap = movement_cap
        self.base_obs_size = net_cfg["obs_size"]
        cfg = default_config()
        for k, v in net_cfg.get("env_overrides", {}).items():
            cfg[k] = v
        cfg.clearance_mm = net_cfg["clearance_mm"]
        cfg.use_force_obs = net_cfg["use_force_obs"]
        cfg.naconmax = 20 * 256
        for k, v in overrides.items():
            cfg[k] = v
        super().__init__(cfg)

    @property
    def action_size(self):
        return self.outputs

    def reset(self, rng):
        state = super().reset(rng)
        guess = jp.zeros(2)
        info = {**state.info, "aim_guess": guess,
                "old_action": jp.zeros(3), "executed_action": jp.zeros(3),
                "movement_correction": jp.zeros(3),
                "learner_terminated": jp.array(False),
                "learner_truncated": jp.array(False)}
        return state.replace(
            obs={"state": jp.concatenate([state.obs["state"], guess])}, info=info)

    def next_guess(self, state, action):
        if self.outputs == 3:
            return state.info['aim_guess']
        aim_action, _ = split_controls(action, self.outputs)
        return advance_guess(state.info["aim_guess"], aim_action, self.aim)

    def step(self, state, action):
        guess = self.next_guess(state, action)
        raw_obs = state.obs["state"][:self.base_obs_size]
        corrected = corrected_observation(raw_obs, guess)
        # Deterministic policy ignores this key; preserve simulator RNG stream.
        robot_action, _ = self.frozen_policy(corrected, jax.random.PRNGKey(0))
        old_action, _ = self.frozen_policy({"state": raw_obs}, jax.random.PRNGKey(0))
        movement_correction = jp.zeros(3)
        if self.outputs != 2:
            _, movement = split_controls(action, self.outputs)
            executed = add_movement_correction(robot_action, movement,
                                               self.movement_cap, self._config.action_scale)
            movement_correction = (executed - robot_action) * self._config.action_scale
            robot_action = executed
        nxt = super().step(state, robot_action)
        ended = state.info["learner_terminated"] | state.info["learner_truncated"]
        reward, terminated, truncated = learner_outcome(
            ended, state.info["succeeded"] > 0.5, nxt.info["succeeded"] > 0.5,
            nxt.done > 0.5, nxt.info["step_count"] >= self._config.episode_length)
        terminated = state.info["learner_terminated"] | terminated
        truncated = state.info["learner_truncated"] | truncated
        info = {**nxt.info, "aim_guess": guess,
                "old_action": old_action, "executed_action": robot_action,
                "movement_correction": movement_correction,
                "learner_terminated": terminated, "learner_truncated": truncated}
        return nxt.replace(
            obs={"state": jp.concatenate([nxt.obs["state"], guess])},
            reward=reward, done=(terminated | truncated).astype(jp.float32), info=info)


class OracleAimEnv(AimCorrectedEnv):
    """Test-only oracle: true constant bias is privileged information."""

    def next_guess(self, state, action):
        del action
        return jp.clip(state.info["bias"][:2], -self.aim.guess_cap, self.aim.guess_cap)


def zero_nudge(observation, key):
    del key
    return jp.zeros(observation["state"].shape[:-1] + (2,)), {}
