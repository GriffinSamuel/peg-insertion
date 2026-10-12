"""Policy loading and evaluation rollouts shared by train.py and evaluate.py.

Brax's own eval wrapper *sums* metrics over an episode, so episode-level
statistics (did it succeed, when, peak force, final error) are read from
`state.info` at the last step of our own fixed-length rollout instead.
There is no auto-reset here: every env runs exactly one full episode.
"""

import functools
import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from brax.io import model as brax_model
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
import jax
import jax.numpy as jp
import numpy as np

from mujoco_playground._src import wrapper as pg_wrapper
from brax.envs.wrappers import training as brax_training

from peg_insertion.env import PegInsertion, default_config, domain_randomize

INFO_KEYS = ('succeeded', 'first_success_step', 'peak_force', 'xy_err', 'depth')


def make_env(clearance_mm: float, use_force_obs: bool, **overrides) -> PegInsertion:
  cfg = default_config()
  cfg.clearance_mm = float(clearance_mm)
  cfg.use_force_obs = bool(use_force_obs)
  for k, v in overrides.items():
    cfg[k] = v
  return PegInsertion(cfg)


def save_policy(run_dir: Path, params: Any, network_config: Dict[str, Any]) -> None:
  run_dir.mkdir(parents=True, exist_ok=True)
  brax_model.save_params(str(run_dir / 'params.pkl'), params)
  (run_dir / 'network_config.json').write_text(json.dumps(network_config, indent=2))


def load_policy(run_dir: Path, deterministic: bool = True) -> tuple[Callable, Dict]:
  """Rebuilds the PPO policy from params.pkl + network_config.json."""
  net_cfg = json.loads((Path(run_dir) / 'network_config.json').read_text())
  params = brax_model.load_params(str(Path(run_dir) / 'params.pkl'))
  preprocess = (
      running_statistics.normalize
      if net_cfg.get('normalize_observations', True)
      else lambda x, y: x
  )
  networks = ppo_networks.make_ppo_networks(
      observation_size={'state': (net_cfg['obs_size'],)},
      action_size=net_cfg['act_size'],
      preprocess_observations_fn=preprocess,
      policy_hidden_layer_sizes=tuple(net_cfg['policy_hidden_layer_sizes']),
      value_hidden_layer_sizes=tuple(net_cfg['value_hidden_layer_sizes']),
      policy_obs_key='state',
      value_obs_key='state',
  )
  make_policy = ppo_networks.make_inference_fn(networks)
  return make_policy(params, deterministic=deterministic), net_cfg


def _wrap(env: PegInsertion, n: int, key: jax.Array, randomize: bool):
  if randomize:
    rand_fn = functools.partial(domain_randomize, rng=jax.random.split(key, n))
    return pg_wrapper.BraxDomainRandomizationVmapWrapper(env, rand_fn)
  return brax_training.VmapWrapper(env)


def rollout_stats(
    env: PegInsertion,
    policy: Callable,
    n_episodes: int,
    seed: int = 0,
    randomize: bool = True,
    batch: int = 256,
    episode_length: Optional[int] = None,
) -> Dict[str, np.ndarray]:
  """Runs n_episodes full episodes; returns final-step info arrays."""
  episode_length = episode_length or env._config.episode_length
  out: Dict[str, list] = {k: [] for k in INFO_KEYS}
  done_episodes, chunk = 0, 0
  while done_episodes < n_episodes:
    n = min(batch, n_episodes - done_episodes)
    key = jax.random.PRNGKey(seed * 100_003 + chunk)
    k_rand, k_reset, k_act = jax.random.split(key, 3)
    wenv = _wrap(env, n, k_rand, randomize)
    info = _run(wenv, policy, n, episode_length, k_reset, k_act)
    for k in INFO_KEYS:
      out[k].append(np.array(info[k]))
    done_episodes += n
    chunk += 1
  return {k: np.concatenate(v) for k, v in out.items()}


def _run(wenv, policy, n, episode_length, k_reset, k_act):
  @jax.jit
  def run(k_reset, k_act):
    state = wenv.reset(jax.random.split(k_reset, n))

    def body(carry, _):
      state, key = carry
      key, sub = jax.random.split(key)
      action, _ = policy(state.obs, sub)
      return (wenv.step(state, action), key), None

    (state, _), _ = jax.lax.scan(body, (state, k_act), None, length=episode_length)
    return {k: state.info[k] for k in INFO_KEYS}

  return run(k_reset, k_act)


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
  """95% Wilson score interval for a binomial proportion."""
  if n == 0:
    return (0.0, 0.0)
  p = successes / n
  denom = 1 + z**2 / n
  centre = (p + z**2 / (2 * n)) / denom
  half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
  return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))


def summarize(stats: Dict[str, np.ndarray]) -> Dict[str, Any]:
  succ = stats['succeeded'] > 0.5
  n = int(succ.size)
  k = int(succ.sum())
  lo, hi = wilson_ci(k, n)
  steps = stats['first_success_step'][succ]
  return {
      'episodes': n,
      'successes': k,
      'success_rate': k / n if n else 0.0,
      'ci_lo': lo,
      'ci_hi': hi,
      'steps_mean': float(steps.mean()) if k else float('nan'),
      'steps_median': float(np.median(steps)) if k else float('nan'),
      'peak_force_mean': float(stats['peak_force'].mean()),
      'final_xy_err_mean_mm': float(stats['xy_err'].mean() * 1000),
      'final_depth_mean_mm': float(stats['depth'].mean() * 1000),
  }
