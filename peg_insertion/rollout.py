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
# Physical-validity check computed outside the env: the largest per-axis
# lateral offset of the tip while it is > 2 mm below the hole entrance. Inside
# the hole this can never exceed the clearance (plus contact compliance), so a
# larger value would mean the peg passed through a wall (e.g. dropped contacts).
DEEP_Z = 0.002
WALL_TOL = 0.0005
EXTRA_KEYS = ('max_lateral_when_deep',)


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
  out: Dict[str, list] = {k: [] for k in INFO_KEYS + EXTRA_KEYS}
  done_episodes, chunk = 0, 0
  while done_episodes < n_episodes:
    n = min(batch, n_episodes - done_episodes)
    key = jax.random.PRNGKey(seed * 100_003 + chunk)
    k_rand, k_reset, k_act = jax.random.split(key, 3)
    wenv = _wrap(env, n, k_rand, randomize)
    info = _run(wenv, policy, n, episode_length, k_reset, k_act,
                env._tip_site, env._entrance_site)
    for k in INFO_KEYS + EXTRA_KEYS:
      out[k].append(np.array(info[k]))
    done_episodes += n
    chunk += 1
  stats = {k: np.concatenate(v) for k, v in out.items()}
  stats['clearance'] = np.full(stats['succeeded'].shape, env.clearance)
  return stats


def _run(wenv, policy, n, episode_length, k_reset, k_act, tip_site, ent_site):
  @jax.jit
  def run(k_reset, k_act):
    state = wenv.reset(jax.random.split(k_reset, n))

    def body(carry, _):
      state, key, max_lat = carry
      key, sub = jax.random.split(key)
      action, _ = policy(state.obs, sub)
      state = wenv.step(state, action)
      rel = state.data.site_xpos[:, tip_site] - state.data.site_xpos[:, ent_site]
      lateral = jp.max(jp.abs(rel[:, :2]), axis=-1)
      max_lat = jp.maximum(max_lat, jp.where(rel[:, 2] < -DEEP_Z, lateral, 0.0))
      return (state, key, max_lat), None

    (state, _, max_lat), _ = jax.lax.scan(
        body, (state, k_act, jp.zeros(n)), None, length=episode_length)
    return {**{k: state.info[k] for k in INFO_KEYS}, 'max_lateral_when_deep': max_lat}

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
  through_wall = stats['max_lateral_when_deep'] > stats['clearance'] + WALL_TOL
  nan_eps = int(np.isnan(stats['peak_force']).sum())
  return {
      'episodes': n,
      'successes': k,
      'success_rate': k / n if n else 0.0,
      'ci_lo': lo,
      'ci_hi': hi,
      'steps_mean': float(steps.mean()) if k else float('nan'),
      'steps_median': float(np.median(steps)) if k else float('nan'),
      'peak_force_mean': float(np.nanmean(stats['peak_force'])),
      'final_xy_err_mean_mm': float(np.nanmean(stats['xy_err']) * 1000),
      'final_depth_mean_mm': float(np.nanmean(stats['depth']) * 1000),
      'still_inserted_at_end': float(np.mean(stats['depth'] >= 0.024)),
      'through_wall_episodes': int(through_wall.sum()),
      'max_lateral_when_deep_mm': float(np.max(stats['max_lateral_when_deep']) * 1000),
      'nan_episodes': nan_eps,
  }
