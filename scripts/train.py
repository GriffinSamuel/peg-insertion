# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# MODIFIED for the peg-insertion project (2026): derived from MuJoCo
# Playground's learning/train_jax_ppo.py. Changes: osmesa instead of EGL;
# PegInsertion env/PPO config/domain randomizer imported directly; full_reset
# auto-resets; clearance/force flags; per-run log dirs; params + network
# config saved for reloading; final success-rate evaluation; vision, wandb and
# rscope paths removed.

"""Train a PPO agent on PegInsertion with Brax + MJX."""

import functools
import json
import os
import shutil
import sys
import time
import warnings
from pathlib import Path

xla_flags = os.environ.get('XLA_FLAGS', '')
if '--xla_gpu_triton_gemm_any' not in xla_flags:
  xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ.setdefault('MUJOCO_GL', 'osmesa')

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from absl import app  # noqa: E402
from absl import flags  # noqa: E402
from absl import logging  # noqa: E402
from brax.training.agents.ppo import networks as ppo_networks  # noqa: E402
from brax.training.agents.ppo import train as ppo  # noqa: E402
import jax  # noqa: E402

from mujoco_playground import wrapper  # noqa: E402

import peg_insertion  # noqa: E402,F401  (registers the envs)
from peg_insertion import rollout  # noqa: E402
from peg_insertion.configs import ppo_config  # noqa: E402
from peg_insertion.env import domain_randomize  # noqa: E402

try:
  import tensorboardX
except ImportError:
  tensorboardX = None

logging.set_verbosity(logging.WARNING)
warnings.filterwarnings('ignore', category=RuntimeWarning, module='jax')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='jax')
warnings.filterwarnings('ignore', category=UserWarning, module='absl')

_RUN_NAME = flags.DEFINE_string('run_name', None, 'Run name (logs/<run_name>).')
_CLEARANCE_MM = flags.DEFINE_float('clearance_mm', 2.0, 'Hole clearance (mm).')
_USE_FORCE_OBS = flags.DEFINE_boolean('use_force_obs', False, 'Add wrist F/T obs.')
_IMPL = flags.DEFINE_enum('impl', 'warp', ['jax', 'warp'], 'MJX implementation (warp ~4x faster here)')
_ENV_OVERRIDES = flags.DEFINE_string(
    'env_overrides', None, 'JSON dict of extra env config overrides.'
)
_DOMAIN_RANDOMIZATION = flags.DEFINE_boolean(
    'domain_randomization', True, 'Use domain randomization (on by default).'
)
_USE_TB = flags.DEFINE_boolean('use_tb', True, 'TensorBoard logging.')
_SEED = flags.DEFINE_integer('seed', 0, 'Random seed')
_NUM_TIMESTEPS = flags.DEFINE_integer('num_timesteps', None, 'Timesteps')
_NUM_EVALS = flags.DEFINE_integer('num_evals', None, 'Number of evaluations')
_NUM_ENVS = flags.DEFINE_integer('num_envs', None, 'Number of environments')
_NUM_EVAL_ENVS = flags.DEFINE_integer('num_eval_envs', None, 'Eval environments')
_BATCH_SIZE = flags.DEFINE_integer('batch_size', None, 'Batch size')
_REWARD_SCALING = flags.DEFINE_float('reward_scaling', None, 'Reward scaling')
_ENTROPY_COST = flags.DEFINE_float('entropy_cost', None, 'Entropy cost')
_LEARNING_RATE = flags.DEFINE_float('learning_rate', None, 'Learning rate')
_DISCOUNTING = flags.DEFINE_float('discounting', None, 'Discounting')
_UNROLL_LENGTH = flags.DEFINE_integer('unroll_length', None, 'Unroll length')
_NUM_MINIBATCHES = flags.DEFINE_integer('num_minibatches', None, 'Minibatches')
_NUM_UPDATES_PER_BATCH = flags.DEFINE_integer(
    'num_updates_per_batch', None, 'Updates per batch'
)
_LR_SCHEDULE = flags.DEFINE_enum(
    'learning_rate_schedule', None, ['NONE', 'ADAPTIVE_KL'],
    'Brax PPO learning-rate schedule (ADAPTIVE_KL adapts LR to a target KL).'
)
_DESIRED_KL = flags.DEFINE_float('desired_kl', None, 'Target KL for ADAPTIVE_KL.')
_FINAL_EVAL_EPISODES = flags.DEFINE_integer(
    'final_eval_episodes', 256, 'Episodes for the final success rate.'
)
_RENDER = flags.DEFINE_boolean('render', True, 'Render one rollout video at the end.')

_PPO_OVERRIDES = {
    'num_timesteps': _NUM_TIMESTEPS,
    'num_evals': _NUM_EVALS,
    'num_envs': _NUM_ENVS,
    'num_eval_envs': _NUM_EVAL_ENVS,
    'batch_size': _BATCH_SIZE,
    'reward_scaling': _REWARD_SCALING,
    'entropy_cost': _ENTROPY_COST,
    'learning_rate': _LEARNING_RATE,
    'discounting': _DISCOUNTING,
    'unroll_length': _UNROLL_LENGTH,
    'num_minibatches': _NUM_MINIBATCHES,
    'num_updates_per_batch': _NUM_UPDATES_PER_BATCH,
    'learning_rate_schedule': _LR_SCHEDULE,
    'desired_kl': _DESIRED_KL,
}


def build_ppo_params():
  params = ppo_config()
  for key, flag in _PPO_OVERRIDES.items():
    if flag.value is not None:
      params[key] = flag.value
  return params


def build_env_overrides(num_envs: int) -> dict:
  overrides = {
      'impl': _IMPL.value,
      'clearance_mm': _CLEARANCE_MM.value,
      'use_force_obs': _USE_FORCE_OBS.value,
      'naconmax': 20 * num_envs,
  }
  if _ENV_OVERRIDES.value:
    overrides.update(json.loads(_ENV_OVERRIDES.value))
  return overrides


def main(argv):
  del argv
  if not _RUN_NAME.value:
    raise app.UsageError('--run_name is required')
  run = _RUN_NAME.value
  logdir = ROOT / 'logs' / run
  ckptdir = ROOT / 'checkpoints' / run
  logdir.mkdir(parents=True, exist_ok=True)
  ckptdir.mkdir(parents=True, exist_ok=True)

  ppo_params = build_ppo_params()
  env_overrides = build_env_overrides(ppo_params.num_envs)
  env_overrides_eval = dict(env_overrides, naconmax=20 * ppo_params.num_eval_envs)

  env = rollout.make_env(**_env_kwargs(env_overrides))
  eval_env = rollout.make_env(**_env_kwargs(env_overrides_eval))
  print(f'Environment config:\n{env._config}')
  print(f'PPO parameters:\n{ppo_params}', flush=True)
  (logdir / 'config.json').write_text(json.dumps({
      'env_config': env._config.to_dict(),
      'env_overrides': env_overrides,
      'ppo_params': ppo_params.to_dict(),
      'seed': _SEED.value,
      'domain_randomization': _DOMAIN_RANDOMIZATION.value,
      'argv': sys.argv,
  }, indent=2, default=str))

  training_params = dict(ppo_params)
  network_kwargs = dict(training_params.pop('network_factory'))
  num_eval_envs = training_params.pop('num_eval_envs', 128)
  network_factory = functools.partial(
      ppo_networks.make_ppo_networks, **network_kwargs
  )
  if _DOMAIN_RANDOMIZATION.value:
    training_params['randomization_fn'] = domain_randomize

  writer = None
  if _USE_TB.value and tensorboardX is not None:
    writer = tensorboardX.SummaryWriter(str(logdir))

  times = [time.monotonic()]
  curve_path = logdir / 'progress.csv'
  curve_path.write_text('step,minutes,eval_reward,eval_success,eval_insert_reward\n')

  def progress(num_steps, metrics):
    times.append(time.monotonic())
    with open(logdir / 'metrics.jsonl', 'a') as f:
      f.write(json.dumps({'step': int(num_steps), **{
          k: float(v) for k, v in metrics.items()}}) + '\n')
    with open(curve_path, 'a') as f:
      f.write(
          f"{num_steps},{(times[-1] - times[0]) / 60:.2f},"
          f"{metrics.get('eval/episode_reward', float('nan')):.4f},"
          f"{metrics.get('eval/episode_success', float('nan')):.4f},"
          f"{metrics.get('eval/episode_insert_reward', float('nan')):.4f}\n"
      )
    if writer is not None:
      for key, value in metrics.items():
        writer.add_scalar(key, value, num_steps)
      writer.flush()
    elapsed = times[-1] - times[0]
    print(
        f'[{elapsed / 60:6.1f} min] step {num_steps:>10d}  '
        f"reward={metrics.get('eval/episode_reward', float('nan')):8.3f}  "
        f"success={metrics.get('eval/episode_success', float('nan')):.3f}  "
        f"insert={metrics.get('eval/episode_insert_reward', float('nan')):7.2f}",
        flush=True,
    )

  train_fn = functools.partial(
      ppo.train,
      **training_params,
      network_factory=network_factory,
      seed=_SEED.value,
      save_checkpoint_path=str(ckptdir),
      wrap_env_fn=functools.partial(
          wrapper.wrap_for_brax_training, full_reset=True
      ),
      num_eval_envs=num_eval_envs,
  )
  make_inference_fn, params, _ = train_fn(
      environment=env, progress_fn=progress, eval_env=eval_env
  )
  wall = time.monotonic() - times[0]
  jit_time = times[1] - times[0] if len(times) > 1 else float('nan')
  train_time = times[-1] - times[1] if len(times) > 2 else float('nan')
  # progress() runs once after the initial (compile-heavy) eval at step 0 and
  # once per training epoch, so train_time covers ~num_timesteps env steps.
  sps = ppo_params.num_timesteps / train_time if train_time > 0 else float('nan')
  print(f'Done training in {wall / 60:.1f} min (first eval/JIT {jit_time:.0f} s)', flush=True)

  network_config = {
      'obs_size': int(env.observation_size['state'][0]),
      'act_size': int(env.action_size),
      'policy_hidden_layer_sizes': list(network_kwargs['policy_hidden_layer_sizes']),
      'value_hidden_layer_sizes': list(network_kwargs['value_hidden_layer_sizes']),
      'normalize_observations': bool(ppo_params.normalize_observations),
      'clearance_mm': _CLEARANCE_MM.value,
      'use_force_obs': _USE_FORCE_OBS.value,
      'env_overrides': env_overrides,
  }
  rollout.save_policy(logdir, params, network_config)

  # Final success rate from our own rollout (episode stats from state.info).
  policy = jax.jit(make_inference_fn(params, deterministic=True))
  final_env = rollout.make_env(**_env_kwargs(env_overrides_eval))
  stats = rollout.rollout_stats(
      final_env, policy, _FINAL_EVAL_EPISODES.value, seed=10_000 + _SEED.value,
      randomize=_DOMAIN_RANDOMIZATION.value, batch=num_eval_envs,
  )
  summary = rollout.summarize(stats)
  final = {
      'run': run,
      'clearance_mm': _CLEARANCE_MM.value,
      'use_force_obs': _USE_FORCE_OBS.value,
      'seed': _SEED.value,
      **summary,
      'wall_time_s': round(wall, 1),
      'jit_time_s': round(jit_time, 1),
      'sps': round(sps, 1),
      'num_envs': int(ppo_params.num_envs),
      'num_timesteps': int(ppo_params.num_timesteps),
  }
  (logdir / 'final_metrics.json').write_text(json.dumps(final, indent=2))
  print(f'FINAL {json.dumps(final)}', flush=True)

  results = ROOT / 'results'
  (results / 'policies' / run).mkdir(parents=True, exist_ok=True)
  shutil.copy(logdir / 'final_metrics.json', results / f'final_metrics_{run}.json')
  (results / 'curves').mkdir(parents=True, exist_ok=True)
  shutil.copy(curve_path, results / 'curves' / f'{run}.csv')
  for name in ('params.pkl', 'network_config.json'):
    shutil.copy(logdir / name, results / 'policies' / run / name)

  if _RENDER.value:
    try:
      sys.path.insert(0, str(ROOT / 'scripts'))
      import render  # pylint: disable=g-import-not-at-top

      render.render_policy(final_env, policy, logdir / 'rollout.mp4',
                           episodes=1, seed=_SEED.value)
    except Exception as exc:  # a renderer failure must never lose a finished run
      print(f'Rendering failed (run results are saved): {exc!r}', flush=True)
  print('ALL DONE', flush=True)


def _env_kwargs(overrides: dict) -> dict:
  kw = dict(overrides)
  return {
      'clearance_mm': kw.pop('clearance_mm'),
      'use_force_obs': kw.pop('use_force_obs'),
      **kw,
  }


if __name__ == '__main__':
  app.run(main)
