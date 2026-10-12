"""PPO hyperparameters for PegInsertion.

Built from Playground's tuned PandaPickCubeCartesian config (same arm, same
delta-position IK control scheme), so we start from settings known to work for
this robot and only change what the task needs.
"""

from ml_collections import config_dict


def ppo_config() -> config_dict.ConfigDict:
  from mujoco_playground.config import manipulation_params

  cfg = manipulation_params.brax_ppo_config('PandaPickCubeCartesian', 'jax')
  cfg.num_timesteps = 10_000_000
  cfg.episode_length = 200
  cfg.num_evals = 10
  cfg.num_eval_envs = 128
  cfg.network_factory.policy_hidden_layer_sizes = (256, 256)
  cfg.network_factory.value_hidden_layer_sizes = (256, 256, 256)
  # Kept from the base: num_envs=1024, batch_size=256, unroll_length=10,
  # num_minibatches=8, num_updates_per_batch=8, discounting=0.97,
  # learning_rate=5e-4, entropy_cost=7.5e-3, reward_scaling=0.1,
  # normalize_observations=True, action_repeat=1, max_grad_norm=1.0.
  return cfg
