"""Peg-insertion task for MuJoCo Playground (MJX + Brax PPO).

Importing this package registers two environments with Playground:
  PegInsertion       position-only observations
  PegInsertionForce  the same task with wrist force/torque observations
"""

from peg_insertion import jax_compat

jax_compat.install()

from mujoco_playground._src.manipulation import register_environment  # noqa: E402

from peg_insertion.env import PegInsertion, default_config, domain_randomize, force_config

register_environment('PegInsertion', PegInsertion, default_config)
register_environment('PegInsertionForce', PegInsertion, force_config)

__all__ = ['PegInsertion', 'default_config', 'force_config', 'domain_randomize']
