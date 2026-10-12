"""Compatibility shim: Brax 0.14.2 (latest) vs JAX 0.11.

JAX 0.11 removed `jax.device_put_replicated`, which Brax's PPO trainer still
calls once to copy the initial training state onto each device. Blackwell
GPUs need a recent JAX, so instead of pinning an old JAX we restore the
function using the drop-in replacement from JAX's pmap migration guide:
stack N copies along a new leading axis and shard that axis over the devices.
"""

import jax
import jax.numpy as jnp
import numpy as np


def _device_put_replicated(x, devices):
  devices = list(devices)
  n = len(devices)
  stacked = jax.tree.map(lambda a: jnp.stack([jnp.asarray(a)] * n), x)
  mesh = jax.sharding.Mesh(np.array(devices), ('devices',))
  sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('devices'))
  return jax.device_put(stacked, sharding)


def install() -> None:
  if 'device_put_replicated' not in vars(jax):
    try:
      jax.device_put_replicated  # noqa: B018
    except AttributeError:
      jax.device_put_replicated = _device_put_replicated
