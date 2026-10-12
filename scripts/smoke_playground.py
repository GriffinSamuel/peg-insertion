"""Phase 1.5 smoke test: Playground's own Franka env on the GPU."""
import os; os.environ.setdefault("MUJOCO_GL", "osmesa")
import time, jax, jax.numpy as jp
from mujoco_playground import registry
# Force the JAX backend: the env's default_config has impl='warp'.
env = registry.load("PandaPickCubeCartesian", config_overrides={"impl": "jax"})
reset, step = jax.jit(env.reset), jax.jit(env.step)
t = time.time(); s = reset(jax.random.PRNGKey(0)); s = step(s, jp.zeros(env.action_size))
jax.block_until_ready(s.obs); print("first step incl. compile (s):", round(time.time() - t, 1))
t = time.time()
for _ in range(100): s = step(s, jp.zeros(env.action_size))
jax.block_until_ready(s.obs); print("100 steps (s):", round(time.time() - t, 3))
print("SMOKE OK")
