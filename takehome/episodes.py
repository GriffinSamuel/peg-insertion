"""Practice collector and test trace runner using the existing rollout machinery.

The test RNG partition, domain randomization, full 200-step scan, statistics
and physical-validity measurement follow peg_insertion.rollout. Actor/critic
parameters are dynamic JIT arguments so evaluations do not recompile at each
learning-curve point. No test path exposes a replay insertion method.
"""
from functools import partial

from takehome import runtime  # noqa: F401
import jax
import jax.numpy as jp
import numpy as np
from peg_insertion import rollout

TRACE_KEYS = ("obs", "guess", "next_guess", "true_bias", "q", "force", "success",
              "terminated", "truncated", "valid", "step", "old_action",
              "executed_action", "depth", "xy_err", "tip_height", "action", "movement")


class EpisodeRunner:
    trace_keys = TRACE_KEYS

    def extra_trace(self, state, nxt):
        """Analysis subclasses may record poses without changing control."""
        return {}

    def __init__(self, env, learner):
        self.env = env
        self.learner = learner
        self.compiled = jax.jit(self._run, static_argnames=("n",))

    def _run(self, actor, critics, seed_key, episode_ids, source, noise_std, *, n):
        env, learner = self.env, self.learner
        k_rand, k_reset, k_act = jax.random.split(seed_key, 3)
        wrapped = rollout._wrap(env, n, k_rand, randomize=True)
        initial = wrapped.reset(jax.random.split(k_reset, n))

        def body(carry, _):
            state, key, max_lat = carry
            key, ka = jax.random.split(key)
            obs = state.obs["state"]
            mean_action = (learner.choose(actor, critics, obs) if hasattr(learner, 'choose')
                           else learner.act(actor, obs))
            # During fill-up the old policy gets only small random nudges.
            mean_action = jp.where(source == 0, jp.zeros_like(mean_action), mean_action)
            action = jp.clip(mean_action + noise_std * jax.random.normal(ka, mean_action.shape), -1, 1)
            physical_action = action * learner.action_scale
            if getattr(learner.config, 'privileged_critic', False):
                q = jp.mean(learner.q_values(critics, obs, physical_action,
                                             state.info['bias'][:, :2]), axis=0)
            else:
                q = jp.mean(learner.q_values(critics, obs, physical_action), axis=0)
            nxt = wrapped.step(state, action)
            rel = nxt.data.site_xpos[:, env._tip_site] - nxt.data.site_xpos[:, env._entrance_site]
            lateral = jp.max(jp.abs(rel[:, :2]), axis=-1)
            max_lat = jp.maximum(max_lat, jp.where(rel[:, 2] < -rollout.DEEP_Z, lateral, 0))
            valid = ~(state.info["learner_terminated"] | state.info["learner_truncated"])
            trace = {
                "obs": obs, "action": physical_action, "reward": nxt.reward,
                "next_obs": nxt.obs["state"],
                "terminated": nxt.info["learner_terminated"],
                "truncated": nxt.info["learner_truncated"],
                "success": (nxt.info["succeeded"] > 0.5) & (state.info["succeeded"] < 0.5),
                "episode_id": episode_ids, "step": state.info["step_count"],
                "source": jp.full((n,), source, dtype=jp.int32),
                "old_action": nxt.info["old_action"],
                "executed_action": nxt.info["executed_action"],
                "movement": nxt.info['movement_correction'],
                "true_bias": state.info["bias"][:,:2],
                "next_true_bias": nxt.info['bias'][:, :2],
                "guess": state.info["aim_guess"], "next_guess": nxt.info["aim_guess"],
                "force": jp.linalg.norm(nxt.data.sensordata[:, env._force_slice], axis=-1),
                "q": q, "valid": valid, "depth": nxt.info["depth"],
                "xy_err": nxt.info["xy_err"], "tip_height": rel[:, 2],
            }
            trace.update(self.extra_trace(state, nxt))
            return (nxt, key, max_lat), trace

        (final, _, max_lat), trace = jax.lax.scan(
            body, (initial, k_act, jp.zeros(n)), None, length=env._config.episode_length)
        stats = {k: final.info[k] for k in rollout.INFO_KEYS}
        stats["max_lateral_when_deep"] = max_lat
        stats["clearance"] = jp.full((n,), env.clearance)
        return stats, trace

    def practice(self, state, seed, batch_index, episode_ids, source):
        key = jax.random.PRNGKey(seed * 100_003 + batch_index)
        std = self.learner.config.fill_noise if source == "fill" else self.learner.config.learner_noise
        stats, trace = self.compiled(state.actor, state.critics, key, jp.array(episode_ids),
                                     jp.array(0 if source == "fill" else 1), jp.array(std), n=10)
        stats, trace = jax.device_get((stats, trace))
        self.check_finite(stats, trace)
        return stats, trace

    def test(self, state, *, episodes=500, seed=123, output=None):
        """Separate test-only runner, exactly the original batch and key schedule."""
        chunks, traces = [], []
        done = chunk = 0
        while done < episodes:
            n = min(256, episodes - done)
            key = jax.random.PRNGKey(seed * 100_003 + chunk)
            stats, trace = self.compiled(
                state.actor, state.critics, key, jp.arange(done, done + n),
                jp.array(1), jp.array(0.0), n=n)
            stats, trace = jax.device_get((stats, trace))
            self.check_finite(stats, trace)
            chunks.append(stats)
            traces.append({k: trace[k] for k in self.trace_keys})
            done += n
            chunk += 1
        stats = {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
        trace = {k: np.concatenate([c[k] for c in traces], axis=1) for k in self.trace_keys}
        if output is not None:
            np.savez_compressed(output, **trace, test_only=np.array(True),
                                learned_from=np.array(False), test_seed=np.array(seed),
                                episode_success=stats["succeeded"])
        summary = rollout.summarize(stats)
        summary.update(test_only=True, learned_from=False, test_seed=seed)
        return summary, stats, trace

    @staticmethod
    def check_finite(stats, trace):
        if not all(np.isfinite(v).all() for v in (*stats.values(), *trace.values())):
            raise RuntimeError("Tripwire: invalid numbers in simulation, learner action or Q rating")
