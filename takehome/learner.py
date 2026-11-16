"""Deterministic off-policy actor-critic with layer-normalized Q ensemble.

The Q output is a discounted return, not a calibrated success probability.
Ordinary training uses only observation, action, reward, next observation and
termination. True-bias labels are admitted only by the explicit cheating flag.
"""
import inspect

from takehome import runtime  # noqa: F401
from brax.training.acme import running_statistics
from flax import linen as nn, struct
import jax
import jax.numpy as jp
import optax
from takehome.config import action_scales


class Actor(nn.Module):
    outputs: int
    hidden: int

    @nn.compact
    def __call__(self, x):
        for _ in range(2):
            x = nn.relu(nn.Dense(self.hidden)(x))
        return jp.tanh(nn.Dense(self.outputs, kernel_init=nn.initializers.zeros,
                               bias_init=nn.initializers.zeros)(x))


class Critic(nn.Module):
    hidden: int

    @nn.compact
    def __call__(self, x):
        for _ in range(2):
            x = nn.relu(nn.LayerNorm()(nn.Dense(self.hidden)(x)))
        return nn.Dense(1, kernel_init=nn.initializers.normal(0.01))(x)[..., 0]


@struct.dataclass
class LearnerState:
    actor: object
    critics: object
    target_actor: object
    target_critics: object
    actor_opt: object
    critic_opt: object
    key: object
    updates: object


def draw_indices(key, batch_size, fill_count, total_count):
    """Actual compiled sampler: exactly half of each when both exist."""
    ka, kb, kc = jax.random.split(key, 3)
    half = batch_size // 2
    left = jax.random.randint(ka, (half,), 0, jp.maximum(fill_count, 1))
    right = fill_count + jax.random.randint(
        kb, (half,), 0, jp.maximum(total_count - fill_count, 1))
    mixed = jp.concatenate([left, right])
    single = jax.random.randint(kc, (batch_size,), 0, total_count)
    return jp.where((fill_count > 0) & (fill_count < total_count), mixed, single)


class Learner:
    def __init__(self, config, env):
        self.config = config
        self.base_size = env.base_obs_size
        params = inspect.getclosurevars(env.frozen_policy).nonlocals["params"]
        self.normalizer = params[0]
        self.actor_net = Actor(config.outputs, config.hidden)
        self.critic_net = Critic(config.hidden)
        self.optimizer = optax.chain(optax.clip_by_global_norm(10.0),
                                    optax.adam(config.learning_rate))
        self.action_scale = jp.array(action_scales(config.outputs, config.step_cap, config.movement_cap))
        self.train_block = jax.jit(self._train_block)

    def encode(self, obs):
        base = running_statistics.normalize({"state": obs[..., :self.base_size]},
                                            self.normalizer)["state"]
        return jp.concatenate([base, obs[..., self.base_size:] / self.config.guess_cap], axis=-1)

    def act(self, params, obs):
        output = self.actor_net.apply(params, self.encode(obs))
        if self.config.supervised_reference:
            # The cheating reference predicts bias directly; turn that estimate
            # into the same bounded incremental correction used by the others.
            return jp.clip((output * self.config.guess_cap - obs[..., -2:]) /
                           self.config.step_cap, -1, 1)
        return output

    def q_values(self, params, obs, physical_action):
        x = jp.concatenate([self.encode(obs), physical_action / self.action_scale], axis=-1)
        return jax.vmap(lambda p: self.critic_net.apply(p, x))(params)

    def initialize(self, seed):
        key, ka, kq = jax.random.split(jax.random.PRNGKey(seed), 3)
        obs = jp.zeros((1, self.base_size + 2))
        actor = self.actor_net.init(ka, self.encode(obs))
        x = jp.concatenate([self.encode(obs), jp.zeros((1, self.config.outputs))], -1)
        critics = jax.vmap(lambda k: self.critic_net.init(k, x))(
            jax.random.split(kq, self.config.ensemble))
        return LearnerState(actor, critics, actor, critics,
                            self.optimizer.init(actor), self.optimizer.init(critics),
                            key, jp.array(0, dtype=jp.int32))

    def _train_block(self, state, data, fill_count, total_count, num_updates):
        cfg = self.config

        def step(_, carry):
            st, loss_sum, q_sum, finite = carry
            key, ks, kt = jax.random.split(st.key, 3)
            ids = draw_indices(ks, cfg.batch_size, fill_count, total_count)
            batch = {k: v[ids] for k, v in data.items()}
            next_action = self.act(st.target_actor, batch["next_obs"]) * self.action_scale
            next_q = self.q_values(st.target_critics, batch["next_obs"], next_action)
            subset = jax.random.permutation(kt, cfg.ensemble)[:min(2, cfg.ensemble)]
            # True termination stops bootstrapping; truncation deliberately does
            # not, as specified. This is not a finite-episode probability.
            target = jax.lax.stop_gradient(batch["reward"] + cfg.discount *
                (1.0 - batch["terminated"]) * jp.min(next_q[subset], axis=0))

            def critic_loss(params):
                q = self.q_values(params, batch["obs"], batch["action"])
                return jp.mean((q - target[None, :]) ** 2), jp.mean(q)

            (loss, mean_q), grads = jax.value_and_grad(critic_loss, has_aux=True)(st.critics)
            delta, critic_opt = self.optimizer.update(grads, st.critic_opt, st.critics)
            critics = optax.apply_updates(st.critics, delta)

            def actor_step(operand):
                actor, actor_opt, target_actor = operand

                def actor_loss(p):
                    if cfg.supervised_reference:
                        prediction = self.actor_net.apply(p, self.encode(batch['obs']))
                        return jp.mean((prediction - batch['true_bias'] / cfg.guess_cap) ** 2)
                    action = self.act(p, batch["obs"]) * self.action_scale
                    return -jp.mean(self.q_values(critics, batch["obs"], action))

                grad = jax.grad(actor_loss)(actor)
                delta, actor_opt = self.optimizer.update(grad, actor_opt, actor)
                actor = optax.apply_updates(actor, delta)
                target_actor = optax.incremental_update(actor, target_actor, cfg.target_rate)
                return actor, actor_opt, target_actor

            actor, actor_opt, target_actor = jax.lax.cond(
                (st.updates + 1) % cfg.updates_per_step == 0,
                actor_step, lambda x: x, (st.actor, st.actor_opt, st.target_actor))
            new = st.replace(actor=actor, critics=critics, target_actor=target_actor,
                target_critics=optax.incremental_update(critics, st.target_critics, cfg.target_rate),
                actor_opt=actor_opt, critic_opt=critic_opt, key=key, updates=st.updates + 1)
            return new, loss_sum + loss, q_sum + mean_q, finite & jp.isfinite(loss) & jp.isfinite(mean_q)

        st, loss, q, finite = jax.lax.fori_loop(
            0, num_updates, step, (state, jp.array(0.0), jp.array(0.0), jp.array(True)))
        return st, {"critic_loss": loss / num_updates, "mean_q": q / num_updates,
                    "finite": finite}
