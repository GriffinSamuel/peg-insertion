"""Goal 7 switches, kept separate from the completed plain comparison recipe."""
from dataclasses import dataclass
import inspect
from takehome import runtime  # noqa: F401
import jax
import jax.numpy as jp
import numpy as np
import optax
from takehome.config import TrainConfig
from takehome.correction import AimCorrectedEnv
from takehome.learner import Learner, LearnerState, draw_indices


@dataclass(frozen=True)
class VariantConfig(TrainConfig):
    coordination: str = 'joint'
    briefing: bool = False
    privileged_critic: bool = False
    final_decider: bool = False
    near_hole: bool = False
    contact_cap: bool = False
    uncertainty_cap: bool = False
    classification: bool = False
    n_step: int = 1
    return_floor: bool = False
    force_penalty: bool = False
    sibling_fill: bool = False
    test_seed: int = 123
    baseline_path: str = 'results/takehome/baseline.json'

    def __post_init__(self):
        super().__post_init__()
        if self.coordination not in ('joint', 'side_by_side', 'aim_then_move', 'move_then_aim'):
            raise ValueError('Unknown coordination')
        if (self.coordination != 'joint' or self.final_decider or self.contact_cap or
                self.uncertainty_cap or self.force_penalty) and self.outputs != 5:
            raise ValueError('This movement experiment requires combined outputs')
        if self.briefing and not self.fill_episodes:
            raise ValueError('Cannot brief without own fill-up data')
        if self.privileged_critic and self.final_decider:
            raise ValueError('Final decider cannot use privileged information')
        if self.coordination != 'joint' and self.final_decider:
            raise ValueError('Final decider requires one joint critic, not independent component critics')
        if self.coordination in ('aim_then_move', 'move_then_aim') and (200-self.fill_episodes) % 20:
            raise ValueError('Sequential halves must each contain whole ten-episode batches')
        if self.force_penalty and self.classification:
            raise ValueError('Force penalty and classification are incompatible')
        if self.n_step not in (1, 3):
            raise ValueError('Credit length must be the predeclared one or three')
        if self.supervised_reference:
            raise ValueError('Cheating supervised reference is not a Goal 7 candidate')


def near_hole_gate(obs):
    return (jp.abs(obs[..., 2]) < .015) & (jp.linalg.norm(obs[..., :2], axis=-1) < .01)


def decider_candidates(action):
    aim = action.at[..., 2:].set(0)
    move = action.at[..., :2].set(0)
    return jp.stack([jp.zeros_like(action), aim, move, action, action*.5,
                     jp.clip(action*1.5, -1, 1), aim*.5+move, aim+move*.5], axis=-2)


class VariantEnv(AimCorrectedEnv):
    def __init__(self, policy_dir, aim, *, config, **kwargs):
        self.variant_config = config
        super().__init__(policy_dir, aim, outputs=config.outputs,
                         movement_cap=config.movement_cap, **kwargs)
        closure = inspect.getclosurevars(self.frozen_policy).nonlocals
        self._policy_parameters = closure['params']
        self._policy_network = closure['policy_network']
        self._distribution = closure['parametric_action_distribution']

    def next_guess(self, state, action):
        guess = super().next_guess(state, action)
        if self.variant_config.near_hole:
            guess = jp.where(near_hole_gate(state.obs['state']), guess, jp.zeros_like(guess))
        return guess

    def movement_limit(self, state, corrected):
        cap = jp.full((3,), self.movement_cap)
        cfg = self.variant_config
        if cfg.near_hole:
            cap = cap * near_hole_gate(state.obs['state'])
        if cfg.contact_cap:
            force = jp.linalg.norm(state.data.sensordata[self._force_slice])
            cap = cap * jp.where(force >= 5., 2., .5)
        if cfg.uncertainty_cap:
            logits = self._policy_network.apply(*self._policy_parameters[:2], corrected)
            std = self._distribution.create_dist(logits).scale
            cap = cap * jp.clip(2 * std, .5, 2.)
        return cap


def prepare_training_data(replay, cfg):
    """Derive targets only within each of this run's counted practice episodes."""
    data = replay.training_arrays()
    if cfg.privileged_critic:
        data = {**data, 'true_bias': replay.arrays['true_bias'],
                'next_true_bias': replay.arrays.get('next_true_bias', replay.arrays['true_bias'])}
    if not (cfg.n_step > 1 or cfg.return_floor or cfg.force_penalty):
        return data
    n = replay.count
    # Replay is step-major inside each ten-episode batch. Explicit episode/step
    # sorting prevents a target from crossing episodes or padding rows.
    arrays = replay.arrays
    rewards = arrays['reward'].copy()
    if cfg.force_penalty:
        rewards[:n] -= .001 * np.maximum(arrays['force'][:n] - 15., 0.) * .02
    data = {**data, 'reward': rewards}
    discounts = np.full(replay.capacity, cfg.discount, dtype=np.float32)
    returns = np.zeros(replay.capacity, dtype=np.float32)
    if cfg.n_step > 1:
        data['next_obs'] = data['next_obs'].copy()
        data['terminated'] = data['terminated'].copy()
        if cfg.privileged_critic:
            data['next_true_bias'] = data['next_true_bias'].copy()
    for episode in sorted(replay.episode_ids):
        ids = np.flatnonzero(arrays['episode_id'][:n] == episode)
        ids = ids[np.argsort(arrays['step'][ids])]
        running = 0.
        for i in ids[::-1]:
            running = float(rewards[i]) + cfg.discount * running
            returns[i] = running
        if cfg.n_step > 1:
            original_rewards = rewards[ids].copy()
            for pos, i in enumerate(ids):
                last = min(pos + cfg.n_step, len(ids))
                segment = ids[pos:last]
                powers = cfg.discount ** np.arange(len(segment))
                data['reward'][i] = float(np.dot(original_rewards[pos:last], powers))
                endpoint = segment[-1]
                data['next_obs'][i] = arrays['next_obs'][endpoint]
                data['terminated'][i] = arrays['terminated'][endpoint]
                if cfg.privileged_critic:
                    data['next_true_bias'][i] = arrays.get('next_true_bias', arrays['true_bias'])[endpoint]
                discounts[i] = cfg.discount ** len(segment)
    if cfg.n_step > 1:
        data['discount'] = discounts
    if cfg.return_floor:
        data['return_floor'] = returns
    return data


class VariantLearner(Learner):
    def __init__(self, config, env):
        super().__init__(config, env)
        self.train_block = jax.jit(self._variant_train_block)

    def q_logits(self, params, obs, physical_action, bias=None):
        x = jp.concatenate([self.encode(obs), physical_action / self.action_scale], axis=-1)
        if self.config.privileged_critic:
            if bias is None:
                raise ValueError('Privileged critic requires explicitly supplied analysis/training bias')
            x = jp.concatenate([x, bias / self.config.guess_cap], axis=-1)
        return jax.vmap(lambda p: self.critic_net.apply(p, x))(params)

    def q_values(self, params, obs, physical_action, bias=None):
        logits = self.q_logits(params, obs, physical_action, bias)
        return jax.nn.sigmoid(logits) if self.config.classification else logits

    def choose(self, actor, critics, obs):
        action = self.act(actor, obs)
        if not self.config.final_decider:
            return action
        options = decider_candidates(action)
        tiled_obs = jp.broadcast_to(obs[..., None, :], options.shape[:-1] + (obs.shape[-1],))
        q = jp.mean(self.q_values(critics, tiled_obs, options * self.action_scale), axis=0)
        index = jp.argmax(q, axis=-1)
        return jp.take_along_axis(options, index[..., None, None], axis=-2)[..., 0, :]

    def initialize(self, seed):
        if not self.config.privileged_critic:
            return super().initialize(seed)
        key, ka, kq = jax.random.split(jax.random.PRNGKey(seed), 3)
        obs = jp.zeros((1, self.base_size + 2))
        actor = self.actor_net.init(ka, self.encode(obs))
        x = jp.concatenate([self.encode(obs), jp.zeros((1, self.config.outputs + 2))], -1)
        critics = jax.vmap(lambda k: self.critic_net.init(k, x))(
            jax.random.split(kq, self.config.ensemble))
        return LearnerState(actor, critics, actor, critics,
                            self.optimizer.init(actor), self.optimizer.init(critics), key,
                            jp.array(0, dtype=jp.int32))

    def _variant_train_block(self, state, data, fill_count, total_count, num_updates,
                             actor_enabled=True):
        cfg = self.config

        def step(_, carry):
            st, loss_sum, q_sum, finite = carry
            key, ks, kt = jax.random.split(st.key, 3)
            ids = draw_indices(ks, cfg.batch_size, fill_count, total_count)
            batch = {k: v[ids] for k, v in data.items()}
            next_action = self.choose(st.target_actor, st.target_critics, batch['next_obs']) * self.action_scale
            next_q = self.q_values(st.target_critics, batch['next_obs'], next_action,
                                   batch.get('next_true_bias'))
            subset = jax.random.permutation(kt, cfg.ensemble)[:min(2, cfg.ensemble)]
            target = batch['reward'] + batch.get('discount', cfg.discount) * (
                1. - batch['terminated']) * jp.min(next_q[subset], axis=0)
            if cfg.return_floor:
                target = jp.maximum(target, batch['return_floor'])
            target = jax.lax.stop_gradient(target)

            def critic_loss(params):
                logits = self.q_logits(params, batch['obs'], batch['action'], batch.get('true_bias'))
                if cfg.classification:
                    loss = jp.mean(optax.sigmoid_binary_cross_entropy(logits, jp.clip(target, 0, 1)[None, :]))
                    q = jax.nn.sigmoid(logits)
                else:
                    q = logits
                    loss = jp.mean((q - target[None, :]) ** 2)
                return loss, jp.mean(q)

            (loss, mean_q), grads = jax.value_and_grad(critic_loss, has_aux=True)(st.critics)
            delta, critic_opt = self.optimizer.update(grads, st.critic_opt, st.critics)
            critics = optax.apply_updates(st.critics, delta)

            def actor_step(operand):
                actor, actor_opt, target_actor = operand
                def actor_loss(p):
                    action = self.act(p, batch['obs']) * self.action_scale
                    return -jp.mean(self.q_values(critics, batch['obs'], action, batch.get('true_bias')))
                grad = jax.grad(actor_loss)(actor)
                delta, actor_opt = self.optimizer.update(grad, actor_opt, actor)
                actor = optax.apply_updates(actor, delta)
                return actor, actor_opt, optax.incremental_update(actor, target_actor, cfg.target_rate)

            actor, actor_opt, target_actor = jax.lax.cond(
                actor_enabled & ((st.updates + 1) % cfg.updates_per_step == 0), actor_step,
                lambda x: x, (st.actor, st.actor_opt, st.target_actor))
            new = st.replace(actor=actor, critics=critics, target_actor=target_actor,
                target_critics=optax.incremental_update(critics, st.target_critics, cfg.target_rate),
                actor_opt=actor_opt, critic_opt=critic_opt, key=key, updates=st.updates + 1)
            return new, loss_sum + loss, q_sum + mean_q, finite & jp.isfinite(loss) & jp.isfinite(mean_q)

        st, loss, q, finite = jax.lax.fori_loop(
            0, num_updates, step, (state, jp.array(0.), jp.array(0.), jp.array(True)))
        return st, {'critic_loss': loss / num_updates, 'mean_q': q / num_updates, 'finite': finite}
