"""Independent actor/critic pairs for the three Goal 7 coordination variants."""
from dataclasses import replace
from takehome import runtime  # noqa: F401
from flax import struct
import jax
import jax.numpy as jp
from takehome.config import action_scales
from takehome.variants import VariantLearner


@struct.dataclass
class CoordinatedState:
    aim: object
    movement: object

    @property
    def actor(self):
        return (self.aim.actor, self.movement.actor)

    @property
    def critics(self):
        return (self.aim.critics, self.movement.critics)

    @property
    def updates(self):
        return self.aim.updates + self.movement.updates


class CoordinatedLearner:
    def __init__(self, config, env):
        self.config = config
        self.base_size = env.base_obs_size
        # Environment-only movement switches remain on the parent configuration.
        common = dict(coordination='joint', final_decider=False, contact_cap=False,
                      uncertainty_cap=False, force_penalty=False)
        self.aim = VariantLearner(replace(config, outputs=2, **common), env)
        self.movement = VariantLearner(replace(config, outputs=3, **common), env)
        self.normalizer = self.aim.normalizer
        self.action_scale = jp.array(action_scales(5, config.step_cap, config.movement_cap))

    def encode(self, obs):
        return self.aim.encode(obs)

    def initialize(self, seed):
        # Separate state/parameters/optimizers; matching sampler streams make
        # each pair study the same selected transitions, without sharing weights.
        return CoordinatedState(self.aim.initialize(seed), self.movement.initialize(seed))

    def act(self, params, obs):
        return jp.concatenate([self.aim.act(params[0], obs),
                               self.movement.act(params[1], obs)], axis=-1)

    def choose(self, actor, critics, obs):
        from takehome.variants import decider_candidates
        action = self.act(actor, obs)
        if not self.config.final_decider:
            return action
        options = decider_candidates(action)
        tiled_obs = jp.broadcast_to(obs[..., None, :], options.shape[:-1] + (obs.shape[-1],))
        q = jp.mean(self.q_values(critics, tiled_obs, options * self.action_scale), axis=0)
        return jp.take_along_axis(options, jp.argmax(q, axis=-1)[..., None, None], axis=-2)[..., 0, :]

    def q_values(self, params, obs, physical_action, bias=None):
        return jp.concatenate([
            self.aim.q_values(params[0], obs, physical_action[..., :2], bias),
            self.movement.q_values(params[1], obs, physical_action[..., 2:], bias)], axis=0)

    def train_block(self, state, data, fill_count, total_count, num_updates,
                    actor_enabled=True, practice_end=200):
        cfg = self.config
        midpoint = cfg.fill_episodes + (cfg.total_episodes - cfg.fill_episodes) // 2
        first_half = practice_end <= midpoint
        train_aim = cfg.coordination == 'side_by_side' or (
            cfg.coordination == 'aim_then_move' and first_half) or (
            cfg.coordination == 'move_then_aim' and not first_half)
        train_move = cfg.coordination == 'side_by_side' or not train_aim
        if not actor_enabled:  # critic briefing, when combined later
            train_aim = train_move = True
        aim, move = state.aim, state.movement
        metrics = []
        if train_aim:
            aim, m = self.aim.train_block(aim, {**data, 'action': data['action'][..., :2]},
                fill_count, total_count, num_updates, actor_enabled)
            metrics.append(m)
        if train_move:
            move, m = self.movement.train_block(move, {**data, 'action': data['action'][..., 2:]},
                fill_count, total_count, num_updates, actor_enabled)
            metrics.append(m)
        combined = {k: jp.mean(jp.stack([m[k] for m in metrics])) for k in ('critic_loss', 'mean_q')}
        combined['finite'] = jp.all(jp.stack([m['finite'] for m in metrics]))
        return CoordinatedState(aim, move), combined
