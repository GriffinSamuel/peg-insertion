"""Goal 7 adapter around the same persistent-budget experiment loop."""
from takehome import runtime  # noqa: F401
import jax
from takehome.correction import AimConfig, AimCorrectedEnv, policy_digest
from takehome.coordination import CoordinatedLearner
from takehome.episodes import EpisodeRunner
from takehome.learner import Learner
from takehome.train import ExperimentRunner, write_json
from takehome.variants import VariantEnv, VariantLearner, prepare_training_data


class EpisodeCollection:
    def __init__(self, parent):
        self.parent = parent
        self.main = EpisodeRunner(parent.env, parent.learner)
        self.siblings = {}
        self.last_policy = parent.config.policy

    def practice(self, state, seed, batch_index, episode_ids, source):
        self.last_policy = self.parent.config.policy
        runner = self.main
        if source == 'fill' and self.parent.config.sibling_fill:
            self.last_policy = ('force_0.5mm_s0', 'force_0.5mm_s3', 'force_0.5mm_s4')[batch_index % 3]
            if self.last_policy not in self.siblings:
                env = self.parent.make_env(self.last_policy)
                self.siblings[self.last_policy] = (EpisodeRunner(env, self.parent.learner),
                                                   policy_digest(env.frozen_policy))
            runner = self.siblings[self.last_policy][0]
        return runner.practice(state, seed, batch_index, episode_ids, source)

    def test(self, *args, **kwargs):
        return self.main.test(*args, **kwargs)


class VariantExperimentRunner(ExperimentRunner):
    def __init__(self, config):
        if not any(d.platform == 'gpu' for d in jax.devices()):
            raise RuntimeError('Tripwire: GPU not visible')
        self.config = config
        self.env = self.make_env(config.policy)
        if config.coordination != 'joint':
            self.learner = CoordinatedLearner(config, self.env)
        elif any((config.briefing, config.privileged_critic, config.final_decider,
                  config.classification, config.n_step > 1, config.return_floor, config.force_penalty)):
            self.learner = VariantLearner(config, self.env)
        else:
            self.learner = Learner(config, self.env)
        self.episodes = EpisodeCollection(self)

    def make_env(self, policy_name):
        cfg = self.config
        path = runtime.ROOT / 'results/policies' / policy_name
        aim = AimConfig(cfg.step_cap, cfg.guess_cap)
        if cfg.near_hole or cfg.contact_cap or cfg.uncertainty_cap:
            return VariantEnv(path, aim, config=cfg)
        return AimCorrectedEnv(path, aim, outputs=cfg.outputs, movement_cap=cfg.movement_cap)

    def update(self, state, replay, new_steps, source, practice_end):
        data = prepare_training_data(replay, self.config)
        briefing = source == 'fill'
        updates = (replay.fill_count if briefing else new_steps) * self.config.updates_per_step
        args = (state, data, replay.fill_count, replay.count, updates)
        if isinstance(self.learner, CoordinatedLearner):
            return self.learner.train_block(*args, actor_enabled=not briefing, practice_end=practice_end)
        if isinstance(self.learner, VariantLearner):
            return self.learner.train_block(*args, actor_enabled=not briefing)
        if briefing:
            raise ValueError('Briefing requires the actor-freezing learner')
        return self.learner.train_block(*args)

    def run(self, seed, output):
        result = super().run(seed, output)
        if self.config.sibling_fill:
            checks = {name: policy_digest(runner.env.frozen_policy) == before
                      for name, (runner, before) in self.episodes.siblings.items()}
            if not all(checks.values()):
                raise RuntimeError('Frozen sibling policy changed')
            if checks:
                result['frozen_siblings_unchanged'] = checks
                write_json(output / 'result.json', result)
        return result
