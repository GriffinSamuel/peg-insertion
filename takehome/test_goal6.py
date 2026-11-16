"""Goal 6 controls and label boundaries; synthetic data spend no episodes."""
from dataclasses import replace
from pathlib import Path
from takehome import runtime  # noqa: F401
import jax
import jax.numpy as jp
import numpy as np
import pytest
from takehome.config import TrainConfig, action_scales
from takehome.correction import AimCorrectedEnv, add_movement_correction, split_controls
from takehome.learner import Learner


def test_movement_cap_and_zero_control():
    old = jp.array([.95, -.2, .1])
    np.testing.assert_array_equal(add_movement_correction(old, jp.zeros(3), .0002, .002), old)
    new = add_movement_correction(old, jp.array([2., -3., .5]), .0002, .002)
    np.testing.assert_allclose(new, [1., -.3, .15], atol=1e-7)
    assert np.max(np.abs(np.asarray(new-old)) * .002) <= .00020001
    aim, move = split_controls(jp.arange(5.), 5)
    np.testing.assert_array_equal(aim, [0, 1])
    np.testing.assert_array_equal(move, [2, 3, 4])
    np.testing.assert_array_equal(split_controls(jp.ones(3), 3)[0], [0, 0])


@pytest.mark.parametrize('outputs', [3, 5])
def test_mixed_units_and_zero_init(outputs):
    env = AimCorrectedEnv(Path('results/policies/force_0.5mm_s2'), outputs=outputs)
    cfg = TrainConfig(outputs=outputs)
    learner = Learner(cfg, env)
    changed = Learner(replace(cfg, step_cap=.0002, movement_cap=.0004), env)
    state = learner.initialize(999)
    obs = jp.zeros((4, env.base_obs_size + 2))
    np.testing.assert_array_equal(learner.act(state.actor, obs), np.zeros((4, outputs)))
    control = jp.ones((4, outputs)) * .4
    np.testing.assert_allclose(learner.q_values(state.critics, obs, control * learner.action_scale),
                               changed.q_values(state.critics, obs, control * changed.action_scale))
    assert env.action_size == outputs
    assert tuple(np.asarray(learner.action_scale).round(7)) == pytest.approx(
        action_scales(outputs, cfg.step_cap, cfg.movement_cap))


def test_supervised_reference_uses_labels_only_when_explicit():
    env = AimCorrectedEnv(Path('results/policies/force_0.5mm_s2'))
    cfg = TrainConfig(supervised_reference=True)
    learner = Learner(cfg, env)
    state = learner.initialize(999)
    obs = jp.zeros((128, env.base_obs_size + 2))
    np.testing.assert_array_equal(learner.act(state.actor, obs), np.zeros((128, 2)))
    labels = jp.broadcast_to(jp.array([.0006, -.0006]), (128, 2))
    data = dict(obs=obs, next_obs=obs, action=jp.zeros((128, 2)), reward=jp.zeros(128),
                terminated=jp.ones(128), true_bias=labels)
    before = learner.actor_net.apply(state.actor, learner.encode(obs))
    state, metrics = learner.train_block(state, data, 64, 128, 40)
    after = learner.actor_net.apply(state.actor, learner.encode(obs))
    assert bool(metrics['finite'])
    assert float(jp.mean((after-labels/cfg.guess_cap)**2)) < float(jp.mean((before-labels/cfg.guess_cap)**2))
    with pytest.raises(ValueError):
        TrainConfig(outputs=5, supervised_reference=True)
