"""Goal 7 short contracts on synthetic arrays; no practice episodes collected."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from takehome import runtime  # noqa: F401
import jax
import jax.numpy as jp
import numpy as np
import pytest
from takehome.correction import AimConfig, AimCorrectedEnv
from takehome.learner import Learner
from takehome.variants import (VariantConfig, VariantEnv, VariantLearner,
                               decider_candidates, near_hole_gate, prepare_training_data)
from takehome.coordination import CoordinatedLearner


@pytest.fixture(scope='module')
def env():
    return AimCorrectedEnv(Path('results/policies/force_0.5mm_s2'))


def synthetic_data(outputs=2, privileged=False):
    obs = jp.zeros((128, 18))
    data = dict(obs=obs, next_obs=obs, action=jp.zeros((128, outputs)),
                reward=jp.ones(128)*.2, terminated=jp.zeros(128))
    if privileged:
        data.update(true_bias=jp.ones((128, 2))*.0005, next_true_bias=jp.ones((128, 2))*.0005)
    return data


@pytest.mark.parametrize('changes', [dict(fill_episodes=0, briefing=True),
    dict(outputs=5, privileged_critic=True, final_decider=True),
    dict(outputs=5, force_penalty=True, classification=True),
    dict(outputs=5, coordination='aim_then_move', fill_episodes=10),
    dict(outputs=5, coordination='side_by_side', final_decider=True)])
def test_incompatible_combinations_rejected(changes):
    with pytest.raises(ValueError):
        VariantConfig(**changes)


def test_main_equivalence_and_briefing_freezes_actor(env):
    cfg = VariantConfig()
    regular, variant = Learner(cfg, env), VariantLearner(cfg, env)
    initial = regular.initialize(999)
    one, _ = regular.train_block(initial, synthetic_data(), 64, 128, 8)
    two, _ = variant.train_block(initial, synthetic_data(), 64, 128, 8)
    for a, b in zip(jax.tree_util.tree_leaves(one), jax.tree_util.tree_leaves(two)):
        np.testing.assert_allclose(a, b, atol=1e-6, rtol=1e-5)
    briefed, metrics = variant.train_block(initial, synthetic_data(), 128, 128, 8, False)
    assert bool(metrics['finite'])
    for a, b in zip(jax.tree_util.tree_leaves(initial.actor), jax.tree_util.tree_leaves(briefed.actor)):
        np.testing.assert_array_equal(a, b)
    assert any(not np.array_equal(a, b) for a, b in zip(
        jax.tree_util.tree_leaves(initial.critics), jax.tree_util.tree_leaves(briefed.critics)))


def test_privileged_critic_never_changes_actor_input(env):
    learner = VariantLearner(VariantConfig(privileged_critic=True), env)
    state = learner.initialize(999)
    data = synthetic_data(privileged=True)
    assert learner.encode(data['obs']).shape == (128, 18)
    with pytest.raises(ValueError):
        learner.q_values(state.critics, data['obs'], data['action'])
    updated, metrics = learner.train_block(state, data, 64, 128, 4)
    assert bool(metrics['finite'])
    assert learner.act(updated.actor, data['obs']).shape == (128, 2)


def test_decider_candidates_are_bounded_and_zero_initially(env):
    learner = VariantLearner(VariantConfig(outputs=5, final_decider=True), env)
    state = learner.initialize(999)
    obs = jp.zeros((4, 18))
    np.testing.assert_array_equal(learner.choose(state.actor, state.critics, obs), np.zeros((4, 5)))
    values = decider_candidates(jp.ones((4, 5))*.8)
    assert values.shape == (4, 8, 5)
    assert np.max(np.abs(values)) <= 1
    np.testing.assert_array_equal(values[:, 0], np.zeros((4, 5)))
    np.testing.assert_array_equal(values[:, 1, 2:], np.zeros((4, 3)))
    np.testing.assert_array_equal(values[:, 2, :2], np.zeros((4, 2)))


def test_classification_loss_is_bounded_and_finite(env):
    learner = VariantLearner(VariantConfig(classification=True), env)
    state, metrics = learner.train_block(learner.initialize(999), synthetic_data(), 64, 128, 8)
    q = learner.q_values(state.critics, jp.zeros((4, 18)), jp.zeros((4, 2)))
    assert bool(metrics['finite']) and np.all((q > 0) & (q < 1))


def test_nstep_return_floor_and_force_penalty_do_not_cross_episodes():
    arrays = dict(obs=np.zeros((6, 18)), next_obs=np.arange(6)[:, None]*np.ones((6, 18)),
        action=np.zeros((6, 2)), reward=np.array([0., 0., 0., 0., 1., 0.]),
        terminated=np.array([False, False, False, False, True, False]),
        episode_id=np.array([0, 1, 0, 1, 0, 1]), step=np.array([0, 0, 1, 1, 2, 2]),
        force=np.array([15., 20., 15., 20., 15., 20.]))
    replay = SimpleNamespace(count=6, capacity=6, episode_ids={0, 1}, arrays=arrays,
        training_arrays=lambda: {k: arrays[k] for k in ('obs','next_obs','action','reward','terminated')})
    data = prepare_training_data(replay, VariantConfig(n_step=3, return_floor=True))
    np.testing.assert_allclose(data['reward'][[0, 2, 4]], [.99**2, .99, 1])
    np.testing.assert_allclose(data['reward'][[1, 3, 5]], 0)
    np.testing.assert_allclose(data['return_floor'], [.99**2, 0, .99, 0, 1, 0])
    np.testing.assert_array_equal(data['terminated'], [True, False, True, False, True, False])
    np.testing.assert_allclose(data['discount'][:2], .99**3)
    np.testing.assert_allclose(data['next_obs'][0], 4)
    np.testing.assert_allclose(data['next_obs'][1], 5)
    penalized = prepare_training_data(replay, VariantConfig(outputs=5, force_penalty=True))
    np.testing.assert_allclose(penalized['reward'][[1, 3, 5]], -.0001)
    np.testing.assert_array_equal(arrays['reward'], [0, 0, 0, 0, 1, 0])


@pytest.mark.parametrize('mode', ['side_by_side', 'aim_then_move', 'move_then_aim'])
def test_coordination_has_separate_states_and_freezes_inactive_part(env, mode):
    learner = CoordinatedLearner(VariantConfig(outputs=5, coordination=mode), env)
    initial = learner.initialize(999)
    np.testing.assert_array_equal(learner.act(initial.actor, jp.zeros((3, 18))), np.zeros((3, 5)))
    state, metrics = learner.train_block(initial, synthetic_data(5), 64, 128, 4, practice_end=120)
    assert bool(metrics['finite'])
    if mode == 'aim_then_move':
        frozen_old, frozen_new = initial.movement, state.movement
        assert int(state.aim.updates) == 4
    elif mode == 'move_then_aim':
        frozen_old, frozen_new = initial.aim, state.aim
        assert int(state.movement.updates) == 4
    else:
        assert int(state.aim.updates) == int(state.movement.updates) == 4
        return
    for a, b in zip(jax.tree_util.tree_leaves(frozen_old), jax.tree_util.tree_leaves(frozen_new)):
        np.testing.assert_array_equal(a, b)
    end, _ = learner.train_block(state, synthetic_data(5), 64, 128, 4, practice_end=130)
    assert int(end.aim.updates) == int(end.movement.updates) == 4


def test_handover_and_dynamic_caps_use_available_sensors():
    np.testing.assert_array_equal(near_hole_gate(jp.array([[0,0,.01], [0,0,.02], [.02,0,0]])),
                                  [True, False, False])
    cfg = VariantConfig(outputs=5, contact_cap=True)
    env = VariantEnv(Path('results/policies/force_0.5mm_s2'), AimConfig(), config=cfg)
    sensors = jp.zeros(env.mj_model.nsensordata)
    state = SimpleNamespace(data=SimpleNamespace(sensordata=sensors), obs={'state': jp.zeros(18)})
    corrected = {'state': jp.zeros(16)}
    np.testing.assert_allclose(env.movement_limit(state, corrected), .0001)
    force = sensors.at[env._force_slice].set(10.)
    state.data.sensordata = force
    np.testing.assert_allclose(env.movement_limit(state, corrected), .0004)
    uncertainty = VariantEnv(Path('results/policies/force_0.5mm_s2'), AimConfig(),
                             config=replace(cfg, contact_cap=False, uncertainty_cap=True))
    cap = uncertainty.movement_limit(state, corrected)
    assert cap.shape == (3,) and np.all(cap >= .00009999) and np.all(cap <= .00040001)
