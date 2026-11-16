"""Analysis poses supplement the ordinary test trace without altering actions."""
from takehome import runtime  # noqa: F401
import numpy as np
from takehome.variants import VariantConfig
from takehome.variant_runner import VariantExperimentRunner
from takehome.pose_episodes import PoseEpisodeRunner
from takehome.episodes import TRACE_KEYS
from takehome.correction import policy_digest


def test_pose_recording_retains_finite_zero_action_and_test_provenance():
    experiment = VariantExperimentRunner(VariantConfig(outputs=3))
    before = policy_digest(experiment.env.frozen_policy)
    recorder = PoseEpisodeRunner(experiment.env, experiment.learner)
    state = experiment.learner.initialize(999)
    summary, stats, trace = recorder.test(state, episodes=4, seed=123)
    assert set(TRACE_KEYS).issubset(trace)
    assert trace['qpos'].shape == (200, 4, experiment.env.mj_model.nq)
    assert trace['mocap_pos'].shape[:2] == (200, 4)
    assert trace['true_rel'].shape == (200, 4, 3)
    np.testing.assert_array_equal(trace['action'], np.zeros((200, 4, 3)))
    np.testing.assert_array_equal(trace['pose_succeeded'][-1], stats['succeeded'])
    assert summary['episodes'] == 4 and summary['test_seed'] == 123
    assert summary['test_only'] and not summary['learned_from']
    assert all(np.isfinite(value).all() for value in trace.values())
    assert policy_digest(experiment.env.frozen_policy) == before
