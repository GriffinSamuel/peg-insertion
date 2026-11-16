"""The selected movement recipe must also accept the ten-input checkpoint."""
from takehome import runtime  # noqa: F401
import numpy as np
from takehome.variants import VariantConfig
from takehome.variant_runner import VariantExperimentRunner
from takehome.correction import policy_digest


def test_position_only_policy_rollout_is_finite_frozen_and_has_twelve_inputs():
    runner = VariantExperimentRunner(VariantConfig(policy='pos_0.5mm_s1', outputs=3))
    before = policy_digest(runner.env.frozen_policy)
    state = runner.learner.initialize(999)
    summary, _, trace = runner.episodes.test(state, episodes=4, seed=123)
    assert runner.env.base_obs_size == 10
    assert trace['obs'].shape == (200, 4, 12)
    assert trace['action'].shape == (200, 4, 3)
    np.testing.assert_array_equal(trace['action'], np.zeros_like(trace['action']))
    assert summary['test_only'] and not summary['learned_from'] and summary['episodes'] == 4
    assert all(np.isfinite(value).all() for value in trace.values())
    assert policy_digest(runner.env.frozen_policy) == before
