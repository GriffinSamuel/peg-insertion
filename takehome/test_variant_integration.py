"""Integration probes are test-only; no replay insertion or practice allocation."""
from pathlib import Path
from takehome import runtime  # noqa: F401
import numpy as np
import pytest
from takehome.variants import VariantConfig
from takehome.variant_runner import VariantExperimentRunner
from takehome.episodes import EpisodeRunner
from takehome.correction import policy_digest


@pytest.mark.parametrize('cfg', [VariantConfig(privileged_critic=True),
    VariantConfig(outputs=5, final_decider=True, near_hole=True, contact_cap=True, uncertainty_cap=True)])
def test_feature_rollouts_are_finite_and_never_learned(cfg):
    runner = VariantExperimentRunner(cfg)
    before = policy_digest(runner.env.frozen_policy)
    state = runner.learner.initialize(999)
    summary, stats, trace = runner.episodes.test(state, episodes=4, seed=123)
    assert summary['test_only'] and not summary['learned_from']
    assert summary['episodes'] == 4 and summary['through_wall_episodes'] == 0
    assert all(np.isfinite(v).all() for v in trace.values())
    np.testing.assert_array_equal(trace['action'], np.zeros_like(trace['action']))
    np.testing.assert_array_equal(trace['guess'], np.zeros_like(trace['guess']))
    assert trace['obs'].shape == (200, 4, 18)
    assert policy_digest(runner.env.frozen_policy) == before


def test_sibling_selection_is_fresh_own_fill_not_old_results(monkeypatch):
    runner = VariantExperimentRunner(VariantConfig(sibling_fill=True))
    calls = []
    def fake_practice(self, state, seed, batch_index, episode_ids, source):
        calls.append((self, seed, batch_index, episode_ids, source))
        return {}, {}
    monkeypatch.setattr(EpisodeRunner, 'practice', fake_practice)
    policies = []
    for batch in range(4):
        runner.episodes.practice(None, 0, batch, list(range(batch*10, batch*10+10)), 'fill')
        policies.append(runner.episodes.last_policy)
    assert policies == ['force_0.5mm_s0','force_0.5mm_s3','force_0.5mm_s4','force_0.5mm_s0']
    assert len(calls) == 4 and all(call[0].learner is runner.learner for call in calls)
    runner.episodes.practice(None, 0, 4, list(range(40, 50)), 'online')
    assert runner.episodes.last_policy == 'force_0.5mm_s2'
    assert calls[-1][0] is runner.episodes.main
    assert set(runner.episodes.siblings) == set(policies)
