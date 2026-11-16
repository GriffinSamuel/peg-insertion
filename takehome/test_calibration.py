"""Synthetic calibration contracts, without reading experimental outcomes."""
import numpy as np
import pytest
from takehome.calibration import (feature_matrix, fit_probability, predict_probability,
                                  reliability_bins, evaluate_trace)


def test_remaining_time_and_control_units_are_explicit():
    cfg = dict(outputs=5,step_cap=.0001,movement_cap=.0002)
    obs = np.zeros((2,18)); action=np.array([[.0001,.0001,.0002,.0002,.0002]]*2)
    x=feature_matrix(obs,action,np.array([0,199]),cfg)
    np.testing.assert_array_equal(x[:,-6:-1],np.ones((2,5)))
    np.testing.assert_allclose(x[:,-1],[1,.005])


def test_logistic_predictor_and_reliability_have_correct_direction():
    rng=np.random.default_rng(12)
    x=rng.normal(size=(500,3)); y=rng.random(500)<1/(1+np.exp(-2*x[:,0]))
    model=fit_probability(x,y)
    pred=predict_probability(model,np.array([[-2,0,0],[2,0,0]]))
    assert pred[0]<.2 and pred[1]>.8
    assert model['extra_practice_episodes']==0 and not model['true_bias_used']
    bins=reliability_bins(np.array([.9,.9,.8,.8]),np.array([1,0,1,0]))
    assert sum(r['count'] for r in bins['bins'])==4
    assert bins['average_assessment']=='overconfident'
    assert abs(bins['signed_bias']-.35)<1e-12


def test_confidence_evaluation_rejects_training_traces():
    with pytest.raises(ValueError,match='test-only'):
        evaluate_trace({},dict(test_only=False,learned_from=True))
