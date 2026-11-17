"""Synthetic final-report contracts; no fresh random key or episode is used."""
from takehome import followup_runtime  # noqa: F401
import pytest
from takehome.followup_final_worker import configurations
from takehome.followup_final_report import _paired_comparison


def choice(**changes):
    data=dict(camera_condition='normal',hand_search=False,hidden_information=False,
              overrides={'outputs':3})
    return {**data,**changes}


def test_plain_chosen_method_reuses_one_configuration():
    assert [n for n,_ in configurations(choice())]==['plain_movement']
    assert [n for n,_ in configurations(choice(overrides={'outputs':3,'movement_cap':.0001}))]==[
        'chosen','plain_movement']


@pytest.mark.parametrize('change',[{'camera_condition':'fixed'},{'hand_search':True},
                                   {'hidden_information':True}])
def test_ineligible_confirmation_is_rejected(change):
    with pytest.raises(ValueError):configurations(choice(**change))


def test_paired_differences_and_identical_method_reuse_are_not_fake_replicates():
    plain=[dict(seed=i,successes=100+i,success_percent=(100+i)/5) for i in range(5)]
    chosen=[dict(seed=i,successes=105+i,success_percent=(105+i)/5) for i in range(5)]
    compared=_paired_comparison(dict(chosen=chosen,plain_movement=plain),False)
    assert compared['mean_chosen_minus_plain_percentage_points']==1.
    assert compared['sample_sd_chosen_minus_plain_percentage_points']==0.
    reused=_paired_comparison(dict(plain_movement=plain),True)
    assert reused['mean_chosen_minus_plain_percentage_points']==0.
    assert reused['independent_method_comparison'] is False
    assert all(r['same_seed_data_reused'] for r in reused['per_seed_differences'])
