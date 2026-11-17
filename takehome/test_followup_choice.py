"""Pure final-selection contracts; no test episode keys or simulation."""
from takehome.followup_choice import choose


def _old():
    return {name:[dict(seed=i,successes=int(value*5),success_percent=value)
                  for i,value in enumerate(values)]
            for name,values in [('aim',[60,62,64,66,68]),('movement',[80,82,84,86,88])]}


def _record(name, mean, spread=2, phase='C', condition='normal', hidden=False, seeds=5):
    entry=dict(name=name,phase=phase,condition=condition,status='finished',changes=2,
               hand_search=False,hidden_information=hidden,overrides=dict(outputs=3,ensemble=1))
    summary=dict(record=dict(mean_percent=mean,spread_pp=spread,
                 seeds=[dict(seed=i,successes=int(mean*5)) for i in range(seeds)]))
    return entry,summary


def test_simpler_baseline_wins_within_top_spread():
    e,s=_record('C_one_critic',86,3)
    result=choose(dict(settings=[e]),{e['name']:s},_old())
    assert result['name']=='original_movement'
    assert result['eligibility_threshold_percent']==83


def test_clear_gain_wins_and_ineligible_results_do_not_compete():
    pairs=[_record('C_one_critic',90),_record('C_privileged',99,hidden=True),
           _record('F_teacher',99,phase='F'),_record('D_example',99,condition='fixed'),
           _record('C_unconfirmed',99,seeds=3)]
    result=choose(dict(settings=[e for e,_ in pairs]),{e['name']:s for e,s in pairs},_old())
    assert result['name']=='C_one_critic'
    assert len(result['excluded'])==4
