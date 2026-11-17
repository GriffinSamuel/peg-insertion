"""Combination eligibility and incompatibility contracts; no simulated episodes."""
from takehome.followup_e_manifest import trick_stacks
import pytest


def _entry(name, overrides, phase='C', verdict='HELPED', value=90):
    e=dict(name=phase+'_'+name,phase=phase,status='finished',changes=2,
           overrides=dict(outputs=3 if phase=='C' else 2,
                          camera_condition='normal' if phase=='C' else 'fixed',**overrides),
           condition='normal' if phase=='C' else 'fixed',hand_search=False,
           hidden_information=bool(overrides.get('privileged_critic')),
           base='movement' if phase=='C' else 'B_aim_fixed')
    s=dict(record=dict(verdict=verdict,mean_percent=value,
                      seeds=[dict(seed=i) for i in range(5)]))
    return e,s


def test_only_top_three_helped_singles_are_stacked():
    pairs=[_entry('fill_20',{'fill_episodes':20},value=94),
           _entry('one_critic',{'ensemble':1},value=93),
           _entry('one_update',{'updates_per_step':1},value=92),
           _entry('return_floor',{'return_floor':True},value=91),
           _entry('promising',{'movement_cap':.0001},verdict='PROMISING',value=99)]
    rows,selection=trick_stacks(dict(settings=[e for e,_ in pairs]),{e['name']:s for e,s in pairs})
    assert len(rows)==4
    assert selection['C']['selected']==['C_fill_20','C_one_critic','C_one_update']
    assert all(r['status']=='planned' and r['base']=='C_fill_20' for r in rows if 'C_fill_20' in r['components'])
    assert all(r['overrides']['outputs']==3 and r['condition']=='normal' for r in rows)


def test_incompatible_fill_and_briefing_are_listed_without_running():
    pairs=[_entry('fill_0',{'fill_episodes':0},value=94),
           _entry('fill_20',{'fill_episodes':20},value=93),
           _entry('briefing',{'briefing':True},value=92)]
    rows,_=trick_stacks(dict(settings=[e for e,_ in pairs]),{e['name']:s for e,s in pairs})
    assert len(rows)==4
    assert sum(r['status']=='planned' for r in rows)==1
    assert all(r['reason'] for r in rows if r['status']=='skipped')


def test_hidden_information_and_fixed_context_remain_labelled():
    pairs=[_entry('privileged',{'privileged_critic':True},phase='D',value=99),
           _entry('one_update',{'updates_per_step':1},phase='D',value=98)]
    rows,_=trick_stacks(dict(settings=[e for e,_ in pairs]),{e['name']:s for e,s in pairs})
    assert len(rows)==1 and rows[0]['hidden_information']
    assert rows[0]['overrides']['outputs']==2 and rows[0]['condition']=='fixed'


def test_misregistered_context_cannot_be_silently_transferred():
    entry,summary=_entry('one_update',{'updates_per_step':1})
    entry['condition']='fixed'
    with pytest.raises(ValueError,match='original base and camera'):
        trick_stacks(dict(settings=[entry]),{entry['name']:summary})
