"""Selection contracts use invented rows only, never any confirmation result."""
from takehome.exploration import select_candidate, merge_changes


def row(name, mean, std, changes, eligible=True):
    return dict(name=name, mean=mean, std=std, changes_from_main=changes, eligible=eligible)


def test_choose_simplest_within_top_spread_and_break_tie_by_mean():
    rows = [row('aim',.60,.02,0), row('movement',.85,.02,1),
            row('cap',.84,.01,1), row('combo',.88,.04,3), row('cheat',.99,.001,1,False)]
    selected, top, _, _ = select_candidate(rows, False)
    assert top['name'] == 'combo'
    assert selected['name'] == 'movement'


def test_skip_rule_confirms_main_and_main_wins_near_ties():
    rows = [row('aim',.65,.08,0), row('tuned',.70,.06,1)]
    assert select_candidate(rows, False)[0]['name'] == 'aim'
    assert select_candidate([row('aim',.60,.01,0),row('tuned',.90,.01,1)], True)[0]['name'] == 'aim'


def test_combining_rejects_conflicting_values_and_information_leaks():
    assert merge_changes([{'overrides':{'fill_episodes':0}}, {'overrides':{'briefing':True}}])[0] is None
    assert merge_changes([{'overrides':{'step_cap':.00005}}, {'overrides':{'step_cap':.0002}}])[0] is None
    assert merge_changes([{'overrides':{'privileged_critic':True}}, {'overrides':{'final_decider':True}}])[0] is None
    assert merge_changes([{'overrides':{'outputs':5,'contact_cap':True}}, {'overrides':{'n_step':3}}])[0] == dict(outputs=5,contact_cap=True,n_step=3)
