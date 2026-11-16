"""Predeclared movement-only settings; returns metadata without running jobs."""
from takehome.goal7_manifest import screening_settings


def settings():
    result=[]
    skip={'side_by_side':'Needs separate aim and movement learners; movement-only has no aim part.',
          'aim_then_move':'Needs an aim component to train first.',
          'move_then_aim':'Needs an aim component to train second.',
          'aim_cap_50um':'An aim nudge cap cannot change a movement-only learner.',
          'aim_cap_200um':'An aim nudge cap cannot change a movement-only learner.'}
    descriptions={
        'briefing':'Let the scorekeeper study its own fill-up data before the corrector learns.',
        'privileged_critic':'Give the scorekeeper the hidden camera error during practice only.',
        'one_critic':'Use one scorekeeper instead of four.',
        'one_update':'Study each new step once instead of four times.',
        'movement_cap_400um':'Allow movement corrections up to 0.4 mm per axis.',
        'movement_cap_100um':'Limit movement corrections to 0.1 mm per axis.',
        'final_decider':'Let the learned scorekeeper choose among zero and scaled corrections.',
        'near_hole':'Allow correction only within 15 mm height and 10 mm sideways camera distance.',
        'contact_cap':'Halve the movement cap below 5 N contact and double it above 5 N.',
        'uncertainty_cap':'Scale the movement cap by the original policy action spread.',
        'classification':'Use a bounded 0-to-1 scorekeeper with a different error formula.',
        'three_step':'Combine three consecutive rewards before updating a rating.',
        'return_floor':'Keep a learning target at least as large as the return actually observed.',
        'force_penalty':'Penalize force above 15 N during practice using the fixed original cost.',
        'sibling_fill':'Collect fill-up data with the three specified sibling policies.',
        'side_by_side':'Train separate aim and movement parts in parallel.',
        'aim_then_move':'Train the aim part first and the movement part second.',
        'move_then_aim':'Train the movement part first and the aim part second.',
        'aim_cap_50um':'Limit an aim nudge to 0.05 mm per axis.',
        'aim_cap_200um':'Allow an aim nudge up to 0.2 mm per axis.',
    }
    original=screening_settings()
    original.append(dict(name='movement_cap_100um',label='Movement cap0.1mm',overrides={'movement_cap':.0001}))
    for row in original:
        name=row['name']; overrides={**row['overrides'],'outputs':3}
        reason=skip.get(name,'')
        description=(f"Use {overrides['fill_episodes']} of 200 practice episodes for fill-up."
                     if name.startswith('fill_') else descriptions[name])
        result.append(dict(name='C_'+name,phase='C',description=description,overrides=overrides,
            base='movement',seeds=[0,1,2],status='skipped' if reason else 'planned',condition='normal',
            hand_search=False,hidden_information=bool(overrides.get('privileged_critic')),
            conditional=False,changes=2,reason=reason,counted=False))
    assert len(result)==25 and sum(r['status']=='planned' for r in result)==20
    return result
