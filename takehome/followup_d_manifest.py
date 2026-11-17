"""Fixed-camera aim singles at the original Goal 7 numeric values."""
from takehome.goal7_manifest import screening_settings
from takehome.followup_c_manifest import settings as movement_settings


def settings():
    descriptions={e['name'][2:]:e['description'] for e in movement_settings()}
    descriptions.update(aim_cap_50um='Limit each aim nudge to 0.05 mm.',
                        aim_cap_200um='Allow each aim nudge up to 0.2 mm.')
    skip={
        'side_by_side':'Needs a movement component; this base is aim-only.',
        'aim_then_move':'Needs a movement component to train second.',
        'move_then_aim':'Needs a movement component to train first.',
        'movement_cap_400um':'A movement cap cannot alter an aim-only learner.',
        'contact_cap':'Scales the absent movement correction, not the aim nudge.',
        'uncertainty_cap':'Scales the absent movement correction, not the aim nudge.',
    }
    result=[]
    for original in screening_settings():
        name=original['name'];reason=skip.get(name,'')
        result.append(dict(name='D_'+name,phase='D',description=descriptions[name],
            overrides={**original['overrides'],'outputs':2,'camera_condition':'fixed'},
            base='B_aim_fixed',seeds=[0,1,2],status='skipped' if reason else 'planned',
            condition='fixed',hand_search=False,
            hidden_information=bool(original['overrides'].get('privileged_critic')),
            conditional=False,changes=2,reason=reason,counted=False))
    assert len(result)==24 and sum(r['status']=='planned' for r in result)==18
    return result
