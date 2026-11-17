"""The three specified calibration/per-attempt splits, with four plain methods."""
SPLITS=(('mostly_fixed',.00065,.000375),('even',.00053,.00053),('mostly_episode',.000375,.00065))


def settings():
    result=[]
    for label,fixed,fresh in SPLITS:
        common=dict(camera_condition='mixed',mixed_fixed_std=fixed,mixed_episode_std=fresh)
        prefix='M_'+label+'_'
        for method,overrides,base in (
            ('aim',{'outputs':2},'floor'),
            ('heat',{'outputs':2,'constant_fill':True,'heatmap':True},prefix+'aim'),
            ('movement',{'outputs':3},'floor'),
            ('combined',{'outputs':5},prefix+'movement')):
            result.append(dict(name=prefix+method,phase='B_mixed',
                description={'aim':'Plain aim corrector.','heat':'Own-practice across-episode aim heat map.',
                    'movement':'Plain movement corrector.','combined':'Aim and movement learned together.'}[method],
                overrides={**common,**overrides},base=base,seeds=[0,1,2],status='planned',
                condition='mixed',condition_label=f'mixed: fixed {fixed*1000:g} mm, per-attempt {fresh*1000:g} mm',
                hand_search=False,hidden_information=False,conditional=False,changes=1,reason='',counted=False))
    return result
