"""Required joint and sequential comparisons, separate from optional trick stacks."""
from itertools import combinations

from takehome.followup_config import FollowupConfig


def required_settings():
    rows=[]
    for condition in ('normal','fixed'):
        for mode in ('joint','aim_then_move','move_then_aim'):
            name=f'E_{condition}_{mode}'
            base=('movement' if condition=='normal' else 'B_movement_fixed') if mode=='joint' else f'E_{condition}_joint'
            rows.append(dict(name=name,phase='E',
                description={'joint':'Learn aim and movement together with one scorekeeper.',
                    'aim_then_move':'Learn aim for80 episodes, then movement for80 while aim stays frozen.',
                    'move_then_aim':'Learn movement for80 episodes, then aim for80 while movement stays frozen.'}[mode],
                overrides=dict(outputs=5,camera_condition=condition,coordination=mode),
                base=base,seeds=[0,1,2],status='planned',condition=condition,
                hand_search=False,hidden_information=False,conditional=False,
                changes=1 if mode=='joint' else 2,reason='',counted=False))
    return rows


def trick_stacks(registry, summaries):
    """Apply the predeclared combination rule without launching or selecting fresh tests."""
    entries={entry['name']:entry for entry in registry['settings']}
    rows=[]; selection={}
    for phase, outputs, condition in (('C',3,'normal'),('D',2,'fixed')):
        eligible=[]
        for entry in entries.values():
            summary=summaries.get(entry['name'],{})
            record=summary.get('record',{})
            if (entry['phase']==phase and entry['status']=='finished'
                    and record.get('verdict')=='HELPED'):
                expected_base='movement' if phase=='C' else 'B_aim_fixed'
                if (entry['condition']!=condition or entry['base']!=expected_base
                        or entry['overrides'].get('outputs',2)!=outputs
                        or entry['overrides'].get('camera_condition','normal')!=condition
                        or entry['hand_search']):
                    raise ValueError('A stack single must retain its original base and camera condition')
                if sorted(r['seed'] for r in record['seeds']) != list(range(5)):
                    raise ValueError('HELPED requires all five choosing seeds')
                eligible.append(entry)
        eligible.sort(key=lambda e:(-summaries[e['name']]['record']['mean_percent'],
                                    e['changes'],e['name']))
        chosen=eligible[:3]
        selection[phase]=dict(eligible=[e['name'] for e in eligible],
                             selected=[e['name'] for e in chosen])
        for size in (2,3):
            for group in combinations(chosen,size):
                overrides=dict(outputs=outputs,camera_condition=condition)
                reason='';seen={}
                for entry in group:
                    for key,value in entry['overrides'].items():
                        if key in ('outputs','camera_condition'):continue
                        if key in seen and seen[key]!=value:
                            reason=f'Incompatible values for {key}: {seen[key]} and {value}.'
                        seen[key]=value
                        overrides[key]=value
                if not reason:
                    try:FollowupConfig(**overrides)
                    except ValueError as exc:reason=str(exc)
                names=[e['name'] for e in group]
                rows.append(dict(name='E_'+phase+'_stack_'+'__'.join(n[2:] for n in names),
                    phase='E',description='Combine the individually helpful changes: '+', '.join(names)+'.',
                    overrides=overrides,base=names[0],seeds=[0,1,2],
                    status='skipped' if reason else 'planned',condition=condition,
                    hand_search=False,hidden_information=any(e['hidden_information'] for e in group),
                    conditional=True,changes=1+size,reason=reason,counted=False,
                    components=names,original_base=group[0]['base']))
    return rows,selection
