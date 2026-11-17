"""Apply the fixed final-choice rule to completed development evidence only."""
import statistics


def choose(registry, summaries, old_references):
    if registry.get('status')=='halted_tripwire':
        raise ValueError('An active tripwire prevents selection')
    candidates=[];excluded=[]
    for name, outputs, changes in (('aim',2,0),('movement',3,1)):
        rows=old_references[name]
        if sorted(r['seed'] for r in rows)!=list(range(5)):
            raise ValueError('Preserved baseline requires five choosing seeds')
        values=[r['success_percent'] for r in rows]
        candidates.append(dict(name='original_'+name,mean_percent=statistics.mean(values),
            spread_pp=statistics.stdev(values),changes=changes,overrides=dict(outputs=outputs),
            camera_condition='normal',hand_search=False,hidden_information=False,
            seed_successes=[dict(seed=r['seed'],successes=r['successes']) for r in rows],
            source='Preserved original five-seed development baseline.'))
    for entry in registry['settings']:
        name=entry['name'];record=summaries.get(name,{}).get('record',{})
        reason=None
        if entry['phase']=='F':reason='All touch comparisons are excluded, including taught networks.'
        elif entry['condition']!='normal':reason='The fresh comparison uses the normal camera condition.'
        elif entry['hand_search']:reason='Hand-written search is ineligible.'
        elif entry['hidden_information']:reason='Hidden simulator information is ineligible.'
        elif entry['status']!='finished':reason='No completed setting.'
        elif sorted(r['seed'] for r in record.get('seeds',[]))!=list(range(5)):
            reason='Only five-seed development evidence is eligible under the prelogged reading.'
        if reason:
            excluded.append(dict(name=name,reason=reason));continue
        candidates.append(dict(name=name,mean_percent=record['mean_percent'],
            spread_pp=record['spread_pp'],changes=entry['changes'],overrides=entry['overrides'],
            camera_condition='normal',hand_search=False,hidden_information=False,
            seed_successes=[dict(seed=r['seed'],successes=r['successes']) for r in record['seeds']],
            source='Completed follow-up development setting.'))
    top=min(candidates,key=lambda c:(-c['mean_percent'],c['changes'],c['name']))
    threshold=top['mean_percent']-top['spread_pp']
    within=[c for c in candidates if c['mean_percent']>=threshold]
    chosen=min(within,key=lambda c:(c['changes'],-c['mean_percent'],c['name']))
    return dict(**chosen,rule='Highest development mean; within one spread of that top mean, '
                'fewest changes; then highest mean; stable name only for an exact tie.',
                development_test_seed=123,development_episodes_per_seed=500,
                fresh_practice_seeds=list(range(200,205)),fresh_test_seed=1001,
                top_average_setting=top['name'],top_average_percent=top['mean_percent'],
                top_spread_pp=top['spread_pp'],eligibility_threshold_percent=threshold,
                candidates=candidates,excluded=excluded,
                no_fresh_confirmation_run_before_choice=True)
