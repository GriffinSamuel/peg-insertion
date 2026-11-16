"""Lead's predeclared verdict rules and a single-writer running scoreboard."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from takehome.followup_runtime import ROOT,OUT
import json,statistics
import numpy as np
from takehome.followup_metrics import summarize_seed
from takehome.followup_scoreboard import render,_atomic_write


def read(path):return json.loads(Path(path).read_text())
def save(path,data):_atomic_write(Path(path),json.dumps(data,indent=2,allow_nan=False)+'\n')
def mean(values):return statistics.mean(values) if values else None
def spread(values):return statistics.stdev(values) if len(values)>1 else 0.
def optional_mean(rows,key):return mean([r[key] for r in rows if r.get(key) is not None])


def old_references():
    path=OUT/'references/old_learning.json'
    if path.exists():return read(path)
    designations=read(ROOT/'results/takehome/benchmark_manifest.json')['headline_designations']
    refs={}
    for kind in ('aim','movement'):
        refs[kind]=[]
        for seed in range(5):
            directory=(ROOT/'results/takehome/benchmark'/designations[str(seed)] if kind=='aim'
                       else ROOT/'results/takehome/goal6/movement'/f'seed_{seed}')
            refs[kind].append(summarize_seed(directory))
    save(path,refs)
    return refs


def metrics_for(entry):
    rows=[]
    for seed in entry['seeds']:
        directory=OUT/entry['phase']/entry['name']/f'seed_{seed}'
        if not (directory/'result.json').exists():continue
        path=directory/'metrics.json'
        row=read(path) if path.exists() else summarize_seed(directory)
        if not path.exists():save(path,row)
        if not row['episode_success_vector_matches_final']:
            raise RuntimeError('Tripwire: final outcome vector and reported count disagree')
        rows.append(row)
    return rows


def base_for(entry,seeds,entries,refs):
    base=entry['base']
    if base in refs:
        return [r for r in refs[base] if r['seed'] in seeds]
    if base=='floor':
        rows=[]
        for seed in seeds:
            cond=entry['condition']
            fixed=f'_seed_{seed}' if cond=='fixed' else ''
            path=OUT/'references'/f'{cond}{fixed}_test_123.json'
            if cond=='normal':path=ROOT/'results/takehome/baseline.json'
            if not path.exists():continue
            summary=read(path)['conditions']['normal']
            rows.append(dict(seed=seed,successes=summary['successes'],test_episodes=500,
                             success_percent=summary['success_rate']*100))
        return rows
    return [r for r in metrics_for(entries[base]) if r['seed'] in seeds]


def verdict(rows,base,expected,failed):
    if failed or len(rows)!=expected or len(base)!=len(rows):return 'DID NOT RUN'
    values=[r['success_percent'] for r in rows]; bases=[r['success_percent'] for r in base]
    paired={r['seed']:r['success_percent'] for r in base}
    if any(r['success_percent']<.5*paired[r['seed']] for r in rows):return 'UNSTABLE'
    delta=mean(values)-mean(bases); threshold=max(spread(values),spread(bases))
    if delta>threshold:return 'HELPED' if len(rows)==5 else 'PROMISING'
    if delta < -threshold:return 'HURT'
    return 'NO REAL DIFFERENCE'


def identical_check(entry,rows):
    if len(rows)<3:return
    vectors=[]
    for row in rows:
        p=Path(row['trace_path'])
        with np.load(p) as z:vectors.append(z['episode_success']>.5)
    if all(np.array_equal(vectors[0],v) for v in vectors[1:]):
        references=[read(ROOT/read(Path(r['result_path']))['config']['baseline_path']) for r in rows]
        if all(b['conditions']['normal']['successes']==0 and b['conditions']['no_camera_bias']['successes']==0 for b in references):
            return
        raise RuntimeError('Tripwire: every learning seed has exactly the same per-episode outcomes: '+entry['name'])


def collect():
    registry=read(OUT/'registry.json'); entries={e['name']:e for e in registry['settings']}
    refs=old_references(); records=[]; summaries={}
    previous={r['name']:r for r in read(OUT/'scoreboard_records.json')['records']}
    old_summaries=read(OUT/'setting_summaries.json') if (OUT/'setting_summaries.json').exists() else {}
    for entry in registry['settings']:
        if entry['status']=='planned':continue
        rows=metrics_for(entry)
        complete=len(rows)==len(entry['seeds'])
        failed=entry['status'] in ('failed','skipped')
        if not complete and not failed:
            if entry['name'] in previous:
                records.append(previous[entry['name']])
                if entry['name'] in old_summaries:summaries[entry['name']]=old_summaries[entry['name']]
            continue
        seeds=[r['seed'] for r in rows] or entry['seeds']
        base=base_for(entry,seeds,entries,refs)
        if not failed:identical_check(entry,rows)
        values=[r['success_percent'] for r in rows]; bases=[r['success_percent'] for r in base]
        v=verdict(rows,base,len(entry['seeds']),failed)
        base_text='; '.join(f"s{r['seed']} {r['successes']}/500 ({r['success_percent']:.1f}%)" for r in base)
        if entry['base']=='floor' and entry['condition']!='fixed' and base:
            r=base[0]
            base_text=f"One shared test123 floor: {r['successes']}/500 ({r['success_percent']:.1f}%); reused, not rerun per practice seed"
        if bases:base_text+=f'; mean {mean(bases):.2f}% ± {spread(bases):.2f}pp'
        used=0
        for p in (OUT/entry['phase']/entry['name']).glob('seed_*/budget.json'):
            budget=read(p);used+=len(budget['episodes'])
            if len(budget['episodes'])>200:raise RuntimeError('Tripwire: budget exceeded')
        record=dict(phase=entry['phase'],name=entry['name'],description=entry['description'],base=entry['base'],
            base_result=base_text or None,seeds=[dict(seed=r['seed'],successes=r['successes'],episodes=500) for r in rows],
            mean_percent=mean(values),spread_pp=spread(values) if values else None,
            delta_pp=mean(values)-mean(bases) if values and bases else None,verdict=v,
            aim_start_mm=optional_mean(rows,'initial_guess_error_mean_mm'),
            aim_success_mm=optional_mean(rows,'guess_error_at_first_valid_success_mean_mm'),
            steps_to_insert=optional_mean(rows,'steps_mean'),peak_force_n=optional_mean(rows,'peak_force_mean'),
            practice_episodes=used,hand_search=entry['hand_search'],hidden_information=entry['hidden_information'],
            camera_condition='fixed for the whole run' if entry['condition']=='fixed' else entry['condition'],reason=entry.get('reason') or None,
            base_mean_percent=mean(bases),base_spread_pp=spread(bases) if bases else None)
        records.append(record)
        gain=optional_mean(rows,'paired_guess_error_reduction_mean_mm')
        summaries[entry['name']]=dict(record=record,per_seed=rows,paired_error_reduction_mm=gain,
            working_aim=(len(rows)==5 and entry['overrides'].get('outputs',2)==2 and entry['condition']=='normal'
                and mean(values)-61.4>spread(values) and gain is not None and gain>0),
            extend=(len(rows)==3 and len(base)==3 and (
                mean(values)-mean(bases)>max(spread(values),spread(bases)) or
                (entry['phase']=='B' and entry['condition']=='normal' and entry['overrides'].get('outputs',2)==2
                 and mean(values)-61.4>spread(values) and gain is not None and gain>0))))
        save(OUT/entry['phase']/entry['name']/'summary.json',summaries[entry['name']])
    prior=read(OUT/'scoreboard_records.json')
    prior['records']=records
    counted=sum(bool(e.get('counted')) for e in entries.values())
    prior['new_settings_attempted']=counted;prior['whole_project_settings_attempted']=28+counted
    save(OUT/'scoreboard_records.json',prior)
    save(OUT/'setting_summaries.json',summaries)
    render(OUT/'scoreboard_records.json',OUT)
    print(json.dumps({'completed_rows':len(records),'new_settings_attempted':counted,
                      'promotions':[n for n,s in summaries.items() if s['extend'] and len(entries[n]['seeds'])==3]}),flush=True)
    return summaries

if __name__=='__main__':collect()
