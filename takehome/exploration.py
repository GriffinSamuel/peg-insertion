"""Lead-authored development summaries, predeclared selection and confirmation."""
import argparse
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/takehome'


def read(path):
    return json.loads((ROOT / path).read_text())


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def summarize():
    methods = read('results/takehome/goal6_manifest.json')['methods']
    specs = {}
    for method in methods:
        name = method['name']
        specs[name] = dict(name=name, label=method['label'], round='plain comparison (reused)',
            base='aim', changes_from_main=0 if name == 'aim' else 1,
            eligible=not method['cheats_on_information'], seeds=list(range(5)),
            results=method['results'], overrides={'outputs': {'aim':2,'movement':3,'combined':5,
                'supervised_cheating_reference':2}[name]},
            output=str(Path(method['results'][0]).parent.parent))
        if name == 'supervised_cheating_reference':
            specs[name]['overrides']['supervised_reference'] = True
    for filename in ('goal7_screening_manifest.json', 'goal7_round2_manifest.json'):
        path = OUT / filename
        if not path.exists():
            continue
        for spec in json.loads(path.read_text())['settings']:
            specs[spec['name']] = {**spec, 'eligible': True}
    rows = []
    for spec in specs.values():
        paths = spec.get('results', [f"{spec['output']}/seed_{s}/result.json" for s in spec['seeds']])
        runs = []
        for path in paths:
            result = read(path)
            f = result['final']
            if result['practice_episodes'] != 200 or f['episodes'] != 500 or f['test_seed'] != 123:
                raise ValueError('Selection requires final200 practice /500 development tests at123')
            runs.append(dict(seed=result['seed'], successes=f['successes'], episodes=f['episodes'],
                success_rate=f['success_rate'], peak_force_mean=f['peak_force_mean'], result_path=path))
        if sorted(r['seed'] for r in runs) != spec['seeds']:
            raise ValueError('Missing or duplicated development practice seed')
        rates = [r['success_rate'] for r in runs]
        rows.append({**spec, 'results': runs, 'mean': statistics.mean(rates),
                     'std': statistics.stdev(rates)})
    lookup = {row['name']: row for row in rows}
    for row in rows:
        seeds = {r['seed'] for r in row['results']}
        base = lookup[row['base']]
        base_rates = [r['success_rate'] for r in base['results'] if r['seed'] in seeds]
        main_rates = [r['success_rate'] for r in lookup['aim']['results'] if r['seed'] in seeds]
        row['matched_base_mean'] = statistics.mean(base_rates)
        row['matched_base_std'] = statistics.stdev(base_rates)
        row['gain_over_base'] = row['mean'] - row['matched_base_mean']
        row['clear_gain_over_base'] = row['gain_over_base'] > max(row['std'], row['matched_base_std'])
        row['clear_gain_over_main'] = (row['mean'] - statistics.mean(main_rates) >
                                      max(row['std'], statistics.stdev(main_rates)))
        margin = max(row['std'], row['matched_base_std'])
        row['difference_below_spread_is_unproven'] = abs(row['gain_over_base']) <= margin
        row['difference_status'] = ('gain beyond spread' if row['gain_over_base'] > margin else
            ('loss beyond spread' if row['gain_over_base'] < -margin else 'difference below spread, unproven'))
    baseline = read('results/takehome/baseline.json')
    summary = dict(settings=rows, floor=baseline['conditions']['normal']['success_rate'],
        ceiling=baseline['conditions']['no_camera_bias']['success_rate'],
        total_settings_tried=len(rows), test_seed=123,
        generated_utc=datetime.now(timezone.utc).isoformat())
    (OUT / 'goal7_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    return summary


def merge_changes(rows):
    changes = {}
    for row in rows:
        for key, value in row['overrides'].items():
            if key in changes and changes[key] != value:
                return None, f'conflicting values for {key}'
            changes[key] = value
    if changes.get('briefing') and changes.get('fill_episodes', 40) == 0:
        return None, 'briefing needs own fill-up'
    if changes.get('privileged_critic') and changes.get('final_decider'):
        return None, 'final decider cannot see true bias'
    if changes.get('force_penalty') and changes.get('classification'):
        return None, 'negative force reward incompatible with binary target'
    coord = changes.get('coordination', 'joint')
    if coord != 'joint' and changes.get('final_decider'):
        return None, 'independent component critics do not provide a joint decider'
    if coord in ('aim_then_move', 'move_then_aim') and (200-changes.get('fill_episodes', 40)) % 20:
        return None, 'sequential halves need whole batches of ten'
    return changes, None


def plan_round2():
    summary = summarize()
    rows = summary['settings']
    eligible = [r for r in rows if r['eligible']]
    skip = not any(r['clear_gain_over_main'] for r in eligible if r['name'] != 'aim')
    promising = sorted([r for r in rows if r['round'] == 'screening' and
                        r['clear_gain_over_base']], key=lambda r: (-r['mean'], r['name']))[:3]
    settings, skipped = [], []
    if not skip:
        for row in promising:
            settings.append({k: row[k] for k in ('name','label','base','changes_from_main','overrides','output')})
            settings[-1].update(seeds=list(range(5)), round='screening plus five-seed follow-up')
        for size in (2, 3):
            for combination in itertools.combinations(promising, size):
                changes, reason = merge_changes(combination)
                names = [r['name'] for r in combination]
                if reason:
                    skipped.append(dict(components=names, reason=reason))
                    continue
                name = 'combo_' + '_'.join(names)
                settings.append(dict(name=name, label=' + '.join(r['label'] for r in combination),
                    base='combined' if changes.get('outputs', 2) == 5 else 'aim',
                    changes_from_main=len(changes), overrides=changes, seeds=list(range(5)),
                    round='combination', output=f'results/takehome/goal7/round2/{name}'))
    unique = {r['name'] for r in rows} | {s['name'] for s in settings}
    if len(unique) > 45:
        raise ValueError('Setting limit exceeded')
    path = OUT / 'goal7_round2_manifest.json'
    jobs = [dict(name=s['name'], args=['takehome/train_variants.py','--manifest',str(path.relative_to(ROOT)),
                '--setting',s['name']], log=f"results/takehome/goal7/logs/round2_{s['name']}.log") for s in settings]
    payload = dict(settings=settings, jobs=jobs, skipped_by_no_gain_rule=skip,
        promising_screening_names=[r['name'] for r in promising], incompatible_combinations=skipped,
        total_settings_after_round2=len(unique), suite_summary='results/takehome/goal7/round2_suite.json',
        planned_utc=datetime.now(timezone.utc).isoformat())
    write_new(path, payload)
    print(json.dumps(payload, indent=2))


def select_candidate(rows, skip_round2):
    candidates = [r for r in rows if r['eligible']]
    top = max(candidates, key=lambda r: r['mean'])
    within = [r for r in candidates if r['mean'] >= top['mean'] - top['std']]
    if skip_round2:
        selected = next(r for r in candidates if r['name'] == 'aim')
        reason = 'No change beat main beyond both sample spreads; written skip rule confirms main.'
    else:
        selected = min(within, key=lambda r: (r['changes_from_main'], -r['mean'], r['name']))
        reason = 'Fewest changes within one sample spread of the highest development mean; higher mean breaks ties.'
    return selected, top, within, reason


def choose():
    if (OUT / 'goal7_choice.json').exists():
        raise ValueError('Selection already written; cannot silently choose again')
    summary = summarize()
    round2 = read('results/takehome/goal7_round2_manifest.json')
    selected, top, within, reason = select_candidate(summary['settings'], round2['skipped_by_no_gain_rule'])
    # Check immediately before writing. References and trained confirmation
    # artifacts are forbidden until this immutable choice exists.
    if list(OUT.glob('**/seed_10[0-4]/result.json')) or (OUT / 'confirmation_baseline.json').exists():
        raise ValueError('Confirmation artifacts already exist before written selection')
    payload = dict(selected=selected, highest_mean_setting=top['name'], highest_mean=top['mean'],
        highest_mean_spread=top['std'], within_one_spread=[r['name'] for r in within], reason=reason,
        total_settings_tried=summary['total_settings_tried'],
        written_utc=datetime.now(timezone.utc).isoformat(), confirmation_previously_unused=True,
        confirmation_practice_seeds=list(range(100,105)), confirmation_test_seed=456)
    write_new(OUT / 'goal7_choice.json', payload)
    print(json.dumps(payload, indent=2))


def confirmation_manifest():
    choice = read('results/takehome/goal7_choice.json')
    selected = choice['selected']
    methods = [('aim','Plain aim',{'outputs':2}),('movement','Plain movement',{'outputs':3}),
               ('combined','Plain aim and movement',{'outputs':5})]
    if selected['name'] not in ('aim','movement','combined'):
        methods.append((selected['name'], 'Final tuned method: '+selected['label'],selected['overrides']))
    settings = []
    for name, label, overrides in methods:
        settings.append(dict(name=name, label=label, round='confirmation',
            overrides={**overrides, 'test_seed':456,'baseline_path':'results/takehome/confirmation_baseline.json'},
            seeds=list(range(100,105)), output=f'results/takehome/goal7/confirmation/{name}',
            final_method=name == selected['name'], settings_tried_for_selection=(
                choice['total_settings_tried'] if name == selected['name'] else 1)))
    path = OUT / 'goal7_confirmation_manifest.json'
    jobs = [dict(name=s['name'],args=['takehome/train_variants.py','--manifest',str(path.relative_to(ROOT)),
        '--setting',s['name']],log=f"results/takehome/goal7/logs/confirmation_{s['name']}.log") for s in settings]
    write_new(path, dict(settings=settings,jobs=jobs,choice='results/takehome/goal7_choice.json',
                        suite_summary='results/takehome/goal7/confirmation_suite.json'))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage',choices=['summarize','plan-round2','choose','confirmation-manifest'])
    stage = p.parse_args().stage
    {'summarize':summarize,'plan-round2':plan_round2,'choose':choose,
     'confirmation-manifest':confirmation_manifest}[stage]()
