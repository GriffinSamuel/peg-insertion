"""Write the lead's predeclared Goal 7 screening manifest, without running it."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def screening_settings():
    settings = []
    def add(name, label, changes, movement=False):
        overrides = {'outputs': 5, **changes} if movement else changes
        settings.append(dict(name=name, label=label, overrides=overrides,
            base='combined' if movement else 'aim', changes_from_main=2 if movement else 1,
            seeds=[0, 1, 2], round='screening'))
    for n in (0, 10, 20, 80, 120):
        add(f'fill_{n}', f'Fill {n} episodes', {'fill_episodes': n})
    for mode in ('side_by_side', 'aim_then_move', 'move_then_aim'):
        add(mode, mode.replace('_', ' '), {'coordination': mode}, movement=True)
    add('briefing', 'Critic briefing on own fill', {'briefing': True})
    add('privileged_critic', 'True bias to critic during training', {'privileged_critic': True})
    add('one_critic', 'One critic', {'ensemble': 1})
    add('one_update', 'One update per new step', {'updates_per_step': 1})
    for cap in (.00005, .0002):
        add(f'aim_cap_{round(cap*1e6)}um', f'Aim cap {cap*1000:g} mm', {'step_cap': cap})
    add('movement_cap_400um', 'Movement cap 0.4 mm', {'movement_cap': .0004}, movement=True)
    add('final_decider', 'Final critic decider', {'final_decider': True}, movement=True)
    add('near_hole', 'Hand over near hole', {'near_hole': True})
    add('contact_cap', 'Contact-dependent movement cap', {'contact_cap': True}, movement=True)
    add('uncertainty_cap', 'Old-policy uncertainty cap', {'uncertainty_cap': True}, movement=True)
    add('classification', 'Classification critic loss', {'classification': True})
    add('three_step', 'Three-step credit', {'n_step': 3})
    add('return_floor', 'Observed return floor', {'return_floor': True})
    add('force_penalty', 'Force penalty (peak force reported)', {'force_penalty': True}, movement=True)
    add('sibling_fill', 'Sibling-policy fill-up', {'sibling_fill': True})
    assert len(settings) == 24
    return settings


def main():
    settings = screening_settings()
    path = ROOT / 'results/takehome/goal7_screening_manifest.json'
    jobs = []
    for setting in settings:
        name = setting['name']
        setting['output'] = f'results/takehome/goal7/screening/{name}'
        jobs.append(dict(name=name,
            args=['takehome/train_variants.py', '--manifest', str(path.relative_to(ROOT)),
                  '--setting', name], log=f'results/takehome/goal7/logs/{name}.log'))
    payload = dict(settings=settings, jobs=jobs, test_seed=123, final_test_episodes=500,
        prior_settings=4, new_settings=24, total_settings_after_screening=28,
        suite_summary='results/takehome/goal7/screening_suite.json')
    with path.open('x') as f:
        json.dump(payload, f, indent=2)
        f.write('\n')


if __name__ == '__main__':
    main()
