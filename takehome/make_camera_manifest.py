"""Materialize the already declared eight camera conditions after Goal 10."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/takehome'
PROBLEMS = ('bias_2x', 'bias_4x', 'one_sided', 'drift', 'jitter', 'rotation', 'delay', 'height')


def main():
    gate = json.loads((OUT / 'goal10_checks.json').read_text())
    if not gate.get('all_goal_checks_pass'):
        raise ValueError('Goal 10 must pass before camera experiments')
    choice = json.loads((OUT / 'goal7_choice.json').read_text())['selected']
    original = json.loads((ROOT / choice['results'][0]['result_path']).read_text())['config']
    jobs, settings = [], []
    for problem in PROBLEMS:
        factor = {'bias_2x': 2, 'bias_4x': 4}.get(problem, 1)
        context = dict(camera_problem=problem, step_cap=original['step_cap'] * factor,
                       guess_cap=original['guess_cap'] * factor, test_seed=123,
                       baseline_path=f'results/takehome/goal11/{problem}/baseline.json')
        settings.append(dict(name=problem, label=f'Final method under {problem}',
            seeds=[0, 1, 2], overrides={**choice['overrides'], **context},
            output=f'results/takehome/goal11/{problem}/trained',
            round='camera conditions, no further tuning'))
        jobs.append(dict(name=problem, args=['takehome/run_camera.py', '--problem', problem],
                         log=f'results/takehome/goal11/logs/{problem}.log'))
    payload = dict(jobs=jobs, settings=settings,
                   suite_summary='results/takehome/goal11/suite.json',
                   practice_episodes_per_condition=600,
                   total_new_practice_episodes=4800,
                   new_tuned_settings=0,
                   choice='results/takehome/goal7_choice.json')
    path = OUT / 'goal11_manifest.json'
    with path.open('x') as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(path.relative_to(ROOT))


if __name__ == '__main__':
    main()
