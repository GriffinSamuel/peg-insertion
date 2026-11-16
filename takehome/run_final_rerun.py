"""Goal 13 fresh final-method run, fixed to confirmation practice seed 100."""
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from takehome.runtime import ROOT  # noqa: E402
from takehome.variants import VariantConfig  # noqa: E402
from takehome.variant_runner import VariantExperimentRunner  # noqa: E402
from takehome.train import write_json  # noqa: E402


def main():
    out = ROOT / 'results/takehome'
    if not json.loads((out / 'goal12_checks.json').read_text()).get('all_goal_checks_pass'):
        raise ValueError('Goal 12 must finish before the final rerun')
    choice = json.loads((out / 'goal7_choice.json').read_text())
    selected = choice['selected']
    confirmed = json.loads((out / 'goal7/confirmation/confirmation_summary.json').read_text())
    reference = next(row for row in confirmed['methods'] if row['name'] == selected['name'])
    config = VariantConfig(**{**selected['overrides'], 'test_seed': 456,
                             'baseline_path': 'results/takehome/confirmation_baseline.json'})
    original = out / 'goal7/confirmation' / selected['name'] / 'seed_100'
    def originals():
        return {name: hashlib.sha256((original / name).read_bytes()).hexdigest()
                for name in ('result.json', 'budget.json', 'model_0200.pkl')}
    hashes_before = originals()
    runner = VariantExperimentRunner(config)
    result = runner.run(100, out / 'goal13/fresh/seed_100')
    if originals() != hashes_before:
        raise RuntimeError('Original confirmation artifacts changed during the fresh rerun')
    rate = result['final']['success_rate']
    delta = abs(rate - reference['mean'])
    payload = dict(method=selected['name'], practice_seed=100, test_seed=456,
        practice_episodes=result['practice_episodes'], final_test_episodes=500,
        successes=result['final']['successes'], success_rate=rate,
        confirmation_mean=reference['mean'], confirmation_sample_spread=reference['std'],
        absolute_difference=delta, within_one_spread=delta <= reference['std'],
        result_path='results/takehome/goal13/fresh/seed_100/result.json',
        budget_path='results/takehome/goal13/fresh/seed_100/budget.json',
        original_confirmation_unchanged=True,
        original_confirmation_hashes=hashes_before,
        repeatability_instruction='If outside the spread, log the limitation and continue under Griffin\'s explicit instruction; never replace a result or retune.')
    write_json(out / 'goal13/rerun_comparison.json', payload)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == '__main__':
    main()
