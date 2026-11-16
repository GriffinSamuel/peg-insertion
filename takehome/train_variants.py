"""Run only a named, predeclared setting and seed group from a suite manifest."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from takehome.runtime import ROOT  # noqa: E402
from takehome.variants import VariantConfig  # noqa: E402
from takehome.variant_runner import VariantExperimentRunner  # noqa: E402
import numpy as np  # noqa: E402


def check_seed_outcomes(output, seeds, config):
    baseline = json.loads((ROOT / config.baseline_path).read_text())
    zero_policy = all(baseline['conditions'][k]['success_rate'] == 0
                      for k in ('normal', 'no_camera_bias'))
    outcomes = []
    for seed in seeds:
        with np.load(output / f'seed_{seed}/test_0200_episodes.npz') as archive:
            outcomes.append(archive['succeeded'])
    if len(outcomes) > 1 and not zero_policy and all(np.array_equal(outcomes[0], x) for x in outcomes[1:]):
        raise RuntimeError('Tripwire: every learning seed has identical per-episode test outcomes')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--setting', required=True)
    a = p.parse_args()
    manifest = json.loads(a.manifest.read_text())
    matches = [s for s in manifest['settings'] if s['name'] == a.setting]
    if len(matches) != 1:
        raise ValueError('Setting must appear exactly once in manifest')
    setting = matches[0]
    cfg = VariantConfig(**setting['overrides'])
    seeds = setting['seeds']
    if any(s in range(100, 105) for s in seeds) or cfg.test_seed == 456:
        choice = ROOT / 'results/takehome/goal7_choice.json'
        if not choice.exists():
            raise ValueError('Confirmation forbidden before written selection')
        if cfg.test_seed != 456 or any(s not in range(100, 105) for s in seeds):
            raise ValueError('Confirmation seed pairing is fixed')
    elif any(s not in range(5) for s in seeds) or cfg.test_seed != 123:
        raise ValueError('Development seed pairing is fixed')
    output = ROOT / setting['output']
    if not output.resolve().is_relative_to(ROOT / 'results/takehome'):
        raise ValueError('Output must be in results/takehome')
    todo = [s for s in seeds if not (output / f'seed_{s}/result.json').exists()]
    if not todo:
        check_seed_outcomes(output, seeds, cfg)
        print('All named results exist; no practice collected.')
        return
    runner = VariantExperimentRunner(cfg)
    for seed in todo:
        result = runner.run(seed, output / f'seed_{seed}')
        print(json.dumps(dict(setting=a.setting, seed=seed, final=result['final'])), flush=True)
    check_seed_outcomes(output, seeds, cfg)


if __name__ == '__main__':
    main()
