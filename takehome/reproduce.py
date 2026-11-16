"""Reproduce the written final method in a separately named output directory."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from takehome.runtime import ROOT  # noqa: E402
from takehome.variants import VariantConfig  # noqa: E402
from takehome.variant_runner import VariantExperimentRunner  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed', type=int, choices=range(100,105), default=100)
    p.add_argument('--output', type=Path, required=True,
                   help='Fresh directory under results/takehome, or the same directory to resume')
    args = p.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / 'results/takehome/reproductions'):
        p.error('Reproductions must use their own results/takehome/reproductions directory')
    choice = json.loads((ROOT / 'results/takehome/goal7_choice.json').read_text())
    config = VariantConfig(**{**choice['selected']['overrides'], 'test_seed':456,
        'baseline_path':'results/takehome/confirmation_baseline.json'})
    result = VariantExperimentRunner(config).run(args.seed, output / f'seed_{args.seed}')
    print(json.dumps(dict(seed=args.seed, result=result['final'],
                          practice_episodes=result['practice_episodes']), indent=2))


if __name__ == '__main__':
    main()
