"""Count unique test measurements and actual ledgers without counting reuse."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from takehome.status import snapshot  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/takehome'


def collect():
    base_path = OUT / 'goal7_pre_screen_checks.json'
    base = json.loads(base_path.read_text())
    paths = set()
    for phase in ('goal7', 'goal10', 'goal11', 'goal13'):
        for pattern in ('**/test_[0-9][0-9][0-9][0-9].json',
                        '**/extra_ceiling_[0-9][0-9][0-9][0-9].json'):
            paths.update((OUT / phase).glob(pattern))
    for pattern in ('confirmation_baseline_normal.json',
                    'confirmation_baseline_no_camera_bias.json',
                    'goal8/seed_*/point_*/test_summary.json',
                    'goal8/seed_*/point_*/extra_ceiling.json',
                    'goal9/before_traces.json', 'goal9/after_seed_*_poses.json',
                    'goal9/extra_ceiling_after_seed_*_poses.json',
                    'goal10/*/baseline_normal.json', 'goal10/*/baseline_no_camera_bias.json',
                    'goal11/*/floor.json', 'goal11/*/oracle.json',
                    'goal11/*/transfer_seed_[0-2].json',
                    'goal11/*/extra_ceiling_transfer_seed_[0-2].json'):
        paths.update(OUT.glob(pattern))
    rows = []
    for path in sorted(paths):
        data = json.loads(path.read_text())
        if not data.get('test_only') or data.get('learned_from') is not False:
            raise ValueError(f'Test-only accounting provenance failed: {path}')
        episodes = data['episodes']
        if not isinstance(episodes, int) or episodes <= 0:
            raise ValueError(f'Invalid test episode count: {path}')
        rows.append(dict(path=str(path.relative_to(ROOT)), episodes=episodes,
                         seed=data['test_seed']))
    # Integration tests do not write normal experiment summaries. Each executed
    # test command gets one explicit evidence file, never a guessed multiplier.
    contracts = []
    for path in sorted(OUT.glob('*_test_episode_accounting.json')):
        item = json.loads(path.read_text())
        if item.get('executed') is not True:
            raise ValueError('Cannot count a test command that has not run')
        contracts.append(dict(path=str(path.relative_to(ROOT)), **item))
    new_tests = sum(r['episodes'] for r in rows) + sum(r['full_test_episodes'] for r in contracts)
    return dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
        full_test_episodes_total=base['full_test_episodes_total']+new_tests,
        full_tests_before_goal7=base['full_test_episodes_total'],
        new_full_test_episodes=new_tests,
        reset_only_probes_total=base['reset_only_probes_total']+
            sum(r.get('reset_only_probes', 0) for r in contracts),
        base_source=str(base_path.relative_to(ROOT)), measurements=rows,
        additional_contract_commands=contracts, practice=snapshot(),
        counting_rule='Unique measurement JSONs; copied baselines, curve copies and reused traces are not new tests.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUT/'episode_accounting.json')
    args = parser.parse_args()
    path = args.output.resolve()
    if not path.is_relative_to(OUT):
        raise ValueError('Accounting output must stay under results/takehome')
    data = collect()
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in data.items() if k not in ('measurements','additional_contract_commands')}, indent=2))
