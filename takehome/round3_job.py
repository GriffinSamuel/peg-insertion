"""Run named Round 3 settings on given seeds in one GPU process; one summary.json per setting."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from takehome.followup_runtime import ROOT  # noqa: E402
import argparse, json  # noqa: E402
from takehome.round3_runner import Round3Runner, R3, write_json  # noqa: E402
from takehome.round3_search import SearchRunner  # noqa: E402
from takehome.round3_settings import SETTINGS, make_config  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--names', nargs='+', required=True)
    p.add_argument('--seeds', type=int, nargs='+', required=True)
    p.add_argument('--test-seed', type=int, default=123)
    a = p.parse_args()
    runner = None
    for name in a.names:
        task = SETTINGS[name][0]
        extra = {} if a.test_seed == 123 else {
            'test_seed': a.test_seed,
            'baseline_path': f'results/takehome/round3/references/normal_test_{a.test_seed}.json'}
        cfg = make_config(name, **extra)
        if runner is None or runner.config != cfg:
            runner = SearchRunner(cfg) if cfg.search != 'none' else Round3Runner(cfg)
        base = R3 / f'task{task}' / name
        for seed in a.seeds:
            r = runner.run(seed, base / f'seed_{seed}')
            print(json.dumps({'setting': name, 'seed': seed, 'successes': r['final']['successes'],
                              'last': r['final_last']['successes'],
                              'rollbacks': len(r['rollbacks']), 'seconds': round(r['seconds_excluding_imports'])}), flush=True)
        results = {}
        for s in a.seeds:
            rp = base / f'seed_{s}' / 'result.json'
            if rp.exists():
                results[str(s)] = json.loads(rp.read_text())['final']
        write_json(base / f'summary_{"_".join(map(str, a.seeds))}.json', {'name': name, 'finals': results})


if __name__ == '__main__':
    main()
