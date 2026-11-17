"""Supervise the four disjoint fresh-seed workers and persist any tripwire halt."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from takehome.followup_runtime import ROOT, OUT
from takehome.followup_collect import read, save, identical_check
from takehome.followup_final_worker import configurations
from takehome.followup_final_report import _load_method_seed, FRESH_SEEDS
import subprocess
import time


def check_completed_methods(configs, checked):
    for method, config in configs:
        if method in checked:
            continue
        if not all((OUT/'G'/method/f'seed_{s}'/'result.json').exists() for s in FRESH_SEEDS):
            continue
        rows = [_load_method_seed(method, config, seed) for seed in FRESH_SEEDS]
        identical_check({'name': 'G_' + method}, rows)
        checked.add(method)
        print('VALIDATED', method, [r['successes'] for r in rows], flush=True)


def run(name):
    registry = read(OUT/'registry.json')
    if registry.get('status') == 'halted_tripwire':
        raise RuntimeError('Tripwire halt is active')
    if not {'A', 'B', 'B_mixed', 'C', 'D', 'E', 'F'}.issubset(registry['completed_phases']):
        raise ValueError('Finish all preceding phases before fresh confirmation')
    configs = configurations(read(OUT/'G/choice.json'))
    active = []
    done = []
    checked = set()
    started = time.perf_counter()
    failure = None
    try:
        for shard in range(4):
            path = OUT/'logs'/f'{name}_shard_{shard}.log'
            path.parent.mkdir(parents=True, exist_ok=True)
            log = path.open('x')
            try:
                process = subprocess.Popen(
                    [sys.executable, '-B', 'takehome/followup_final_worker.py', '--shard', str(shard)],
                    cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            except BaseException:
                log.close()
                raise
            active.append((shard, process, log, path))
            print('START', shard, process.pid, flush=True)
        while active:
            for item in list(active):
                shard, process, log, path = item
                code = process.poll()
                if code is None:
                    continue
                log.close()
                active.remove(item)
                done.append(dict(shard=shard, exit_code=code))
                print('FINISH', shard, code, flush=True)
                if code:
                    detail = 'Tripwire: fresh worker failed' if 'Tripwire:' in path.read_text() else 'Fresh worker failed'
                    raise RuntimeError(f'{detail}: shard {shard}, exit {code}; see {path.relative_to(OUT)}')
                check_completed_methods(configs, checked)
            if active:
                time.sleep(1)
        check_completed_methods(configs, checked)
        if checked != {method for method, _ in configs}:
            raise RuntimeError('Workers exited but a confirmation method is incomplete')
    except BaseException as exc:
        failure = repr(exc)
        if 'Tripwire:' in str(exc):
            registry['status'] = 'halted_tripwire'
            registry['halted_at'] = 'G'
            save(OUT/'registry.json', registry)
        print('FAILURE', failure, flush=True)
        raise
    finally:
        for _, process, _, _ in active:
            process.terminate()
        for shard, process, log, _ in active:
            try:
                code = process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                code = process.wait()
            log.close()
            done.append(dict(shard=shard, exit_code=code, stopped_by_parent=True))
        save(OUT/f'{name}_suite.json', dict(jobs=done, failure=failure,
            tripwire=bool(failure and 'Tripwire:' in failure),
            checked_methods=sorted(checked), seconds=time.perf_counter()-started))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    args = parser.parse_args()
    if not args.name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in args.name):
        parser.error('name must be a simple lowercase log name')
    run(args.name)
