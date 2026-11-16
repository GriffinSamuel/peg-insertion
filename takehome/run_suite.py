"""Detached parent for predeclared commands, at most four concurrent GPU jobs."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--workers', type=int, default=4, choices=(1, 2, 4))
    a = p.parse_args()
    manifest = json.loads(a.manifest.read_text())
    pending = list(manifest['jobs'])
    active, finished = [], []
    started = time.perf_counter()
    failure = False
    while pending or active:
        while pending and len(active) < a.workers and not failure:
            job = pending.pop(0)
            path = ROOT / job['log']
            if not path.resolve().is_relative_to(ROOT / 'results/takehome'):
                raise ValueError('Log outside allowed outputs')
            path.parent.mkdir(parents=True, exist_ok=True)
            log = path.open('a')
            proc = subprocess.Popen([sys.executable, '-B', *job['args']], cwd=ROOT,
                                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            active.append((job, proc, log, time.perf_counter()))
            print('START', job['name'], proc.pid, flush=True)
        for item in list(active):
            job, proc, log, t = item
            code = proc.poll()
            if code is None:
                continue
            log.close()
            row = {'name': job['name'], 'exit_code': code, 'seconds': time.perf_counter()-t}
            finished.append(row)
            print('FINISH', json.dumps(row), flush=True)
            active.remove(item)
            if code:
                failure = True
                # A failed child may be a tripwire; stop other collection promptly.
                for _, other, _, _ in active:
                    other.terminate()
                pending.clear()
        if active:
            time.sleep(1)
    out = ROOT / manifest['suite_summary']
    out.write_text(json.dumps({'jobs': finished, 'seconds': time.perf_counter()-started,
        'finished_utc': datetime.now(timezone.utc).isoformat(), 'passed': not failure}, indent=2)+'\n')
    raise SystemExit(1 if failure else 0)


if __name__ == '__main__':
    main()
