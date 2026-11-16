"""Read-only experiment accounting, with one small replaceable status summary."""
import json
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/takehome'


def snapshot():
    practice = complete = runs = 0
    for path in OUT.glob('**/budget.json'):
        cfg = path.parent / 'config.json'
        if not cfg.exists():
            continue  # synthetic unit-test ledgers do not describe experiments
        budget = json.loads(path.read_text())
        practice += len(budget['episodes'])
        runs += 1
        complete += (path.parent / 'result.json').exists()
    names = {'aim'}
    for method in ('movement','combined','supervised_cheating_reference'):
        if list((OUT / 'goal6' / method).glob('seed_*/budget.json')):
            names.add(method)
    for filename in ('goal7_screening_manifest.json','goal7_round2_manifest.json'):
        path = OUT / filename
        if path.exists():
            for spec in json.loads(path.read_text())['settings']:
                if list((ROOT / spec['output']).glob('seed_*/budget.json')):
                    names.add(spec['name'])
    result = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
                  practice_episodes_reserved=practice, experiment_ledgers=runs,
                  completed_experiments=complete, settings_tried=len(names),
                  setting_names=sorted(names))
    (OUT / 'live_status.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__ == '__main__':
    print(json.dumps(snapshot(),indent=2))
