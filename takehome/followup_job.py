"""One narrow setting task; batch its predeclared seeds in one GPU process."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from takehome.followup_runtime import ROOT,OUT
import argparse,json
from dataclasses import replace
from takehome.followup_config import FollowupConfig
from takehome.followup_runner import FollowupRunner
from takehome.followup_reference import measure_reference
from takehome.followup_paths import run_path


def main():
    p=argparse.ArgumentParser();p.add_argument('--setting',required=True);p.add_argument('--seeds',type=int,nargs='+',required=True)
    a=p.parse_args()
    registry=json.loads((OUT/'registry.json').read_text())
    if registry.get('status')=='halted_tripwire':
        raise RuntimeError('Tripwire halt is active; user direction is required before any experiment')
    entry=next(x for x in registry['settings'] if x['name']==a.setting)
    if entry.get('status') not in ('running','finished'):
        raise ValueError('Setting must be authorized in registry before launch')
    if not set(a.seeds).issubset(set(entry['seeds'])):
        raise ValueError('Unregistered seeds')
    cfg=FollowupConfig(**entry['overrides'])
    runner=None; last_cfg=None
    for seed in a.seeds:
        output=run_path(entry,seed)
        if (output/'result.json').exists():
            print('SKIP completed',output,flush=True);continue
        current=replace(cfg,fixed_bias_seed=seed) if cfg.camera_condition in ('fixed','mixed') else cfg
        path=measure_reference(current)
        current=replace(current,baseline_path=str(path.relative_to(ROOT)))
        if current!=last_cfg:
            runner=FollowupRunner(current);last_cfg=current
        result=runner.run(seed,output)
        print(json.dumps({'setting':a.setting,'seed':seed,'successes':result['final']['successes'],
                          'seconds':result['seconds_excluding_imports']}),flush=True)

if __name__=='__main__':main()
