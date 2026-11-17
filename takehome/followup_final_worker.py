"""Locked fresh-seed confirmation; four workers own disjoint experiment paths."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from takehome.followup_runtime import ROOT,OUT
import argparse,json
from dataclasses import replace
from takehome.followup_config import FollowupConfig
from takehome.followup_runner import FollowupRunner
from takehome.followup_reference import reference_path


def configurations(choice):
    if choice.get('camera_condition')!='normal' or choice.get('hand_search') or choice.get('hidden_information'):
        raise ValueError('Fresh choice must be an eligible normal-condition learning method')
    chosen=FollowupConfig(**{**choice['overrides'],'test_seed':1001})
    if chosen.camera_condition!='normal' or chosen.privileged_critic or chosen.within_grid:
        raise ValueError('Fresh choice contains forbidden context or information/search')
    plain=FollowupConfig(outputs=3,test_seed=1001)
    if chosen==plain:
        return [('plain_movement',plain)]
    return [('chosen',chosen),('plain_movement',plain)]


def main():
    p=argparse.ArgumentParser();p.add_argument('--shard',required=True,type=int,choices=range(4));a=p.parse_args()
    choice=json.loads((OUT/'G/choice.json').read_text())
    registry=json.loads((OUT/'registry.json').read_text())
    if registry.get('status')=='halted_tripwire':raise RuntimeError('Tripwire halt active')
    configs=configurations(choice)
    tasks=[(name,cfg,seed) for name,cfg in configs for seed in range(200,205)]
    runner=None;last=None
    for index,(name,cfg,seed) in enumerate(tasks):
        if index%4!=a.shard:continue
        ref=reference_path(cfg)
        if not ref.exists():raise ValueError('Measure the single fresh reference before parallel confirmation')
        cfg=replace(cfg,baseline_path=str(ref.relative_to(ROOT)))
        output=OUT/'G'/name/f'seed_{seed}'
        if (output/'result.json').exists():
            print('SKIP',output,flush=True);continue
        if cfg!=last:runner=FollowupRunner(cfg);last=cfg
        result=runner.run(seed,output)
        print(json.dumps(dict(method=name,seed=seed,successes=result['final']['successes'])),flush=True)


if __name__=='__main__':main()
