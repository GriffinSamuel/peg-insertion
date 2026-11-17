"""Measure one context's references before any parallel learner can share them."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from takehome.followup_runtime import OUT
from dataclasses import replace
import argparse,json
from takehome.followup_config import FollowupConfig
from takehome.followup_reference import measure_reference


def main():
    p=argparse.ArgumentParser();p.add_argument('--setting',required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    p.add_argument('--extension',action='store_true');a=p.parse_args()
    registry=json.loads((OUT/'registry.json').read_text())
    if registry.get('status')=='halted_tripwire':raise RuntimeError('Tripwire halt active')
    entry=next(e for e in registry['settings'] if e['name']==a.setting)
    if not set(a.seeds).issubset(set(entry['seeds'])):
        summary=json.loads((OUT/entry['phase']/entry['name']/'summary.json').read_text())
        if not (a.extension and set(a.seeds)=={3,4} and summary['record']['verdict']=='PROMISING'):
            raise ValueError('Unregistered seeds require the prescribed PROMISING extension')
    cfg=FollowupConfig(**entry['overrides'])
    for seed in a.seeds:
        print(measure_reference(replace(cfg,fixed_bias_seed=seed)),flush=True)


if __name__=='__main__':main()
