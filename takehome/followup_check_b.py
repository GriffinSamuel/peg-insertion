"""Pre-run checks, using a separate counted ten-episode code-test ledger."""
from pathlib import Path
import sys
import json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from takehome.followup_runtime import ROOT,OUT
import jax
import numpy as np
from takehome.followup_config import FollowupConfig
from takehome.followup_runner import FollowupRunner,write_json


def main():
    destination=OUT/'B/checks'
    destination.mkdir(parents=True,exist_ok=True)
    if (destination/'checks_direct.json').exists():
        print('SKIP completed checks'); return
    print('devices',jax.devices(),flush=True)
    normal=FollowupRunner(FollowupConfig())
    state=normal.learner.initialize(0)
    assert np.all(np.asarray(normal.learner.act(state.actor,np.zeros((1,18),np.float32)))==0)
    if (destination/'zero.json').exists():
        zero=json.loads((destination/'zero.json').read_text())
    else:
        zero,stats,trace=normal.episodes.test(state,episodes=500,seed=123)
        write_json(destination/'zero.json',zero)
    if abs(zero['success_rate']-.614)>.03:
        raise RuntimeError('Tripwire: follow-up zero wrapper changes original behavior by over3points')
    cfg=FollowupConfig(code_test=True,total_episodes=10,fill_episodes=0,memory=True,success_labels=True)
    trial=FollowupRunner(cfg)
    result=trial.run(0,destination/'code_direct_seed_0')
    assert result['practice_episodes']==10 and not result['test_data_learned_from']
    assert result['frozen_policy_unchanged']
    write_json(destination/'checks_direct.json',{'gpu':[str(d) for d in jax.devices()],
        'zero':zero,'zero_tolerance_passed':True,'code_test_practice_episodes':10,
        'code_test_successes':result['final']['successes'],'frozen_policy_unchanged':True,
        'test_data_learned_from':False})
    print('PASS follow-up zero test and10-episode memory/label code trial',flush=True)

if __name__=='__main__':main()
