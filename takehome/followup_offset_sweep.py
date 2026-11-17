"""Prescribed, test-only 7x7 constant-aim sweep; four disjoint worker shards."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from takehome.followup_runtime import ROOT, OUT
import argparse
import json
import time
import jax
import jax.numpy as jp
import numpy as np
from takehome.correction import AimConfig, policy_digest
from takehome.followup_config import FollowupConfig
from takehome.followup_env import FollowupEnv
from takehome.followup_learner import FollowupLearner
from takehome.followup_episodes import FollowupEpisodes
from takehome.train import write_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--shard', type=int, required=True, choices=range(4))
    args=parser.parse_args()
    if not any(d.platform=='gpu' for d in jax.devices()):
        raise RuntimeError('Tripwire: GPU not visible')
    folder=OUT/'B/offset_sweep'
    folder.mkdir(parents=True,exist_ok=True)
    cfg=FollowupConfig()
    env=FollowupEnv(ROOT/'results/policies'/cfg.policy,AimConfig(),config=cfg,
                    obs_hole_bias_std=0.)
    learner=FollowupLearner(cfg,env)
    episodes=FollowupEpisodes(env,learner)
    state=learner.initialize(0)
    before=policy_digest(env.frozen_policy)
    axis=np.linspace(-.0009,.0009,7)
    for index in range(args.shard,49,4):
        path=folder/f'cell_{index:02d}'/'test_0000.json'
        if path.exists():
            print('SKIP',index,flush=True)
            continue
        path.parent.mkdir(parents=True,exist_ok=True)
        x,y=divmod(index,7)
        guess=jp.asarray([axis[x],axis[y]])
        t0=time.perf_counter()
        summary,stats,trace=episodes.test(state.replace(start_guess=guess),episodes=256,seed=123)
        assert np.allclose(trace['guess'],np.asarray(guess),rtol=0,atol=1e-9)
        assert np.allclose(trace['next_guess'],np.asarray(guess),rtol=0,atol=1e-9)
        assert np.count_nonzero(trace['true_bias'])==0
        assert np.count_nonzero(trace['action'])==0
        assert policy_digest(env.frozen_policy)==before
        if not np.isfinite(summary['steps_mean']):
            summary['steps_mean']=summary['steps_median']=None
        summary.update(cell=index,grid_xy=[x,y],constant_guess_mm=(np.asarray(guess)*1000).tolist(),
            input_shift_mm=(-np.asarray(guess)*1000).tolist(),practice_episodes=0,
            seconds=time.perf_counter()-t0,frozen_policy_unchanged=True,
            used_for_learner_selection=False,full_trace_saved=False)
        write_json(path,summary)
        print(json.dumps(dict(cell=index,guess_mm=summary['constant_guess_mm'],
                             successes=summary['successes'],seconds=summary['seconds'])),flush=True)


if __name__=='__main__':main()
