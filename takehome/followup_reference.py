"""Test-only floor/ceiling for a predeclared follow-up camera condition."""
from takehome.followup_runtime import ROOT,OUT
from takehome.correction import AimConfig
from takehome.followup_env import FollowupEnv
from takehome.followup_learner import FollowupLearner
from takehome.followup_episodes import FollowupEpisodes
from takehome.train import write_json
from dataclasses import replace
import json
import numpy as np


def reference_path(cfg):
    if cfg.camera_condition=='normal' and cfg.test_seed==123:
        return ROOT/'results/takehome/baseline.json'
    fixed=f'_seed_{cfg.fixed_bias_seed}' if cfg.camera_condition=='fixed' else ''
    return OUT/'references'/f'{cfg.camera_condition}{fixed}_test_{cfg.test_seed}.json'


def measure_reference(cfg):
    path=reference_path(cfg)
    if path.exists():
        return path
    if cfg.test_seed==1001 and not (OUT/'G/choice.json').exists():
        raise ValueError('Fresh reference forbidden before PhaseG choice')
    path.parent.mkdir(parents=True,exist_ok=True)
    plain=replace(cfg,outputs=2,drift_gate=False,pullback=False,memory=False,success_labels=False,
                  fast_contact=False,constant_fill=False,heatmap=False,within_grid=False,
                  near_hole=False,contact_cap=False,uncertainty_cap=False,final_decider=False,
                  privileged_critic=False,coordination='joint',classification=False)
    conditions={}
    for name,oracle in [('normal',False),('no_camera_bias',True)]:
        kwargs={'obs_hole_bias_std':0.} if cfg.camera_condition=='normal' and oracle else {}
        env=FollowupEnv(ROOT/'results/policies'/cfg.policy,AimConfig(),config=plain,
                        oracle=oracle and cfg.camera_condition!='normal',**kwargs)
        learner=FollowupLearner(plain,env)
        runner=FollowupEpisodes(env,learner)
        summary,stats,_=runner.test(learner.initialize(0),episodes=500,seed=cfg.test_seed)
        if not np.isfinite(summary['steps_mean']):
            summary['steps_mean']=summary['steps_median']=None
        summary.update(reference_only=True,oracle_correction=oracle)
        np.savez_compressed(path.with_name(path.stem+'_'+name+'_episodes.npz'),**stats)
        conditions[name]=summary
    write_json(path,dict(policy=cfg.policy,test_seed=cfg.test_seed,episodes_per_condition=500,
        camera_condition=cfg.camera_condition,fixed_bias_seed=cfg.fixed_bias_seed,
        practice_episodes=0,conditions=conditions,test_only=True,learned_from=False,
        ceiling_definition='Oracle cancels this condition\'s camera bias; no learned controller'))
    return path
