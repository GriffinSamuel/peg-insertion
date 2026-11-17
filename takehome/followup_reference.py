"""Test-only floor/ceiling for a predeclared follow-up camera condition."""
from takehome.followup_runtime import ROOT,OUT
from takehome.correction import AimConfig
from takehome.followup_env import FollowupEnv
from takehome.followup_learner import FollowupLearner
from takehome.followup_episodes import FollowupEpisodes
from takehome.train import write_json
from takehome.followup_paths import condition_key
from dataclasses import replace
import json
import numpy as np


def reference_path(cfg):
    if cfg.camera_condition=='normal' and cfg.test_seed==123:
        return ROOT/'results/takehome/baseline.json'
    fixed=f'_seed_{cfg.fixed_bias_seed}' if cfg.camera_condition in ('fixed','mixed') else ''
    return OUT/'references'/f'{condition_key(cfg)}{fixed}_test_{cfg.test_seed}.json'


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
        zero_error=oracle and cfg.camera_condition in ('normal','mixed')
        kwargs={'obs_hole_bias_std':0.} if zero_error else {}
        reference_cfg=(replace(plain,camera_condition='normal',mixed_fixed_std=0.,mixed_episode_std=0.)
                       if zero_error else plain)
        env=FollowupEnv(ROOT/'results/policies'/cfg.policy,AimConfig(),config=reference_cfg,
                        oracle=oracle and not zero_error,**kwargs)
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
        mixed_fixed_std=cfg.mixed_fixed_std,mixed_episode_std=cfg.mixed_episode_std,
        practice_episodes=0,conditions=conditions,test_only=True,learned_from=False,
        ceiling_definition=('Zero camera error, both components removed; no learned controller'
            if cfg.camera_condition=='mixed' else 'Oracle cancels this condition\'s camera bias; no learned controller')))
    return path
