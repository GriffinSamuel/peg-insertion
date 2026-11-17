"""The Phase F 500-episode hand-written comparison, with no learned episodes."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from takehome.followup_runtime import ROOT, OUT
import argparse
import json
import time
import jax
import numpy as np
from takehome.correction import AimConfig, policy_digest
from takehome.followup_config import FollowupConfig
from takehome.followup_learner import FollowupLearner
from takehome.followup_metrics import summarize_trace
from takehome.followup_touch import TouchEnv, TouchEpisodes, TOUCH_STEP_CAP, SPIRAL_SPACING


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--method', choices=['blind_spiral'], required=True)
    args=parser.parse_args()
    registry=json.loads((OUT/'registry.json').read_text())
    if 'E' not in registry['completed_phases'] or registry.get('status')=='halted_tripwire':
        raise ValueError('Touch comparisons wait until Phase E is complete')
    if not (OUT/'F/signal_diagnosis.json').exists():
        raise ValueError('Review the saved-trace signal diagnosis first')
    if not any(d.platform=='gpu' for d in jax.devices()):
        raise RuntimeError('Tripwire: GPU not visible')
    output=OUT/'F'/('hand_'+args.method)
    output.mkdir(parents=True, exist_ok=True)
    if (output/'result.json').exists():
        print('SKIP', output, flush=True);return
    if (output/'test_0000.json').exists():
        raise ValueError('A partial hand comparison exists; inspect and recover it without retesting')
    started=time.perf_counter()
    cfg=FollowupConfig(step_cap=TOUCH_STEP_CAP)
    env=TouchEnv(ROOT/'results/policies'/cfg.policy, AimConfig(cfg.step_cap,cfg.guess_cap),
                 config=cfg, touch_method=args.method)
    # This initialized actor is never trained and never chooses the aim: TouchEnv
    # supplies it from observed contact/history. Its Q ratings have no meaning.
    learner=FollowupLearner(cfg,env)
    before=policy_digest(env.frozen_policy)
    runner=TouchEpisodes(env,learner)
    summary, stats, trace=runner.test(learner.initialize(0), episodes=500, seed=123,
                                     output=output/'test_0000_traces.npz')
    if not np.isfinite(summary['steps_mean']):
        summary['steps_mean']=summary['steps_median']=None
    summary.update(practice_episodes=0, hand_search=True, hidden_information=False,
                   camera_condition='normal', reference_only=True)
    (output/'test_0000.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    np.savez_compressed(output/'test_0000_episodes.npz', **stats)
    trace.update(episode_success=stats['succeeded'], test_only=np.array(True),
                 learned_from=np.array(False), test_seed=np.array(123))
    metrics=summarize_trace(trace, summary, outputs=2)
    if not metrics['episode_success_vector_matches_final']:
        raise RuntimeError('Tripwire: hand reference outcome vector disagrees with final count')
    if policy_digest(env.frozen_policy)!=before:
        raise RuntimeError('Tripwire: frozen policy changed')
    result=dict(method=args.method, final=summary, metrics=metrics, practice_episodes=0,
        no_learning=True, test_data_learned_from=False, frozen_policy_unchanged=True,
        hand_search=True, hidden_information=False, uses_q_ratings=False,
        aim_final_error_mm=float(np.linalg.norm(trace['next_guess'][-1]-trace['true_bias'][-1],axis=-1).mean()*1000),
        timeout_episodes=int(np.any(trace['truncated'],axis=0).sum()),
        search_used_episodes=int(np.any(trace['touch_active'],axis=0).sum()),
        mean_search_steps=float(trace['touch_steps'][-1].mean()),
        step_cap_mm=TOUCH_STEP_CAP*1000, spiral_spacing_mm=SPIRAL_SPACING*1000,
        seconds_excluding_imports=time.perf_counter()-started,
        guess_error_analysis_uses_hidden_bias_only_after_rollout=True)
    (output/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result), flush=True)


if __name__=='__main__':
    main()
