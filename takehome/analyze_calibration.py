"""Goal 8: independent episode probabilities and raw Q, on test episodes only."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from takehome.runtime import ROOT  # noqa: E402
import numpy as np  # noqa: E402
from takehome.artifacts import load_model, evaluation_runner  # noqa: E402
from takehome.calibration import (fit_from_practice,evaluate_trace,reliability_bins,raw_rating_bins)  # noqa: E402
from takehome.train import write_json  # noqa: E402
from takehome.evaluation_checks import verify_ceiling  # noqa: E402


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seeds',nargs='+',type=int,default=list(range(100,105)))
    a=p.parse_args()
    out=ROOT/'results/takehome/goal8'; out.mkdir(parents=True,exist_ok=True)
    check=ROOT/'results/takehome/goal7_checks.json'
    if not check.exists() or not json.loads(check.read_text()).get('all_goal_checks_pass'):
        raise ValueError('Goal 7 must finish before confidence analysis')
    choice=json.loads((ROOT/'results/takehome/goal7_choice.json').read_text())
    method=choice['selected']['name']
    base=ROOT/'results/takehome/goal7/confirmation'/method
    pooled={}; records=[]; runner=None; new_tests=0
    for seed in a.seeds:
        run=base/f'seed_{seed}'
        original_budget=(run/'budget.json').read_bytes()
        result=json.loads((run/'result.json').read_text())
        for point in result['curve'][1:]:
            end=point['practice_episodes']
            folder=out/f'seed_{seed}'/f'point_{end:04d}'
            folder.mkdir(parents=True,exist_ok=True)
            model_path=folder/'probability_model.json'
            if model_path.exists():
                model=json.loads(model_path.read_text())
            else:
                model=fit_from_practice(run,end)
                write_json(model_path,model)
            if end==200:
                trace_path=run/'test_0200_traces.npz'
            else:
                trace_path=folder/'test_traces.npz'
                if not trace_path.exists():
                    cfg,state=load_model(run/f'model_{end:04d}.pkl')
                    if runner is None:
                        runner=evaluation_runner(cfg)
                    summary,_,_=runner.episodes.test(state,episodes=256,seed=123,output=trace_path)
                    if not np.isfinite(summary['steps_mean']):
                        summary['steps_mean']=summary['steps_median']=None
                    summary.update(practice_episodes=end,reason='Goal8 confidence over training requires traces')
                    baseline=json.loads((ROOT/cfg.baseline_path).read_text())
                    summary['ceiling_verification']=verify_ceiling(runner.episodes,state,summary,
                        outputs=cfg.outputs,ceiling=baseline['conditions']['no_camera_bias']['success_rate'],
                        output=folder/'extra_ceiling.json',existing=run/f'extra_ceiling_{end:04d}.json')
                    write_json(folder/'test_summary.json',summary)
                    new_tests+=256+(summary['ceiling_verification'] or {}).get('new_test_episodes',0)
            with np.load(trace_path) as archive:
                trace={k:archive[k] for k in archive.files}
            summary,pred,y,q=evaluate_trace(model,trace)
            summary.update(seed=seed,practice_episodes=end,trace_path=str(trace_path.relative_to(ROOT)),
                probability_model=str(model_path.relative_to(ROOT)),test_episodes=len(trace['episode_success']),
                predictor_uses_raw_q=False,predictor_uses_true_bias=False,
                raw_q_uses_privileged_bias=bool(model['config'].get('privileged_critic')))
            write_json(folder/'analysis.json',summary)
            records.append(summary)
            pooled.setdefault(end,[]).append((pred,y,q))
            print(json.dumps(dict(seed=seed,practice=end,assessment=summary['average_assessment'],
                                  mean_predicted=summary['mean_predicted'],observed=summary['observed'])),flush=True)
        if (run/'budget.json').read_bytes()!=original_budget:
            raise RuntimeError('Practice budget changed during test-only analysis')
    pooled_rows=[]
    for end,arrays in sorted(pooled.items()):
        pred,y,q=[np.concatenate([item[i] for item in arrays]) for i in range(3)]
        summary=reliability_bins(pred,y)
        summary.update(practice_episodes=end,raw_q_bins=raw_rating_bins(q,y),seed_count=len(arrays))
        pooled_rows.append(summary)
    # Group-specific summaries let several disjoint seed groups be analyzed
    # without writing the same file. The normal invocation uses all five seeds.
    name='summary.json' if a.seeds==list(range(100,105)) else 'summary_'+'_'.join(map(str,a.seeds))+'.json'
    write_json(out/name,dict(method=method,seeds=a.seeds,per_seed=records,pooled=pooled_rows,
        practice_episodes_added=0,new_test_episodes_this_invocation=new_tests,
        training='own counted replay at each checkpoint, no later practice or tests',
        probability_predictor='regularized logistic regression on observation, action and remaining time',
        raw_q_interpretation='discounted bootstrapped return, not episode success probability'))


if __name__=='__main__':
    main()
