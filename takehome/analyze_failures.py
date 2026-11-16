"""Goal 9 fixed failure partition and separately recorded illustrative videos."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from takehome.runtime import ROOT  # noqa: E402
import numpy as np  # noqa: E402
from takehome.config import TrainConfig  # noqa: E402
from takehome.train import ExperimentRunner,write_json  # noqa: E402
from takehome.artifacts import load_model,evaluation_runner  # noqa: E402
from takehome.pose_episodes import PoseEpisodeRunner  # noqa: E402
from takehome.failure_rules import classify,ORDER,RULES  # noqa: E402
from takehome.evaluation_checks import verify_ceiling  # noqa: E402


def load_trace(path):
    with np.load(path,allow_pickle=False) as a:
        trace={k:a[k] for k in a.files}
    if not bool(trace['test_only']) or bool(trace['learned_from']):
        raise ValueError('Failure analysis needs test-only traces')
    return trace


def record(runner,state,path,*,outputs=None,ceiling=None,existing_check=None):
    if path.exists():
        return load_trace(path)
    summary,_,_=runner.test(state,episodes=500,seed=456,output=path)
    if not np.isfinite(summary['steps_mean']):
        summary['steps_mean']=summary['steps_median']=None
    summary.update(illustrative_repeat=True,not_used_to_replace_headline=True)
    if outputs is not None:
        summary['ceiling_verification']=verify_ceiling(runner,state,summary,outputs=outputs,
            ceiling=ceiling,output=path.with_name('extra_ceiling_'+path.stem+'.json'),existing=existing_check)
    write_json(path.with_suffix('.json'),summary)
    return load_trace(path)


def main():
    check=ROOT/'results/takehome/goal8_checks.json'
    if not check.exists() or not json.loads(check.read_text()).get('all_goal_checks_pass'):
        raise ValueError('Goal8 must finish before failure analysis')
    choice=json.loads((ROOT/'results/takehome/goal7_choice.json').read_text())
    method=choice['selected']['name']
    ceiling=json.loads((ROOT/'results/takehome/confirmation_baseline.json').read_text())['conditions']['no_camera_bias']['success_rate']
    base=ROOT/'results/takehome/goal7/confirmation'/method
    out=ROOT/'results/takehome/goal9';out.mkdir(parents=True,exist_ok=True)
    unlearned=ExperimentRunner(TrainConfig())
    before_path=out/'before_traces.npz'
    before_trace=record(PoseEpisodeRunner(unlearned.env,unlearned.learner),
                        unlearned.learner.initialize(0),before_path)
    before=classify(before_trace)
    before.update(trace_path=str(before_path.relative_to(ROOT)),test_seed=456,
                  basis='Separately labelled original-policy traced repeat; not a replacement baseline')
    after=[]
    for seed in range(100,105):
        path=base/f'seed_{seed}/test_0200_traces.npz'
        summary=classify(load_trace(path))
        summary.update(practice_seed=seed,test_seed=456,trace_path=str(path.relative_to(ROOT)),
                       basis='Original final confirmation test')
        after.append(summary)
    pooled=dict(episodes=sum(r['episodes'] for r in after),successes=sum(r['successes'] for r in after),
        failures=sum(r['failures'] for r in after),
        counts={k:sum(r['counts'][k] for r in after) for k in ORDER})
    assert sum(pooled['counts'].values())==pooled['failures']
    write_json(out/'failures.json',dict(method=method,before=before,after_per_seed=after,
        after_pooled=pooled,rules=RULES,order=list(ORDER),practice_episodes_added=0,
        headline_baseline='results/takehome/confirmation_baseline.json'))
    jobs=[];examples=[]
    def add_examples(group,seed,summary,trace_path,limits):
        for label in ORDER:
            existing=sum(e['group']==group and e['type']==label for e in examples)
            needed=max(0,limits[label]-existing)
            ids=[i for i,value in enumerate(summary['labels']) if value==label][:needed]
            for episode in ids:
                stem=f'{group}_{seed}_{label}_{episode}'
                video=f'results/takehome/goal9/videos/{stem}.mp4'
                examples.append(dict(group=group,practice_seed=seed,type=label,episode_id=episode,
                    trace_path=str(trace_path.relative_to(ROOT)),video=video,illustrative_repeat=True))
                jobs.append(dict(name=stem,args=['takehome/render_failure.py','--trace',str(trace_path.relative_to(ROOT)),
                    '--episode',str(episode),'--policy','force_0.5mm_s2','--title',f'{group}: {label}',
                    '--output',video],log=f'results/takehome/goal9/videos/{stem}.log'))
    add_examples('before','unlearned',before,before_path,{k:min(2,before['counts'][k]) for k in ORDER})
    wanted={k:min(2,pooled['counts'][k]) for k in ORDER}
    pose=None;repeat_summaries=[]
    for seed in range(100,105):
        if all(sum(e['group']=='after' and e['type']==k for e in examples)>=wanted[k] for k in ORDER):
            break
        cfg,state=load_model(base/f'seed_{seed}/model_0200.pkl')
        if pose is None:
            evaluator=evaluation_runner(cfg)
            pose=PoseEpisodeRunner(evaluator.env,evaluator.learner)
        trace_path=out/f'after_seed_{seed}_poses.npz'
        trace=record(pose,state,trace_path,outputs=cfg.outputs,ceiling=ceiling,
                     existing_check=base/f'seed_{seed}/extra_ceiling_0200.json')
        summary=classify(trace)
        repeat_summaries.append(dict(seed=seed,episodes=500,counts=summary['counts'],
            successes=summary['successes'],trace_path=str(trace_path.relative_to(ROOT))))
        add_examples('after',seed,summary,trace_path,wanted)
    missing={k:max(0,wanted[k]-sum(e['group']=='after' and e['type']==k for e in examples)) for k in ORDER}
    write_json(out/'video_manifest.json',dict(jobs=jobs,examples=examples,pose_repeat_summaries=repeat_summaries,
        missing_after_examples=missing,repeatability_limitation='Illustrative repeats may differ from scored episodes',
        suite_summary='results/takehome/goal9/video_suite.json'))
    print(json.dumps(dict(failure_count_before=before['failures'],failure_count_after=pooled['failures'],
                          videos=len(jobs),missing=missing)),flush=True)


if __name__=='__main__':
    main()
