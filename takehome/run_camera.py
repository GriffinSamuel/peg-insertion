"""One predeclared Goal 11 condition: references, transfer and three fresh runs."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from takehome.runtime import ROOT  # noqa: E402
import numpy as np  # noqa: E402
from takehome.artifacts import load_model  # noqa: E402
from takehome.camera import CameraConfig,CameraEnv,CameraExperimentRunner,PROBLEMS,FORMULAS,cap_multiplier  # noqa: E402
from takehome.correction import AimConfig  # noqa: E402
from takehome.episodes import EpisodeRunner  # noqa: E402
from takehome.learner import Learner  # noqa: E402
from takehome.train import write_json  # noqa: E402
from takehome.train_variants import check_seed_outcomes  # noqa: E402
from takehome.evaluation_checks import verify_ceiling  # noqa: E402


def evaluate(runner,state,path,*,ceiling=None,outputs=None,**metadata):
    if path.exists():
        return json.loads(path.read_text())
    summary,_,_=runner.test(state,episodes=500,seed=123,output=path.with_suffix('.npz'))
    if not np.isfinite(summary['steps_mean']):
        summary['steps_mean']=summary['steps_median']=None
    summary.update(metadata)
    if outputs is not None:
        summary['ceiling_verification']=verify_ceiling(runner,state,summary,outputs=outputs,
            ceiling=ceiling,output=path.with_name('extra_ceiling_'+path.name))
    write_json(path,summary)
    return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--problem',choices=PROBLEMS,required=True)
    a=p.parse_args()
    gate=ROOT/'results/takehome/goal10_checks.json'
    if not gate.exists() or not json.loads(gate.read_text()).get('all_goal_checks_pass'):
        raise ValueError('Goal10 must finish before camera experiments')
    choice=json.loads((ROOT/'results/takehome/goal7_choice.json').read_text())
    selected=choice['selected']
    sources={r['seed']:ROOT/r['result_path'] for r in selected['results']}
    original_cfg,_=load_model(sources[0].parent/'model_0200.pkl')
    factor=cap_multiplier(a.problem)
    out=ROOT/'results/takehome/goal11'/a.problem;out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():
        print('SKIP completed camera condition',a.problem);return
    context=dict(camera_problem=a.problem,step_cap=original_cfg.step_cap*factor,
        guess_cap=original_cfg.guess_cap*factor,test_seed=123,
        baseline_path=str((out/'baseline.json').relative_to(ROOT)))
    cfg=CameraConfig(**{**asdict(original_cfg),**context})
    declaration=dict(problem=a.problem,formula=FORMULAS[a.problem],config=cfg.to_dict(),
        cap_multiplier=factor,step_cap_m=cfg.step_cap,guess_cap_m=cfg.guess_cap,
        final_method=selected['name'],test_seed=123,practice_seeds=[0,1,2])
    declaration_path=out/'declaration.json'
    if declaration_path.exists() and json.loads(declaration_path.read_text())!=declaration:
        raise ValueError('Camera declaration changed after a condition started')
    write_json(declaration_path,declaration)
    reference_cfg=CameraConfig(camera_problem=a.problem,step_cap=cfg.step_cap,guess_cap=cfg.guess_cap)
    references={}
    for name,oracle in [('floor',False),('oracle',True)]:
        env=CameraEnv(ROOT/'results/policies'/cfg.policy,AimConfig(cfg.step_cap,cfg.guess_cap),
                       config=reference_cfg,oracle=oracle)
        learner=Learner(reference_cfg,env)
        references[name]=evaluate(EpisodeRunner(env,learner),learner.initialize(0),out/f'{name}.json',
            practice_episodes_added=0,cheats_on_information=oracle,
            label='True-error oracle (cheats: actual XY error)' if oracle else 'Original policy under camera problem')
    ceiling=json.loads((ROOT/'results/takehome/baseline.json').read_text())['conditions']['no_camera_bias']
    write_json(out/'baseline.json',dict(policy=cfg.policy,test_seed=123,practice_episodes=0,
        conditions=dict(normal=references['floor'],no_camera_bias=ceiling),
        ceiling_source='Goal1: no constant bias and none of the added problem'))
    runner=CameraExperimentRunner(cfg)
    transfer=[];trained=[]
    for seed in (0,1,2):
        _,state=load_model(sources[seed].parent/'model_0200.pkl')
        transfer.append(evaluate(runner.episodes,state,out/f'transfer_seed_{seed}.json',
            ceiling=ceiling['success_rate'],outputs=cfg.outputs,
            seed=seed,practice_episodes_added=0,trained_source=str(sources[seed].relative_to(ROOT)),
            scaled_step_cap_m=cfg.step_cap,scaled_guess_cap_m=cfg.guess_cap))
    for seed in (0,1,2):
        trained.append(runner.run(seed,out/f'trained/seed_{seed}'))
    check_seed_outcomes(out/'trained',[0,1,2],cfg)
    write_json(out/'result.json',dict(declaration=declaration,floor=references['floor'],oracle=references['oracle'],
        ceiling=ceiling,transfer=transfer,retrained=[dict(seed=r['seed'],final=r['final'],
            result_path=str((out/f"trained/seed_{r['seed']}/result.json").relative_to(ROOT))) for r in trained],
        practice_episodes_added=600,settings_counted_in_goal7=False))
    print(json.dumps(dict(problem=a.problem,floor=references['floor']['success_rate'],
        oracle=references['oracle']['success_rate'],transfer=[r['success_rate'] for r in transfer],
        retrained=[r['final']['success_rate'] for r in trained])),flush=True)


if __name__=='__main__':
    main()
