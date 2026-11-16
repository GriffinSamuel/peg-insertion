"""Rebuild the final report from audited small result files and lead analysis."""
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from takehome import report_tables as tables  # noqa: E402

OUT = ROOT / 'results/takehome'
SOURCES = {}


def read(name):
    path = OUT / name
    value = json.loads(path.read_text())
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return value


def source(name, title=None):
    return f'[{title or name}](results/takehome/{name})'


def figure(name, title):
    return f'![{title}](results/takehome/{name})'


def main():
    if not read('goal11_checks.json')['all_goal_checks_pass']:
        raise ValueError('Goal11 must finish before the final write-up')
    data = {key: read(name) for key, name in tables.SOURCES.items()}
    baseline = read('baseline.json')['conditions']
    mechanism = read('mechanism_check.json')['conditions']
    confirmation = data['confirmation']
    choice = read('goal7_choice.json')
    final = next(row for row in confirmation['methods'] if row['final_method'])
    protocol = read('protocol.json')
    accounting = read('episode_accounting.json')
    diagnosis = read('repeatability_conclusion.json')
    timings = read('benchmark_timings.json')['groups']
    force = read('goal7/force_penalty_comparison.json')
    calibration_checks = read('goal8_checks.json')
    videos = read('goal9/video_manifest.json')
    checks9 = read('goal9_checks.json')
    transfer_analysis = read('goal10_interpretation.json')
    camera_analysis = read('goal11_interpretation.json')
    figures = read('figure_rebuild_inventory.json')
    decisions = read('report_choice_inventory.json')
    environment = read('environment.json')
    read('method_sources.json')
    sections = []
    def add(text):
        sections.append(text.strip())

    add(f'''# Improving a frozen peg-insertion policy with 200 practice episodes

The selected method corrects the robot's hand movements. On the fresh confirmation test it achieved **{100*final['mean']:.2f}%** success, with a sample standard deviation of **{100*final['std']:.2f} percentage points** across five practice seeds. The original policy scored {100*confirmation['floor']:.1f}%; removing its constant camera bias gave {100*confirmation['ceiling']:.1f}%. Each learner used exactly 200 new practice episodes, including fill-up. The chosen method had the benefit of a {choice['total_settings_tried']}-setting search. {source(tables.SOURCES['confirmation'], 'Confirmation results')}.

The original idea was to learn the camera error and correct the policy's aim. That mechanism works when given the true error, but the learned aim corrector did not establish a clear gain at this budget. Direct movement correction was stronger. This is a simulation result with noticeable variation between runs, not evidence of reliable real-robot performance.''')

    add('''## Question and experimental contract

Can a small learner improve a frozen, partly successful peg-insertion policy within 200 practice episodes? The starting checkpoint is `force_0.5mm_s2`. A frozen policy keeps the same network weights throughout; a new network supplies bounded corrections around it. The old policy, simulator, original results and tests remain unchanged.

The budget counts every episode used for learning, including the first 40 episodes used to fill the memory bank. The remaining 160 episodes use the learner. A persistent counter reserves batches of 10 before collection and refuses episode 201. Test episodes are labelled separately and never enter the memory bank. Practice and tests use the same original randomization of friction, mass and controller gains. Each experiment has its own budget; the entire search is much larger than 200 episodes.

Success uses the original environment rule: the peg ever reaches 24 mm depth in a 30 mm hole, while meeting the original 10.5 mm radial in-hole threshold. The test also retains the original wall-passage diagnostic. A task episode lasts at most 200 control steps of 20 ms each. These operational rules are kept unchanged; the radial success threshold is not a newly imposed tighter geometric tolerance.

Development uses practice seeds 0-4 and test seed 123; screening and transfer use practice seeds 0-2. Final tests have 500 episodes per seed. Learning curves show the original policy at zero practice, then 256 test episodes every 20 practice episodes after fill-up, and the full 500 at 200. Confirmation uses practice seeds 100-104 and test seed 456, only after the final choice was written. Its original-policy and no-bias references are measured afresh. Per the logged reading of the definitions, intermediate confirmation curves still use 256 tests at seed 123; their endpoints use 500 at 456.

Spread means sample standard deviation across practice seeds. A difference smaller than the larger matched spread is labelled unproven. This is the predeclared descriptive rule, not a formal significance test or a confidence interval. All results use the final 200-episode checkpoint, including bad runs.''')
    add(source('protocol.json', 'Exact protocol, units, seeds and success rule'))

    add('''## Method and reasons

An actor is the network that proposes a correction. A critic, also called the scorekeeper or Q-function, predicts discounted future reward for that choice. A replay buffer is the memory bank of past practice steps. This is an off-policy actor-critic method: it learns from stored behavior, including earlier versions of itself. XY denotes sideways position and Z denotes height.

The main learner has two 64-unit hidden layers in its actor and in each of four critics. Layer normalization rescales internal activations in each critic to help keep training stable. Critics learn with squared error; the actor increases the mean critic rating. Four critic updates are made per valid new learner step, and the actor updates every fourth critic update. The target uses the smaller rating from two randomly selected target critics. Target networks are slowly updated copies, mixed at rate 0.005 to make training targets less abrupt. Adam, a training rule that adapts the step size for each network weight, uses learning rate 0.0003. The combined size of the learning gradients is capped at 10; discount is 0.99. Discounting values earlier success more than later success. These choices adapt repeated replay and critic-ensemble ideas from [REDQ](https://arxiv.org/abs/2101.05982) and equal-source replay with normalized critics from [RLPD](https://proceedings.mlr.press/v202/ball23a.html). This implementation uses a deterministic actor and is not an exact implementation of either paper; its fill-up is charged to the budget.

Whenever both fill-up and learner data exist, each 128-step study batch contains exactly 64 of each. If only one source exists, all 128 come from it. The old policy's input normalization is reused; corrections and the running guess are divided by their physical caps before entering the critic. Gaussian (bell-shaped) practice noise has normalized standard deviation 0.5 during fill-up and 0.2 thereafter, clipped to the action bounds. Test actions contain no added exploration noise. Each actor's output layer starts at exactly zero.

The aim learner adds a capped XY nudge to a running guess and subtracts that guess from the camera reading. Its nudge cap is 0.1 mm per axis and its guess cap is 3 mm per axis. The movement learner adds up to 0.2 mm per axis to the old policy's movement command before the original action bounds are applied. The original action scale is 2 mm. Combined control has two aim outputs and three movement outputs. The final method uses only the three movement outputs, so its camera guess stays zero.

The selected movement method has this control flow. Dashed arrows are learning updates, not extra information available during testing.

```mermaid
flowchart LR
    O[Normal sensor reading] --> F[Frozen old policy]
    O --> A[Movement learner]
    F --> S[Add correction and clip to original action bounds]
    A --> S
    S --> E[Unchanged simulator]
    E --> O
    E -->|Practice steps only| R[Counted replay buffer]
    R -.-> Q[Critic ensemble]
    Q -.-> A
```

Reward is one at the first original-environment success and zero otherwise. Success ends the learner's episode. A time limit is stored separately and retains the critic's continuation target. The memory bank retains episode and step IDs, fill/learner source, old policy action, termination and truncation, and true camera bias for analysis. That true bias is excluded from normal actor and critic inputs. Only the explicitly labelled privileged-critic and supervised-reference experiments may use it. Tests never train either the learner or the later confidence predictor.''')
    add(source('protocol.json', 'Method parameters')+'; '+source('method_sources.json', 'Primary paper references'))
    add(tables.table(['Method','Learned outputs','Information available','Practice budget'],[
        ['Aim only','Two XY nudges to the running camera-error guess','Normal sensors and own guess','40 fill + 160 learner episodes'],
        ['Movement only (selected)','Three XYZ additions to the old movement command','Normal sensors; aim guess stays zero','40 fill + 160 learner episodes'],
        ['Combined','Two aim nudges and three movement additions','Normal sensors and own guess','40 fill + 160 learner episodes'],
        ['Supervised reference (cheats on information)','Two aim nudges trained against the true error','True camera-error labels during learning','Same counted 200 episodes'],
    ]))

    add('## Starting line and mechanism checks')
    rows=[]
    for label,row in [('Original policy',baseline['normal']),('No constant camera bias',baseline['no_camera_bias']),('Zero correction',mechanism['zero']),('True-error oracle (cheats on information)',mechanism['oracle'])]:
        rows.append([label,tables.result_cell(row['successes'],row['episodes']),row['test_seed']])
    add(tables.table(['Check','Successes / tests','Test seed'],rows))
    add(f'''The zero-correction and true-error checks passed their written tolerances. This supports the wrapper and the usefulness of correct aim, but does not show that a small learner can infer that error from its observations. The normal camera has fixed XY bias with standard deviation 0.75 mm per episode and stepwise white noise with standard deviation 0.2 mm. Removing constant bias retains white noise. {source('baseline.json')}; {source('mechanism_check.json')}; {source('protocol.json')}.''')

    add('## Development results, including all seeds')
    add(tables.table(['Method',*[f'Seed {s}' for s in range(5)],'Seeds','Mean ± sample SD'],tables.goal6_table(data['goal6'])))
    add(f'''All counts above use 500 tests at seed 123. Aim-only averaged 65.16%, a 3.76-point gain over the 61.4% floor, smaller than its 7.74-point seed spread. Movement averaged 83.36% and combined averaged 78.60%. Movement was best; adding aim did not improve it. The supervised reference uses true camera-error labels on the same counted practice and is explicitly a reference that cheats on information. It is not the perfect-error oracle and is not a guaranteed upper bound. {source(tables.SOURCES['goal6'])}.''')
    add(figure('goal5_learning_curves.png','Main aim learning curves'))
    add(figure('goal6/comparison.png','Development method comparison'))
    add(figure('goal6/learning_curves.png','All development comparison learning curves'))

    add('## Exploration and choice before confirmation')
    add(f'''The study considered {choice['total_settings_tried']} settings: the four earlier methods, including the ineligible supervised reference, plus 24 new screening settings. All requested exploration families were covered. Each new setting changed one feature from aim-only, or from combined control when the feature needed movement. The 40-fill case reused the main result. The table keeps all seeds, peak forces and differences smaller than the matched spread. Seeds 3 and 4 are marked not run for the three-seed screen, as specified in advance.''')
    add(tables.table(['Setting',*[f'Seed {s}' for s in range(5)],'Seeds','Mean ± sample SD','Difference from relevant base','Mean peak force by seed'],tables.goal7_table(data['goal7'])))
    add(f'''No new trick beat its relevant base beyond both matched sample spreads. Therefore no trick qualified for extension or combination, and round 2 had no jobs. This is the written eligibility rule, not an omitted experiment for time. Some movement-capable methods did beat main aim, so the separate fallback that would force aim-only did not apply. The fixed rule selected the highest mean, then the simplest method within one spread of that top mean. Plain movement was the only eligible method in that band. The choice was written before any confirmation reference or learner ran. {source('goal7_choice.json')}; {source('goal7_round2_manifest.json')}; {source('goal7_checks.json')}.

The force-penalty screen reduced the matched mean peak force from 91.79 N to 88.16 N, while mean success fell from 77.47% to 67.47%. Both differences are smaller than the corresponding seed spread; neither is established as a reliable effect. {source('goal7/force_penalty_comparison.json')}.''')
    add(figure('goal7/exploration.png','Every explored setting'))

    add('## Fresh confirmation: the headline')
    add(tables.table(['Method and selection advantage',*[f'Seed {s}' for s in range(100,105)],'Seeds','Mean ± sample SD'],tables.confirmation_table(confirmation)))
    add(f'''The final movement result is {100*final['mean']:.2f}%, {100*(final['mean']-confirmation['floor']):.2f} percentage points above the fresh original-policy reference. Its five runs span 72.0%-88.6%. It is below its development average, and remains the headline. Aim-only's 1.96-point and combined's 6.88-point gains over the floor are below their respective spreads. The final method benefited from selection across 28 settings; the other two plain comparisons each used one untuned setting. All use their final checkpoint, not their best learning-curve point. The early aim/combined collapses and recovery remain visible; these observations do not identify a cause. {source(tables.SOURCES['confirmation'])}; {source('confirmation_baseline.json')}.''')
    add(figure('goal7/confirmation/confirmation.png','Fresh confirmation'))
    add(figure('goal7/confirmation/confirmation_learning_curves.png','All confirmation learning curves'))

    add('## Can confidence be trusted?')
    add('''Raw Q is not an episode-success probability: it discounts later success and uses an estimated continuation value at a time limit. A separate logistic predictor was fitted on each run's own counted practice, using normal observations, proposed action and remaining time. Logistic regression maps a weighted combination of those inputs to a probability. It uses no true bias, raw Q or test labels, and adds no practice. All 45 fixed fits converged. Intermediate diagnostics need full test traces, so those local traces are the explicit later-analysis exception to final-only storage.

The final predictor is underconfident on average by 5.23 percentage points. It predicts 69.54% against 74.77% observed over 392,329 valid steps from the 2,500 confirmation episodes. Only steps up to learner termination are valid. These are step-weighted statistics: longer failures contribute more steps. They must not be substituted for the 80.20% episode-level headline. Expected calibration error, the weighted absolute gap across ten probability bins, is 6.32 points. The Brier score, mean squared probability error with zero as perfect, is 0.15235.

Of 495 failed episodes, 355 contain a prediction of at least 80%. Only 15 show the predeclared drop of at least 25 points between adjacent five-step averages. These are counts of selected moments, not an alarm precision estimate. Missing drops are explicitly null. The raw-Q plot has no probability-identity line and is not monotonic against eventual success after pooling states and times.''')
    add(tables.table(['Seed/group','Episode result','Valid steps','Mean prediction','Observed step-weighted rate','Calibration error','Brier','Signed bias'],tables.calibration_table(data['calibration'])))
    add(source('goal8/summary.json')+'; '+source('goal8_checks.json')+'; '+source('goal8/confident_failed.csv','Confident failed moments')+'; '+source('goal8/confidence_drops.csv','Confidence drops'))
    for name,title in [('reliability_final.png','Probability reliability'),('raw_q_final.png','Raw critic rating'),('calibration_over_training.png','Confidence through training')]:
        add(figure('goal8/'+name,title))

    add('## Failure cases')
    names,rows=tables.failures_table(data['failures'])
    add(tables.table(['Group','Practice seed','Successes / tests',*names],rows))
    add('''The before trace is a separate 303/500 original-policy repeat at seed 456, not a replacement for the 304/500 headline reference. After counts use the original final confirmation traces. The rules below apply in order, and the first match wins. A residual timeout excludes failures assigned an earlier pattern; other remains available even when its count is zero.''')
    add('\n'.join(f'{i}. **{name.replace("_"," ")}**: {data["failures"]["rules"][name]}' for i,name in enumerate(names,1)))
    add('''These are operational patterns, not proven causes. The near-contact proxy requires distance and force together; it does not mean the peg never approached the hole. Wrong-way correction cannot occur without a correction, so its increase from zero is not a controlled estimate of harm. Its outward-direction calculation retains camera white noise.

Thirteen illustrative videos cover every type that occurred: six before and seven after, with one example for the single rare after near-contact failure and no invented examples for zero-case types. They replay recorded poses through the existing video code. All have readable guess/true-error overlays and passed metadata and decoded-frame checks. The movement-only guess correctly remains zero. Separate pose repeats score 303/500 before and 382,391,419,349/500 for after practice seeds 100-103; these outcomes stay separate from the headline. All ending previews and temporal samples of two representative videos were reviewed. No gross wall passage was visible in those samples; the saved physical diagnostics report none. This does not prove perfect simulator fidelity.''')
    add(source('goal9/failures.json')+'; '+source('goal9/video_manifest.json')+'; '+source('goal9/video_audit.json')+'; '+source('goal9_checks.json'))
    add(figure('goal9/failure_counts.png','Failure counts, each seed and pooled'))
    add(figure('goal9/before_examples.png','Original-policy video previews'))
    add(figure('goal9/after_examples.png','Final-method video previews'))

    add('## Other policies, with the final recipe unchanged')
    add(tables.table(['Policy',*[f'Seed {s}' for s in range(3)],'Seeds','Mean ± sample SD','Own floor','Own ceiling'],tables.transfer_table(data['transfer'])))
    add('Each policy has its own references measured first, and each fresh practice seed has its own 200-episode budget. Position-only input has 10 original sensor numbers instead of 16; the same wrapper accepts both. These are fresh training runs with the same recipe, not zero-practice transfer of the main policy\'s learner weights.')
    add('\n\n'.join(transfer_analysis['paragraphs']))
    add(source(tables.SOURCES['transfer'])+'; '+source('goal10_interpretation.json'))
    add(figure('goal10/transfer.png','The three other policies'))

    add('## Different camera problems')
    add('''The unchanged final method was tested under eight predeclared conditions. Transfer reuses normal-condition models from practice seeds 0-2 with no new learning. Retrain starts fresh under that condition with 200 own practice episodes per seed. The floor is the original policy under the new problem; the ceiling reuses Goal 1's no-bias, no-added-problem test. Each true-error oracle cheats on information and corrects the actual XY error at that moment; it cannot directly repair a Z error. Every displayed final score uses 500 tests at seed 123.

For larger offsets, the aim nudge and guess caps scale with the offset in both transfer and retrain, as required. The final movement-only method does not use those aim outputs; its 0.2 mm movement cap stays unchanged. Height error remains included with all three movement outputs. The expectation that height must fail applied to aim-only control and is not imposed on the selected method.''')
    add(camera_analysis['formula_symbols'])
    add(tables.table(['Condition','Predeclared formula and caps'],[[r['problem'],r['formula_and_caps']] for r in camera_analysis['declarations']]))
    add(tables.table(['Condition','Floor','Ceiling','Oracle (cheats)',*[f'Transfer s{s}' for s in range(3)],'Transfer mean ± SD',*[f'Retrain s{s}' for s in range(3)],'Retrain mean ± SD'],tables.camera_table(data['camera'])))
    add('\n\n'.join(camera_analysis['paragraphs']))
    add(source(tables.SOURCES['camera'])+'; '+source('goal11_interpretation.json')+'; '+source('goal11_manifest.json'))
    add(figure('goal11/camera_comparison.png','All eight camera conditions'))

    add('## Repeatability, full accounting and limits')
    add(f'''The authorized repeatability investigation lasted {diagnosis['elapsed_seconds']//60} minutes {diagnosis['elapsed_seconds']%60} seconds before Goal 6. Even the original evaluator, before learning, produced 303,309,307 successes on three identical 500-episode seed-123 runs. Deterministic GPU atomic modes made captured resets agree but did not make full contact trajectories repeat; the lowest-capacity mode also produced overflow warnings and was rejected. No safe complete fix was established within the time box. Production execution, tests, seeds and budgets were left unchanged, and work continued under Griffin's explicit instruction. {source('repeatability_conclusion.json')}; {source('repeatability_findings.md','Detailed diagnosis')}.

Across the study so far, the ledger contains **{accounting['practice']['practice_episodes_reserved']:,} practice episodes** in {accounting['practice']['experiment_ledgers']} experiment ledgers and **{accounting['full_test_episodes_total']:,} full test-only episodes**. Reset-only probes total {accounting['reset_only_probes_total']}; shorter physics probes are listed separately in the final test inventory. There are 28 searched settings. Other-policy and camera experiments reuse the chosen recipe and are listed as new conditions, not additional tuning settings. Reused references and traces are counted once. {source('episode_accounting.json')}.

The result is simulation-only, uses one laptop GPU, and has few practice seeds. Reusing a development test for many choices creates selection advantage, which is why the independent confirmation headline and number of tried settings are shown. The spread rule does not correct for multiple comparisons or provide a formal statistical guarantee. A single no-bias reference is a useful comparison, not a hard upper bound for methods that change hand movements. Confidence models learn from changing practice behavior; their errors need not transfer to a new robot or camera. Failure labels are descriptive and depend on thresholds. There is no evidence here of hardware safety or real-world transfer.''')

    add('## Runtime and every additional run')
    add(tables.table(['Benchmark','Processes','Runs','Total seconds','Seconds per run'],[[r['name'],r['process_count'],r['experiments'],f"{r['seconds']:.2f}",f"{r['seconds_per_experiment']:.2f}"] for r in timings]))
    add('Four concurrent GPU processes gave the best measured throughput, so suites use up to four independent workers and reuse one process across seeds of a setting. Helpers shorten routine code, tables, plots and checking; they do not create more GPU capacity. Startup and compilation are included in the benchmark. '+source('benchmark_timings.json'))
    designated={r['result_path'] for setting in data['goal7']['settings'] if setting['name']=='aim' for r in setting['results']}
    extras=[]
    for path in sorted((OUT/'benchmark').glob('*/seed_*/result.json')):
        rel=str(path.relative_to(ROOT))
        if rel not in designated:
            result=json.loads(path.read_text()); SOURCES[rel]=hashlib.sha256(path.read_bytes()).hexdigest()
            extras.append([path.parent.parent.name,result['seed'],result['practice_episodes'],tables.result_cell(result['final']['successes'],500),f'[{rel}]({rel})'])
    add('The timing repeats below are retained separately and never replace the designated main runs.')
    add(tables.table(['Timing group','Practice seed','Practice episodes','Final test result','Source'],extras))
    trial=read('goal4_trial/seed_999/result.json')
    add(f'''The code trial used practice seed 999, {trial['practice_episodes']} practice episodes and {trial['final']['successes']}/{trial['final']['episodes']} test successes. It was a code check, not a searched setting. Diagnostic repeatability outcomes, including bad runs, are all below. They use no learning; the small probes are intentionally separate from the main 500-episode test. {source('goal4_trial/seed_999/result.json')}; {source('repeatability_conclusion.json')}.''')
    add(tables.table(['Diagnostic','Full test episodes','Success counts by repeat','Reset-only probes'],[[r['tag'],r['test_only_episodes'],', '.join(map(str,r['success_counts'])),r['reset_only_probes']] for r in diagnosis['runs']]))
    extras=[]
    for path in sorted(OUT.glob('**/extra_ceiling*.json')):
        if not any(part in ('goal7','goal8','goal9','goal10','goal11','goal13') for part in path.parts):
            continue
        result=read(str(path.relative_to(OUT)))
        if result.get('test_only'):
            extras.append([str(path.relative_to(OUT)),tables.result_cell(result['successes'],result['episodes']),result['test_seed'],result.get('through_wall_episodes'),result['used_for_selection']])
    if extras:
        add('Every independent ceiling-exceedance check is retained below. These extra seed-789 tests and physical diagnostics verify a movement-capable result; they do not replace or select a score.')
        add(tables.table(['Check file','Successes / tests','Test seed','Flagged wall-passage episodes','Used for selection'],extras))
        add(source('ceiling_reviews/summary.json','Recorded-trajectory physical reviews')+'. Rebuild the inspection charts from local traces with `source scripts/env.sh && python -B takehome/plot_ceiling_checks.py`.')
    else:
        add('No independent ceiling-exceedance check was required in the recorded runs.')
    add('All phase reports, preflight results and passed/failed implementation checks remain in [TAKEHOME_PROGRESS.md](TAKEHOME_PROGRESS.md). The synthetic failure-rule fixture was corrected once to represent its intended timeout; no experimental rule or outcome changed. No exploration family was dropped for time.')

    add('## Rebuild the figures and preserve local artifacts')
    add('''Run from the project root inside WSL. Every command starts with the project environment. Statistical figures rebuild from committed small summaries. Full videos, full traces, replay arrays and model checkpoints remain on disk only and are ignored by git. Video previews therefore require the local videos; video rendering requires the local pose traces. A fresh clone can reproduce a new independent final-method run with its own ledger using:

```bash
source scripts/env.sh && python -B takehome/reproduce.py --seed 100 --output results/takehome/reproductions/local_check
```

Reusing that output directory resumes or skips the same run; choose a new directory for another fresh experiment. It does not replace an earlier result. The final artifact manifest records which inputs are local-only.''')
    add(read('figure_rebuild_commands.json')['markdown'])
    add('The environment and installed versions are recorded in '+source('environment.json')+'. Detached jobs survive a closed terminal but not a WSL shutdown. Account rate limits were unavailable and no fixed session deadline was visible; uninterrupted execution cannot be guaranteed. Preflight evidence is in '+source('preflight_verification.json')+'.')

    add('## Final validation and remaining questions')
    add('Read the headline and fresh-confirmation table first, then the method and the repeatability limits. The policy and camera sections show where the gain does and does not carry over. The full decision appendix and dated progress log preserve the implementation choices and every fallback for review.')
    final_checks_path=OUT/'goal13_checks.json'
    if final_checks_path.exists():
        check=read('goal13_checks.json')
        rerun=read('goal13/rerun_comparison.json')
        add(check['report_paragraph'])
        add(f"Fresh practice seed {rerun['practice_seed']} scored {rerun['successes']}/500 ({100*rerun['success_rate']:.2f}%), compared with confirmation mean {100*rerun['confirmation_mean']:.2f}% and spread {100*rerun['confirmation_sample_spread']:.2f} points. Within one spread: **{'yes' if rerun['within_one_spread'] else 'no'}**. The original confirmation files were unchanged. "+source('goal13/rerun_comparison.json'))
    else:
        add('Goal 13 validation is pending. The report does not yet claim that the final full test suite or fresh rerun has passed.')
    add('''The continuation fallback used was Griffin's explicit instruction to log unresolved repeatability and carry on. The written search eligibility rule left no qualifying combinations, so round 2 had no jobs; no exploration family was cut for time. The hovering policy's identical zero outcomes used the explicit zero-floor-and-ceiling exemption. Movement results above the no-bias reference used the required extra test and physical inspection. The mechanism-check fallback was not needed. Goal 9 showed the one available example of its rare failure type and no examples for types with zero cases, with those limits reported rather than inventing footage. The exact 50/50 sampling rule uses its authorized single-buffer exceptions when the other buffer is empty.''')
    add('''Open questions are why aim-only learning collapses early despite a strong true-error oracle, which simulator operations still prevent exact repetition, whether the movement gain survives more independent seeds and real hardware, and whether an episode-probability model can give useful early failure warnings. These are not answered by the present experiments. No further tuning was done after selection.

AI assistance was used throughout. The lead Codex agent designed the learner and protocol implementation, diagnosed problems, interpreted results, reviewed helper code and wrote this report. Cheaper helper agents performed narrow routine implementation, experiment execution, aggregation, plotting, rendering and audits. They did not choose the winner, change seeds or budgets, or edit the decision log. Griffin supplied and revised the plan, resolved the sampling exceptions and authorized continuation with the repeatability limitation. Griffin should review and be able to explain the report before submitting it.''')

    add('## Complete decision log and plan gaps')
    add('The dated entries below include historical choices and their reasons, including every PLAN GAP. Later explicit decisions supersede earlier planning assumptions; historical counts are not the final cumulative totals. Exact source text and line numbers are captured in '+source('report_choice_inventory.json')+'.')
    decision_text=(ROOT/'TAKEHOME_DECISIONS.md').read_text()
    bullets=[line for line in decision_text.splitlines() if line.lstrip().startswith('- ')]
    add('\n'.join(bullets))

    report='\n\n'.join(sections)+'\n'
    report=report.replace('\u2014','; ')
    (ROOT/'TAKEHOME_REPORT.md').write_text(report)
    provenance=dict(result_sources=SOURCES,decision_bullets=len(bullets),
        plan_gap_entries=sum('PLAN GAP' in line for line in bullets),
        has_em_dash='\u2014' in report,report_sha256=hashlib.sha256(report.encode()).hexdigest(),
        generated_from='takehome/write_report.py',number_policy='Tables derive from raw summaries; prose measurements link to result files; all parameter choices are recorded in protocol and the complete decision inventory.')
    (OUT/'report_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps(dict(report='TAKEHOME_REPORT.md',sources=len(SOURCES),decision_entries=len(bullets))))


if __name__ == '__main__':
    main()
