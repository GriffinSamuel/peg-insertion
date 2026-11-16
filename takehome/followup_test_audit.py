"""Count real test-only rollouts made by the unchanged software tests."""
from datetime import datetime,timezone
from functools import wraps
import json
from takehome.followup_runtime import OUT


class TestAudit:
    def __init__(self):
        self.events=[]
        self.passed_nodes=[]
        self.failed_nodes=[]

    def install(self):
        from takehome.episodes import EpisodeRunner
        from takehome.followup_episodes import FollowupEpisodes
        from peg_insertion import rollout
        import numpy as np
        for cls in (EpisodeRunner,FollowupEpisodes):
            original=cls.test
            def wrap(method,kind):
                @wraps(method)
                def recorded(runner,*args,**kwargs):
                    result=method(runner,*args,**kwargs)
                    summary=result[0]
                    self.events.append(dict(kind=kind,episodes=summary['episodes'],
                        successes=summary['successes'],test_seed=summary['test_seed'],
                        test_only=True,learned_from=False))
                    return result
                return recorded
            cls.test=wrap(original,cls.__name__)
        original_rollout=rollout.rollout_stats
        @wraps(original_rollout)
        def recorded_rollout(*args,**kwargs):
            result=original_rollout(*args,**kwargs)
            successes=np.asarray(result['succeeded'])>.5
            self.events.append(dict(kind='original_rollout_stats',episodes=int(len(successes)),
                successes=int(successes.sum()),test_seed=kwargs.get('seed',123),
                test_only=True,learned_from=False))
            return result
        rollout.rollout_stats=recorded_rollout

    def pytest_runtest_logreport(self,report):
        if report.when=='call':
            if report.passed:self.passed_nodes.append(report.nodeid)
            elif report.failed:self.failed_nodes.append(report.nodeid)

    def save(self,exit_code):
        # Counts are from the unchanged, inspected test bodies. These probes
        # reset or step briefly and never feed a learner or full test score.
        probe_counts={'tests/test_env.py::test_reset_step_shapes':2,
                      'tests/test_env.py::test_random_steps_stable':1,
                      'tests/test_env.py::test_auto_reset_resamples_hole':6,
                      'takehome/test_correction.py::test_reset_and_frozen_policy_after_test_rollout':8}
        probes={n:probe_counts[n] for n in self.passed_nodes if n in probe_counts}
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
        folder=OUT/'validation'/stamp;folder.mkdir(parents=True)
        data=dict(exit_code=int(exit_code),passed_nodes=self.passed_nodes,failed_nodes=self.failed_nodes,
                  events=self.events,reset_only_probes=sum(probes.values()),reset_probe_sources=probes,
                  reset_count_scope='completed passing probe tests; failed probe counts unknown')
        (folder/'audit.json').write_text(json.dumps(data,indent=2)+'\n')
        if self.events:
            summary=dict(episodes=sum(e['episodes'] for e in self.events),
                successes=sum(e['successes'] for e in self.events),test_seed=123,
                test_only=True,learned_from=False,software_validation=True,events=self.events)
            (folder/'test_0000.json').write_text(json.dumps(summary,indent=2)+'\n')
