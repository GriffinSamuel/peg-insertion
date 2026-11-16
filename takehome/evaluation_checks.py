"""Apply the written ceiling rule to later test-only evaluation rollouts."""
from pathlib import Path
import json
import numpy as np
from takehome.train import write_json


def verify_ceiling(runner, state, summary, *, outputs, ceiling, output, existing=None):
    rate = summary['success_rate']
    if outputs == 2:
        if ceiling > .5 and rate > ceiling + .05:
            raise RuntimeError('Tripwire: aim-only learner exceeds ceiling by more than five points')
        return None
    if rate <= ceiling:
        return None
    # A prior check of this exact saved model already supplies the required
    # independent test seed. Reusing it does not collect or count new episodes.
    for path in (existing, output):
        if path is not None and Path(path).exists():
            checked = json.loads(Path(path).read_text())
            if not (checked['test_seed'] == 789 and checked['episodes'] == 500
                    and checked['test_only'] and not checked['learned_from']
                    and checked['used_for_selection'] is False):
                raise ValueError('Existing ceiling verification lacks the fixed test contract')
            return dict(path=str(path), reused=True, new_test_episodes=0)
    output = Path(output)
    extra, stats, trace = runner.test(state, episodes=500, seed=789,
                                      output=output.with_suffix('.npz'))
    if not np.isfinite(extra['steps_mean']):
        extra['steps_mean'] = extra['steps_median'] = None
    examples = np.flatnonzero(stats['succeeded'] > .5)[:5]
    extra.update(used_for_selection=False, inspected_episode_ids=examples.tolist(),
        max_lateral_when_deep_examples_m=stats['max_lateral_when_deep'][examples].tolist(),
        max_abs_control_examples=[float(np.max(np.abs(trace['executed_action'][:,i])))
                                  for i in examples],
        reason='Movement-capable evaluation exceeded its fixed ceiling reference')
    write_json(output, extra)
    return dict(path=str(output), reused=False, new_test_episodes=500)
