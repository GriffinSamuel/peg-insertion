"""Policy-specific, test-only floor/ceiling via the unchanged original rollout."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from takehome.runtime import ROOT  # noqa: E402
import jax  # noqa: E402
import numpy as np  # noqa: E402
from peg_insertion import rollout  # noqa: E402
from takehome.train import write_json  # noqa: E402


def measure(policy_name, test_seed, output):
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT / 'results/takehome'):
        raise ValueError('Output outside allowed directory')
    if output.exists():
        return json.loads(output.read_text())
    if test_seed == 456 and not (ROOT / 'results/takehome/goal7_choice.json').exists():
        raise ValueError('Confirmation reference forbidden before written choice')
    if not any(d.platform == 'gpu' for d in jax.devices()):
        raise RuntimeError('Tripwire: GPU missing')
    output.parent.mkdir(parents=True, exist_ok=True)
    policy, cfg = rollout.load_policy(ROOT / 'results/policies' / policy_name)
    overrides = {k: v for k, v in cfg.get('env_overrides', {}).items()
                 if k not in ('clearance_mm', 'use_force_obs', 'naconmax')}
    conditions = {}
    for name, extra in [('normal', {}), ('no_camera_bias', {'obs_hole_bias_std': 0.})]:
        partial = output.with_name(output.stem + '_' + name + '.json')
        if partial.exists():
            conditions[name] = json.loads(partial.read_text())
            continue
        started = time.perf_counter()
        env = rollout.make_env(cfg['clearance_mm'], cfg['use_force_obs'],
                               naconmax=20*256, **{**overrides, **extra})
        stats = rollout.rollout_stats(env, jax.jit(policy), 500, seed=test_seed, randomize=True, batch=256)
        if not all(np.isfinite(v).all() for v in stats.values()):
            raise RuntimeError('Tripwire: invalid reference simulation numbers')
        summary = rollout.summarize(stats)
        if not np.isfinite(summary['steps_mean']):
            summary['steps_mean'] = summary['steps_median'] = None
        summary.update(test_only=True, learned_from=False, test_seed=test_seed,
                       eval_seconds=time.perf_counter()-started, all_stats_finite=True)
        np.savez_compressed(partial.with_suffix('.npz'), **stats)
        write_json(partial, summary)
        conditions[name] = summary
        print(name, json.dumps(summary), flush=True)
    result = dict(policy=policy_name, test_seed=test_seed, episodes_per_condition=500,
                  practice_episodes=0, conditions=conditions)
    write_json(output, result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--policy', default='force_0.5mm_s2')
    p.add_argument('--test-seed', type=int, default=123)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    measure(a.policy, a.test_seed, a.output)
