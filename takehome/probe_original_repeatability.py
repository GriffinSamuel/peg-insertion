"""Repeat the unchanged original evaluator; diagnostic tests never enter replay."""
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from takehome.runtime import ROOT
import jax
import numpy as np
from peg_insertion import rollout


def main():
    policy, cfg = rollout.load_policy(ROOT / 'results/policies/force_0.5mm_s2')
    overrides = {k: v for k, v in cfg['env_overrides'].items()
                 if k not in ('clearance_mm', 'use_force_obs', 'naconmax')}
    env = rollout.make_env(cfg['clearance_mm'], cfg['use_force_obs'],
                           naconmax=20 * 256, **overrides)
    output = ROOT / 'results/takehome/repeatability/original_evaluator'
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for repeat in range(3):
        start = time.perf_counter()
        stats = rollout.rollout_stats(env, jax.jit(policy), 500, seed=123,
                                      randomize=True, batch=256)
        assert all(np.isfinite(v).all() for v in stats.values())
        summary = rollout.summarize(stats)
        records.append({'repeat': repeat, **summary, 'seconds': time.perf_counter() - start,
                        'success_bits': stats['succeeded'].astype(int).tolist()})
        np.savez_compressed(output / f'episodes_{repeat}.npz', **stats)
        result = {'test_only': True, 'learned_from': False, 'practice_episodes': 0,
                  'test_seed': 123, 'test_only_episodes': 500 * len(records),
                  'original_evaluator_unchanged': True, 'repeats': records}
        (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps({k: v for k, v in records[-1].items() if k != 'success_bits'}), flush=True)


if __name__ == '__main__':
    main()
