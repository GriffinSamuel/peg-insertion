"""Read-only simulation probes: never insert their data into replay or learn."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from takehome.runtime import ROOT
if os.environ.get('TAKEHOME_PROBE_WARP_DETERMINISTIC') == '1':
    import warp as wp
    wp.config.deterministic = getattr(wp.DeterministicMode,
        os.environ.get('TAKEHOME_PROBE_WARP_MODE', 'RUN_TO_RUN'))
    wp.config.deterministic_debug = True
    wp.config.deterministic_max_records = int(os.environ.get('TAKEHOME_PROBE_MAX_RECORDS', '0'))
import jax
import numpy as np
from peg_insertion import rollout
from takehome.config import TrainConfig
from takehome.correction import AimCorrectedEnv
from takehome.episodes import EpisodeRunner
from takehome.learner import Learner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--test-episodes', type=int, choices=(0, 500), default=0)
    args = parser.parse_args()
    output = ROOT / 'results/takehome/repeatability' / args.tag
    output.mkdir(parents=True, exist_ok=True)
    env = AimCorrectedEnv(ROOT / 'results/policies/force_0.5mm_s2')
    learner = Learner(TrainConfig(), env)
    runner = EpisodeRunner(env, learner)
    state = learner.initialize(0)
    if args.test_episodes:
        records = []
        for repeat in range(args.repeats):
            started = time.perf_counter()
            summary, stats, trace = runner.test(state, episodes=500, seed=123,
                output=output / f'test_{repeat}_traces.npz')
            np.savez_compressed(output / f'test_{repeat}_episodes.npz', **stats)
            records.append({'repeat': repeat, **summary,
                'seconds': time.perf_counter() - started,
                'trace_hashes': {k: hashlib.sha256(v.tobytes()).hexdigest()
                                 for k, v in trace.items()}})
            result = {'purpose': 'repeatability diagnosis only', 'test_only': True,
                'learned_from': False, 'practice_episodes': 0,
                'test_only_episodes': 500 * len(records), 'test_seed': 123,
                'repeats': records, 'reset_only_probes': 0}
            (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
            print(json.dumps(records[-1]), flush=True)
        return
    seed_key = jax.random.PRNGKey(0)
    k_rand, k_reset, _ = jax.random.split(seed_key, 3)
    wrapped = rollout._wrap(env, 10, k_rand, randomize=True)
    reset = jax.jit(wrapped.reset)
    keys = jax.random.split(k_reset, 10)
    records = []
    for repeat in range(args.repeats):
        started = time.perf_counter()
        initial = jax.device_get(reset(keys))
        arrays = {'obs': np.asarray(initial.obs['state'])}
        for name in ['qpos', 'qvel', 'qacc', 'sensordata', 'site_xpos',
                     'qfrc_bias', 'qfrc_constraint', 'cfrc_int', 'cfrc_ext', 'cacc']:
            if hasattr(initial.data, name):
                arrays[name] = np.asarray(getattr(initial.data, name))
        np.savez_compressed(output / f'reset_{repeat}.npz', **arrays)
        stats, trace = runner.compiled(state.actor, state.critics, seed_key,
            jax.numpy.arange(10), jax.numpy.array(0), jax.numpy.array(0.5), n=10)
        stats, trace = jax.device_get((stats, trace))
        runner.check_finite(stats, trace)
        np.savez_compressed(output / f'trace_{repeat}.npz', **trace,
                            test_only=True, learned_from=False)
        records.append({'repeat': repeat, 'seconds': time.perf_counter() - started,
                        'successes': int(np.sum(stats['succeeded'])),
                        'reset_hashes': {k: hashlib.sha256(v.tobytes()).hexdigest()
                                         for k, v in arrays.items()}})
        print(json.dumps(records[-1]), flush=True)
    result = {'purpose': 'repeatability diagnosis only', 'test_only': True,
              'learned_from': False, 'practice_episodes': 0,
              'test_only_episodes': 10 * args.repeats,
              'reset_only_probes': 10 * args.repeats, 'seed': 0,
              'repeats': records, 'devices': [str(d) for d in jax.devices()]}
    (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
