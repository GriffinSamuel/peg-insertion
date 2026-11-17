"""Round 3 runner: the follow-up training loop plus version keeping, rollback and shaping.

Thin on purpose: it reuses FollowupRunner for the environment, learner, episode
counter and test runs. Only run() and update() are replaced.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from takehome.followup_runtime import ROOT  # noqa: E402  (must come first)
import json  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
import jax  # noqa: E402
import numpy as np  # noqa: E402
from takehome.budget import EpisodeBudget  # noqa: E402
from takehome.correction import policy_digest  # noqa: E402
from takehome.replay import Replay  # noqa: E402
from takehome.variants import prepare_training_data  # noqa: E402
from takehome.followup_runner import FollowupRunner, write_json, save_checkpoint, validate_state  # noqa: E402
from takehome.round3_config import (ROLLBACK_WINDOW, ROLLBACK_DROP, DEPTH_SCALE,  # noqa: E402
                                    TORQUE_SCALE)

R3 = ROOT / 'results/takehome/round3'
HOLE_SUCCESS_DEPTH = 0.8 * 0.030     # same rule as the environment's success test
DEPTH_IDX, TORQUE_IDX = 9, slice(13, 16)


def potential(obs, mode):
    """Score of a situation from the robot's own readings only (no hidden values)."""
    score = np.zeros(obs.shape[0], dtype=np.float32)
    if mode in ('depth', 'both'):
        score += DEPTH_SCALE * np.clip(obs[:, DEPTH_IDX] / HOLE_SUCCESS_DEPTH, 0., 1.)
    if mode in ('torque', 'both'):
        score -= TORQUE_SCALE * np.minimum(np.linalg.norm(obs[:, TORQUE_IDX], axis=-1), 1.)
    return score


def shaped_rewards(replay, mode, discount):
    """reward + discount*potential(next) - potential(this); terminal potential is 0."""
    n = replay.count
    a = replay.arrays
    reward = a['reward'][:n].copy()
    nxt = potential(a['next_obs'][:n], mode)
    nxt = np.where(a['terminated'][:n] > .5, 0., nxt)
    reward += (discount * nxt - potential(a['obs'][:n], mode)).astype(reward.dtype)
    return reward


def episode_successes(trajectory):
    ok = (np.asarray(trajectory['success']) > .5) & np.asarray(trajectory['valid']).astype(bool)
    return ok.any(axis=0).astype(float)


def pick_version(versions, learner_success):
    """versions: list of end-episode numbers; learner_success: {batch_end: [10 per-episode 0/1]}.
    Score of the version saved at e = mean success of practice learner episodes in
    batches e and e+10 (for e=200: batches 190 and 200). Ties go to the later version."""
    best, best_score = None, -1.
    for e in versions:
        window = [e, e + 10] if e + 10 in learner_success else [e - 10, e]
        vals = np.concatenate([learner_success[b] for b in window if b in learner_success])
        score = float(vals.mean())
        if score >= best_score:
            best, best_score = e, score
    return best, best_score


class Round3Runner(FollowupRunner):
    def update(self, state, replay, new_steps, source, practice_end):
        cfg = self.config
        data = prepare_training_data(replay, cfg)
        if cfg.shaping != 'none':
            reward = data['reward'].copy()
            reward[:replay.count] = shaped_rewards(replay, cfg.shaping, cfg.discount)
            data = {**data, 'reward': reward}
        steps = replay.fill_count if source == 'fill' else new_steps
        return self.learner.train_block(state, data, replay.fill_count, replay.count,
                                        steps * cfg.updates_per_step, actor_enabled=source != 'fill')

    def run(self, seed, output):
        cfg = self.config
        test_seed = cfg.test_seed
        final_check = seed in range(300, 305) or test_seed == 2002
        if final_check:
            if not (R3 / 'final_choice.json').exists():
                raise ValueError('Fresh seeds forbidden before the written final choice')
            if seed not in range(300, 305) or test_seed != 2002:
                raise ValueError('Fresh practice and test seeds must stay paired')
        elif seed not in range(5) or test_seed != 123:
            raise ValueError('Choosing uses practice 0-4 and test 123 only')
        output = Path(output).resolve()
        if not output.is_relative_to(R3):
            raise ValueError('Output must be within results/takehome/round3/')
        if (output / 'result.json').exists():
            print('SKIP completed', output, flush=True)
            return json.loads((output / 'result.json').read_text())
        if output.exists():   # an unfinished run of our own: start over in a clean folder
            shutil.rmtree(output)
        output.mkdir(parents=True)
        write_json(output / 'config.json', cfg.to_dict())
        started = time.perf_counter()
        before = policy_digest(self.env.frozen_policy)
        ledger = EpisodeBudget(output / 'budget.json', limit=cfg.total_episodes)
        replay = Replay(ledger)
        baseline = json.loads((ROOT / cfg.baseline_path).read_text())
        if not str(baseline['policy']).endswith(cfg.policy) or baseline['test_seed'] != test_seed:
            raise ValueError('Reference policy and test seed must match this experiment')
        curve = [{'practice_episodes': 0, 'label': 'original_policy', **baseline['conditions']['normal']}]
        timings, losses, rollbacks = [], [], []
        learner_success = {}        # batch end -> per-episode 0/1 (learner episodes only)
        snapshots = {}              # batch end -> host copy of state AFTER that block's update
        pre_update = {}             # batch end -> state that COLLECTED that batch
        state = self.learner.initialize(seed)
        best_window, best_state, since_restore = -1., None, 0
        for start in range(0, cfg.total_episodes, 10):
            end = start + 10
            source = 'fill' if end <= cfg.fill_episodes else 'online'
            ids = list(range(start, end))
            t0 = time.perf_counter()
            assert ledger.reserve(10, source) == ids
            collector = jax.device_get(state)
            _, trajectory = self.episodes.practice(state, seed, start // 10, ids, source)
            np.savez_compressed(output / f'practice_{end:04d}.npz', **trajectory)
            n = replay.add(trajectory, ids, source, test_only=False)
            if source == 'online':
                learner_success[end] = episode_successes(trajectory)
                pre_update[end] = collector
            collect_seconds = time.perf_counter() - t0
            restored = False
            if cfg.rollback and source == 'online':
                since_restore += 1
                if end - 10 in learner_success and since_restore >= ROLLBACK_WINDOW // 10:
                    window = float(np.concatenate([learner_success[end - 10], learner_success[end]]).mean())
                    if window > best_window:
                        best_window, best_state = window, pre_update[end]
                    elif best_state is not None and window < best_window - ROLLBACK_DROP:
                        state = jax.tree_util.tree_map(jax.numpy.asarray, best_state)
                        restored, since_restore = True, 0
                        rollbacks.append({'at_episode': end, 'window': window, 'best_window': best_window})
            update_seconds = 0.
            if source == 'online' and not restored:
                t0 = time.perf_counter()
                state, metrics = self.update(state, replay, n, source, end)
                metrics = jax.device_get(metrics)
                validate_state(state)
                if not bool(metrics['finite']):
                    raise RuntimeError('Tripwire: invalid numbers during gradient updates')
                losses.append({'practice_episodes': end, 'source': source,
                               **{k: float(v) for k, v in metrics.items()}})
                update_seconds = time.perf_counter() - t0
            elif source == 'fill':
                pass   # fill-up only stocks the memory bank, as in the base recipe
            if source == 'online':
                snapshots[end] = jax.device_get(state)
            test_seconds = 0.
            if end >= cfg.fill_episodes and end % 20 == 0:
                t0 = time.perf_counter()
                episodes = 500 if end == 200 else 256
                trace_path = output / 'test_0200_traces.npz' if episodes == 500 else None
                summary, stats, test_trace = self.episodes.test(
                    state, episodes=episodes, seed=test_seed if episodes == 500 else 123, output=trace_path)
                if not np.isfinite(summary['steps_mean']):
                    summary['steps_mean'] = summary['steps_median'] = None
                point = {'practice_episodes': end, **summary}
                write_json(output / f'test_{end:04d}.json', point)
                curve.append(point)
                self.ceiling_check(summary, baseline, state, output, end)
                test_seconds = time.perf_counter() - t0
            timings.append({'practice_episodes': end, 'source': source, 'valid_steps': n,
                            'collect_seconds': collect_seconds, 'update_seconds': update_seconds,
                            'test_seconds': test_seconds})
            print(json.dumps({'seed': seed, 'practice_episodes': end, 'restored': restored,
                              'seconds': round(time.perf_counter() - started, 1)}), flush=True)
        replay.save(output / 'replay.npz')
        final_last = curve[-1]
        final = final_last
        picked = None
        if cfg.selection == 'pick':
            versions = sorted(snapshots)
            picked_end, picked_score = pick_version(versions, learner_success)
            picked = {'version_end': picked_end, 'practice_window_success': picked_score}
            if picked_end != 200:
                pstate = jax.tree_util.tree_map(jax.numpy.asarray, snapshots[picked_end])
                summary, _, _ = self.episodes.test(pstate, episodes=500, seed=test_seed,
                                                   output=output / 'test_picked_traces.npz')
                if not np.isfinite(summary['steps_mean']):
                    summary['steps_mean'] = summary['steps_median'] = None
                final = {'practice_episodes': 200, 'picked_version': picked_end, **summary}
            else:
                final = {**final_last, 'picked_version': 200}
            save_checkpoint(output / 'model_picked.pkl', {'config': cfg.to_dict(),
                                                         'state': snapshots[picked_end]})
        unchanged = policy_digest(self.env.frozen_policy) == before
        if not unchanged:
            raise RuntimeError('Frozen policy parameters changed')
        result = {'seed': seed, 'config': cfg.to_dict(), 'curve': curve, 'final': final,
                  'final_last': final_last, 'picked': picked, 'rollbacks': rollbacks,
                  'learner_practice_success': {str(k): v.tolist() for k, v in learner_success.items()},
                  'practice_episodes': ledger.used, 'fill_episodes': cfg.fill_episodes,
                  'replay_steps': replay.count, 'timings': timings,
                  'seconds_excluding_imports': time.perf_counter() - started,
                  'frozen_policy_unchanged': unchanged, 'test_data_learned_from': False,
                  'finished_utc': datetime.now(timezone.utc).isoformat()}
        write_json(output / 'result.json', result)
        return result

    def ceiling_check(self, summary, baseline, state, output, end):
        cfg = self.config
        ceiling = baseline['conditions']['no_camera_bias']['success_rate']
        if cfg.outputs == 2 and cfg.camera_condition == 'normal' and ceiling > .5 \
                and summary['success_rate'] > ceiling + .05:
            raise RuntimeError('Tripwire: aim-only learner exceeds ceiling by more than five points')
        if cfg.outputs != 2 and summary['success_rate'] > ceiling:
            extra, extra_stats, extra_trace = self.episodes.test(
                state, episodes=500, seed=789, output=output / f'extra_ceiling_{end:04d}_traces.npz')
            if not np.isfinite(extra['steps_mean']):
                extra['steps_mean'] = extra['steps_median'] = None
            extra['used_for_selection'] = False
            extra['through_wall_flag'] = extra.get('through_wall_episodes', 0)
            write_json(output / f'extra_ceiling_{end:04d}.json', extra)
