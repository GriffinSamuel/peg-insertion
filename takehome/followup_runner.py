"""Run named-policy experiments with a hard episode ledger and saved models."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from takehome.followup_runtime import ROOT, OUT  # noqa: E402
import jax  # noqa: E402
import numpy as np  # noqa: E402
from takehome.budget import EpisodeBudget  # noqa: E402
from takehome.config import TrainConfig  # noqa: E402
from takehome.correction import AimConfig, AimCorrectedEnv, policy_digest  # noqa: E402
from takehome.episodes import EpisodeRunner  # noqa: E402
from takehome.learner import Learner  # noqa: E402
from takehome.replay import Replay  # noqa: E402


from takehome.followup_env import FollowupEnv, offset_grid
from takehome.followup_learner import FollowupLearner, success_targets
from takehome.followup_episodes import FollowupEpisodes
from takehome.followup_coordination import FollowupCoordinatedLearner
from takehome.variants import prepare_training_data

# Capture code at process import, rather than after a long run has finished.
SOURCE_HASHES = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in (ROOT / 'takehome').glob('*.py')}

def write_json(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write("\n")
    tmp.replace(path)


def save_checkpoint(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def validate_state(state):
    if not all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(state)):
        raise RuntimeError("Tripwire: invalid learner parameters or optimizer state")


class FollowupRunner:
    """One set of compiled functions reused across seeds in the same process."""
    def __init__(self, config):
        if not any(d.platform == "gpu" for d in jax.devices()):
            raise RuntimeError("Tripwire: GPU not visible")
        self.config=config
        self.env=self.make_env(config.policy)
        self.learner=(FollowupLearner(config,self.env) if config.coordination=='joint'
                      else FollowupCoordinatedLearner(config,self.env))
        self.episodes=FollowupCollection(self)

    def make_env(self,policy):
        cfg=self.config
        return FollowupEnv(ROOT/'results/policies'/policy,AimConfig(cfg.step_cap,cfg.guess_cap),config=cfg)

    def after_collect(self,state,replay):
        if not self.config.heatmap:
            return state
        grid=np.asarray(offset_grid())
        attempts=np.zeros(len(grid),dtype=np.int32)
        successes=np.zeros(len(grid),dtype=np.int32)
        a=replay.arrays
        for episode in sorted(replay.episode_ids):
            ids=np.flatnonzero(a['episode_id'][:replay.count]==episode)
            first=ids[np.argmin(a['step'][ids])]
            cell=int(np.argmin(np.linalg.norm(grid-a['guess'][first],axis=-1)))
            attempts[cell]+=1
            successes[cell]+=int(a['success'][ids].any())
        ratings=(successes+1.)/(attempts+2.)
        candidates=np.flatnonzero(ratings==ratings.max())
        cell=candidates[np.argmin(np.linalg.norm(grid[candidates],axis=-1))]
        return state.replace(start_guess=jax.numpy.asarray(grid[cell]))

    def update(self,state,replay,new_steps,source,practice_end):
        data=prepare_training_data(replay,self.config)
        if self.config.success_labels:
            labels,valid=success_targets(replay)
            data={**data,'success_label':labels,'label_valid':valid}
        steps=replay.fill_count if source=='fill' else new_steps
        kwargs={'practice_end':practice_end} if self.config.coordination!='joint' else {}
        return self.learner.train_block(state,data,replay.fill_count,replay.count,
            steps*self.config.updates_per_step,actor_enabled=source!='fill',**kwargs)

    def run(self, seed, output):
        cfg = self.config
        test_seed = getattr(cfg, 'test_seed', 123)
        if seed in range(200,205) or test_seed==1001:
            if not (OUT/'G/choice.json').exists():
                raise ValueError('Fresh seeds forbidden before written Phase G choice')
            if seed not in range(200,205) or test_seed!=1001:
                raise ValueError('Fresh practice and test seeds must stay paired')
        elif seed not in range(5) or test_seed!=123:
            raise ValueError('Choosing uses practice0–4/test123 only')
        output = Path(output).resolve()
        if not output.is_relative_to(OUT):
            raise ValueError("Output must be within results/takehome/followup/")
        output.mkdir(parents=True, exist_ok=True)
        result_path = output / "result.json"
        if result_path.exists():
            print("SKIP completed experiment", output, flush=True)
            return json.loads(result_path.read_text())
        config_path = output / "config.json"
        if config_path.exists() and json.loads(config_path.read_text()) != cfg.to_dict():
            raise ValueError("Existing experiment uses a different configuration")
        write_json(config_path, cfg.to_dict())
        started = time.perf_counter()
        before = policy_digest(self.env.frozen_policy)
        ledger = EpisodeBudget(output / "budget.json", limit=cfg.total_episodes)
        replay = Replay(ledger)
        checkpoint_path = output / "checkpoint.pkl"
        baseline = json.loads((ROOT / getattr(cfg, 'baseline_path',
                                              'results/takehome/baseline.json')).read_text())
        if not str(baseline['policy']).endswith(cfg.policy) or baseline['test_seed'] != test_seed:
            raise ValueError('Reference policy and final test seed must match this experiment')
        if cfg.camera_condition in ('fixed','mixed'):
            if (baseline.get('camera_condition')!=cfg.camera_condition or
                    baseline.get('fixed_bias_seed')!=cfg.fixed_bias_seed):
                raise ValueError('Reference must match this calibration draw')
        if cfg.camera_condition=='mixed' and any(baseline.get(k)!=getattr(cfg,k)
                for k in ('mixed_fixed_std','mixed_episode_std')):
            raise ValueError('Reference must match both mixed components')
        curve = [{"practice_episodes": 0, "label": "original_policy",
                  **baseline["conditions"]["normal"]}]
        timings, losses = [], []
        used = 0
        previous_seconds = 0.0
        state = self.learner.initialize(seed)
        if checkpoint_path.exists():
            with checkpoint_path.open("rb") as f:
                checkpoint = pickle.load(f)
            state = jax.tree_util.tree_map(jax.numpy.asarray, checkpoint["state"])
            used, curve, timings, losses = (checkpoint[k] for k in ("used", "curve", "timings", "losses"))
            previous_seconds = checkpoint["seconds"]
            for end in range(10, used + 1, 10):
                source = "fill" if end <= cfg.fill_episodes else "online"
                with np.load(output / f"practice_{end:04d}.npz") as archive:
                    data = {k: archive[k] for k in archive.files}
                replay.add(data, list(range(end - 10, end)), source, test_only=False)
        if ledger.used < used:
            raise RuntimeError("Checkpoint exceeds persisted episode budget")
        for start in range(used, cfg.total_episodes, 10):
            end = start + 10
            source = "fill" if end <= cfg.fill_episodes else "online"
            ids = list(range(start, end))
            batch_file = output / f"practice_{end:04d}.npz"
            t0 = time.perf_counter()
            if ledger.used > start:
                if not batch_file.exists():
                    raise RuntimeError("Interrupted reservation has no saved data; refusing to recollect spent episodes")
                with np.load(batch_file) as archive:
                    trajectory = {k: archive[k] for k in archive.files}
            else:
                assert ledger.reserve(10, source) == ids
                _, trajectory = self.episodes.practice(state, seed, start // 10, ids, source)
                np.savez_compressed(batch_file, **trajectory)
            n = replay.add(trajectory, ids, source, test_only=False)
            state = self.after_collect(state, replay)
            collect_seconds = time.perf_counter() - t0
            update_seconds = 0.0
            if source == "online" or (getattr(cfg, 'briefing', False) and end == cfg.fill_episodes):
                t0 = time.perf_counter()
                state, metrics = self.update(state, replay, n, source, end)
                metrics = jax.device_get(metrics)
                validate_state(state)
                if not bool(metrics["finite"]):
                    raise RuntimeError("Tripwire: invalid numbers during gradient updates")
                losses.append({"practice_episodes": end, 'source': source,
                               **{k: float(v) for k, v in metrics.items()}})
                update_seconds = time.perf_counter() - t0
            test_seconds = 0.0
            if (not cfg.code_test and end >= cfg.fill_episodes and end % 20 == 0) or end == cfg.total_episodes:
                t0 = time.perf_counter()
                episodes = 8 if cfg.code_test else (500 if end == 200 else 256)
                # Keep full traces only for the final 500-test evaluation.
                # Later analysis goals may explicitly request additional traces.
                trace_path = output / f"test_{end:04d}_traces.npz" if episodes == 500 else None
                summary, stats, test_trace = self.episodes.test(
                    state, episodes=episodes, seed=test_seed if episodes == 500 else 123, output=trace_path)
                summary['trace_shapes'] = {key: list(test_trace[key].shape)
                    for key in ('guess', 'true_bias', 'q', 'force')}
                if not np.isfinite(summary["steps_mean"]):
                    # With zero successes, mean insertion time is undefined, not
                    # a numerical failure. Serialize it as null, never NaN.
                    summary["steps_mean"] = summary["steps_median"] = None
                point = {"practice_episodes": end, **summary}
                write_json(output / f"test_{end:04d}.json", point)
                np.savez_compressed(output / f"test_{end:04d}_episodes.npz", **stats)
                curve.append(point)
                ceiling = baseline["conditions"]["no_camera_bias"]["success_rate"]
                if cfg.outputs == 2 and not cfg.final_decider and cfg.camera_condition=='normal' and ceiling > 0.5 and summary["success_rate"] > ceiling + 0.05:
                    raise RuntimeError("Tripwire: aim-only learner exceeds ceiling by more than five points")
                if (cfg.outputs != 2 or cfg.final_decider or cfg.camera_condition in ('fixed','one-sided','mixed')) and summary['success_rate'] > ceiling:
                    extra, extra_stats, extra_trace = self.episodes.test(
                        state, episodes=500, seed=789,
                        output=output / f'extra_ceiling_{end:04d}_traces.npz')
                    if not np.isfinite(extra['steps_mean']):
                        extra['steps_mean'] = extra['steps_median'] = None
                    examples = np.flatnonzero(extra_stats['succeeded'] > .5)[:5]
                    extra['inspected_episode_ids'] = examples.tolist()
                    extra['max_lateral_when_deep_examples_m'] = extra_stats[
                        'max_lateral_when_deep'][examples].tolist()
                    extra['max_abs_control_examples'] = [float(np.max(np.abs(
                        extra_trace['executed_action'][:, i]))) for i in examples]
                    extra['used_for_selection'] = False
                    extra['camera_condition'] = cfg.camera_condition
                    extra['reason'] = 'Required physical check above the zero-error reference; revised consistent-camera exception applies to aim-only'
                    write_json(output / f'extra_ceiling_{end:04d}.json', extra)
                save_checkpoint(output / f"model_{end:04d}.pkl",
                                {"config": cfg.to_dict(), "state": jax.device_get(state)})
                test_seconds = time.perf_counter() - t0
            timings.append({"practice_episodes": end, "source": source,
                            'source_policy': getattr(self.episodes, 'last_policy', cfg.policy),
                            "valid_steps": n, "collect_seconds": collect_seconds,
                            "update_seconds": update_seconds, "test_seconds": test_seconds})
            save_checkpoint(checkpoint_path, {"state": jax.device_get(state), "used": end,
                "curve": curve, "timings": timings, "losses": losses,
                "seconds": previous_seconds + time.perf_counter() - started})
            write_json(output / "progress.json", {"seed": seed, "practice_episodes": end,
                       "gradient_updates": int(state.updates), "latest": timings[-1]})
            print(json.dumps({"seed": seed, "practice_episodes": end,
                              "gradient_updates": int(state.updates),
                              "seconds": round(time.perf_counter() - started, 2)}), flush=True)
        replay.save(output / "replay.npz")
        unchanged = policy_digest(self.env.frozen_policy) == before
        sibling_audit=self.episodes.verify_frozen_siblings()
        if not unchanged:
            raise RuntimeError("Frozen policy parameters changed")
        method = 'supervised_cheating_reference' if cfg.supervised_reference else {2:'aim',3:'movement',5:'combined'}[cfg.outputs]
        result = {"seed": seed, "config": cfg.to_dict(), "curve": curve,
                  "method": method, "cheats_on_information": cfg.privileged_critic,
                  "final": curve[-1], "practice_episodes": ledger.used,
                  "fill_episodes": cfg.fill_episodes,
                  "learner_episodes": cfg.total_episodes - cfg.fill_episodes,
                  "gradient_updates": int(state.updates), "replay_steps": replay.count,
                  "fill_steps": replay.fill_count, "losses": losses, "timings": timings,
                  "seconds_excluding_imports": previous_seconds + time.perf_counter() - started,
                  "frozen_policy_unchanged": unchanged, "test_data_learned_from": False,
                  "frozen_sibling_policies": sibling_audit,
                  "code_test": cfg.code_test, "setting_counted": not cfg.code_test,
                  "finished_utc": datetime.now(timezone.utc).isoformat(),
                  "source_hashes": SOURCE_HASHES}
        write_json(result_path, result)
        return result



class FollowupCollection:
    def __init__(self,parent):
        self.parent=parent
        self.main=FollowupEpisodes(parent.env,parent.learner)
        self.siblings={}
        self.sibling_digests={}
        self.last_policy=parent.config.policy

    def practice(self,state,seed,batch_index,episode_ids,source):
        self.last_policy=self.parent.config.policy
        runner=self.main
        if source=='fill' and self.parent.config.sibling_fill:
            self.last_policy=('force_0.5mm_s0','force_0.5mm_s3','force_0.5mm_s4')[batch_index%3]
            if self.last_policy not in self.siblings:
                env=self.parent.make_env(self.last_policy)
                self.siblings[self.last_policy]=FollowupEpisodes(env,self.parent.learner)
                self.sibling_digests[self.last_policy]=policy_digest(env.frozen_policy)
            runner=self.siblings[self.last_policy]
        return runner.practice(state,seed,batch_index,episode_ids,source)

    def test(self,*args,**kwargs):
        return self.main.test(*args,**kwargs)

    def verify_frozen_siblings(self):
        audit={name:policy_digest(runner.env.frozen_policy)==self.sibling_digests[name]
               for name,runner in self.siblings.items()}
        if not all(audit.values()):
            raise RuntimeError('Tripwire: frozen sibling policy changed')
        return audit
