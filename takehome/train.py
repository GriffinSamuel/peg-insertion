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
from takehome.runtime import ROOT  # noqa: E402
import jax  # noqa: E402
import numpy as np  # noqa: E402
from takehome.budget import EpisodeBudget  # noqa: E402
from takehome.config import TrainConfig  # noqa: E402
from takehome.correction import AimConfig, AimCorrectedEnv, policy_digest  # noqa: E402
from takehome.episodes import EpisodeRunner  # noqa: E402
from takehome.learner import Learner  # noqa: E402
from takehome.replay import Replay  # noqa: E402


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


class ExperimentRunner:
    """One set of compiled functions reused across seeds in the same process."""
    def __init__(self, config):
        if not any(d.platform == "gpu" for d in jax.devices()):
            raise RuntimeError("Tripwire: GPU not visible")
        if config.outputs != 2:
            raise ValueError("3/5-output networks are available; movement environment adapters belong to Goal 6")
        self.config = config
        self.env = AimCorrectedEnv(ROOT / "results/policies" / config.policy,
                                   AimConfig(config.step_cap, config.guess_cap))
        self.learner = Learner(config, self.env)
        self.episodes = EpisodeRunner(self.env, self.learner)

    def run(self, seed, output):
        cfg = self.config
        output = Path(output).resolve()
        if not output.is_relative_to(ROOT / "results/takehome"):
            raise ValueError("Output must be within results/takehome/")
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
        baseline = json.loads((ROOT / "results/takehome/baseline.json").read_text())
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
            collect_seconds = time.perf_counter() - t0
            update_seconds = 0.0
            if source == "online":
                t0 = time.perf_counter()
                state, metrics = self.learner.train_block(
                    state, replay.training_arrays(), replay.fill_count, replay.count,
                    n * cfg.updates_per_step)
                metrics = jax.device_get(metrics)
                validate_state(state)
                if not bool(metrics["finite"]):
                    raise RuntimeError("Tripwire: invalid numbers during gradient updates")
                losses.append({"practice_episodes": end,
                               **{k: float(v) for k, v in metrics.items()}})
                update_seconds = time.perf_counter() - t0
            test_seconds = 0.0
            if (not cfg.code_test and end >= cfg.fill_episodes and end % 20 == 0) or end == cfg.total_episodes:
                t0 = time.perf_counter()
                episodes = 8 if cfg.code_test else (500 if end == 200 else 256)
                summary, stats, _ = self.episodes.test(
                    state, episodes=episodes, seed=123, output=output / f"test_{end:04d}_traces.npz")
                if not np.isfinite(summary["steps_mean"]):
                    # With zero successes, mean insertion time is undefined, not
                    # a numerical failure. Serialize it as null, never NaN.
                    summary["steps_mean"] = summary["steps_median"] = None
                point = {"practice_episodes": end, **summary}
                write_json(output / f"test_{end:04d}.json", point)
                np.savez_compressed(output / f"test_{end:04d}_episodes.npz", **stats)
                curve.append(point)
                ceiling = baseline["conditions"]["no_camera_bias"]["success_rate"]
                if ceiling > 0.5 and summary["success_rate"] > ceiling + 0.05:
                    raise RuntimeError("Tripwire: aim-only learner exceeds ceiling by more than five points")
                save_checkpoint(output / f"model_{end:04d}.pkl",
                                {"config": cfg.to_dict(), "state": jax.device_get(state)})
                test_seconds = time.perf_counter() - t0
            timings.append({"practice_episodes": end, "source": source,
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
        if not unchanged:
            raise RuntimeError("Frozen policy parameters changed")
        result = {"seed": seed, "config": cfg.to_dict(), "curve": curve,
                  "final": curve[-1], "practice_episodes": ledger.used,
                  "fill_episodes": cfg.fill_episodes,
                  "learner_episodes": cfg.total_episodes - cfg.fill_episodes,
                  "gradient_updates": int(state.updates), "replay_steps": replay.count,
                  "fill_steps": replay.fill_count, "losses": losses, "timings": timings,
                  "seconds_excluding_imports": previous_seconds + time.perf_counter() - started,
                  "frozen_policy_unchanged": unchanged, "test_data_learned_from": False,
                  "code_test": cfg.code_test, "setting_counted": not cfg.code_test,
                  "finished_utc": datetime.now(timezone.utc).isoformat(),
                  "source_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in (ROOT / "takehome").glob("*.py")}}
        write_json(result_path, result)
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="force_0.5mm_s2")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--output", type=Path, default=ROOT / "results/takehome/main")
    parser.add_argument("--fill-episodes", type=int, default=40)
    parser.add_argument("--updates-per-step", type=int, default=4)
    parser.add_argument("--ensemble", type=int, default=4)
    parser.add_argument("--outputs", type=int, choices=(2, 3, 5), default=2)
    parser.add_argument("--step-cap", type=float, default=0.0001)
    parser.add_argument("--guess-cap", type=float, default=0.003)
    parser.add_argument("--code-test", action="store_true")
    args = parser.parse_args()
    if any(seed in range(100, 105) for seed in args.seeds):
        parser.error("Confirmation seeds are reserved until Goal 7 selection")
    config = TrainConfig(policy=args.policy, outputs=args.outputs,
        fill_episodes=args.fill_episodes, total_episodes=10 if args.code_test else 200,
        updates_per_step=args.updates_per_step, ensemble=args.ensemble,
        step_cap=args.step_cap, guess_cap=args.guess_cap, code_test=args.code_test)
    todo = [s for s in args.seeds if not (args.output / f"seed_{s}" / "result.json").exists()]
    if not todo:
        print("All requested results already exist; no episodes collected.")
        return
    runner = ExperimentRunner(config)
    for seed in todo:
        result = runner.run(seed, args.output / f"seed_{seed}")
        print(json.dumps({"seed": seed, "success_rate": result["final"]["success_rate"],
                          "seconds": result["seconds_excluding_imports"]}), flush=True)


if __name__ == "__main__":
    main()
