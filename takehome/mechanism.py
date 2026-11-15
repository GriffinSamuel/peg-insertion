"""Goal 3: same fixed test for zero correction and true-bias oracle."""
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from takehome.runtime import ROOT  # noqa: E402
import jax  # noqa: E402
import numpy as np  # noqa: E402
from peg_insertion import rollout  # noqa: E402
from takehome.correction import AimCorrectedEnv, OracleAimEnv, policy_digest, zero_nudge  # noqa: E402


def main():
    output = ROOT / "results/takehome/mechanism_check.json"
    if output.exists():
        print("Existing result; skipping:", output)
        return
    if not any(d.platform == "gpu" for d in jax.devices()):
        raise RuntimeError("Tripwire: GPU not visible")
    baseline = json.loads((output.parent / "baseline.json").read_text())
    results = {}
    for name, cls in (("zero", AimCorrectedEnv), ("oracle", OracleAimEnv)):
        partial = output.with_name(f"mechanism_{name}.json")
        if partial.exists():
            results[name] = json.loads(partial.read_text())
            continue
        env = cls(ROOT / "results/policies/force_0.5mm_s2")
        before = policy_digest(env.frozen_policy)
        t0 = time.perf_counter()
        print("Starting", name, "500 test-only episodes, seed 123", flush=True)
        stats = rollout.rollout_stats(env, zero_nudge, 500, seed=123,
                                      randomize=True, batch=256)
        summary = {**rollout.summarize(stats), "test_only": True, "learned_from": False,
                   "test_seed": 123, "eval_seconds": time.perf_counter() - t0,
                   "all_stats_finite": all(bool(np.isfinite(v).all()) for v in stats.values()),
                   "frozen_policy_unchanged": policy_digest(env.frozen_policy) == before}
        if name == "zero":
            original = np.load(output.parent / "baseline_normal_episodes.npz")
            summary["same_per_episode_success_as_baseline"] = bool(
                np.array_equal(stats["succeeded"], original["succeeded"]))
        np.savez_compressed(output.with_name(f"mechanism_{name}_episodes.npz"), **stats)
        with partial.open("x") as f:
            json.dump(summary, f, indent=2, allow_nan=False)
            f.write("\n")
        print(name, json.dumps(summary), flush=True)
        if not summary["all_stats_finite"] or not summary["frozen_policy_unchanged"]:
            raise RuntimeError("Tripwire: invalid numbers or changed frozen policy")
        results[name] = summary
        if name == "zero" and abs(summary["success_rate"] - baseline["conditions"]["normal"]["success_rate"]) > 0.03:
            raise RuntimeError("Tripwire: zero correction changed baseline success rate")
    result = {"conditions": results, "practice_episodes": 0, "settings_tried": 0,
              "checks": {
                  "zero_within_3_points": abs(results["zero"]["success_rate"] - baseline["conditions"]["normal"]["success_rate"]) <= 0.03,
                  "oracle_within_5_points": abs(results["oracle"]["success_rate"] - baseline["conditions"]["no_camera_bias"]["success_rate"]) <= 0.05}}
    with output.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
