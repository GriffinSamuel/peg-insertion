"""Goal 1: fixed, test-only evaluation using the project's rollout code."""

import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Keep library-generated outputs within the permitted output directory.
CACHE = ROOT / "results/takehome/preflight_cache"
for key, sub in {
    "XDG_CACHE_HOME": "xdg",
    "JAX_COMPILATION_CACHE_DIR": "jax",
    "WARP_CACHE_PATH": "warp",
    "MPLCONFIGDIR": "mpl",
}.items():
    os.environ[key] = str(CACHE / sub)

import jax  # noqa: E402
import numpy as np  # noqa: E402
from peg_insertion import rollout  # noqa: E402


def main():
    started = time.perf_counter()
    devices = jax.devices()
    print("JAX devices:", devices, flush=True)
    if not any(d.platform == "gpu" for d in devices):
        raise RuntimeError("Tripwire: GPU is not visible")
    output = ROOT / "results/takehome/baseline.json"
    if output.exists():
        print("Existing result; skipping:", output, flush=True)
        return
    policy_dir = ROOT / "results/policies/force_0.5mm_s2"
    policy, cfg = rollout.load_policy(policy_dir)
    overrides = {
        k: v for k, v in cfg.get("env_overrides", {}).items()
        if k not in ("clearance_mm", "use_force_obs", "naconmax")
    }
    results = {}
    for name, extra in (("normal", {}), ("no_camera_bias", {"obs_hole_bias_std": 0.0})):
        partial = output.with_name(f"baseline_{name}.json")
        if partial.exists():
            results[name] = json.loads(partial.read_text())
            print("Reusing completed condition:", name, flush=True)
            continue
        env = rollout.make_env(
            cfg["clearance_mm"], cfg["use_force_obs"],
            naconmax=20 * 256, **{**overrides, **extra},
        )
        t0 = time.perf_counter()
        print("Starting", name, "500 test-only episodes, seed 123", flush=True)
        stats = rollout.rollout_stats(
            env, jax.jit(policy), 500, seed=123, randomize=True, batch=256,
        )
        finite = all(bool(np.isfinite(v).all()) for v in stats.values())
        summary = {
            **rollout.summarize(stats),
            "test_only": True, "learned_from": False, "test_seed": 123,
            "obs_hole_bias_std": float(env._config.obs_hole_bias_std),
            "obs_noise_std": float(env._config.obs_noise_std),
            "all_stats_finite": finite,
            "eval_seconds": time.perf_counter() - t0,
        }
        np.savez_compressed(output.with_name(f"baseline_{name}_episodes.npz"), **stats)
        with partial.open("x") as f:
            json.dump(summary, f, indent=2, allow_nan=False)
            f.write("\n")
        print(name, json.dumps(summary), flush=True)
        if not finite:
            raise RuntimeError("Tripwire: invalid simulation numbers")
        results[name] = summary
    result = {
        "policy": str(policy_dir.relative_to(ROOT)),
        "test_seed": 123, "episodes_per_condition": 500,
        "practice_episodes": 0, "settings_tried": 0,
        "conditions": results,
        "checks": {
            "gpu_visible": True,
            "normal_within_5_points": abs(results["normal"]["success_rate"] - 0.628) <= 0.05,
            "ceiling_within_5_points": abs(results["no_camera_bias"]["success_rate"] - 0.940) <= 0.05,
        },
        "seconds_excluding_imports": time.perf_counter() - started,
    }
    with output.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
