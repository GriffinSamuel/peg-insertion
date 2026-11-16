#!/usr/bin/env python3
"""Create Goal 10 transfer and policy-reference manifests after the gates pass."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "takehome"
CHOICE_PATH = RESULTS / "goal7_choice.json"
CHECKS_PATH = RESULTS / "goal9_checks.json"
TRAIN_MANIFEST = RESULTS / "goal10_manifest.json"
REFERENCE_MANIFEST = RESULTS / "goal10_reference_manifest.json"
POLICIES = ("force_0.5mm_s4", "pos_0.5mm_s1", "force_0.5mm_s1")
SEEDS = [0, 1, 2]


def read_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def write_new(path, payload):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")


def build_manifests():
    if not CHOICE_PATH.is_file():
        raise FileNotFoundError(f"Required written choice is missing: {CHOICE_PATH}")
    if not CHECKS_PATH.is_file():
        raise FileNotFoundError(f"Required Goal 9 checks are missing: {CHECKS_PATH}")
    checks = read_json(CHECKS_PATH)
    if checks.get("all_goal_checks_pass") is not True:
        raise ValueError("Goal 9 gate failed: all_goal_checks_pass must be true")

    choice = read_json(CHOICE_PATH)
    selected = choice.get("selected")
    if not isinstance(selected, dict) or not isinstance(selected.get("overrides"), dict):
        raise ValueError("Written choice must contain selected.overrides")
    algorithm_overrides = dict(selected["overrides"])

    settings = []
    training_jobs = []
    reference_jobs = []
    for policy in POLICIES:
        baseline_path = f"results/takehome/goal10/{policy}/baseline.json"
        overrides = {
            **algorithm_overrides,
            "policy": policy,
            "test_seed": 123,
            "baseline_path": baseline_path,
        }
        # Verify that policy-specific evaluation context is the only difference.
        copied_algorithm = {k: v for k, v in overrides.items()
                            if k not in ("policy", "test_seed", "baseline_path")}
        if copied_algorithm != algorithm_overrides:
            raise AssertionError("Transfer manifest changed an algorithm override")

        name = policy
        output = f"results/takehome/goal10/{policy}/trained"
        settings.append({
            "name": name,
            "label": f"Final method on {policy}",
            "overrides": overrides,
            "base": selected.get("name", "selected Goal 7 method"),
            "changes_from_main": selected.get("changes_from_main"),
            "seeds": list(SEEDS),
            "round": "transfer",
            "output": output,
        })
        training_jobs.append({
            "name": name,
            "args": ["takehome/train_variants.py", "--manifest",
                     "results/takehome/goal10_manifest.json", "--setting", name],
            "log": f"results/takehome/goal10/logs/{policy}.log",
            "requires": [baseline_path],
        })
        reference_jobs.append({
            "name": policy,
            "args": ["takehome/reference.py", "--policy", policy,
                     "--test-seed", "123", "--output", baseline_path],
            "log": f"results/takehome/goal10/logs/reference_{policy}.log",
        })

    train_manifest = {
        "settings": settings,
        "jobs": training_jobs,
        "test_seed": 123,
        "final_test_episodes": 500,
        "suite_summary": "results/takehome/goal10/training_suite.json",
        "choice": "results/takehome/goal7_choice.json",
        "reference_manifest": "results/takehome/goal10_reference_manifest.json",
        "execution_order": "Run all policy reference jobs successfully before training jobs.",
        "practice_episodes_per_seed": 200,
    }
    reference_manifest = {
        "jobs": reference_jobs,
        "test_seed": 123,
        "episodes_per_condition": 500,
        "practice_episodes": 0,
        "suite_summary": "results/takehome/goal10/reference_suite.json",
        "must_run_before": "results/takehome/goal10_manifest.json",
    }
    if len(settings) != 3 or any(setting["seeds"] != SEEDS for setting in settings):
        raise AssertionError("Expected exactly three policies with seeds 0, 1, and 2")
    if TRAIN_MANIFEST.exists() or REFERENCE_MANIFEST.exists():
        raise FileExistsError("Refusing to overwrite an existing Goal 10 manifest")
    return train_manifest, reference_manifest


def main():
    train_manifest, reference_manifest = build_manifests()
    write_new(REFERENCE_MANIFEST, reference_manifest)
    write_new(TRAIN_MANIFEST, train_manifest)
    print(json.dumps({"training_manifest": str(TRAIN_MANIFEST.relative_to(ROOT)),
                      "reference_manifest": str(REFERENCE_MANIFEST.relative_to(ROOT)),
                      "policies": list(POLICIES), "seeds": SEEDS}, indent=2))


if __name__ == "__main__":
    main()
