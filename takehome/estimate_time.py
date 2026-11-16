"""Goal 5 planning estimates from measured throughput, with explicit assumptions."""
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/takehome"


def main():
    timings = json.loads((OUT / "benchmark_timings.json").read_text())["groups"]
    if len(timings) != 4:
        raise RuntimeError("Need all four measured benchmark groups before estimating")
    solo = next(g for g in timings if g["name"] == "solo")["seconds_per_experiment"] / 60
    fastest = min(timings, key=lambda g: g["seconds_per_experiment"])
    fast = fastest["seconds_per_experiment"] / 60
    runs = [json.loads(p.read_text()) for p in (OUT / "benchmark").glob("*/seed_*/result.json")]
    final_eval = statistics.mean(r["timings"][-1]["test_seconds"] for r in runs) / 60
    # Separate evaluation estimates include a modest process/compilation allowance.
    evaluation = max(0.5, final_eval + 0.4)
    # goal, likely/worst new training runs, likely/worst extra 500-episode tests,
    # other serial CPU minutes, lead/routine implementation minutes, complexity.
    specs = [
        (6, 15, 15, 0, 0, 5, 25, 1.0, "15 new learners; reuse main/floor/ceiling"),
        (7, 118, 235, 2, 2, 10, 90, 1.25,
         "72 screening + 26 follow-up + 20 confirmation; upper bound 235"),
        (8, 0, 0, 0, 5, 10, 20, 1.0, "5 saved-data analyses; no fresh practice"),
        (9, 0, 0, 1, 6, 30, 30, 1.0, "20 videos at 1.5 min each; one before-trace test"),
        (10, 9, 9, 6, 6, 5, 10, 1.25, "3 policies x 3 seeds; six floor/ceiling tests"),
        (11, 24, 24, 40, 40, 10, 60, 1.25, "8 problems x 3 retrains; 24 transfer + 8 floor + 8 oracle tests"),
        (12, 0, 0, 0, 0, 10, 45, 1.0, "Report and figure reproduction; no new experiment"),
        (13, 1, 1, 0, 0, 20, 15, 1.25, "1 confirmation-seed rerun plus old/new tests and audit"),
    ]
    rows = []
    for goal, train, worst_train, tests, worst_tests, cpu, implementation, factor, description in specs:
        serial_minutes = train * solo * factor + tests * evaluation + cpu + implementation
        # Helpers reduce routine implementation overhead, never GPU work itself.
        # Rendering is assumed to use three CPU workers; other analysis two.
        cpu_workers = 3 if goal == 9 else 2
        fast_minutes = train * fast * factor + tests * evaluation + cpu / cpu_workers + implementation * 0.75
        # Worst-case allowance: twice solo cost for future methods, no GPU
        # concurrency benefit, and twice the non-training work.
        worst_minutes = worst_train * solo * 2 + worst_tests * evaluation * 2 + 2 * (cpu + implementation)
        rows.append({"goal": goal, "description": description,
                     "new_training_runs_likely": train, "new_training_runs_upper": worst_train,
                     "additional_tests_likely": tests, "additional_tests_upper": worst_tests,
                     "training_minutes_per_run_estimate": solo * factor if train else None,
                     "evaluation_minutes_estimate": evaluation if tests else None,
                     "cpu_minutes_serial": cpu, "implementation_minutes_serial": implementation,
                     "serial_hours_likely": serial_minutes / 60,
                     "fastest_setup_hours_likely": fast_minutes / 60,
                     "worst_case_hours": worst_minutes / 60})
    cuts = [
        ("Skip Goal 7 combination round", 26, 0, 0, 30),
        ("Screen 12 new settings instead of 24", 36, 0, 0, 45),
        ("Retrain on four camera problems instead of eight", 12, 20, 5, 30),
        ("Drop the position-only and hovering-policy extensions", 6, 4, 3, 5),
        ("Render one example per failure type per side instead of two", 0, 0, 15, 5),
    ]
    cut_rows = [{"proposal": name, "training_runs_saved": train, "tests_saved": tests,
                 "estimated_hours_saved": (train * fast * 1.25 + tests * evaluation + cpu / 3 + implementation * 0.75) / 60}
                for name, train, tests, cpu, implementation in cuts]
    result = {"measured_solo_minutes_per_run": solo, "fastest_measured_setup": fastest["name"],
              "fastest_measured_minutes_per_run": fast, "rows": rows, "cut_list": cut_rows,
              "totals": {k: sum(r[k] for r in rows) for k in
                         ("serial_hours_likely", "fastest_setup_hours_likely", "worst_case_hours")},
              "assumptions": [
                  "Estimates, not runtime guarantees; laptop must remain awake.",
                  "Likely Goal 7: 24 new single-change settings x 3 seeds, four combinations x 5 plus two extra seeds for three promising singles, then up to four methods x 5 confirmation seeds.",
                  "Goal 7 upper bound: up to 43 new configurations x 5 seeds plus 20 confirmation runs, treating two of the 45 configurations as already available references.",
                  "Future complex learners estimated at 1.25 times main-run cost; worst case allows twice solo cost and no GPU concurrency advantage.",
                  "Writing/review/plot overhead is a planning estimate; helpers assumed to shorten that portion by 25%. They do not multiply GPU capacity.",
                  "Saved-data analyses and rendering estimates are not benchmarked here; videos assume 20 x 1.5 min and three CPU workers.",
                  "GPU throughput comparisons use small groups and different seeds, so the fastest observed setup is provisional. All forecasts include implementation/review time and leave evaluation work serial.",
                  "Worst case is a planning allowance, not an upper bound on interruptions or debugging. No confirmation seeds have been used for this estimate.",
                  "Cuts are proposals for Griffin, not changes already made. Savings are approximate and not all independent."]}
    (OUT / "time_estimate.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# Remaining-work estimate at the Goal 5 check-in", "",
             "These are planning estimates based on the completed benchmark, not measured runtimes for future methods.", "",
             "| Goal | New 200-episode runs | Extra 500-episode tests | Work | Minutes per training run | Hours serial | Hours with fastest setup and helpers |",
             "|---|---:|---:|---|---:|---:|---:|"]
    for r in rows:
        minutes = r["training_minutes_per_run_estimate"]
        minutes_label = f"{minutes:.2f}" if minutes is not None else "n/a"
        lines.append(f"| {r['goal']} | {r['new_training_runs_likely']} | {r['additional_tests_likely']} | {r['description']} | {minutes_label} | {r['serial_hours_likely']:.2f} | {r['fastest_setup_hours_likely']:.2f} |")
    lines += ["", f"Extra 500-episode tests are estimated at {evaluation:.2f} minutes each, including a startup allowance. Training-run times include their prescribed learning-curve tests. Analysis-only rows include code, review and CPU work.",
              "", f"Likely total: {result['totals']['serial_hours_likely']:.1f} hours serial, or {result['totals']['fastest_setup_hours_likely']:.1f} hours using the fastest measured setup and helpers. Conservative worst-case planning allowance: {result['totals']['worst_case_hours']:.1f} hours.",
              "", "## Cut list (requires a plan change at the check-in)", ""]
    lines += [f"{i}. {r['proposal']}: about {r['estimated_hours_saved']:.2f} hours saved." for i, r in enumerate(cut_rows, 1)]
    lines += ["", "## Assumptions", ""] + [f"- {x}" for x in result["assumptions"]]
    (OUT / "time_estimate.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
