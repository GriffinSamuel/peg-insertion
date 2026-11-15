"""Execute the predeclared Goal 5 GPU schedule; retain every real experiment."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/takehome"


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def check_identical_headline_outcomes(manifest):
    import numpy as np
    paths = [BASE / "benchmark" / p / "test_0200_episodes.npz"
             for p in manifest["headline_designations"].values()]
    if all(p.exists() for p in paths):
        successes = [np.load(p)["succeeded"] for p in paths]
        if all(np.array_equal(successes[0], s) for s in successes[1:]):
            raise RuntimeError("Tripwire: all five learned seeds have identical outcomes on every test episode")


def main():
    manifest = json.loads((BASE / "benchmark_manifest.json").read_text())
    timing_path = BASE / "benchmark_timings.json"
    timings = json.loads(timing_path.read_text()) if timing_path.exists() else {"groups": []}
    log_dir = BASE / "benchmark_logs"
    log_dir.mkdir(exist_ok=True)
    for group in manifest["groups"]:
        if any(g["name"] == group["name"] for g in timings["groups"]):
            continue
        check_identical_headline_outcomes(manifest)
        processes, logs, finished = [], [], []
        start = time.perf_counter()
        utc = datetime.now(timezone.utc).isoformat()
        print("START benchmark group", group["name"], group["processes"], flush=True)
        for seeds in group["processes"]:
            suffix = "_".join(map(str, seeds))
            path = log_dir / f"{group['name']}_seeds_{suffix}.log"
            log = path.open("a")
            logs.append(log)
            cmd = [sys.executable, "-B", str(ROOT / "takehome/train.py"),
                   "--seeds", *map(str, seeds),
                   "--output", str(BASE / "benchmark" / group["name"])]
            p = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            processes.append((p, seeds, path))
        write_json(BASE / "benchmark_active.json", {"group": group["name"],
                   "started_utc": utc,
                   "processes": [{"pid": p.pid, "seeds": seeds, "log": str(path.relative_to(ROOT))}
                                 for p, seeds, path in processes]})
        try:
            while processes:
                for p, seeds, path in list(processes):
                    status = p.poll()
                    if status is not None:
                        if status:
                            raise RuntimeError(f"Experiment process failed ({status}); inspect {path}")
                        finished.append({"seeds": seeds, "seconds_from_group_start": time.perf_counter() - start,
                                         "exit_code": status, "log": str(path.relative_to(ROOT))})
                        processes.remove((p, seeds, path))
                if processes:
                    time.sleep(2)
        except BaseException:
            for p, _, _ in processes:
                p.terminate()
            for p, _, _ in processes:
                p.wait()
            raise
        finally:
            for log in logs:
                log.close()
        seconds = time.perf_counter() - start
        count = sum(len(s) for s in group["processes"])
        row = {"name": group["name"], "process_count": len(group["processes"]),
               "experiments": count, "seconds": seconds,
               "seconds_per_experiment": seconds / count, "processes": finished,
               "started_utc": utc, "finished_utc": datetime.now(timezone.utc).isoformat()}
        timings["groups"].append(row)
        write_json(timing_path, timings)
        print("DONE", json.dumps(row), flush=True)
        check_identical_headline_outcomes(manifest)
    print("All nine predeclared experiments completed.", flush=True)


if __name__ == "__main__":
    main()
