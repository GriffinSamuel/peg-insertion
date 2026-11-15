"""Launch a detached command and preserve its exit status and elapsed time."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in args.name):
        parser.error("name must contain only lowercase letters, digits, _ or -")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("supply a command after --")
    base = ROOT / "results/takehome" / args.name
    if args.worker:
        started = time.perf_counter()
        timestamp = datetime.now(timezone.utc).isoformat()
        result = subprocess.run(command, cwd=ROOT, check=False)
        payload = {"command": command, "started_utc": timestamp,
                   "finished_utc": datetime.now(timezone.utc).isoformat(),
                   "seconds": time.perf_counter() - started, "exit_code": result.returncode}
        with base.with_suffix(".exit.json").open("x") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
        raise SystemExit(result.returncode)
    with base.with_suffix(".log").open("x") as log:
        child = subprocess.Popen(
            [sys.executable, "-B", str(Path(__file__).resolve()),
             "--worker", "--name", args.name, "--", *command],
            cwd=ROOT, start_new_session=True, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT)
    with base.with_suffix(".pid").open("x") as f:
        f.write(str(child.pid) + "\n")
    print(json.dumps({"pid": child.pid, "log": str(base.with_suffix(".log"))}))


if __name__ == "__main__":
    main()
