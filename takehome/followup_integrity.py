#!/usr/bin/env python3
"""Snapshot and verify pre-follow-up project artifacts without modifying them."""

import argparse
import base64
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FOLLOWUP = ROOT / "results" / "takehome" / "followup"
MANIFEST = FOLLOWUP / "preservation_manifest.json"
REPORT = ROOT / "TAKEHOME_REPORT.md"
CHUNK_SIZE = 1024 * 1024

# These directory names and suffixes are treated as cache or temporary data.
# No file extension or scientific-result filename is excluded.
EXCLUDED_DIRECTORY_NAMES = {
    ".cache", ".pytest_cache", "__pycache__", "cache", "caches",
    "mplconfig", "preflight_cache", "pytest_cache", "tmp", "temp",
}
EXCLUDED_DIRECTORY_SUFFIXES = ("_tmp", ".tmp")
EXCLUDED_RESULTS_SUBTREE = "results/takehome/followup"
PROTECTED_RESULT_FILENAMES = ("*.npz", "*.pkl", "*.mp4", "result.json")
TRACKED_PATHSPECS = (
    "takehome", "peg_insertion", "scripts", "tests", "policies",
    "results/policies",
)


def relative(path):
    return path.relative_to(ROOT).as_posix()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def excluded_directory(name):
    lowered = name.lower()
    return (lowered in EXCLUDED_DIRECTORY_NAMES
            or any(lowered.endswith(suffix) for suffix in EXCLUDED_DIRECTORY_SUFFIXES))


def add_candidate(candidates, path, category):
    try:
        path.relative_to(ROOT)
    except ValueError:
        raise ValueError(f"refusing path outside project root: {path}")
    if not os.path.lexists(path):
        return
    rel = relative(path)
    if rel == EXCLUDED_RESULTS_SUBTREE or rel.startswith(EXCLUDED_RESULTS_SUBTREE + "/"):
        return
    categories = candidates.setdefault(rel, set())
    categories.add(category)


def tracked_candidates(candidates):
    command = ["git", "ls-files", "-z", "--", *TRACKED_PATHSPECS]
    completed = subprocess.run(command, cwd=ROOT, check=True, stdout=subprocess.PIPE)
    for raw in completed.stdout.split(b"\0"):
        if not raw:
            continue
        path = ROOT / os.fsdecode(raw)
        add_candidate(candidates, path, "tracked_source_or_policy")


def result_candidates(candidates):
    if not RESULTS.exists():
        return
    for current, directory_names, filenames in os.walk(RESULTS, topdown=True, followlinks=False):
        current_path = Path(current)
        kept_directories = []
        for name in directory_names:
            path = current_path / name
            rel = relative(path)
            if rel == EXCLUDED_RESULTS_SUBTREE or rel.startswith(EXCLUDED_RESULTS_SUBTREE + "/"):
                continue
            if excluded_directory(name):
                continue
            if path.is_symlink():
                add_candidate(candidates, path, "results_tree")
                continue
            kept_directories.append(name)
        directory_names[:] = kept_directories
        for name in filenames:
            path = current_path / name
            add_candidate(candidates, path, "results_tree")


def collect_candidates():
    candidates = {}
    tracked_candidates(candidates)
    result_candidates(candidates)
    add_candidate(candidates, REPORT, "top_level_report")
    return candidates


def file_record(rel, categories):
    path = ROOT / rel
    before = path.lstat()
    is_symlink = stat.S_ISLNK(before.st_mode)
    target = os.readlink(path) if is_symlink else None
    if is_symlink:
        digest = sha256_bytes(os.fsencode(target))
    elif stat.S_ISREG(before.st_mode):
        digest = sha256_file(path)
    else:
        raise ValueError(f"unsupported non-regular artifact: {rel}")
    after = path.lstat()
    if (before.st_size, before.st_mtime_ns, before.st_mode) != (
            after.st_size, after.st_mtime_ns, after.st_mode):
        raise RuntimeError(f"artifact changed during snapshot: {rel}")
    return {
        "relative_path": rel,
        "included_by": sorted(categories),
        "size": before.st_size,
        "mtime_ns": before.st_mtime_ns,
        "sha256": digest,
        "symlink_target": target,
    }


def exclusions_record():
    return {
        "excluded_subtrees": [EXCLUDED_RESULTS_SUBTREE],
        "excluded_cache_temp_directory_names": sorted(EXCLUDED_DIRECTORY_NAMES),
        "excluded_cache_temp_directory_suffixes": list(EXCLUDED_DIRECTORY_SUFFIXES),
        "excluded_extensions": [],
        "protected_result_name_patterns_not_excluded": list(PROTECTED_RESULT_FILENAMES),
    }


def snapshot():
    allowed_output_root = ROOT / "results" / "takehome"
    if FOLLOWUP.is_symlink() or not FOLLOWUP.resolve().is_relative_to(allowed_output_root.resolve()):
        raise ValueError("follow-up output directory must resolve inside results/takehome and not be a symlink")
    FOLLOWUP.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(MANIFEST):
        raise FileExistsError(f"snapshot manifest already exists: {relative(MANIFEST)}")
    candidates = collect_candidates()
    report_bytes = REPORT.read_bytes()
    if "TAKEHOME_REPORT.md" not in candidates:
        raise RuntimeError("TAKEHOME_REPORT.md was not included in snapshot candidates")
    records = [file_record(rel, categories)
               for rel, categories in sorted(candidates.items())]
    report_record = next(record for record in records
                         if record["relative_path"] == "TAKEHOME_REPORT.md")
    if report_record["sha256"] != sha256_bytes(report_bytes):
        raise RuntimeError("report changed while its exact-byte snapshot was being prepared")
    manifest = {
        "schema_version": 1,
        "mode": "snapshot",
        "file_count": len(records),
        "exclusions": exclusions_record(),
        "original_report": {
            "relative_path": "TAKEHOME_REPORT.md",
            "sha256": sha256_bytes(report_bytes),
            "size": len(report_bytes),
            "bytes_base64": base64.b64encode(report_bytes).decode("ascii"),
        },
        "files": records,
    }
    with MANIFEST.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    return {"manifest": relative(MANIFEST), "file_count": len(records),
            "exclusions": manifest["exclusions"]}


def acceptable_report_append(current, original):
    if not current.startswith(original):
        return False, "report no longer begins with the exact snapshotted bytes"
    suffix = current[len(original):]
    if not suffix.startswith(b"\n\n"):
        return False, "report append must begin with one blank-line paragraph separator"
    try:
        paragraph_block = suffix[2:].decode("utf-8")
    except UnicodeDecodeError:
        return False, "appended report paragraph is not UTF-8"
    if paragraph_block.endswith("\n"):
        paragraph = paragraph_block[:-1]
    else:
        paragraph = paragraph_block
    if not paragraph or "\n\n" in paragraph or paragraph.startswith("\n"):
        return False, "append must contain exactly one non-empty paragraph"
    if paragraph_block.endswith("\n\n"):
        return False, "append must contain exactly one trailing newline at most"
    if not re.search(r"\]\(TAKEHOME_FOLLOWUP_REPORT\.md\)", paragraph):
        return False, "appended paragraph must link to TAKEHOME_FOLLOWUP_REPORT.md"
    followup_report = ROOT / "TAKEHOME_FOLLOWUP_REPORT.md"
    if not followup_report.is_file():
        return False, "TAKEHOME_FOLLOWUP_REPORT.md link target does not exist"
    return True, None


def verify(allow_report_append=False):
    if not MANIFEST.is_file():
        return {"passed": False, "failure_count": 1,
                "failures": [{"path": relative(MANIFEST), "reason": "snapshot manifest is missing"}],
                "checked_count": 0}
    try:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"passed": False, "failure_count": 1,
                "failures": [{"path": relative(MANIFEST),
                              "reason": f"cannot parse manifest: {type(exc).__name__}: {exc}"}],
                "checked_count": 0}
    failures = []
    records = manifest.get("files")
    if not isinstance(records, list):
        records = []
        failures.append({"path": relative(MANIFEST), "reason": "manifest files must be a list"})
    if manifest.get("file_count") != len(records):
        failures.append({"path": relative(MANIFEST), "reason": "manifest file_count does not match files list"})

    original_report = None
    report_descriptor = manifest.get("original_report", {})
    encoded = report_descriptor.get("bytes_base64")
    try:
        original_report = base64.b64decode(encoded, validate=True)
    except Exception:
        failures.append({"path": "TAKEHOME_REPORT.md", "reason": "original report bytes are missing or invalid base64"})
    if original_report is not None:
        if len(original_report) != report_descriptor.get("size"):
            failures.append({"path": "TAKEHOME_REPORT.md", "reason": "stored report byte length mismatch"})
        if sha256_bytes(original_report) != report_descriptor.get("sha256"):
            failures.append({"path": "TAKEHOME_REPORT.md", "reason": "stored report bytes do not match stored report hash"})

    seen = set()
    checked = 0
    for record in records:
        rel = record.get("relative_path")
        if not isinstance(rel, str) or not rel:
            failures.append({"path": str(rel), "reason": "invalid relative_path"})
            continue
        if rel in seen:
            failures.append({"path": rel, "reason": "duplicate path in manifest"})
            continue
        seen.add(rel)
        rel_path = Path(rel)
        if rel_path.is_absolute() or ".." in rel_path.parts:
            failures.append({"path": rel, "reason": "manifest path escapes project root"})
            continue
        lexical_path = ROOT / rel_path
        if not os.path.lexists(lexical_path):
            failures.append({"path": rel, "reason": "listed file is missing"})
            continue
        try:
            current_stat = lexical_path.lstat()
            current_is_symlink = stat.S_ISLNK(current_stat.st_mode)
            current_target = os.readlink(lexical_path) if current_is_symlink else None
            expected_target = record.get("symlink_target")
            if current_is_symlink != (expected_target is not None):
                failures.append({"path": rel, "reason": "symlink type changed"})
                continue
            if current_target != expected_target:
                failures.append({"path": rel, "reason": "symlink target changed"})
                continue
            if current_is_symlink:
                current_hash = sha256_bytes(os.fsencode(current_target))
            elif stat.S_ISREG(current_stat.st_mode):
                current_hash = sha256_file(lexical_path)
            else:
                failures.append({"path": rel, "reason": "listed artifact is no longer a regular file or symlink"})
                continue
            checked += 1
            if rel == "TAKEHOME_REPORT.md" and allow_report_append and original_report is not None:
                ok, reason = acceptable_report_append(lexical_path.read_bytes(), original_report)
                if ok:
                    continue
                failures.append({"path": rel, "reason": reason})
                continue
            if current_stat.st_size != record.get("size"):
                failures.append({"path": rel, "reason": "size changed"})
            elif current_stat.st_mtime_ns != record.get("mtime_ns"):
                failures.append({"path": rel, "reason": "mtime_ns changed"})
            elif current_hash != record.get("sha256"):
                failures.append({"path": rel, "reason": "SHA256 changed"})
        except Exception as exc:
            failures.append({"path": rel, "reason": f"verification error: {type(exc).__name__}: {exc}"})
    return {"passed": not failures, "failure_count": len(failures),
            "failures": failures, "checked_count": checked,
            "manifest_file_count": manifest.get("file_count"),
            "allow_report_append": bool(allow_report_append)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("snapshot", "verify"))
    parser.add_argument("--allow-report-append", action="store_true",
                        help="verify one exact appended paragraph linking TAKEHOME_FOLLOWUP_REPORT.md")
    args = parser.parse_args(argv)
    if args.mode == "snapshot" and args.allow_report_append:
        parser.error("--allow-report-append applies only to verify mode")
    try:
        result = snapshot() if args.mode == "snapshot" else verify(args.allow_report_append)
    except Exception as exc:
        result = {"passed": False, "failure_count": 1,
                  "failures": [{"path": relative(MANIFEST),
                                "reason": f"{type(exc).__name__}: {exc}"}]}
    if args.mode == "snapshot":
        passed = result.get("failure_count", 0) == 0
        output = {"passed": passed, **result}
    else:
        output = result
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0 if output.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
