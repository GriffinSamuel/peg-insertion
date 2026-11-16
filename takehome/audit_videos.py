#!/usr/bin/env python3
"""Audit Goal 9 illustrative videos and their manifest/source metadata."""

import json
import math
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "takehome"
MANIFEST_PATH = RESULTS / "goal9" / "video_manifest.json"
FAILURES_PATH = RESULTS / "goal9" / "failures.json"
OUTPUT_PATH = RESULTS / "goal9" / "video_audit.json"
EXPECTED_FRAMES = 100
EXPECTED_FPS = 25.0
EXPECTED_DURATION = 4.0


def resolve_under_results(value):
    path = Path(value)
    path = path if path.is_absolute() else ROOT / path
    path = path.resolve()
    if not path.is_relative_to(RESULTS.resolve()):
        raise ValueError(f"Path is outside results/takehome: {value}")
    return path


def ratio(value):
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and "/" in value:
        numerator, denominator = value.split("/", 1)
        return float(numerator) / float(denominator)
    return float(value)


def ffprobe(path):
    command = [
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=nb_frames,nb_read_frames,r_frame_rate,avg_frame_rate,duration",
        "-show_entries", "format=duration", "-of", "json", str(path),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(completed.stdout)
    streams = payload.get("streams", [])
    if len(streams) != 1:
        raise ValueError(f"Expected one video stream, found {len(streams)}")
    stream = streams[0]
    format_data = payload.get("format", {})
    return {
        "nb_frames": int(stream["nb_frames"]) if stream.get("nb_frames") not in (None, "N/A") else None,
        "decoded_frames": int(stream["nb_read_frames"]) if stream.get("nb_read_frames") not in (None, "N/A") else None,
        "r_frame_rate": ratio(stream["r_frame_rate"]),
        "avg_frame_rate": ratio(stream["avg_frame_rate"]),
        "stream_duration_seconds": float(stream["duration"]) if stream.get("duration") not in (None, "N/A") else None,
        "format_duration_seconds": float(format_data["duration"]) if format_data.get("duration") else None,
    }


def check_example(example):
    video_text = example.get("video")
    trace_text = example.get("trace_path")
    video = resolve_under_results(video_text)
    trace = resolve_under_results(trace_text)
    metadata_path = video.with_suffix(".json")
    checks = {}
    details = {"video": str(video.relative_to(ROOT)),
               "metadata": str(metadata_path.relative_to(ROOT)),
               "trace": str(trace.relative_to(ROOT)),
               "episode_id": example.get("episode_id"),
               "group": example.get("group"), "type": example.get("type")}

    checks["video_exists_nonempty"] = video.is_file() and video.stat().st_size > 0
    details["video_bytes"] = video.stat().st_size if video.is_file() else 0
    checks["trace_exists"] = trace.is_file()
    checks["metadata_exists"] = metadata_path.is_file()
    metadata = None
    if checks["metadata_exists"]:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata is None:
        checks.update({"metadata_episode_matches": False, "metadata_trace_matches": False,
                       "no_new_physics": False, "final_trace_success_is_false": False,
                       "metadata_fps_valid": False, "metadata_frames_valid": False,
                       "metadata_duration_is_derived_target": False})
    else:
        checks["metadata_episode_matches"] = (
            metadata.get("source_episode_index_zero_based") == example.get("episode_id"))
        try:
            metadata_trace = resolve_under_results(metadata.get("trace", ""))
            checks["metadata_trace_matches"] = metadata_trace == trace
        except (ValueError, OSError):
            checks["metadata_trace_matches"] = False
        checks["no_new_physics"] = metadata.get("new_physics_simulated") is False
        checks["final_trace_success_is_false"] = metadata.get("observed_final_success_in_trace") is False
        fps = metadata.get("fps")
        frames = metadata.get("rendered_frame_count")
        try:
            fps = float(fps)
            checks["metadata_fps_valid"] = math.isfinite(fps) and fps > 0
        except (TypeError, ValueError):
            fps = None
            checks["metadata_fps_valid"] = False
        checks["metadata_frames_valid"] = isinstance(frames, int) and not isinstance(frames, bool) and frames > 0
        duration = frames / fps if checks["metadata_frames_valid"] and checks["metadata_fps_valid"] else None
        checks["metadata_duration_is_derived_target"] = (
            frames == EXPECTED_FRAMES and fps is not None and
            math.isclose(fps, EXPECTED_FPS, rel_tol=0, abs_tol=1e-8) and
            duration is not None and math.isclose(duration, EXPECTED_DURATION, rel_tol=0, abs_tol=1e-8))
        details["metadata_rendered_frame_count"] = frames
        details["metadata_fps"] = fps
        details["expected_duration_from_metadata_seconds"] = duration

    probe = None
    probe_error = None
    if checks["video_exists_nonempty"]:
        try:
            probe = ffprobe(video)
        except (OSError, subprocess.CalledProcessError, json.JSONDecodeError, KeyError,
                TypeError, ValueError) as exc:
            probe_error = f"{type(exc).__name__}: {exc}"
    details["ffprobe"] = probe
    if probe is None:
        checks.update({"decoded_frame_count_matches_metadata": False,
                       "nb_frames_matches_metadata": False,
                       "fps_matches_metadata": False,
                       "duration_matches_metadata": False})
        if probe_error:
            details["ffprobe_error"] = probe_error
    else:
        metadata_frames = metadata.get("rendered_frame_count") if metadata else None
        metadata_fps = metadata.get("fps") if metadata else None
        expected_duration = (metadata_frames / metadata_fps
                             if isinstance(metadata_frames, int) and isinstance(metadata_fps, (int, float))
                             and metadata_fps > 0 else None)
        checks["decoded_frame_count_matches_metadata"] = probe["decoded_frames"] == metadata_frames
        checks["nb_frames_matches_metadata"] = probe["nb_frames"] == metadata_frames
        checks["fps_matches_metadata"] = (
            metadata_fps is not None and
            math.isclose(probe["avg_frame_rate"], float(metadata_fps), rel_tol=0, abs_tol=1e-8) and
            math.isclose(probe["r_frame_rate"], float(metadata_fps), rel_tol=0, abs_tol=1e-8))
        checks["duration_matches_metadata"] = (
            expected_duration is not None and
            probe["stream_duration_seconds"] is not None and
            probe["format_duration_seconds"] is not None and
            math.isclose(probe["stream_duration_seconds"], expected_duration,
                         rel_tol=0, abs_tol=1e-3) and
            math.isclose(probe["format_duration_seconds"], expected_duration,
                         rel_tol=0, abs_tol=1e-3))

    return {**details, "checks": checks, "passed": all(checks.values())}


def count_contract(manifest, failures):
    categories = failures.get("order")
    examples = manifest.get("examples")
    missing_after = manifest.get("missing_after_examples", {})
    if not isinstance(categories, list) or not isinstance(examples, list):
        raise ValueError("Failure summary order or video manifest examples is missing")
    counts = {}
    for group in ("before", "after"):
        for category in categories:
            counts[(group, category)] = sum(
                1 for example in examples
                if example.get("group") == group and example.get("type") == category)
    checks = []
    before_counts = failures["before"]["counts"]
    after_counts = failures["after_pooled"]["counts"]
    for category in categories:
        desired_before = min(2, before_counts[category])
        actual_before = counts[("before", category)]
        checks.append({"group": "before", "type": category,
                       "desired": desired_before, "actual": actual_before,
                       "logged_missing": 0, "passed": actual_before == desired_before})
        desired_after = min(2, after_counts[category])
        actual_after = counts[("after", category)]
        logged_missing = missing_after.get(category, 0)
        checks.append({"group": "after", "type": category,
                       "desired": desired_after, "actual": actual_after,
                       "logged_missing": logged_missing,
                       "passed": actual_after + logged_missing == desired_after})
    return checks


def main():
    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        failures = json.loads(FAILURES_PATH.read_text(encoding="utf-8"))
        count_checks = count_contract(manifest, failures)
        examples = [check_example(example) for example in manifest["examples"]]
        passed = all(row["passed"] for row in examples) and all(row["passed"] for row in count_checks)
        result = {
            "manifest": str(MANIFEST_PATH.relative_to(ROOT)),
            "failure_summary": str(FAILURES_PATH.relative_to(ROOT)),
            "video_count": len(examples),
            "count_checks": count_checks,
            "videos": examples,
            "passed": passed,
        }
        OUTPUT_PATH.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                               encoding="utf-8")
        print(json.dumps({"video_count": len(examples), "passed": passed,
                          "output": str(OUTPUT_PATH.relative_to(ROOT))}, indent=2))
        return 0 if passed else 1
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        print(f"Goal 9 video audit failed to run: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
