"""Render one recorded trace episode for illustration; no rollout is run."""

import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from takehome import runtime  # noqa: E402,F401
os.environ['JAX_PLATFORMS'] = 'cpu'  # Pose replay uses MuJoCo/OSMesa, not GPU physics.


def project_path(value):
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def path_in_results(path):
    return path.is_relative_to(ROOT / "results" / "takehome")


def load_trace(path, episode):
    required = ("qpos", "mocap_pos", "pose_succeeded", "force",
                "next_guess", "true_bias")
    with np.load(path, allow_pickle=False) as archive:
        missing = [key for key in required if key not in archive]
        if missing:
            raise ValueError(f"trace lacks required arrays: {missing}")
        arrays = {key: np.asarray(archive[key]) for key in required}
    for key, value in arrays.items():
        if value.ndim < 2:
            raise ValueError(f"{key} must have [time, episode, ...] dimensions")
    lengths = {value.shape[0] for value in arrays.values()}
    episode_counts = {value.shape[1] for value in arrays.values()}
    if len(lengths) != 1 or len(episode_counts) != 1:
        raise ValueError("trace arrays have inconsistent time or episode dimensions")
    if not 0 <= episode < next(iter(episode_counts)):
        raise ValueError(f"episode must be zero-based in [0, {next(iter(episode_counts)) - 1}]")
    for key, value in arrays.items():
        if not np.issubdtype(value.dtype, np.number) and value.dtype != np.bool_:
            raise ValueError(f"{key} must be numeric or boolean")
        if not np.isfinite(value).all():
            raise ValueError(f"{key} contains non-finite values")
    if arrays["next_guess"].shape[-1] < 2 or arrays["true_bias"].shape[-1] < 2:
        raise ValueError("next_guess and true_bias must contain XY coordinates")
    return {key: value[:, episode] for key, value in arrays.items()}


def build_render_env(policy_name):
    policy_dir = (ROOT / "results" / "policies" / policy_name).resolve()
    if not policy_dir.is_relative_to(ROOT / "results" / "policies"):
        raise ValueError("policy must be a name under results/policies")
    network_config_path = policy_dir / "network_config.json"
    if not network_config_path.is_file():
        raise FileNotFoundError(f"policy network config is missing: {network_config_path}")
    network_config = json.loads(network_config_path.read_text(encoding="utf-8"))

    # MuJoCo's GL backend must be selected before importing MuJoCo or render.py.
    os.environ["MUJOCO_GL"] = "osmesa"
    scripts_path = ROOT / "scripts"
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(scripts_path))
    import mujoco
    from peg_insertion import xml_builder

    defaults = {
        "clearance_mm": network_config.get("clearance_mm", 2.0),
        "ctrl_dt": 0.02,
        "sim_dt": 0.005,
        "peg_width": xml_builder.PEG_WIDTH,
        "peg_len": xml_builder.PEG_LEN,
        "hole_depth": xml_builder.HOLE_DEPTH,
    }
    overrides = network_config.get("env_overrides", {}) or {}
    for key in defaults:
        if key in overrides:
            defaults[key] = overrides[key]
    clearance_mm = float(network_config.get("clearance_mm", defaults["clearance_mm"]))
    clearance = clearance_mm / 1000.0
    peg_width = float(defaults["peg_width"])
    peg_len = float(defaults["peg_len"])
    hole_depth = float(defaults["hole_depth"])
    assets = xml_builder.get_assets(peg_width, peg_len)
    xml = xml_builder.build_xml(clearance, peg_width, peg_len, hole_depth)
    mj_model = mujoco.MjModel.from_xml_string(xml, assets=assets)
    mj_model.opt.timestep = float(defaults["sim_dt"])
    env = SimpleNamespace(
        clearance=clearance, _pw=peg_width, _pl=peg_len, _hd=hole_depth,
        _model_assets=assets, mj_model=mj_model,
        _config=SimpleNamespace(ctrl_dt=float(defaults["ctrl_dt"])),
    )
    return env, network_config_path, defaults


def add_correction_overlay(frame, next_guess, true_bias, font_factory, episode):
    from PIL import Image, ImageDraw

    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    font = font_factory(20)
    x0, y0 = 555, 16
    width, height = 660, 115
    draw.rounded_rectangle((x0, y0, x0 + width, y0 + height), radius=8,
                           fill=(0, 0, 0, 205), outline=(240, 240, 240, 220), width=1)
    draw.text((x0 + 12, y0 + 8), "Camera error in XY (mm)",
              fill=(255, 255, 255), font=font)
    guess = np.asarray(next_guess[:2], dtype=float) * 1000.0
    bias = np.asarray(true_bias[:2], dtype=float) * 1000.0
    draw.text((x0 + 12, y0 + 39),
              f"Guess: ({guess[0]:+.3f}, {guess[1]:+.3f})",
              fill=(130, 220, 255), font=font)
    draw.text((x0 + 340, y0 + 39),
              f"True error: ({bias[0]:+.3f}, {bias[1]:+.3f})",
              fill=(255, 205, 120), font=font)
    draw.text((x0 + 12, y0 + 76),
              f"Recorded illustrative repeat; source episode {episode} (zero-based)",
              fill=(255,255,255),font=font_factory(16))
    return np.asarray(image)


def render(trace_path, episode, policy, title, output):
    trace_path, output = project_path(trace_path), project_path(output)
    if not trace_path.is_file():
        raise FileNotFoundError(f"trace file is missing: {trace_path}")
    if not path_in_results(output) or output.suffix.lower() != ".mp4":
        raise ValueError("--output must be an MP4 under results/takehome")
    metadata_path = output.with_suffix(".json")
    if output.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if (metadata.get('source_episode_index_zero_based') == episode and
                metadata.get('policy_name_for_render_geometry') == policy and
                project_path(metadata.get('trace', '')) == trace_path and output.stat().st_size > 0):
            return output, metadata_path
        raise ValueError('Existing video metadata does not match the requested recorded episode')
    if output.exists() or metadata_path.exists():
        raise FileExistsError(f"output or adjacent metadata already exists: {output}")

    selected = load_trace(trace_path, episode)
    env, network_config_path, env_config = build_render_env(policy)
    import mediapy
    import render as base_render

    frame_data = {
        "qpos": selected["qpos"],
        "mocap_pos": selected["mocap_pos"],
        "succeeded": selected["pose_succeeded"],
        "force": selected["force"],
    }
    if frame_data["succeeded"].ndim != 1 or frame_data["force"].ndim != 1:
        raise ValueError("pose_succeeded and force must be scalar per time step")
    frames = base_render.frames_for(env, [frame_data], title)
    indices = range(0, len(selected["qpos"]), base_render.RENDER_EVERY)
    rendered_indices = list(indices)
    if len(frames) != len(rendered_indices):
        raise RuntimeError("renderer produced an unexpected number of frames")
    frames = [add_correction_overlay(frame, selected["next_guess"][index],
                                     selected["true_bias"][index], base_render._font, episode)
              for frame, index in zip(frames, rendered_indices)]
    output.parent.mkdir(parents=True, exist_ok=True)
    fps = 1. / (env._config.ctrl_dt * base_render.RENDER_EVERY)
    mediapy.write_video(str(output), frames, fps=fps)
    metadata = {
        "trace": str(trace_path.relative_to(ROOT)) if trace_path.is_relative_to(ROOT) else str(trace_path),
        "source_episode_index_zero_based": episode,
        "policy_name_for_render_geometry": policy,
        "network_config": str(network_config_path.relative_to(ROOT)),
        "title": title,
        "output": str(output.relative_to(ROOT)),
        "fps": fps,
        "rendered_frame_count": len(frames),
        "recorded_illustrative_repeat": True,
        "new_physics_simulated": False,
        "observed_final_success_in_trace": bool(selected["pose_succeeded"][-1]),
        "outcome_note": "This records the selected trace's final success flag; it does not assert identity with a separately scored episode.",
        "environment_geometry_and_timing": env_config,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n",
                             encoding="utf-8")
    return output, metadata_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--episode", type=int, required=True,
                        help="zero-based episode index in the trace NPZ")
    parser.add_argument("--policy", required=True,
                        help="policy name under results/policies, used for render geometry")
    parser.add_argument("--title", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output, metadata = render(args.trace, args.episode, args.policy,
                              args.title, args.output)
    print(json.dumps({"video": str(output.relative_to(ROOT)),
                      "metadata": str(metadata.relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()
