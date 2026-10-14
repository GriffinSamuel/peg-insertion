"""Make the README GIF and the 0.5 mm side-by-side comparison video.

Reads the rendered rollout videos (scripts/render.py), never the policies:
  results/insertion.gif               first episode of force_0.5mm_s0, < 5 MB
  results/videos/compare_0.5mm.mp4    pos_0.5mm_s0 | force_0.5mm_s0, same episodes

Usage: python scripts/make_media.py
"""

import subprocess
import sys
from pathlib import Path

import mediapy
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
VIDEOS = ROOT / 'results' / 'videos'
GIF_SOURCE = VIDEOS / 'force_0.5mm_s0.mp4'
GIF_OUT = ROOT / 'results' / 'insertion.gif'
GIF_MAX_BYTES = 5 * 1024 * 1024
GIF_WIDTHS = (720, 640, 560, 480)  # first width that fits under the limit wins
GIF_FPS = 12.5
EPISODE_FRAMES = 101  # 200 control steps rendered every 2nd step, plus step 0
COMPARE = ('pos_0.5mm_s0', 'force_0.5mm_s0')
COMPARE_OUT = VIDEOS / 'compare_0.5mm.mp4'
COMPARE_SIZE = (960, 540)  # per panel; 1928x540 side by side with an 8 px divider


def make_gif() -> Path:
  for width in GIF_WIDTHS:
    vf = (
        f'trim=end_frame={EPISODE_FRAMES},fps={GIF_FPS},'
        f'scale={width}:-1:flags=lanczos,split[a][b];'
        '[a]palettegen=stats_mode=full[p];[b][p]paletteuse=dither=bayer:bayer_scale=4'
    )
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-i', str(GIF_SOURCE), '-vf', vf, str(GIF_OUT)]
    subprocess.run(cmd, check=True)
    size = GIF_OUT.stat().st_size
    print(f'gif width {width}: {size / 1e6:.2f} MB', flush=True)
    if size < GIF_MAX_BYTES:
      return GIF_OUT
  raise RuntimeError(f'GIF stays over {GIF_MAX_BYTES} bytes even at width {GIF_WIDTHS[-1]}')


def _resize(frames: np.ndarray) -> np.ndarray:
  return np.stack([np.array(Image.fromarray(f).resize(COMPARE_SIZE, Image.LANCZOS))
                   for f in frames])


def make_compare() -> Path:
  left, right = (mediapy.read_video(str(VIDEOS / f'{run}.mp4')) for run in COMPARE)
  if len(left) != len(right):
    raise RuntimeError(f'frame counts differ: {len(left)} vs {len(right)}')
  fps = left.metadata.fps
  gap = np.full((len(left), COMPARE_SIZE[1], 8, 3), 255, dtype=np.uint8)
  frames = np.concatenate([_resize(left), gap, _resize(right)], axis=2)
  mediapy.write_video(str(COMPARE_OUT), frames, fps=fps)
  print(f'wrote {COMPARE_OUT} ({len(frames)} frames, {frames.shape[2]}x{frames.shape[1]})',
        flush=True)
  return COMPARE_OUT


def main() -> int:
  print(f'wrote {make_gif()}', flush=True)
  make_compare()
  return 0


if __name__ == '__main__':
  sys.exit(main())
