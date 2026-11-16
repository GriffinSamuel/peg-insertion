"""Rebuild the saved temporal video-review sheet without new simulation."""
import io
import json
from pathlib import Path
import subprocess
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/takehome/goal9'


def main():
    records = json.loads((OUT / 'motion_review_sources.json').read_text())
    canvas = Image.new('RGB', (1280, 400*((len(records)+1)//2)), 'white')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 16)
    for index, record in enumerate(records):
        path = (ROOT / record['video']).resolve()
        if not path.is_relative_to(OUT):
            raise ValueError('Review source must be a local Goal9 video')
        raw = subprocess.check_output(['ffmpeg', '-loglevel', 'error', '-ss',
            str(record['seconds']), '-i', str(path), '-frames:v', '1', '-f',
            'image2pipe', '-vcodec', 'png', 'pipe:1'])
        frame = Image.open(io.BytesIO(raw)).convert('RGB')
        frame.thumbnail((640, 360))
        x, y = (index % 2)*640, (index // 2)*400
        canvas.paste(frame, (x, y))
        draw.text((x+8, y+365), f"{path.stem}: {record['seconds']:.1f}s",
                  font=font, fill='black')
    canvas.save(OUT / 'motion_review.png')


if __name__ == '__main__':
    main()
