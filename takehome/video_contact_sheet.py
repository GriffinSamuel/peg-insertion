"""Small report previews of local failure videos, without any new simulation."""
import io
import json
from pathlib import Path
import subprocess
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/takehome/goal9'


def main():
    manifest = json.loads((OUT / 'video_manifest.json').read_text())
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
    sources = []
    for group in ('before', 'after'):
        examples = [e for e in manifest['examples'] if e['group'] == group]
        if not examples:
            continue
        columns = 2
        canvas = Image.new('RGB', (1280, 410*((len(examples)+columns-1)//columns)+55), 'white')
        draw = ImageDraw.Draw(canvas)
        draw.text((15,15), f'{group.title()}: recorded illustrative repeats, local videos retain full resolution',
                  font=font, fill='black')
        for index, example in enumerate(examples):
            path = ROOT / example['video']
            if not path.resolve().is_relative_to(OUT) or not path.is_file():
                raise ValueError(f'Missing local video: {path}')
            meta = json.loads(path.with_suffix('.json').read_text())
            seconds = min(3.8, (meta['rendered_frame_count']-1)/meta['fps'])
            frame = subprocess.check_output(['ffmpeg','-loglevel','error','-ss',str(seconds),
                '-i',str(path),'-frames:v','1','-f','image2pipe','-vcodec','png','pipe:1'])
            image = Image.open(io.BytesIO(frame)).convert('RGB').resize((640,360))
            x, y = (index % columns)*640, 55+(index//columns)*410
            canvas.paste(image, (x,y))
            label = f"{example['type']} | seed {example['practice_seed']} | episode {example['episode_id']}"
            draw.text((x+8,y+367), label, font=font, fill='black')
            sources.append(dict(group=group,video=example['video'],time_seconds=seconds,
                                source_episode=example['episode_id']))
        canvas.save(OUT / f'{group}_examples.png', optimize=True)
    (OUT / 'preview_sources.json').write_text(json.dumps(sources,indent=2)+'\n')
    print(json.dumps(dict(previewed_videos=len(sources),new_simulator_episodes=0)))


if __name__ == '__main__':
    main()
