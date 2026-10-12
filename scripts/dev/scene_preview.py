"""Write scene XMLs and render results/scene_preview.png (closeup + overview)."""
import os, sys
sys.path.insert(0, os.getcwd())
os.environ.setdefault('MUJOCO_GL', 'osmesa')
import numpy as np, mujoco, mediapy
from peg_insertion import xml_builder
from peg_insertion.env import PegInsertion
for c in (2.0, 1.0, 0.5):
    print(xml_builder.write_scene(c))
xml = (xml_builder.ASSETS_DIR / 'scene_c2.0.xml').read_text()
m = mujoco.MjModel.from_xml_string(xml, assets=xml_builder.get_assets())
e = PegInsertion()
d = mujoco.MjData(m)
q = e._solve_ik_np(np.array([0.5, 0.0, 0.045]) - e._tip_from_ik, e._home_ref_q)
d.qpos[:7] = q; d.qpos[7:9] = e._grip
mujoco.mj_forward(m, d)
imgs = []
r = mujoco.Renderer(m, 480, 640)
for cam in ('closeup', 'overview'):
    r.update_scene(d, camera=cam); imgs.append(r.render().copy())
mediapy.write_image('results/scene_preview.png', np.concatenate(imgs, axis=1))
print('wrote results/scene_preview.png')
