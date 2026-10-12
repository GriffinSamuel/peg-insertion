"""Dev probe: jit the MJX env, run the scripted controller, report numbers."""
import os, sys, time
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), 'scripts'))
import jax, jax.numpy as jp, numpy as np
from peg_insertion.env import PegInsertion, default_config
import scripted_insert as si

servo = float(sys.argv[1]) if len(sys.argv) > 1 else 0.005
for c in [float(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else (2.0, 1.0, 0.5):
    cfg = default_config(); cfg.clearance_mm = c; cfg.max_servo_error = servo
    env = PegInsertion(cfg)
    t = time.time()
    r = si.run_scripted(env, n=16, return_traj=False)
    print(f"servo {servo} clearance {c}: {r}  ({time.time()-t:.1f}s)", flush=True)
    t = time.time()
    r = si.run_scripted(env, n=16, offset_xy=(0.0015, 0.0))
    print(f"   offset1.5mm: {r}  ({time.time()-t:.1f}s)", flush=True)
