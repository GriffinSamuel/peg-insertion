"""Dev probe: model geometry, orientation, gravity sag and force sensor (CPU MuJoCo)."""
import os, sys, time
sys.path.insert(0, os.getcwd())
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np, mujoco
from peg_insertion import xml_builder
from peg_insertion.env import PegInsertion

t = time.time()
e = PegInsertion()
print("env build s", round(time.time() - t, 1))
m = e.mj_model
print("nq nv nu nbody ngeom npair nsensor", m.nq, m.nv, m.nu, m.nbody, m.ngeom, m.npair, m.nsensordata)
print("tip_rot\n", e._tip_rot, "\nhand rot at ref\n", e._hand_rot_check.round(4))
print("tip_from_ik", e._tip_from_ik, "home_ref_q", e._home_ref_q.round(3))
d = mujoco.MjData(m)
fadr = m.sensor_adr[m.sensor('peg_force').id]
def settle(tip, n=200):
    q = e._solve_ik_np(np.array(tip) - e._tip_from_ik, e._home_ref_q)
    d.qpos[:7] = q; d.qpos[7:9] = e._grip; d.qvel[:] = 0; d.ctrl[:7] = q; d.ctrl[7] = e._grip
    mujoco.mj_forward(m, d)
    for _ in range(n): mujoco.mj_step(m, d)
    tipp = d.site_xpos[m.site('peg_tip').id]
    return tipp, d.sensordata[fadr:fadr+3].copy()
for tgt in ([0.5, 0, 0.08], [0.49, 0.01, 0.031], [0.51, -0.01, 0.031], [0.35, 0.15, 0.10]):
    tip, f = settle(tgt)
    print("target", tgt, "tip", tip.round(5), "sag(mm)", ((tip - tgt) * 1000).round(2), "F", f.round(3), "|F|", round(float(np.linalg.norm(f)), 3))
tip, f = settle([0.35, 0.15, -0.005]); print("pressed tip", tip.round(4), "|F|", np.linalg.norm(f).round(2))
print("contacts", d.ncon)
# scripted straight insertion in plain mujoco at 2 mm (true hole at 0.5, 0)
for c in (2.0, 1.0, 0.5):
    ec = PegInsertion(config_overrides={'clearance_mm': c})
    mc = ec.mj_model; dc = mujoco.MjData(mc)
    q = ec._solve_ik_np(np.array([0.5, 0, 0.06]) - ec._tip_from_ik, ec._home_ref_q)
    dc.qpos[:7] = q; dc.qpos[7:9] = ec._grip; dc.ctrl[:7] = q; dc.ctrl[7] = ec._grip
    mujoco.mj_forward(mc, dc)
    cur = np.array([0.5, 0, 0.06]); peak = 0
    for k in range(80):
        tip = dc.site_xpos[mc.site('peg_tip').id].copy()
        cur = cur + np.clip((np.array([0.5, 0, tip[2] - 0.001]) - cur) , -0.002, 0.002)
        cur = np.clip(cur, tip - 0.005, tip + 0.005)
        qq = ec._solve_ik_np(cur - ec._tip_from_ik, dc.ctrl[:7].copy())
        dc.ctrl[:7] = qq
        for _ in range(4): mujoco.mj_step(mc, dc)
        peak = max(peak, np.linalg.norm(dc.sensordata[fadr:fadr+3]))
    tip = dc.site_xpos[mc.site('peg_tip').id]
    print(f"clearance {c}: final tip {tip.round(4)} depth(mm) {(0.03 - tip[2])*1000:.2f} peak|F| {peak:.1f} ncon {dc.ncon}")
