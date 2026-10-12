"""Dev diagnostic: roll out a trained policy, count terminations/NaNs, show where the tip ends."""
import os, sys, functools
sys.path.insert(0, os.getcwd())
import jax, jax.numpy as jp, numpy as np
import peg_insertion
from peg_insertion import rollout
from peg_insertion.env import domain_randomize
from mujoco_playground._src import wrapper as pw
run = sys.argv[1]; det = (sys.argv[2] if len(sys.argv) > 2 else 'det') == 'det'
d = f'logs/{run}' if os.path.exists(f'logs/{run}/params.pkl') else f'results/archive/lr5e-4_10M/policies/{run}'
policy, cfg = rollout.load_policy(d, deterministic=det)
extra = {k: v for k, v in cfg['env_overrides'].items() if k not in ('clearance_mm','use_force_obs','naconmax')}
env = rollout.make_env(cfg['clearance_mm'], cfg['use_force_obs'], naconmax=20*256, **extra)
n = 256
wenv = pw.BraxDomainRandomizationVmapWrapper(env, functools.partial(domain_randomize, rng=jax.random.split(jax.random.PRNGKey(1), n)))
@jax.jit
def run_(key):
    s = wenv.reset(jax.random.split(key, n))
    def body(c, _):
        s, k = c; k, sub = jax.random.split(k)
        a, _ = policy(s.obs, sub); s2 = wenv.step(s, a)
        tip = s2.data.site_xpos[:, env._tip_site]; ent = s2.data.site_xpos[:, env._entrance_site]
        F = jp.linalg.norm(s2.data.sensordata[:, env._force_slice], axis=-1)
        return (s2, k), (s2.done, jp.isnan(s2.data.qpos).any(-1), tip - ent, F, a, s2.reward)
    (s, _), tr = jax.lax.scan(body, (s, key), None, length=200)
    return s.info, tr, s.info['bias']
info, (done, nan, rel, F, act, rew), bias = jax.tree.map(np.array, run_(jax.random.PRNGKey(5)))
print(run, 'det' if det else 'stoch', 'success', info['succeeded'].mean())
print('episodes with done flag:', (done.sum(0) > 0).mean(), ' with NaN:', (nan.sum(0) > 0).mean())
for t in (0, 25, 50, 100, 150, 199):
    r = rel[t] * 1000
    print(f't={t:3d} rel mm: z mean {r[:,2].mean():6.2f} (min {r[:,2].min():6.2f})  |xy| mean {np.linalg.norm(r[:,:2],axis=-1).mean():5.2f}  F mean {F[t].mean():5.1f}  |a| {np.abs(act[t]).mean(0).round(2)}  rew {rew[t].mean():.3f}')
bxy = np.linalg.norm(bias[:, :2], axis=-1) * 1000
fin = np.linalg.norm(rel[-1][:, :2], axis=-1) * 1000
print('final |xy| vs |bias| corr', np.corrcoef(fin, bxy)[0,1].round(2), ' final z<0.5mm above entrance frac', (rel[-1][:,2] < 0.0005).mean())
