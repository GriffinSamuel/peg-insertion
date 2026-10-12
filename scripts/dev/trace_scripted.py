import os, sys
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), 'scripts'))
import jax, jax.numpy as jp, numpy as np
from peg_insertion.env import PegInsertion, default_config
import scripted_insert as si
env = PegInsertion()
reset = jax.jit(env.reset); step = jax.jit(env.step)
act_fn = jax.jit(lambda s, d: si.scripted_action(env, s, d, jp.zeros(2)))
s = reset(jax.random.PRNGKey(0)); desc = jp.array(False)
for t in range(120):
    a, desc = act_fn(s, desc)
    tip = np.array(s.data.site_xpos[env._tip_site]); ent = np.array(s.data.site_xpos[env._entrance_site])
    if t % 5 == 0 or t < 5:
        print(t, "tip-ent mm", ((tip-ent)*1000).round(2), "cmd-ent mm", ((np.array(s.info['current_pos'])-ent)*1000).round(2), "a", np.array(a).round(2), "desc", bool(desc), "r", float(s.reward), "q", np.array(s.data.qpos[:7]).round(3), flush=True)
    s = step(s, a)
