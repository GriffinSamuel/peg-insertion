"""Contracts for the explicitly revised camera conditions and preserved restart."""
from takehome import followup_runtime  # noqa: F401
import jax.numpy as jp
import numpy as np
from takehome.followup_config import FollowupConfig
from takehome.followup_env import FollowupEnv
from takehome.variants import VariantEnv
from takehome.followup_reference import reference_path
from takehome.followup_paths import run_path
from takehome.test_followup_controls import _env,_state


def test_mixed_bias_keeps_episode_draw_without_rescaling_at_each_step(monkeypatch):
    cfg=FollowupConfig(camera_condition='mixed',mixed_fixed_std=.00065,mixed_episode_std=.000375)
    env=object.__new__(FollowupEnv)
    env.followup_config=cfg
    env.fixed_bias=jp.asarray([.0004,-.0002])
    monkeypatch.setattr(VariantEnv,'_get_obs',lambda self,data,info,action,rng:info['bias'])
    first={'bias':jp.asarray([.00075,-.0015,0.])}
    a=env._get_obs(None,first,None,None)
    np.testing.assert_allclose(a,[.000775,-.00095,0.],rtol=0,atol=1e-9)
    for _ in range(5):
        np.testing.assert_array_equal(env._get_obs(None,first,None,None),a)
    second={'bias':jp.asarray([-.00075,.0015,0.])}
    b=env._get_obs(None,second,None,None)
    np.testing.assert_allclose(b,[.000025,.00055,0.],rtol=0,atol=1e-9)
    np.testing.assert_allclose((a[:2]+b[:2])/2,env.fixed_bias,atol=1e-9)


def test_reference_identity_contains_mixed_split_and_fixed_seed():
    paths=[]
    for fixed,fresh in ((.00065,.000375),(.00053,.00053),(.000375,.00065)):
        for seed in (0,1):
            cfg=FollowupConfig(camera_condition='mixed',mixed_fixed_std=fixed,
                               mixed_episode_std=fresh,fixed_bias_seed=seed)
            paths.append(reference_path(cfg))
    assert len(set(paths))==6


def test_restart_path_never_reuses_halted_ledger():
    base=dict(phase='B',name='B_aim_fixed',seed_paths={'4':'B/B_aim_fixed/seed_4_restart_1'})
    assert run_path(base,4).name=='seed_4_restart_1'
    assert run_path(base,3).name=='seed_3'


def test_aim_near_hole_gate_is_restored_for_fixed_condition_trick():
    env=_env(near_hole=True)
    far=_state(guess=(.001,-.001),z=.02)
    np.testing.assert_array_equal(env.next_guess(far,jp.ones(2)),jp.zeros(2))
    close=_state(guess=(0.,0.),z=.001)
    np.testing.assert_allclose(env.next_guess(close,jp.ones(2)),[.0001,.0001],atol=1e-9)
