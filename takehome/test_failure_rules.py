"""Synthetic ordering/exhaustiveness check; no experimental traces are read."""
import numpy as np
from takehome.failure_rules import classify


def test_failure_rules_partition_and_precedence():
    t,n=200,6
    a=dict(episode_success=np.array([0,0,0,0,0,1]),valid=np.ones((t,n),bool),
        obs=np.zeros((t,n,18)),guess=np.zeros((t,n,2)),next_guess=np.zeros((t,n,2)),
        true_bias=np.zeros((t,n,2)),force=np.ones((t,n))*10,xy_err=np.ones((t,n))*.001,
        tip_height=np.zeros((t,n)),movement=np.zeros((t,n,3)),depth=np.zeros((t,n)),
        truncated=np.ones((t,n),bool),terminated=np.zeros((t,n),bool))
    a['force'][:,0]=0  # no contact wins even if correction is also wrong
    a['next_guess'][:,0:2,0]=.001
    a['depth'][:,3:5]=np.linspace(0,.02,t)[:,None]  # >1mm change in last40 steps avoids rim rule
    a['truncated'][:,4]=False;a['terminated'][:,4]=True
    result=classify(a)
    assert result['labels']==['never_near_hole_contact','wrong_way_correction','stuck_on_rim','timeout','other','success']
    assert sum(result['counts'].values())==5
