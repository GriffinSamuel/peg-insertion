"""Reuse the original independent-pair update schedule with follow-up state."""
from takehome import followup_runtime  # noqa: F401
from dataclasses import replace
from flax import struct
import jax.numpy as jp
from takehome.config import action_scales
from takehome.coordination import CoordinatedLearner
from takehome.followup_learner import FollowupLearner


@struct.dataclass
class FollowupCoordinatedState:
    aim: object
    movement: object

    @property
    def actor(self):return (self.aim.actor,self.movement.actor)
    @property
    def critics(self):return (self.aim.critics,self.movement.critics)
    @property
    def updates(self):return self.aim.updates+self.movement.updates
    @property
    def start_guess(self):return self.aim.start_guess


class FollowupCoordinatedLearner(CoordinatedLearner):
    def __init__(self,config,env):
        self.config=config;self.base_size=env.base_obs_size
        if any((config.memory,config.heatmap,config.success_labels,config.within_grid)):
            raise ValueError('Sequential comparison uses the plain aim and movement components')
        common=dict(coordination='joint',final_decider=False,contact_cap=False,
                    uncertainty_cap=False,force_penalty=False)
        self.aim=FollowupLearner(replace(config,outputs=2,**common),env)
        self.movement=FollowupLearner(replace(config,outputs=3,**common),env)
        self.normalizer=self.aim.normalizer
        self.action_scale=jp.array(action_scales(5,config.step_cap,config.movement_cap))

    def initialize(self,seed):
        return FollowupCoordinatedState(self.aim.initialize(seed),self.movement.initialize(seed))

    def train_block(self,*args,**kwargs):
        state,metrics=super().train_block(*args,**kwargs)
        return FollowupCoordinatedState(state.aim,state.movement),metrics
