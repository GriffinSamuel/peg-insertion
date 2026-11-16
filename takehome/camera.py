"""Goal 11 camera readings, created inside a takehome-only environment subclass."""
from dataclasses import dataclass
from takehome import runtime  # noqa: F401
import jax
import jax.numpy as jp
from takehome.correction import AimConfig
from takehome.variants import VariantConfig, VariantEnv
from takehome.variant_runner import VariantExperimentRunner

PROBLEMS = ('bias_2x','bias_4x','one_sided','drift','jitter','rotation','delay','height')
FORMULAS = {
    'bias_2x': 'r + 2*b + epsilon; b_xy ~ N(0,0.75mm); nudge/guess caps times2',
    'bias_4x': 'r + 4*b + epsilon; b_xy ~ N(0,0.75mm); nudge/guess caps times4',
    'one_sided': 'r_xy + abs(b_xy) + epsilon_xy; Z unchanged',
    'drift': 'r_xy + b_xy + (step/200)*[1mm,-1mm] + epsilon_xy; Z unchanged',
    'jitter': 'r_xy + b_xy + epsilon_xy + eta_xy, eta ~ N(0,(0.6mm)^2 I); Z unchanged',
    'rotation': 'R(10 degrees)*r_xy + b_xy + epsilon_xy; Z unchanged',
    'delay': 'r(max(step-3,0)) + b + epsilon; 60ms geometry delay, present proprioception',
    'height': 'r + b + epsilon + [0,0,2mm]'}


def cap_multiplier(problem):
    return {'bias_2x':2.,'bias_4x':4.}.get(problem,1.)


@dataclass(frozen=True)
class CameraConfig(VariantConfig):
    camera_problem: str = 'normal'

    def __post_init__(self):
        super().__post_init__()
        if self.camera_problem not in ('normal',*PROBLEMS):
            raise ValueError('Unknown predeclared camera problem')


class CameraEnv(VariantEnv):
    def __init__(self,policy_dir,aim,*,config,oracle=False):
        self.camera_problem=config.camera_problem
        self.oracle=oracle
        super().__init__(policy_dir,aim,config=config,
                         obs_hole_bias_std=.00075*cap_multiplier(config.camera_problem))

    def _get_obs(self,data,info,action,rng):
        true_rel=data.site_xpos[self._tip_site]-data.site_xpos[self._entrance_site]
        if 'camera_base_bias' not in info:
            info['camera_base_bias']=info['bias']
            info['camera_history']=jp.broadcast_to(true_rel,(4,3))
        base=info['camera_base_bias']
        info['bias']=base
        original=super()._get_obs(data,info,action,rng)
        history=jp.concatenate([info['camera_history'][1:],true_rel[None,:]],axis=0)
        delta=jp.zeros(3)
        problem=self.camera_problem
        if problem=='one_sided':
            delta=delta.at[:2].set(jp.abs(base[:2])-base[:2])
        elif problem=='drift':
            delta=delta.at[:2].set(info['step_count']/200.*jp.array([.001,-.001]))
        elif problem=='jitter':
            extra=.0006*jax.random.normal(jax.random.fold_in(rng,7001),(2,))
            delta=delta.at[:2].set(extra)
        elif problem=='rotation':
            theta=jp.pi/18
            rotation=jp.array([[jp.cos(theta),-jp.sin(theta)],[jp.sin(theta),jp.cos(theta)]])
            delta=delta.at[:2].set(rotation@true_rel[:2]-true_rel[:2])
        elif problem=='delay':
            delta=history[0]-true_rel
        elif problem=='height':
            delta=delta.at[2].set(.002)
        # Baseline white noise is retained. The recorded effective bias is only
        # the constant error plus the added problem, including any Z component.
        obs=original['state'].at[:3].add(delta)
        obs=obs.at[9].set(jp.maximum(0.,-obs[2]))
        info['camera_history']=history
        info['bias']=base+delta
        return {'state':obs}

    def next_guess(self,state,action):
        if self.oracle:
            return jp.clip(state.info['bias'][:2],-self.aim.guess_cap,self.aim.guess_cap)
        return super().next_guess(state,action)


class CameraExperimentRunner(VariantExperimentRunner):
    def make_env(self,policy_name):
        cfg=self.config
        return CameraEnv(runtime.ROOT/'results/policies'/policy_name,
                         AimConfig(cfg.step_cap,cfg.guess_cap),config=cfg)
