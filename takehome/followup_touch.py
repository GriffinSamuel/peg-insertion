"""Observable-only Phase F search controllers, separate from learning methods."""
from takehome import followup_runtime  # noqa: F401
import jax.numpy as jp
from takehome.followup_env import FollowupEnv, stalled
from takehome.followup_episodes import FollowupEpisodes

SPIRAL_SPACING = .0005
TOUCH_STEP_CAP = .0002


def spiral_proposal(current, center, theta, active, guess_cap=.003):
    growth = SPIRAL_SPACING/(2*jp.pi)
    radius = growth*theta
    turn = jp.minimum(jp.pi/2, TOUCH_STEP_CAP/jp.sqrt(radius*radius+growth*growth))
    next_theta = theta+turn
    target = center+growth*next_theta*jp.stack([jp.cos(next_theta), jp.sin(next_theta)])
    proposal = jp.clip(current+jp.clip(target-current, -TOUCH_STEP_CAP, TOUCH_STEP_CAP),
                       -guess_cap, guess_cap)
    return jp.where(active, proposal, current), jp.where(active, next_theta, theta)


class TouchEnv(FollowupEnv):
    def __init__(self, *args, touch_method='blind_spiral', **kwargs):
        if touch_method != 'blind_spiral':
            raise ValueError('Only the predeclared blind spiral is implemented yet')
        self.touch_method = touch_method
        super().__init__(*args, **kwargs)

    def reset(self, key):
        state = super().reset(key)
        return state.replace(info={**state.info, 'touch_theta': jp.array(0.),
            'touch_center': jp.zeros(2), 'touch_started': jp.array(False),
            'touch_active': jp.array(False), 'touch_steps': jp.array(0, dtype=jp.int32)})

    def touch_update(self, state):
        raw = state.obs['state'][:self.base_obs_size]
        active = (state.info['contact_seen'] & stalled(state.info['reading_history'])
            & (raw[2] < .001) & (state.info['step_count'] >= 10)
            & ~state.info['learner_terminated'] & ~state.info['learner_truncated'])
        center = jp.where(state.info['touch_started'], state.info['touch_center'], state.info['aim_guess'])
        guess, theta = spiral_proposal(state.info['aim_guess'], center,
                                      state.info['touch_theta'], active, self.aim.guess_cap)
        return guess, theta, center, active

    def next_guess(self, state, action):
        del action
        return self.touch_update(state)[0]

    def step(self, state, action):
        _, theta, center, active = self.touch_update(state)
        nxt = super().step(state, action)
        return nxt.replace(info={**nxt.info, 'touch_theta': theta,
            'touch_center': jp.where(active, center, state.info['touch_center']),
            'touch_started': state.info['touch_started'] | active,
            'touch_active': active,
            'touch_steps': state.info['touch_steps']+active.astype(jp.int32)})


class TouchEpisodes(FollowupEpisodes):
    trace_keys = FollowupEpisodes.trace_keys+('touch_active', 'touch_steps')

    def extra_trace(self, state, nxt):
        # Record the physical aim nudge actually supplied by the hand routine.
        return dict(action=nxt.info['aim_guess']-state.info['aim_guess'],
                    touch_active=nxt.info['touch_active'], touch_steps=nxt.info['touch_steps'])
