"""Observable-only aim changes and new camera condition in a separate wrapper."""
from takehome import followup_runtime  # noqa: F401
import jax
import jax.numpy as jp
from takehome.variants import VariantEnv


def offset_grid():
    axis = jp.linspace(-.002, .002, 9)
    x,y = jp.meshgrid(axis,axis,indexing='ij')
    return jp.stack([x.ravel(),y.ravel()],axis=-1)


def contact_seen(raw):
    return jp.linalg.norm(raw[...,10:13] * 20., axis=-1) >= 5.


def stalled(history):
    # Camera Z includes the original 0.2mm noise; average five readings each.
    return jp.mean(history[-10:-5,2]) - jp.mean(history[-5:,2]) < .0002


class FollowupEnv(VariantEnv):
    def __init__(self, policy_dir, aim, *, config, oracle=False, **kwargs):
        self.followup_config=config
        self.oracle=oracle
        self.fixed_bias=.00075*jax.random.normal(
            jax.random.fold_in(jax.random.PRNGKey(config.fixed_bias_seed),8713),(2,))
        self.grid=offset_grid()
        super().__init__(policy_dir,aim,config=config,**kwargs)

    def _get_obs(self,data,info,action,rng):
        # Only the simulator owns the hidden bias. It changes the measurement,
        # never the controller's input schema or target construction.
        cond=self.followup_config.camera_condition
        if cond=='one-sided':
            info['bias']=info['bias'].at[:2].set(jp.abs(info['bias'][:2]))
        elif cond=='fixed':
            info['bias']=info['bias'].at[:2].set(self.fixed_bias)
        return super()._get_obs(data,info,action,rng)

    def augment(self,raw,guess,history):
        parts=[raw,guess]
        if self.followup_config.memory:
            parts += [jp.mean(history,axis=0),history[-1]-history[0]]
        return jp.concatenate(parts)

    @property
    def learner_obs_size(self):
        return self.base_obs_size+2+(2*self.base_obs_size if self.followup_config.memory else 0)

    def reset(self,rng):
        state=super().reset(rng)
        raw=state.obs['state'][:self.base_obs_size]
        history=jp.broadcast_to(raw,(20,self.base_obs_size))
        info={**state.info,'reading_history':history,'contact_seen':jp.array(False),
              'grid_cold':jp.zeros(81,dtype=bool),'grid_target':jp.array(40,dtype=jp.int32),
              'grid_age':jp.array(0,dtype=jp.int32),'constant_fill_active':jp.array(False)}
        return state.replace(obs={'state':self.augment(raw,state.info['aim_guess'],history)},info=info)

    def next_guess(self,state,action):
        cfg=self.followup_config
        if self.oracle:
            return jp.clip(state.info['bias'][:2],-self.aim.guess_cap,self.aim.guess_cap)
        if cfg.outputs==3:
            return state.info['aim_guess']
        current=state.info['aim_guess']
        raw=state.obs['state'][:self.base_obs_size]
        contact=state.info['contact_seen'] | contact_seen(raw)
        cap=self.aim.step_cap * (jp.where(contact,2.,1.) if cfg.fast_contact else 1.)
        proposal=current+cap*jp.clip(action[:2],-1,1)
        if cfg.drift_gate:
            proposal=jp.where(contact & stalled(state.info['reading_history']),proposal,current)
            proposal=jp.where(contact,proposal,jp.zeros_like(proposal))
        if cfg.pullback:
            proposal=proposal-.01*current
        if cfg.within_grid:
            # Search trigger and cell elimination use camera height only.
            active=(raw[2]<.001) & stalled(state.info['reading_history']) & (state.info['step_count']>=10)
            target=self.grid[state.info['grid_target']]
            search=current+jp.clip(target-current,-.0002,.0002)
            proposal=jp.where(active,search,proposal)
        proposal=jp.where(state.info['constant_fill_active'],current,proposal)
        return jp.clip(proposal,-self.aim.guess_cap,self.aim.guess_cap)

    def step(self,state,action):
        nxt=super().step(state,action)
        raw=nxt.obs['state'][:self.base_obs_size]
        history=jp.concatenate([state.info['reading_history'][1:],raw[None,:]],axis=0)
        info={**nxt.info,'reading_history':history,
              'contact_seen':state.info['contact_seen'] | contact_seen(raw)}
        if self.followup_config.within_grid:
            active=(raw[2]<.001) & stalled(history) & (nxt.info['step_count']>=10)
            reached=jp.max(jp.abs(nxt.info['aim_guess']-self.grid[state.info['grid_target']]))<.0001
            age=jp.where(active & reached,state.info['grid_age']+1,0)
            eliminate=active & reached & (age>=5)
            cold=state.info['grid_cold'].at[state.info['grid_target']].set(
                state.info['grid_cold'][state.info['grid_target']] | eliminate)
            dist=jp.linalg.norm(self.grid-nxt.info['aim_guess'],axis=-1)
            target=jp.argmin(jp.where(cold,jp.inf,dist))
            info.update(grid_cold=cold,grid_target=jp.where(eliminate & ~jp.all(cold),target,state.info['grid_target']),
                        grid_age=jp.where(eliminate,0,age))
        return nxt.replace(obs={'state':self.augment(raw,nxt.info['aim_guess'],history)},info=info)
