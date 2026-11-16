"""Follow-up learner: original update recipe plus explicit observable labels."""
from takehome import followup_runtime  # noqa: F401
from brax.training.acme import running_statistics
from flax import struct
import jax
import jax.numpy as jp
import numpy as np
import optax
from takehome.learner import LearnerState, draw_indices
from takehome.variants import VariantLearner, prepare_training_data


@struct.dataclass
class FollowupState(LearnerState):
    start_guess: object


def success_targets(replay):
    """First-success camera XY is an approximate label, never a hidden target."""
    a=replay.arrays
    labels=np.zeros((replay.capacity,2),dtype=np.float32)
    valid=np.zeros(replay.capacity,dtype=np.float32)
    for episode in sorted(replay.episode_ids):
        ids=np.flatnonzero(a['episode_id'][:replay.count]==episode)
        ids=ids[np.argsort(a['step'][ids])]
        good=ids[a['success'][ids].astype(bool)]
        if not len(good):
            continue
        first=good[0]
        label=a['next_obs'][first,:2]
        labels[ids]=label
        valid[ids]=1.
    return labels,valid


class FollowupLearner(VariantLearner):
    def __init__(self,config,env):
        super().__init__(config,env)
        self.obs_size=env.learner_obs_size
        self.train_block=jax.jit(self._followup_train_block)

    def normalized(self,raw):
        return running_statistics.normalize({'state':raw},self.normalizer)['state']

    def encode(self,obs):
        b=self.base_size
        parts=[self.normalized(obs[...,:b]),obs[...,b:b+2]/self.config.guess_cap]
        if self.config.memory:
            mean=obs[...,b+2:2*b+2]
            delta=obs[...,2*b+2:3*b+2]
            parts += [self.normalized(mean), self.normalized(obs[...,:b])-self.normalized(obs[...,:b]-delta)]
        return jp.concatenate(parts,axis=-1)

    def initialize(self,seed):
        key,ka,kq=jax.random.split(jax.random.PRNGKey(seed),3)
        obs=jp.zeros((1,self.obs_size))
        actor=self.actor_net.init(ka,self.encode(obs))
        extra=2 if self.config.privileged_critic else 0
        x=jp.concatenate([self.encode(obs),jp.zeros((1,self.config.outputs+extra))],-1)
        critics=jax.vmap(lambda k:self.critic_net.init(k,x))(jax.random.split(kq,self.config.ensemble))
        return FollowupState(actor,critics,actor,critics,self.optimizer.init(actor),
                             self.optimizer.init(critics),key,jp.array(0,dtype=jp.int32),jp.zeros(2))

    def act(self,params,obs):
        if not self.config.success_labels:
            return super().act(params,obs)
        raw=self.actor_net.apply(params,self.encode(obs))
        prediction=raw[...,:2]*self.config.guess_cap
        current=obs[...,self.base_size:self.base_size+2]
        aim=jp.clip((prediction-current)/self.config.step_cap,-1,1)
        return aim if self.config.outputs==2 else jp.concatenate([aim,raw[...,2:]],axis=-1)

    def choose(self,actor,critics,obs):
        action=self.act(actor,obs)
        if not self.config.final_decider or self.config.outputs==5:
            return super().choose(actor,critics,obs)
        options=jp.stack([jp.zeros_like(action),action,action*.5,jp.clip(action*1.5,-1,1)],axis=-2)
        tiled=jp.broadcast_to(obs[...,None,:],options.shape[:-1]+(obs.shape[-1],))
        q=jp.mean(self.q_values(critics,tiled,options*self.action_scale),axis=0)
        return jp.take_along_axis(options,jp.argmax(q,axis=-1)[...,None,None],axis=-2)[...,0,:]

    def _followup_train_block(self, state, data, fill_count, total_count, num_updates,
                             actor_enabled=True):
        cfg = self.config

        def step(_, carry):
            st, loss_sum, q_sum, finite = carry
            key, ks, kt = jax.random.split(st.key, 3)
            ids = draw_indices(ks, cfg.batch_size, fill_count, total_count)
            batch = {k: v[ids] for k, v in data.items()}
            next_action = self.choose(st.target_actor, st.target_critics, batch['next_obs']) * self.action_scale
            next_q = self.q_values(st.target_critics, batch['next_obs'], next_action,
                                   batch.get('next_true_bias'))
            subset = jax.random.permutation(kt, cfg.ensemble)[:min(2, cfg.ensemble)]
            target = batch['reward'] + batch.get('discount', cfg.discount) * (
                1. - batch['terminated']) * jp.min(next_q[subset], axis=0)
            if cfg.return_floor:
                target = jp.maximum(target, batch['return_floor'])
            target = jax.lax.stop_gradient(target)

            def critic_loss(params):
                logits = self.q_logits(params, batch['obs'], batch['action'], batch.get('true_bias'))
                if cfg.classification:
                    loss = jp.mean(optax.sigmoid_binary_cross_entropy(logits, jp.clip(target, 0, 1)[None, :]))
                    q = jax.nn.sigmoid(logits)
                else:
                    q = logits
                    loss = jp.mean((q - target[None, :]) ** 2)
                return loss, jp.mean(q)

            (loss, mean_q), grads = jax.value_and_grad(critic_loss, has_aux=True)(st.critics)
            delta, critic_opt = self.optimizer.update(grads, st.critic_opt, st.critics)
            critics = optax.apply_updates(st.critics, delta)

            def actor_step(operand):
                actor, actor_opt, target_actor = operand
                def actor_loss(p):
                    action = self.act(p, batch['obs']) * self.action_scale
                    loss = -jp.mean(self.q_values(critics, batch['obs'], action, batch.get('true_bias')))
                    if cfg.success_labels:
                        # Label comes only from successful OWN PRACTICE readings.
                        target = batch['success_label'] / cfg.guess_cap
                        prediction = self.actor_net.apply(p,self.encode(batch['obs']))[..., :2]
                        squared = jp.mean((prediction-target)**2, axis=-1)
                        loss += jp.sum(squared * batch['label_valid']) / jp.maximum(jp.sum(batch['label_valid']),1.)
                    return loss
                grad = jax.grad(actor_loss)(actor)
                delta, actor_opt = self.optimizer.update(grad, actor_opt, actor)
                actor = optax.apply_updates(actor, delta)
                return actor, actor_opt, optax.incremental_update(actor, target_actor, cfg.target_rate)

            actor, actor_opt, target_actor = jax.lax.cond(
                actor_enabled & ((st.updates + 1) % cfg.updates_per_step == 0), actor_step,
                lambda x: x, (st.actor, st.actor_opt, st.target_actor))
            new = st.replace(actor=actor, critics=critics, target_actor=target_actor,
                target_critics=optax.incremental_update(critics, st.target_critics, cfg.target_rate),
                actor_opt=actor_opt, critic_opt=critic_opt, key=key, updates=st.updates + 1)
            return new, loss_sum + loss, q_sum + mean_q, finite & jp.isfinite(loss) & jp.isfinite(mean_q)

        st, loss, q, finite = jax.lax.fori_loop(
            0, num_updates, step, (state, jp.array(0.), jp.array(0.), jp.array(True)))
        return st, {'critic_loss': loss / num_updates, 'mean_q': q / num_updates, 'finite': finite}
