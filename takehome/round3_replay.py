"""Task 3.0 (what does the aim learner's wiggle do?) and 3.1 (hand-set wiggle). Test only: no learning,
no Replay.add, no episode reservations."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from takehome.followup_runtime import ROOT  # noqa: E402
import argparse, json, pickle  # noqa: E402
from dataclasses import replace  # noqa: E402
import jax, jax.numpy as jp, numpy as np  # noqa: E402
from takehome.correction import AimConfig  # noqa: E402
from takehome.followup_config import FollowupConfig  # noqa: E402
from takehome.followup_env import FollowupEnv, contact_seen, stalled  # noqa: E402
from takehome.followup_episodes import FollowupEpisodes  # noqa: E402
from takehome.followup_learner import FollowupLearner  # noqa: E402
from takehome.round3_runner import R3, write_json  # noqa: E402

PERIOD = 10          # steps per wiggle circle (same in 3.1 and 3.2)
CAP = 0.0002         # movement cap (m)


class ReplayEnv(FollowupEnv):
    """mode: full | const | freeze | jitter. const_vec/amp used by const and jitter."""
    def __init__(self, *a, mode='full', const_vec=(0., 0.), amp=0., **k):
        self.mode, self.const_vec, self.amp = mode, jp.asarray(const_vec, dtype=jp.float32), amp
        super().__init__(*a, **k)
        assert not self.oracle

    def reset(self, rng):
        state = super().reset(rng)
        return state.replace(info={**state.info, 'wkey': jax.random.fold_in(rng, 77)})

    def next_guess(self, state, action):
        raw = state.obs['state'][:self.base_obs_size]
        contact = state.info['contact_seen'] | contact_seen(raw)
        if self.mode == 'full':
            return super().next_guess(state, action)
        if self.mode == 'const':
            return self.const_vec
        if self.mode == 'freeze':
            return jp.where(contact, state.info['aim_guess'], super().next_guess(state, action))
        if self.mode == 'jitter':
            ang = jax.random.uniform(jax.random.fold_in(state.info['wkey'], state.info['step_count']),
                                     minval=0., maxval=2 * jp.pi)
            wig = self.amp * jp.stack([jp.cos(ang), jp.sin(ang)])
            return jp.clip(self.const_vec + jp.where(contact, wig, 0.), -self.aim.guess_cap, self.aim.guess_cap)
        raise ValueError(self.mode)


class HandWiggleEnv(FollowupEnv):
    """Old policy + circular movement wiggle once in contact and depth has stalled (3.1)."""
    def __init__(self, *a, amp=0., **k):
        self.amp = amp
        super().__init__(*a, **k)
        assert not self.oracle

    def step(self, state, action):
        raw = state.obs['state'][:self.base_obs_size]
        active = (state.info['contact_seen'] | contact_seen(raw)) & stalled(state.info['reading_history'])
        phase = 2 * jp.pi * (state.info['step_count'] % PERIOD) / PERIOD
        wig = (self.amp / CAP) * jp.stack([jp.cos(phase), jp.sin(phase), jp.zeros(())])
        return super().step(state, jp.where(active, wig, jp.zeros(3)))


def constants(run_dir):
    """Pre-contact mean guess and end-of-episode mean guess from the run's own practice files (episodes 100-200)."""
    pre, end = [], []
    for e in range(110, 201, 10):
        z = np.load(run_dir / f'practice_{e:04d}.npz')
        g, f, v = z['guess'], z['force'], z['valid'].astype(bool)
        for i in range(g.shape[1]):
            ok = np.flatnonzero(v[:, i])
            hit = np.flatnonzero(f[ok, i] >= 5.)
            before = ok[:hit[0]] if len(hit) else ok
            if len(before):
                pre.append(g[before, i].mean(0))
            end.append(g[ok[-1], i])
    return np.mean(pre, 0), np.mean(end, 0)


def run_arm(cfg, env_cls, state, **kw):
    env = env_cls(ROOT / 'results/policies' / cfg.policy, AimConfig(cfg.step_cap, cfg.guess_cap),
                  config=cfg, **kw)
    learner = FollowupLearner(cfg, env)
    ep = FollowupEpisodes(env, learner)
    if state is None:
        state = learner.initialize(0)
    summary, stats, _ = ep.test(state, episodes=500, seed=123)
    return summary, stats['succeeded'] > .5


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--part', choices=['3.0', '3.1'], required=True)
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    a = p.parse_args()
    out = R3 / 'task3'
    out.mkdir(parents=True, exist_ok=True)
    if a.part == '3.0':
        for name, d in (('B_aim_fixed', 'B/B_aim_fixed'), ('D_aim_cap_200um', 'D/D_aim_cap_200um')):
            for s in a.seeds:
                path = out / f'step30_{name}_s{s}.json'
                if path.exists():
                    continue
                run_dir = ROOT / f'results/takehome/followup/{d}/seed_{s}'
                cfg = FollowupConfig(**json.loads((run_dir / 'config.json').read_text()))
                with (run_dir / 'model_0200.pkl').open('rb') as f:
                    state = jax.tree_util.tree_map(jnp_asarray, pickle.load(f)['state'])
                pre, endm = constants(run_dir)
                arms = {'1_full': (ReplayEnv, dict(mode='full')),
                        '2_const_precontact_mean': (ReplayEnv, dict(mode='const', const_vec=pre)),
                        '2b_const_endofepisode_mean': (ReplayEnv, dict(mode='const', const_vec=endm)),
                        '3_freeze_at_contact': (ReplayEnv, dict(mode='freeze')),
                        '4_const_plus_jitter_0.1mm': (ReplayEnv, dict(mode='jitter', const_vec=pre, amp=1e-4))}
                res, wins = {}, {}
                for k, (cls, kw) in arms.items():
                    summ, ok = run_arm(cfg, cls, state, **kw)
                    res[k] = {'successes': summ['successes'], 'rate': summ['success_rate'],
                              'peak_force_mean': summ['peak_force_mean']}
                    wins[k] = ok
                    print(name, s, k, summ['successes'], flush=True)
                np.savez_compressed(out / f'step30_{name}_s{s}_episodes.npz', **wins)
                write_json(path, {'setting': name, 'seed': s, 'test_seed': 123, 'constants_precontact': pre.tolist(),
                                  'constants_endofepisode': endm.tolist(), 'arms': res})
    else:
        base_cfg = FollowupConfig(outputs=3, test_seed=123)
        for amp in (0., 5e-5, 1e-4, 2e-4):
            path = out / f'step31_amp_{int(amp * 1e6)}um.json'
            if path.exists():
                continue
            summ, ok = run_arm(base_cfg, HandWiggleEnv, None, amp=amp)
            np.savez_compressed(out / f'step31_amp_{int(amp * 1e6)}um_episodes.npz', ok=ok)
            write_json(path, {'amp_m': amp, 'successes': summ['successes'], 'rate': summ['success_rate'],
                              'peak_force_mean': summ['peak_force_mean'], 'steps_mean': summ['steps_mean'] if np.isfinite(summ['steps_mean']) else None,
                              'test_seed': 123, 'learning': False, 'uses_hand_routine': True})
            print('3.1', amp, summ['successes'], flush=True)


def jnp_asarray(x):
    return jp.asarray(x)


if __name__ == '__main__':
    main()
