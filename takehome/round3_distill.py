"""Task 4: shrink (policy + movement corrector) into one small network and measure what is lost.
Teachers: followup/G/plain_movement seeds 200-204 (test seed 1001). Test episodes only; no new practice
episodes in the main version. 8-bit and 16-bit are SIMULATED by rounding the weights, not a real runtime."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from takehome.followup_runtime import ROOT  # noqa: E402
import argparse, json, pickle, time  # noqa: E402
import jax, jax.numpy as jp, numpy as np, optax  # noqa: E402
from takehome.correction import AimConfig, add_movement_correction  # noqa: E402
from takehome.followup_config import FollowupConfig  # noqa: E402
from takehome.followup_env import FollowupEnv  # noqa: E402
from takehome.followup_episodes import FollowupEpisodes  # noqa: E402
from takehome.followup_learner import FollowupLearner  # noqa: E402
from takehome.round3_runner import R3, write_json  # noqa: E402

G = ROOT / 'results/takehome/followup/G/plain_movement'
OUT = R3 / 'task4'
SIZES = {'full_h256': 256, 'half_h180': 180, 'quarter_h125': 125}
PRECISIONS = ('fp32', 'fp16', 'int8')


def load_teacher(seed):
    run = G / f'seed_{seed}'
    cfg = FollowupConfig(**json.loads((run / 'config.json').read_text()))
    env = FollowupEnv(ROOT / 'results/policies' / cfg.policy, AimConfig(cfg.step_cap, cfg.guess_cap), config=cfg)
    learner = FollowupLearner(cfg, env)
    with (run / 'model_0200.pkl').open('rb') as f:
        state = jax.tree_util.tree_map(jp.asarray, pickle.load(f)['state'])
    return run, cfg, env, learner, state


def teacher_actions(env, learner, state, obs18):
    """Executed robot action of old policy + corrector, no exploration noise."""
    def one(o):
        raw = o[:env.base_obs_size]
        old, _ = env.frozen_policy({'state': raw}, jax.random.PRNGKey(0))
        corr = learner.act(state.actor, o[None])[0]
        return add_movement_correction(old, corr, env.movement_cap, env._config.action_scale)
    return np.asarray(jax.jit(jax.vmap(one))(jp.asarray(obs18)))


def practice_rows(run):
    xs = []
    for f in sorted(run.glob('practice_*.npz')):
        z = np.load(f)
        xs.append(z['obs'][z['valid'].astype(bool)])
    return np.concatenate(xs)


def init_mlp(key, h):
    sizes = [16, h, h, 3]
    ps = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        key, k = jax.random.split(key)
        ps.append({'w': jax.random.normal(k, (a, b)) * np.sqrt(2. / a), 'b': jnp_zeros(b)})
    return ps


def jnp_zeros(n):
    return jp.zeros(n)


def forward(ps, x):
    for p in ps[:-1]:
        x = jax.nn.relu(x @ p['w'] + p['b'])
    return jp.tanh(x @ ps[-1]['w'] + ps[-1]['b'])


def train_student(x, y, h, seed, steps=8000):
    mean, std = x.mean(0), x.std(0) + 1e-6
    xn = (x - mean) / std
    n_val = len(x) // 10
    perm = np.random.default_rng(seed).permutation(len(x))
    val, tr = perm[:n_val], perm[n_val:]
    ps = init_mlp(jax.random.PRNGKey(seed), h)
    opt = optax.adam(optax.cosine_decay_schedule(2e-3, steps))
    st = opt.init(ps)
    xt, yt = jp.asarray(xn[tr]), jp.asarray(np.clip(y[tr], -.999, .999))

    @jax.jit
    def step(ps, st, key):
        idx = jax.random.randint(key, (512,), 0, xt.shape[0])
        loss, g = jax.value_and_grad(lambda p: jp.mean((forward(p, xt[idx]) - yt[idx]) ** 2))(ps)
        up, st = opt.update(g, st)
        return optax.apply_updates(ps, up), st, loss
    key = jax.random.PRNGKey(seed + 1)
    for i in range(steps):
        key, k = jax.random.split(key)
        ps, st, _ = step(ps, st, k)
    val_mse = float(jp.mean((forward(ps, jp.asarray(xn[val])) - jp.asarray(np.clip(y[val], -.999, .999))) ** 2))
    # fold the input normalisation into layer 1 so the network takes the raw 16 readings
    w0 = ps[0]['w'] / jp.asarray(std)[:, None]
    b0 = ps[0]['b'] - (jp.asarray(mean) / jp.asarray(std)) @ ps[0]['w']
    return [{'w': w0, 'b': b0}] + ps[1:], val_mse


def quantize(ps, precision):
    out = []
    for p in ps:
        w = np.asarray(p['w'], dtype=np.float32)
        if precision == 'fp16':
            w = w.astype(np.float16).astype(np.float32)
        elif precision == 'int8':
            s = np.abs(w).max() / 127.
            w = np.round(w / s).clip(-127, 127) * s
        out.append({'w': jp.asarray(w), 'b': jp.asarray(np.asarray(p['b'], dtype=np.float32))})
    return out


def footprint(ps, precision):
    n = sum(p['w'].size + p['b'].size for p in ps)
    wbytes = {'fp32': 4, 'fp16': 2, 'int8': 1}[precision]
    nb = sum(p['b'].size for p in ps)
    return int(n), int(n * 4 if precision == 'fp32' else (n - nb) * wbytes + nb * 4)


def student_policy(ps):
    def policy(obs, key):
        return forward(ps, obs['state']), {}
    return policy


def eval_student(cfg, ps, label):
    """Test the student as the whole controller (corrector fixed at zero = fresh learner)."""
    env = FollowupEnv(ROOT / 'results/policies' / cfg.policy, AimConfig(cfg.step_cap, cfg.guess_cap), config=cfg)
    learner = FollowupLearner(cfg, env)    # built first: it reads the old policy's normaliser
    env.frozen_policy = student_policy(ps)
    ep = FollowupEpisodes(env, learner)
    summary, _, _ = ep.test(learner.initialize(0), episodes=500, seed=cfg.test_seed)
    return summary


def time_decision(fn, args, n=3000):
    cpu = jax.devices('cpu')[0]
    args = jax.device_put(args, cpu)
    f = jax.jit(fn, device=cpu)
    f(*args).block_until_ready()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        f(*args).block_until_ready()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts) * 1e6)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', type=int, nargs='+', default=[200, 201, 202, 203, 204])
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for seed in a.seeds:
        path = OUT / f'teacher_{seed}.json'
        if path.exists():
            continue
        run, cfg, env, learner, state = load_teacher(seed)
        recorded = json.loads((run / 'result.json').read_text())['final']['successes']
        ep = FollowupEpisodes(env, learner)
        tsum, _, _ = ep.test(state, episodes=500, seed=cfg.test_seed)
        obs = practice_rows(run)
        y = teacher_actions(env, learner, state, obs)
        x = obs[:, :16]
        res = {'seed': seed, 'recorded_successes': recorded, 'reproduced_successes': tsum['successes'],
               'rows': int(len(x)), 'students': {}}
        x0 = jp.zeros((1, 16))
        t_us = time_decision(lambda o: add_movement_correction(
            env.frozen_policy({'state': o[0]}, jax.random.PRNGKey(0))[0],
            learner.act(state.actor, jp.concatenate([o, jp.zeros((1, 2))], -1))[0], env.movement_cap,
            env._config.action_scale), (x0,))
        res['teacher_time_us'] = t_us
        res['teacher_weights'] = 71686 + 18 * 64 + 64 + 64 * 64 + 64 + 64 * 3 + 3
        for name, h in SIZES.items():
            ps, val_mse = train_student(x, y, h, seed)
            for prec in PRECISIONS:
                q = quantize(ps, prec)
                n_w, nbytes = footprint(q, prec)
                summ = eval_student(cfg, q, f'{name}_{prec}')
                us = time_decision(lambda o, q=q: forward(q, o), (x0,))
                res['students'][f'{name}_{prec}'] = {'hidden': h, 'precision': prec, 'weights': n_w, 'bytes': nbytes,
                    'successes': summ['successes'], 'rate': summ['success_rate'], 'val_mse': val_mse,
                    'time_us_cpu_batch1': us}
                print(seed, name, prec, summ['successes'], flush=True)
        write_json(path, res)


if __name__ == '__main__':
    main()
