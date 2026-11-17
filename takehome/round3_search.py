"""Task 3.2: the learner outputs search settings; a fixed routine turns them into capped movement.

p0 wiggle width (0..0.3 mm, relu), p1 p2 sideways lean (+-0.2 mm), p3 extra downward push (0..0.2 mm, relu),
optionally p4..p6 the plain movement correction. The routine is zero before first contact, and relu/zero-start
means an untrained learner behaves exactly like the old policy. Wiggle is a circle with period 10 steps."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from takehome.followup_runtime import ROOT  # noqa: E402
import jax, jax.numpy as jp, numpy as np  # noqa: E402
import optax  # noqa: E402
from takehome.correction import AimConfig  # noqa: E402
from takehome.followup_env import FollowupEnv, contact_seen  # noqa: E402
from takehome.followup_episodes import FollowupEpisodes  # noqa: E402
from takehome.followup_learner import FollowupLearner, FollowupState  # noqa: E402
from takehome.followup_runner import FollowupCollection  # noqa: E402
from takehome.learner import Actor  # noqa: E402
from takehome.round3_runner import Round3Runner  # noqa: E402

PERIOD, WIDTH_MAX = 10, 0.0003
HOLD = 10


def search_dims(mode):
    return 7 if mode == 'every_plus_move' else 4


class SearchEnv(FollowupEnv):
    def __init__(self, *a, mode='every', **k):
        self.mode = mode
        super().__init__(*a, **k)
        assert not self.oracle

    @property
    def action_size(self):
        return search_dims(self.mode)

    def reset(self, rng):
        state = super().reset(rng)
        return state.replace(info={**state.info, 'contact_steps': jp.array(0, dtype=jp.int32),
                                   'held': jp.zeros(4)})

    def step(self, state, action):
        raw = state.obs['state'][:self.base_obs_size]
        contact = state.info['contact_seen'] | contact_seen(raw)
        k = state.info['contact_steps']
        p = jp.clip(action[:4], -1, 1)
        if self.mode == 'hold10':
            p = jp.where(state.info['step_count'] % HOLD == 0, p, state.info['held'])
        width = (WIDTH_MAX / self.movement_cap) * jax.nn.relu(p[0])
        phase = 2 * jp.pi * (k % PERIOD) / PERIOD
        search = jp.stack([p[1] + width * jp.cos(phase), p[2] + width * jp.sin(phase), -jax.nn.relu(p[3])])
        move = jp.where(contact, search, jp.zeros(3))
        if self.mode == 'every_plus_move':
            move = move + jp.clip(action[4:7], -1, 1)
        nxt = super().step(state, jp.clip(move, -1, 1))
        info = {**nxt.info, 'contact_steps': jp.where(contact, k + 1, 0), 'held': p}
        return nxt.replace(info=info)


class SearchLearner(FollowupLearner):
    def __init__(self, config, env):
        super().__init__(config, env)
        self.n_out = search_dims(config.search)
        self.actor_net = Actor(self.n_out, config.hidden)
        self.action_scale = jp.ones(self.n_out)

    def initialize(self, seed):
        key, ka, kq = jax.random.split(jax.random.PRNGKey(seed), 3)
        obs = jp.zeros((1, self.obs_size))
        actor = self.actor_net.init(ka, self.encode(obs))
        x = jp.concatenate([self.encode(obs), jp.zeros((1, self.n_out))], -1)
        critics = jax.vmap(lambda k: self.critic_net.init(k, x))(jax.random.split(kq, self.config.ensemble))
        return FollowupState(actor, critics, actor, critics, self.optimizer.init(actor),
                             self.optimizer.init(critics), key, jp.array(0, dtype=jp.int32), jp.zeros(2))


def macro_data(replay, cfg):
    """Blocks of HOLD steps become one transition (reward summed with discount, discount**length)."""
    a = replay.arrays
    n = replay.count
    rows = {k: [] for k in ('obs', 'action', 'reward', 'next_obs', 'terminated', 'discount')}
    fill_rows = 0
    for episode in sorted(replay.episode_ids):
        ids = np.flatnonzero(a['episode_id'][:n] == episode)
        ids = ids[np.argsort(a['step'][ids])]
        for pos in range(0, len(ids), HOLD):
            block = ids[pos:pos + HOLD]
            g = cfg.discount ** np.arange(len(block))
            rows['obs'].append(a['obs'][block[0]])
            rows['action'].append(a['action'][block[0]])
            rows['reward'].append(float(np.dot(a['reward'][block], g)))
            rows['next_obs'].append(a['next_obs'][block[-1]])
            rows['terminated'].append(a['terminated'][block[-1]])
            rows['discount'].append(cfg.discount ** len(block))
        if episode < cfg.fill_episodes:
            fill_rows = len(rows['obs'])
    out = {}
    for k, v in rows.items():
        arr = np.asarray(v, dtype=np.float32)
        pad = np.zeros((replay.capacity,) + arr.shape[1:], dtype=np.float32)
        pad[:len(arr)] = arr
        out[k] = pad
    return out, fill_rows, len(rows['obs'])


class SearchRunner(Round3Runner):
    def __init__(self, config):
        self.config = config
        self.env = SearchEnv(ROOT / 'results/policies' / config.policy, AimConfig(config.step_cap, config.guess_cap),
                             config=config, mode=config.search)
        self.learner = SearchLearner(config, self.env)
        self.episodes = FollowupCollection(self)

    def update(self, state, replay, new_steps, source, practice_end):
        cfg = self.config
        if cfg.search != 'hold10':
            return super().update(state, replay, new_steps, source, practice_end)
        data, fill_rows, total_rows = macro_data(replay, cfg)
        # Updates proportional to macro rows (same updates per transition row as the base recipe).
        batch_rows = max(1, int(round(new_steps / HOLD)))
        return self.learner.train_block(state, data, fill_rows, total_rows,
                                        batch_rows * cfg.updates_per_step, actor_enabled=True)
