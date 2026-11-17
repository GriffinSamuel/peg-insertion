"""Floor and no-camera-error reference at the fresh test seed 2002 (test only). Needs final_choice.json first."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from takehome.followup_runtime import ROOT  # noqa: E402
import json  # noqa: E402
import numpy as np  # noqa: E402
from takehome.correction import AimConfig  # noqa: E402
from takehome.followup_env import FollowupEnv  # noqa: E402
from takehome.followup_learner import FollowupLearner  # noqa: E402
from takehome.followup_episodes import FollowupEpisodes  # noqa: E402
from takehome.round3_config import Round3Config  # noqa: E402
from takehome.round3_runner import R3, write_json  # noqa: E402

SEED = 2002


def main():
    if not (R3 / 'final_choice.json').exists():
        raise ValueError('Fresh test seed forbidden before the written final choice')
    path = R3 / 'references' / f'normal_test_{SEED}.json'
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = Round3Config(outputs=2, test_seed=SEED)
    conditions = {}
    for name, zero_error in (('normal', False), ('no_camera_bias', True)):
        env = FollowupEnv(ROOT / 'results/policies' / cfg.policy, AimConfig(), config=cfg,
                          **({'obs_hole_bias_std': 0.} if zero_error else {}))
        learner = FollowupLearner(cfg, env)
        summary, stats, _ = FollowupEpisodes(env, learner).test(learner.initialize(0), episodes=500, seed=SEED)
        if not np.isfinite(summary['steps_mean']):
            summary['steps_mean'] = summary['steps_median'] = None
        summary.update(reference_only=True)
        conditions[name] = summary
        print(name, summary['successes'], flush=True)
    write_json(path, dict(policy=cfg.policy, test_seed=SEED, episodes_per_condition=500,
                          camera_condition='normal', fixed_bias_seed=0, mixed_fixed_std=0., mixed_episode_std=0.,
                          practice_episodes=0, conditions=conditions))


if __name__ == '__main__':
    main()
