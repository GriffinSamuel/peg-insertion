"""Practice-only replay with complete per-step provenance and symmetric sampling."""
import numpy as np

TRAIN_KEYS = ("obs", "action", "reward", "next_obs", "terminated")
REQUIRED_KEYS = TRAIN_KEYS + (
    "truncated", "success", "episode_id", "step", "source", "old_action",
    "executed_action", "true_bias", "guess", "force", "q", "valid",
)


def sample_indices(rng, size, fill_count, online_count):
    """NumPy reference used by tests and offline inspection."""
    if size % 2 or size < 2 or fill_count + online_count == 0:
        raise ValueError("Need an even batch and at least one nonempty source")
    if fill_count and online_count:
        return np.concatenate([rng.integers(fill_count, size=size // 2),
                               fill_count + rng.integers(online_count, size=size // 2)])
    return rng.integers(fill_count + online_count, size=size)


class Replay:
    def __init__(self, budget, capacity=40000):
        self.budget = budget
        self.capacity = capacity
        self.count = 0
        self.fill_count = 0
        self.arrays = {}
        self.episode_ids = set()

    def add(self, trajectory, episode_ids, source, *, test_only):
        self.budget.require_practice(episode_ids, source, test_only=test_only)
        if self.episode_ids.intersection(episode_ids):
            raise ValueError("Episode already inserted; cannot collect it twice")
        if source == "fill" and self.count != self.fill_count:
            raise ValueError("Fill-up data must precede online data")
        if set(REQUIRED_KEYS) - trajectory.keys():
            raise ValueError("Replay trajectory is missing required provenance")
        data = {k: np.asarray(v) for k, v in trajectory.items()}
        valid = data["valid"].astype(bool)
        if not set(np.unique(data["episode_id"])).issubset(set(episode_ids)):
            raise ValueError("Trajectory contains an unreserved episode")
        expected_source = 0 if source == "fill" else 1
        if not np.all(data["source"] == expected_source):
            raise ValueError("Trajectory source label does not match reservation")
        n = int(valid.sum())
        if self.count + n > self.capacity:
            raise RuntimeError("Replay capacity would be exceeded")
        for key in REQUIRED_KEYS:
            values = data[key][valid]
            if not np.isfinite(values).all():
                raise RuntimeError("Tripwire: invalid numbers in practice trajectory")
            if key not in self.arrays:
                self.arrays[key] = np.zeros((self.capacity,) + values.shape[1:], values.dtype)
            self.arrays[key][self.count:self.count + n] = values
        self.count += n
        if source == "fill":
            self.fill_count += n
        self.episode_ids.update(episode_ids)
        return n

    def training_arrays(self):
        # True bias and all other analysis-only metadata never reach the learner.
        return {k: self.arrays[k] for k in TRAIN_KEYS}

    def save(self, path):
        np.savez_compressed(path, **{k: v[:self.count] for k, v in self.arrays.items()})
