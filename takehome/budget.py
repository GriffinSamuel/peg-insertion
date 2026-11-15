"""Persistent practice reservations: crashes never make spent episodes free."""
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path


class EpisodeBudget:
    """Every reserved practice episode counts, including incomplete attempts.

    Reserve before touching the simulator. Test episodes have no reservation
    and cannot pass require_practice(), which the replay insertion path uses.
    """

    def __init__(self, path, limit=200):
        if type(limit) is not int or not 1 <= limit <= 200:
            raise ValueError("Practice limit must be an integer from 1 to 200")
        self.path = Path(path)
        self.limit = limit
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._locked():
            if not self.path.exists():
                self._save({"limit": limit, "episodes": []})
            elif self._read()["limit"] != limit:
                raise ValueError("Cannot change the budget of an existing experiment")

    @contextmanager
    def _locked(self):
        with self.path.with_suffix(self.path.suffix + ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _read(self):
        return json.loads(self.path.read_text())

    def _save(self, data):
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(data, indent=2) + "\n")
        temporary.replace(self.path)

    @property
    def used(self):
        return len(self._read()["episodes"])

    def reserve(self, count, source):
        if type(count) is not int or count < 1:
            raise ValueError("Episode count must be a positive integer")
        if source not in ("fill", "online"):
            raise ValueError("Only practice data may be reserved; test data is forbidden")
        with self._locked():
            data = self._read()
            used = len(data["episodes"])
            if used + count > self.limit:
                raise RuntimeError(f"Practice budget exceeded: {used} + {count} > {self.limit}")
            ids = list(range(used, used + count))
            data["episodes"].extend({"id": i, "source": source} for i in ids)
            self._save(data)
            return ids

    def require_practice(self, episode_ids, source, *, test_only):
        if test_only or source not in ("fill", "online"):
            raise ValueError("Test data cannot enter replay or be learned from")
        entries = self._read()["episodes"]
        if not episode_ids or len(set(episode_ids)) != len(episode_ids):
            raise ValueError("Require unique reserved episode IDs")
        for episode_id in episode_ids:
            if (type(episode_id) is not int or not 0 <= episode_id < len(entries)
                    or entries[episode_id]["source"] != source):
                raise ValueError("Unreserved episode or wrong practice source")
