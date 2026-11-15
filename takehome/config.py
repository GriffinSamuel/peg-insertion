"""Only the main learner and the basic settings required by Goal 4."""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class TrainConfig:
    policy: str = "force_0.5mm_s2"
    outputs: int = 2
    fill_episodes: int = 40
    total_episodes: int = 200
    updates_per_step: int = 4
    ensemble: int = 4
    hidden: int = 64
    batch_size: int = 128
    learning_rate: float = 0.0003
    discount: float = 0.99
    target_rate: float = 0.005
    step_cap: float = 0.0001
    guess_cap: float = 0.003
    fill_noise: float = 0.5
    learner_noise: float = 0.2
    code_test: bool = False

    def __post_init__(self):
        if self.outputs not in (2, 3, 5):
            raise ValueError("Output count must be 2, 3 or 5")
        if self.total_episodes != (10 if self.code_test else 200):
            raise ValueError("Exactly 200 practice episodes, or 10 for the code test")
        if not 0 <= self.fill_episodes <= self.total_episodes or self.fill_episodes % 10:
            raise ValueError("Fill-up count must fit the budget and batches of 10")
        if self.ensemble < 1 or self.updates_per_step < 1 or self.hidden < 1:
            raise ValueError("Ensemble, updates and hidden width must be positive")
        if self.batch_size < 2 or self.batch_size % 2:
            raise ValueError("Batch size must be positive and even for exactly 50/50 sampling")
        if not 0 < self.step_cap <= self.guess_cap:
            raise ValueError("Require 0 < step cap <= guess cap")
        if not all(math.isfinite(x) for x in (
            self.step_cap, self.guess_cap, self.learning_rate, self.discount,
            self.target_rate, self.fill_noise, self.learner_noise)):
            raise ValueError("Settings must be finite")

    def to_dict(self):
        return asdict(self)
