"""Round 3 settings: the follow-up config plus a few new, explicit switches."""
from dataclasses import dataclass
from takehome.followup_config import FollowupConfig

# Fixed BEFORE any result was seen (see TAKEHOME_DECISIONS.md, 2026-11-17).
ROLLBACK_WINDOW = 20      # learner episodes in the success-rate window
ROLLBACK_DROP = 0.20      # restore when the window is this far below the best window
DEPTH_SCALE = 0.20        # largest value of the depth score
TORQUE_SCALE = 0.05       # largest (negative) value of the torque score


@dataclass(frozen=True)
class Round3Config(FollowupConfig):
    shaping: str = 'none'        # none | depth | torque | both
    selection: str = 'last'      # last | pick  (pick = best version by practice success)
    rollback: bool = False
    search: str = 'none'         # none | every | hold10 | every_plus_move  (Task 3.2)

    def __post_init__(self):
        super().__post_init__()
        if self.shaping not in ('none', 'depth', 'torque', 'both'):
            raise ValueError('Unknown shaping')
        if self.selection not in ('last', 'pick'):
            raise ValueError('Unknown selection')
        if self.search not in ('none', 'every', 'hold10', 'every_plus_move'):
            raise ValueError('Unknown search')
        if self.search != 'none' and self.outputs != 3:
            raise ValueError('Search settings run on the movement environment (outputs=3 internally)')
