"""Predeclared follow-up settings; original configurations remain frozen."""
from dataclasses import dataclass
from takehome.config import TrainConfig


@dataclass(frozen=True)
class FollowupConfig(TrainConfig):
    coordination: str = 'joint'
    briefing: bool = False
    privileged_critic: bool = False
    final_decider: bool = False
    near_hole: bool = False
    contact_cap: bool = False
    uncertainty_cap: bool = False
    classification: bool = False
    n_step: int = 1
    return_floor: bool = False
    force_penalty: bool = False
    sibling_fill: bool = False
    test_seed: int = 123
    baseline_path: str = 'results/takehome/baseline.json'
    camera_condition: str = 'normal'
    fixed_bias_seed: int = 0
    mixed_fixed_std: float = 0.
    mixed_episode_std: float = 0.
    drift_gate: bool = False
    pullback: bool = False
    memory: bool = False
    success_labels: bool = False
    fast_contact: bool = False
    constant_fill: bool = False
    heatmap: bool = False
    within_grid: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.supervised_reference:
            raise ValueError('Follow-up does not admit supervised hidden labels')
        if self.camera_condition not in ('normal', 'one-sided', 'fixed', 'mixed'):
            raise ValueError('Unknown camera condition')
        if self.camera_condition=='mixed' and (self.mixed_fixed_std,self.mixed_episode_std) not in (
                (.00065,.000375),(.00053,.00053),(.000375,.00065)):
            raise ValueError('Mixed camera must use one of the three prescribed splits')
        if self.camera_condition!='mixed' and (self.mixed_fixed_std or self.mixed_episode_std):
            raise ValueError('Mixed scales apply only to the mixed condition')
        if self.coordination not in ('joint','side_by_side','aim_then_move','move_then_aim'):
            raise ValueError('Unknown coordination')
        if self.outputs == 3 and any((self.drift_gate,self.pullback,self.memory,self.success_labels,
                                      self.fast_contact,self.constant_fill,self.heatmap,self.within_grid)):
            raise ValueError('Phase B switches require an aim output')
        if self.coordination != 'joint' and self.outputs != 5:
            raise ValueError('Coordination requires both parts')
        if self.coordination != 'joint' and self.final_decider:
            raise ValueError('Independent component critics cannot choose joint candidates')
        if self.coordination in ('aim_then_move','move_then_aim') and (200-self.fill_episodes)%20:
            raise ValueError('Sequential halves require complete ten-episode batches')
        if self.privileged_critic and self.final_decider:
            raise ValueError('A test-time decider cannot use hidden information')
        if self.force_penalty and self.classification:
            raise ValueError('Negative force reward incompatible with classification')
        if self.n_step not in (1,3):
            raise ValueError('Only predeclared credit lengths')
        if self.heatmap and not self.constant_fill:
            raise ValueError('Across-episode heat map requires its constant-offset fill')
        if self.briefing and not self.fill_episodes:
            raise ValueError('Briefing needs own fill data')
        if any((self.drift_gate,self.pullback,self.memory,self.success_labels,self.fast_contact,
                self.constant_fill,self.heatmap,self.within_grid)) and self.privileged_critic:
            raise ValueError('Phase B forbids hidden information')
