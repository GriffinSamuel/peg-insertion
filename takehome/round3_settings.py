"""Every Round 3 learning setting, declared in one place. Count toward the 30-setting limit."""
from takehome.round3_config import Round3Config

MOVE = dict(outputs=3, test_seed=123)
AIM = dict(outputs=2, test_seed=123)

# name -> (task, plain sentence, base label, config overrides, uses_hand_routine)
SETTINGS = {
    'R1a_pick_end': ('1', 'Train as usual, then use the saved version whose own practice episodes did best.',
                     'movement', {**MOVE, 'selection': 'pick'}, False),
    'R1b_rollback': ('1', 'If the recent practice success drops 20 points below its best, go back to the best version.',
                     'movement', {**MOVE, 'rollback': True}, False),
    'R1c_both': ('1', 'Rollback during training and pick the best version at the end.',
                 'movement', {**MOVE, 'rollback': True, 'selection': 'pick'}, False),
    'R2a_depth': ('2', 'Add a getting-warmer reward from the insertion depth reading.',
                  'movement', {**MOVE, 'shaping': 'depth'}, False),
    'R2b_torque': ('2', 'Add a getting-warmer reward for less wrist twisting.',
                   'movement', {**MOVE, 'shaping': 'torque'}, False),
    'R2c_both': ('2', 'Add getting-warmer rewards from both depth and low torque.',
                 'movement', {**MOVE, 'shaping': 'both'}, False),
    'R2d_aim_torque': ('2', 'Torque getting-warmer reward on the aim-only corrector (normal camera error).',
                       'aim', {**AIM, 'shaping': 'torque'}, False),
    'R3a_search_every': ('3', 'Learner picks wiggle width, sideways lean and extra push every step after contact.',
                        'movement', {**MOVE, 'search': 'every'}, True),
    'R3b_search_hold10': ('3', 'Learner picks the same search settings once every 10 steps and they are held.',
                         'movement', {**MOVE, 'search': 'hold10'}, True),
    'R3c_search_plus_move': ('3', 'Search settings every step plus the plain movement correction on top.',
                            'movement', {**MOVE, 'search': 'every_plus_move'}, True),
    'RF_plain_movement': ('F', 'Final check: plain movement on fresh practice seeds 300-304, test seed 2002.',
                          'movement', {**MOVE, 'test_seed': 2002,
                                       'baseline_path': 'results/takehome/round3/references/normal_test_2002.json'}, False),
    'R0_wrapper_check': ('0', 'Plain movement through the round 3 runner (checks the wrapper matches the base).',
                         'movement', {**MOVE}, False),
}


def make_config(name, **extra):
    task, _, _, overrides, _ = SETTINGS[name]
    return Round3Config(**{**overrides, **extra})
