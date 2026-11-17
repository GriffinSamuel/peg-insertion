# Task 3.0 findings (test only, 500 episodes per cell, test seed 123, runs 0-2)

| learner | full | constant (pre-contact mean) | constant (end-of-episode mean) | freeze at first contact | constant + 0.1 mm jitter after contact |
|---|---|---|---|---|---|
| B_aim_fixed | 90.0% (455/433/462) | 63.4% (155/444/352) | 45.1% (453/18/206) | 88.9% (451/438/445) | 63.5% (160/446/347) |
| D_aim_cap_200um | 99.2% (493/500/495) | 34.7% (307/0/213) | 6.0% (0/16/74) | 97.1% (481/493/483) | 34.4% (290/0/226) |

Reading: freezing the guess at first contact keeps almost all of the gain (B 90.0 to 88.9%, D 99.2 to 97.1%), so the gain is made before contact, not by moving the guess while touching. Holding one constant guess loses most of the gain in both versions (the learned guess is a path, not one number), and adding a random 0.1 mm jitter to a constant does not bring it back (it is the same as the constant, within noise). The wiggle explanation is not supported. Caveat: tested only on fixed-camera-error learners; the jitter is an i.i.d. random offset each step, not an accumulating random walk.
