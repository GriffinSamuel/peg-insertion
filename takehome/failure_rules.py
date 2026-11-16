"""Fixed ordered, operational failure labels for Goal 9; not causal diagnoses."""
import numpy as np

ORDER = ('never_near_hole_contact','wrong_way_correction','stuck_on_rim','timeout','other')
RULES = {
    'never_near_hole_contact': 'No valid step has XY error <=5mm, absolute tip height <=3mm and force >=5N.',
    'wrong_way_correction': 'At the last valid step, aim-guess error exceeds the zero-guess error by >0.25mm with a nonzero >0.25mm guess, OR the net commanded movement correction points radially outward by >1mm summed over valid near-hole steps.',
    'stuck_on_rim': 'At least20 of the last40 valid steps meet the near-hole contact rule and their insertion-depth range is <1mm.',
    'timeout': 'Final recorded learner-truncated flag is true and learner-terminated flag is false.',
    'other': 'Any remaining failed episode.'}


def classify(trace):
    success = np.asarray(trace['episode_success']) > .5
    valid = np.asarray(trace['valid'],bool)
    n = len(success)
    labels = ['success' if value else None for value in success]
    evidence = []
    for episode in np.flatnonzero(~success):
        times = np.flatnonzero(valid[:,episode])
        if not len(times):
            raise ValueError('A failed episode has no valid recorded step')
        contact = ((np.asarray(trace['xy_err'])[:,episode] <= .005) &
                   (np.abs(np.asarray(trace['tip_height'])[:,episode]) <= .003) &
                   (np.asarray(trace['force'])[:,episode] >= 5.) & valid[:,episode])
        last = times[-1]
        guess = np.asarray(trace.get('next_guess',trace['guess']))[last,episode]
        bias = np.asarray(trace['true_bias'])[last,episode]
        aim_wrong = np.linalg.norm(guess-bias) > np.linalg.norm(bias)+.00025 and np.linalg.norm(guess)>.00025
        # The physical XY direction is estimated from the recorded camera
        # observation minus known bias, so it still contains camera white noise.
        rel = np.asarray(trace['obs'])[:,episode,:2] - np.asarray(trace['true_bias'])[:,episode]
        radius = np.linalg.norm(rel,axis=-1)
        movement = np.asarray(trace.get('movement',np.zeros((*valid.shape,3))))[:,episode,:2]
        radial = np.sum(movement*rel,axis=-1)/np.maximum(radius,1e-9)
        near = valid[:,episode] & (np.abs(np.asarray(trace['tip_height'])[:,episode])<=.015) & (radius<=.01)
        outward = float(radial[near].sum())
        recent = times[-40:]
        rim = int(contact[recent].sum())>=20 and np.ptp(np.asarray(trace['depth'])[recent,episode]) < .001
        timeout = bool(np.asarray(trace['truncated'])[-1,episode]) and not bool(np.asarray(trace['terminated'])[-1,episode])
        matched = [not bool(contact.any()), bool(aim_wrong or outward>.001), bool(rim), timeout, True]
        label = next(name for name,ok in zip(ORDER,matched) if ok)
        labels[episode] = label
        evidence.append(dict(episode_id=int(episode),label=label,near_contact_steps=int(contact.sum()),
            aim_error_m=float(np.linalg.norm(guess-bias)),zero_guess_error_m=float(np.linalg.norm(bias)),
            outward_command_sum_m=outward,timeout=timeout))
    counts = {name:labels.count(name) for name in ORDER}
    if sum(counts.values()) != int((~success).sum()) or any(label is None for label in labels):
        raise ValueError('Failure labels do not form a complete disjoint partition')
    return dict(episodes=n,successes=int(success.sum()),failures=int((~success).sum()),
                counts=counts,labels=labels,evidence=evidence,order=list(ORDER),rules=RULES,
                interpretation='Operational trajectory patterns, not proven causes; outward direction retains camera white noise.')
