"""Phase F saved-pose diagnosis only; no controller fitting or new episodes."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from takehome.followup_runtime import ROOT, OUT
import json
import math
import numpy as np


def pressure_point(force, torque, rotation):
    """Nominal contact point inferred from the observed wrench at the peg base.

    Both wrench components share the sensor's sign convention, which cancels in
    the ratio. The contact plane is 60 mm below the wrist site. Clipping and
    multiple contacts can make this estimate inaccurate; it is only a diagnosis.
    """
    f = np.asarray(force) @ np.asarray(rotation).T
    t = np.asarray(torque) @ np.asarray(rotation).T
    usable = np.abs(f[..., 2]) >= 2.
    denominator = np.where(usable, f[..., 2], 1.)
    xy = np.stack([(-.060*f[..., 0]-t[..., 1])/denominator,
                   (t[..., 0]-.060*f[..., 1])/denominator], axis=-1)
    return xy, usable


def wilson_lower(successes, count):
    if not count:
        return None
    z = 1.96
    p = successes/count
    return float((p+z*z/(2*count)-z*math.sqrt(p*(1-p)/count+z*z/(4*count*count)))/(1+z*z/count))


def describe(trace, rotation):
    obs = trace['obs'][..., :16].astype(np.float64)
    # Sensor readings in obs[t] refer to incoming state; true_rel[t-1] refers
    # to that same state. No future geometry enters the observed controller.
    rel = trace['true_rel'][:-1].astype(np.float64)
    raw = obs[1:]
    valid = trace['valid'][1:].astype(bool)
    force = raw[..., 10:13]*20.
    torque = raw[..., 13:16]*2.
    initial_force = obs[0, :, 10:13]*20.
    initial_torque = obs[0, :, 13:16]*2.
    mean_force = force.copy()
    mean_torque = torque.copy()
    stall = np.zeros(valid.shape, dtype=bool)
    for i in range(9, len(raw)):
        history = raw[i-9:i+1]
        stall[i] = history[:5, :, 2].mean(0)-history[5:, :, 2].mean(0) < .0002
        mean_force[i] = force[i-4:i+1].mean(0)
        mean_torque[i] = torque[i-4:i+1].mean(0)
    cop, vertical = pressure_point(mean_force-initial_force, mean_torque-initial_torque, rotation)
    partial = (np.max(np.abs(rel[..., :2]), axis=-1) > .0005) & (
        np.max(np.abs(rel[..., :2]), axis=-1) < .0205)
    surface = (rel[..., 2] >= -.001) & (rel[..., 2] <= .001)
    contact = np.linalg.norm(force, axis=-1) >= 5.
    mask = valid & partial & surface & contact & stall
    episodes = np.flatnonzero(mask.any(axis=0))
    times = mask.argmax(axis=0)[episodes]
    rows = []
    for time, episode in zip(times, episodes):
        true_xy = rel[time, episode, :2]
        point = cop[time, episode]
        norm = np.linalg.norm(point)*np.linalg.norm(true_xy)
        informative = bool(vertical[time, episode] and norm > 1e-12)
        cosine = float(np.clip(np.dot(point, true_xy)/norm, -1, 1)) if informative else None
        rows.append(dict(episode=int(episode), incoming_control_step=int(time+1),
            true_xy_mm=(true_xy*1000).tolist(), true_height_mm=float(rel[time, episode, 2]*1000),
            force_n=force[time, episode].tolist(), torque_nm=torque[time, episode].tolist(),
            estimated_pressure_xy_mm=(point*1000).tolist(), vertical_force_usable=bool(vertical[time, episode]),
            force_component_saturated=bool(np.any(np.abs(raw[time, episode, 10:13]) >= .9999)),
            torque_component_saturated=bool(np.any(np.abs(raw[time, episode, 13:16]) >= .9999)),
            direction_cosine=cosine,
            direction_error_degrees=float(np.degrees(np.arccos(cosine))) if cosine is not None else None))
    cosines=[r['direction_cosine'] for r in rows if r['direction_cosine'] is not None]
    correct=sum(c > 0 for c in cosines)
    associations=[]
    for component in range(6):
        for axis in range(2):
            x=np.asarray([(r['force_n']+r['torque_nm'])[component] for r in rows])
            y=np.asarray([r['true_xy_mm'][axis] for r in rows])
            correlation=(float(np.corrcoef(x,y)[0,1])
                if len(x)>1 and np.std(x)>1e-12 and np.std(y)>1e-12 else None)
            associations.append(dict(component=('Fx','Fy','Fz','Tx','Ty','Tz')[component],
                position_axis=('x','y')[axis], pearson_correlation=correlation))
    return dict(episodes=len(valid[0]), partial_rest_episode_count=len(rows),
        direction_usable_episode_count=len(cosines), correct_half_plane_count=correct,
        correct_half_plane_fraction=correct/len(cosines) if cosines else None,
        descriptive_wilson_lower=wilson_lower(correct, len(cosines)),
        median_direction_error_degrees=float(np.median([r['direction_error_degrees'] for r in rows
            if r['direction_error_degrees'] is not None])) if cosines else None,
        force_saturated_fraction=float(np.mean([r['force_component_saturated'] for r in rows])) if rows else None,
        torque_saturated_fraction=float(np.mean([r['torque_component_saturated'] for r in rows])) if rows else None,
        raw_wrench_position_associations=associations, rows=rows)


def main():
    registry=json.loads((OUT/'registry.json').read_text())
    if 'E' not in registry['completed_phases'] or registry.get('status')=='halted_tripwire':
        raise ValueError('Touch analysis waits until Phase E is complete without an active tripwire')
    from takehome.correction import AimConfig
    from takehome.followup_config import FollowupConfig
    from takehome.followup_env import FollowupEnv
    # Construct robot metadata only; never reset or step the environment.
    env=FollowupEnv(ROOT/'results/policies/force_0.5mm_s2', AimConfig(), config=FollowupConfig())
    rotation=np.asarray(env._hand_rot_check)
    sources=[ROOT/'results/takehome/goal9/before_traces.npz',
             *sorted((ROOT/'results/takehome/goal9').glob('after_seed_*_poses.npz'))]
    groups=[]
    for path in sources:
        with np.load(path, allow_pickle=False) as archive:
            trace={k:archive[k] for k in ('obs','true_rel','valid','test_only','learned_from','test_seed')}
        if not bool(trace['test_only']) or bool(trace['learned_from']):
            raise ValueError('Expected preserved test-only diagnostic poses')
        groups.append(dict(source=str(path.relative_to(ROOT)), test_seed=int(trace['test_seed']),
                           **describe(trace, rotation)))
    output=OUT/'F';output.mkdir(parents=True, exist_ok=True)
    payload=dict(groups=groups, nominal_sensor_to_world_rotation=rotation.tolist(),
        new_practice_episodes=0, new_test_episodes=0, reset_probes=0, fitted_controller=False,
        used_for_final_method_selection=False,
        method='One first stalled surface-contact sample per episode, with five-reading wrench average.',
        masks='Valid incoming state; actual height within +/-1 mm; square-peg partial overlap 0.5..20.5 mm; observed contact >=5 N; existing ten-reading stall rule.',
        limitations=['Only the observed component-clipped wrench is available.',
            'Nominal fixed wrist rotation is used; actual compliance can change it.',
            'A wrench can reflect multiple contacts, friction, gravity and acceleration.',
            'Descriptive intervals are not a controller-validation test; repeated pose sets share test seeds.',
            'Hidden geometry is used only to describe old saved traces, never to train or control.'])
    (output/'signal_diagnosis.json').write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes=plt.subplots(len(groups), 2, figsize=(10, 3*len(groups)), squeeze=False)
    for index, group in enumerate(groups):
        rows=group['rows']
        for axis in range(2):
            ax=axes[index, axis]
            ax.scatter([r['true_xy_mm'][axis] for r in rows],
                       [r['estimated_pressure_xy_mm'][axis] for r in rows], s=12, alpha=.5)
            ax.axhline(0, color='gray', lw=.6);ax.axvline(0, color='gray', lw=.6)
            ax.set_xlabel('Actual peg offset '+('x' if axis==0 else 'y')+' (mm)')
            ax.set_ylabel('Estimated pressure point (mm)')
            ax.set_title(Path(group['source']).stem+f'; n={len(rows)}')
            ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(output/'signal_diagnosis.png', dpi=150);plt.close(fig)
    print(json.dumps([{k:v for k,v in g.items() if k!='rows'} for g in groups]), flush=True)


if __name__=='__main__':
    main()
