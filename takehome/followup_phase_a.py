"""Saved-data aim diagnosis. No simulation and no fitting on test episodes."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from takehome.followup_runtime import ROOT, OUT
import json
import time
import numpy as np

CONTACT_N = 5.0
WINDOWS = (1, 20, 40)
RIDGE = 0.01


def read_trace(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def avg(values):
    return float(np.mean(values)) if len(values) else None


def first(mask):
    return np.where(mask.any(axis=0), mask.argmax(axis=0), -1)


def histories(obs, t, e, window):
    times = np.maximum(t[:, None] - np.arange(window - 1, -1, -1), 0)
    return obs[times, e[:, None], :16].reshape(len(t), 16 * window).astype(np.float64)


def association(x, y):
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def trace_metrics(trace):
    valid = trace['valid'].astype(bool)
    success_t = first(trace['success'].astype(bool) & valid)
    contact_t = first((trace['force'] >= CONTACT_N) & valid)
    n = valid.shape[1]
    bias = trace['true_bias'][0]
    assert n == 500 and np.allclose(trace['true_bias'], bias[None, :, :])
    assert np.allclose(trace['guess'][0], 0)
    initial_error = np.linalg.norm(trace['guess'][0] - bias, axis=-1) * 1000
    end_guess = trace['next_guess'][-1]
    end_error = np.linalg.norm(end_guess - bias, axis=-1) * 1000
    contact_ids = np.flatnonzero(contact_t >= 0)
    success_ids = np.flatnonzero(success_t >= 0)
    contact_error = np.linalg.norm(trace['next_guess'][contact_t[contact_ids], contact_ids]
                                   - bias[contact_ids], axis=-1) * 1000
    success_error = np.linalg.norm(trace['next_guess'][success_t[success_ids], success_ids]
                                   - bias[success_ids], axis=-1) * 1000
    last_valid = valid.sum(axis=0) - 1
    live_end_guess = trace['next_guess'][last_valid, np.arange(n)]
    norms = np.linalg.norm(end_guess, axis=-1)
    unit = end_guess[norms > 1e-12] / norms[norms > 1e-12, None]
    return dict(test_episodes=n, test_seed=int(trace['test_seed']),
        successes=len(success_ids), contacts=len(contact_ids),
        true_error_mean_mm=avg(np.linalg.norm(bias, axis=-1)*1000),
        guess_error_start_mm=avg(initial_error),
        guess_error_first_contact_mm=avg(contact_error),
        guess_error_success_mm=avg(success_error),
        paired_start_error_success_episodes_mm=avg(initial_error[success_ids]),
        paired_error_reduction_at_success_mm=avg(initial_error[success_ids]-success_error),
        guess_error_end200_mm=avg(end_error),
        guess_error_last_valid_mm=avg(np.linalg.norm(live_end_guess-bias, axis=-1)*1000),
        end_closer_fraction=float(np.mean(end_error < initial_error)),
        first_contact_control_step_mean=avg(contact_t[contact_ids]+1),
        mean_end_guess_xy_mm=(end_guess.mean(axis=0)*1000).tolist(),
        std_end_guess_xy_mm=(end_guess.std(axis=0, ddof=1)*1000).tolist(),
        end_direction_concentration=float(np.linalg.norm(unit.mean(axis=0))) if len(unit) else None,
        end_guess_bias_pearson_xy=[association(end_guess[:,i], bias[:,i]) for i in range(2)])


def prediction_probe(directory, trace, seed):
    # Fit only this run's own counted practice; fixed every-tenth-valid-step sampling.
    budget = json.loads((directory/'budget.json').read_text())
    assert budget['limit'] == 200 and len(budget['episodes']) == 200
    training = []
    episode_ids = set()
    for end in range(10, 201, 10):
        data = read_trace(directory / f'practice_{end:04d}.npz')
        assert data['obs'].shape[:2] == (200, 10)
        assert not bool(data.get('test_only', False))
        episode_ids.update(map(int, np.unique(data['episode_id'])))
        ts, es = np.where(data['valid'].astype(bool) &
                          (np.arange(200)[:, None] % 10 == 0))
        training.append((data['obs'], ts, es, data['true_bias'][ts, es].astype(np.float64)*1000))
    assert episode_ids == set(range(200))
    contacts = first((trace['force'] >= CONTACT_N) & trace['valid'].astype(bool))
    ids = np.flatnonzero((contacts >= 0) & (contacts < 199))
    moments = {'first_contact_reading': (contacts[ids]+1, ids)}
    ids125 = np.flatnonzero(trace['valid'][124])
    moments['control_step125_before_action'] = (np.full(len(ids125), 124), ids125)
    rows = []
    for window in WINDOWS:
        x = np.concatenate([histories(o,t,e,window) for o,t,e,y in training])
        y = np.concatenate([y for o,t,e,y in training])
        mean, scale = x.mean(axis=0), x.std(axis=0)
        scale = np.where(scale > 1e-8, scale, 1.)
        z = (x-mean)/scale
        intercept = y.mean(axis=0)
        gram = z.T @ z / len(z) + RIDGE*np.eye(z.shape[1])
        weights = np.linalg.solve(gram, z.T @ (y-intercept) / len(z))
        assert np.isfinite(weights).all()
        for moment, (ts, es) in moments.items():
            features = histories(trace['obs'], ts, es, window)
            prediction = (features-mean)/scale @ weights + intercept
            target = trace['true_bias'][ts,es].astype(np.float64)*1000
            error = np.linalg.norm(prediction-target, axis=-1)
            denom = np.sum((target-target.mean(axis=0))**2)
            rows.append(dict(seed=seed, window_steps=window, moment=moment,
                train_practice_episodes=200, train_step_samples=len(x),
                evaluation_episodes=len(es), prediction_mean_error_mm=avg(error),
                zero_guess_mean_error_mm=avg(np.linalg.norm(target,axis=-1)),
                training_mean_guess_error_mm=avg(np.linalg.norm(target-intercept,axis=-1)),
                predictive_r2=float(1-np.sum((prediction-target)**2)/denom) if denom > 0 else None,
                fitted_on_test=False, hidden_bias_labels_used_for_diagnostic_fit_only=True))
        print(json.dumps(dict(seed=seed,window=window,fit_samples=len(x))),flush=True)
    return rows


def main():
    started = time.perf_counter()
    output = OUT/'A'
    output.mkdir(parents=True,exist_ok=True)
    metrics, probes = [], []
    for seed in range(100,105):
        directory = ROOT/f'results/takehome/goal7/confirmation/aim/seed_{seed}'
        path = directory/'test_0200_traces.npz'
        trace = read_trace(path)
        assert bool(trace['test_only']) and not bool(trace['learned_from'])
        assert int(trace['test_seed']) == 456
        metrics.append(dict(seed=seed,source=str(path.relative_to(ROOT)),**trace_metrics(trace)))
        probes.extend(prediction_probe(directory,trace,seed))
    before_path = ROOT/'results/takehome/goal9/before_traces.npz'
    before = read_trace(before_path)
    bias = np.linalg.norm(before['true_bias'][0],axis=-1)*1000
    outcomes = before['episode_success'] > .5
    bins = []
    for label, mask in [('under0.5mm',bias<.5),('0.5_to1.5mm',(bias>=.5)&(bias<=1.5)),('over1.5mm',bias>1.5)]:
        bins.append(dict(error_bin=label,episodes=int(mask.sum()),successes=int(outcomes[mask].sum()),
                         success_rate=avg(outcomes[mask]),mean_true_error_mm=avg(bias[mask])))
    payload = dict(aim_metrics=metrics,prediction_probes=probes,old_policy_bias_bins=bins,
        old_policy_source=str(before_path.relative_to(ROOT)),old_policy_test_seed=int(before['test_seed']),
        old_policy_successes=int(outcomes.sum()),old_policy_test_episodes=len(outcomes),
        contact_threshold_n=CONTACT_N,contact_steps_one_based=True,
        contact_and_success_guess='next_guess from the step that first recorded contact/success',
        end200_warning='Includes post-success control in the unchanged full-horizon test; last-valid and success metrics are also reported.',
        predictor=dict(kind='linear ridge',normalized_l2_penalty=RIDGE,windows=list(WINDOWS),
            raw_inputs='16 original readings only; no true error, guess, future reading or test label among inputs',
            train_sampling='Every tenth valid step from all200 own counted practice episodes',
            padding='Repeat first reading at the beginning of each episode',
            interpretation='A fixed linear diagnostic, not a claim about all possible nonlinear predictors',
            old_confirmation_data_not_used_to_select_followup_settings=True),
        new_practice_episodes=0,new_test_episodes=0,seconds=time.perf_counter()-started)
    (output/'diagnosis.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(output/'diagnosis.json'),new_episodes=0,seconds=payload['seconds'])),flush=True)


if __name__ == '__main__':
    main()
