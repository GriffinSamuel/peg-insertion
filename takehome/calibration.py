"""Episode-success predictor fitted only to each run's own counted practice."""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from takehome.config import action_scales


def feature_matrix(obs, action, step, config):
    scales = np.array(action_scales(config['outputs'], config['step_cap'],
                                   config.get('movement_cap', .0002)))
    remaining = (200. - np.asarray(step)) / 200.
    return np.concatenate([obs, action / scales, remaining[..., None]], axis=-1)


def fit_probability(features, outcomes):
    """Fixed regularized logistic predictor; deterministic full-batch fitting."""
    x, y = np.asarray(features, dtype=float), np.asarray(outcomes, dtype=float)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Nonfinite calibration training data')
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.)
    z = np.column_stack([(x-mean)/scale, np.ones(len(x))])
    def objective(w):
        logits = z @ w
        loss = np.mean(np.logaddexp(0., logits) - y*logits) + .0005*np.sum(w[:-1]**2)
        gradient = z.T @ (expit(logits)-y) / len(y)
        gradient[:-1] += .001*w[:-1]
        return loss, gradient
    fit = minimize(objective, np.zeros(z.shape[1]), jac=True, method='L-BFGS-B',
                   options=dict(maxiter=200, ftol=1e-10, gtol=1e-6))
    return dict(mean=mean.tolist(), scale=scale.tolist(), weights=fit.x.tolist(),
                training_rows=len(y), training_success_fraction=float(y.mean()),
                converged=bool(fit.success), optimizer_message=str(fit.message),
                iterations=int(fit.nit), objective=float(fit.fun),
                true_bias_used=False, raw_q_used=False,
                includes_remaining_time=True, extra_practice_episodes=0)


def predict_probability(model, features):
    z = (np.asarray(features)-np.array(model['mean']))/np.array(model['scale'])
    w = np.array(model['weights'])
    return expit(z @ w[:-1] + w[-1])


def fit_from_practice(run_dir, practice_end=200):
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir/'config.json').read_text())
    budget = json.loads((run_dir/'budget.json').read_text())
    if not 0 < practice_end <= 200 or practice_end % 10:
        raise ValueError('Invalid practice cutoff')
    reserved = {entry['id'] for entry in budget['episodes']}
    if not set(range(practice_end)).issubset(reserved):
        raise ValueError('Calibration requested unreserved practice episodes')
    xs, ys = [], []
    for end in range(10, practice_end+1, 10):
        with np.load(run_dir/f'practice_{end:04d}.npz', allow_pickle=False) as a:
            if 'test_only' in a and bool(a['test_only']):
                raise ValueError('Test episodes cannot train a calibrator')
            ids = np.asarray(a['episode_id'])
            if set(np.unique(ids)) != set(range(end-10,end)):
                raise ValueError('Practice provenance mismatch')
            valid = np.asarray(a['valid'], dtype=bool)
            episode_success = np.any(np.asarray(a['success']) & valid, axis=0)
            labels = np.broadcast_to(episode_success[None,:], valid.shape)
            features = feature_matrix(a['obs'],a['action'],a['step'],cfg)
            xs.append(features[valid]); ys.append(labels[valid])
    model = fit_probability(np.concatenate(xs),np.concatenate(ys))
    model.update(practice_episodes=practice_end,source_run=str(run_dir),config=cfg,
                 training_source='own counted practice only')
    return model


def reliability_bins(predicted, outcome, bins=10):
    p, y = np.asarray(predicted), np.asarray(outcome)
    ids = np.minimum((p*bins).astype(int),bins-1)
    rows = []
    for b in range(bins):
        mask = ids == b
        if mask.any():
            rows.append(dict(bin=b, count=int(mask.sum()), predicted=float(p[mask].mean()),
                             observed=float(y[mask].mean())))
    bias = float(p.mean()-y.mean())
    return dict(bins=rows,rows=len(y),mean_predicted=float(p.mean()),observed=float(y.mean()),
        signed_bias=bias, brier_score=float(np.mean((p-y)**2)),
        expected_calibration_error=float(sum(r['count']*abs(r['predicted']-r['observed']) for r in rows)/len(y)),
        average_assessment='overconfident' if bias>.03 else ('underconfident' if bias<-.03 else 'about right on average'))


def raw_rating_bins(ratings, outcomes):
    q, y = np.asarray(ratings), np.asarray(outcomes)
    edges = np.unique(np.quantile(q,np.linspace(0,1,11)))
    ids = np.searchsorted(edges[1:-1],q,side='right')
    return [dict(bin=int(b),count=int((ids==b).sum()),raw_q=float(q[ids==b].mean()),
                 observed=float(y[ids==b].mean())) for b in np.unique(ids)]


def evaluate_trace(model, trace):
    if not bool(trace['test_only']) or bool(trace['learned_from']):
        raise ValueError('Calibration chart requires test-only data')
    features = feature_matrix(trace['obs'], trace['action'],trace['step'],model['config'])
    predicted = predict_probability(model,features)
    valid = np.asarray(trace['valid'],bool)
    success = np.asarray(trace['episode_success']) > .5
    labels = np.broadcast_to(success[None,:],valid.shape)
    summary = reliability_bins(predicted[valid],labels[valid])
    summary['raw_q_bins'] = raw_rating_bins(trace['q'][valid],labels[valid])
    failed = np.flatnonzero(~success)
    moments, drops = [], []
    for episode in failed:
        times = np.flatnonzero(valid[:,episode])
        values = predicted[times,episode]
        peak = int(np.argmax(values)); t = int(times[peak])
        if values[peak] >= .8:
            moments.append(dict(episode_id=int(episode),step=t,predicted=float(values[peak]),
                raw_q=float(trace['q'][t,episode]),guess_m=trace['guess'][t,episode].tolist(),
                true_bias_m=trace['true_bias'][t,episode].tolist(),force_n=float(trace['force'][t,episode])))
        # Compare two adjacent five-step averages; retain null when there is no
        # 25-percentage-point fall, instead of inventing a confidence collapse.
        changes = [(int(times[i]),float(values[i-5:i].mean()-values[i:i+5].mean()))
                   for i in range(5,len(values)-4)]
        qualifying = [item for item in changes if item[1] >= .25]
        drops.append(dict(episode_id=int(episode),step=qualifying[0][0] if qualifying else None,
                          drop=qualifying[0][1] if qualifying else None))
    summary.update(confident_failed_moments=moments,failed_confidence_drops=drops,
                   failed_episodes=len(failed),test_seed=int(trace['test_seed']))
    return summary, predicted[valid], labels[valid], np.asarray(trace['q'])[valid]
