"""Development-fitted Gaussian and history-stratified Markov forward filters.

Full development records may estimate parameters. Evaluation predictions use
only the current structured record, lagged treatment, and questionnaires from
strictly earlier sessions. Issued latent estimates are never revised when
forming the same recall-window endpoint used for the causal BALL student.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def prior_counts(anchors, n):
    times = np.sort(anchors.loc[anchors.observed.astype(bool), 't'].to_numpy(int))
    return np.searchsorted(times, np.arange(n), side='left')


def pattern_groups(anchors, n, edges):
    return np.searchsorted(edges, prior_counts(anchors, n), side='right')


def update_gaussian(mean, covariance, indices, weights, value, variance):
    cross = covariance[:, indices] @ weights
    residual_variance = float(weights @ cross[indices] + variance)
    if not np.isfinite(residual_variance) or residual_variance <= 0:
        raise ValueError('Invalid Gaussian observation variance.')
    mean = mean + cross * ((value - weights @ mean[indices]) / residual_variance)
    covariance = covariance - np.outer(cross, cross) / residual_variance
    return mean, (covariance + covariance.T) * .5


def forward_filter(drift, persistence, innovation_variance, daily, daily_observed,
                   daily_variance, anchors, prior_variance, observation_update=None):
    """Append states in session order, assimilating each anchor one session late.

    The growing covariance retains the correlations needed for recall-window
    measurements. Past internal states can update, but past issued estimates
    stay fixed. No backward pass is performed on evaluation records.
    """
    n = len(daily)
    issued = np.empty(n); issued_variance = np.empty(n)
    mean = np.empty(0); covariance = np.empty((0, 0))
    by_time = {}
    for r in anchors.itertuples(index=False):
        if bool(r.observed):
            by_time.setdefault(int(r.t), []).append(r)
    for k in range(n):
        if k:
            for r in by_time.get(k-1, []):
                if observation_update is not None:
                    mean, covariance = observation_update(mean, covariance, r)
                    continue
                start = max(0, int(r.window_start))
                end = min(k-1, int(r.window_end), int(r.t))
                if end < start:
                    raise ValueError('Empty retained questionnaire recall window.')
                idx = np.arange(start, end+1)
                weights = np.full(len(idx), float(r.loading)/len(idx))
                mean, covariance = update_gaussian(
                    mean, covariance, idx, weights,
                    float(r.value)-float(getattr(r, 'intercept', 0.)),
                    float(r.observation_sd)**2)
            a = persistence[k-1]
            cross = covariance[:, -1]*a
            new = np.empty((k+1, k+1))
            new[:-1, :-1] = covariance
            new[:-1, -1] = cross; new[-1, :-1] = cross
            new[-1, -1] = a*a*covariance[-1, -1]+innovation_variance[k-1]
            mean = np.r_[mean, a*mean[-1]+drift[k-1]]
            covariance = new
        else:
            mean = np.zeros(1); covariance = np.array([[prior_variance]], float)
        if daily_observed[k] and np.isfinite(daily[k]):
            mean, covariance = update_gaussian(mean, covariance, np.array([k]),
                np.ones(1), float(daily[k]), daily_variance)
        issued[k] = mean[k]; issued_variance[k] = covariance[k, k]
    return issued, issued_variance


def person_arrays(data, pid, readout, channel, E):
    comp = data.components[data.components.id.eq(pid)].sort_values('t').copy()
    if 'session_observed' in comp:
        comp = comp[comp.session_observed.astype(bool)]
    daily = data.daily[data.daily.id.eq(pid) & data.daily.t.isin(comp.t)].sort_values('t')
    all_anchors = data.anchors[data.anchors.id.eq(pid)].copy()
    anchors = all_anchors[all_anchors.anchor.eq(channel)].copy()
    # Same-day sessions after the questionnaire are outside its recall window.
    anchors['window_end'] = np.minimum(anchors.window_end, anchors.t)
    b = comp[[f'B{j}' for j in range(data.config.q)]].astype(float).ffill().fillna(0.).to_numpy()
    cols = [f'X{j}' for j in range(data.config.p_daily)]
    maskcols = [f'input_{c}' for c in cols]
    if not set(maskcols).issubset(daily): maskcols = [f'obs_{c}' for c in cols]
    x = daily[cols].to_numpy(float); mask = daily[maskcols].to_numpy(bool) & np.isfinite(x)
    intercept, beta, means, sd = readout
    predicted = intercept + np.where(mask, x, means[None, :]) @ beta
    features = E._dc_transition_features(b, comp.a.to_numpy(int),
        comp.recent_treatment.to_numpy(float), comp.treatment_burden.to_numpy(float),
        data.config.n_treatment_types, basis='linear')
    gaps = comp.dt.to_numpy(float)[:-1]
    if np.any(gaps < 0): raise ValueError('Negative elapsed time.')
    return dict(comp=comp, anchors=anchors, all_anchors=all_anchors,
                daily=predicted, daily_observed=mask.any(axis=1), daily_variance=sd**2,
                features=features, gaps=gaps)


def transition_arrays(person, model, args):
    features = person['features']; gaps = person['gaps']
    theta = model['theta']; nbase = features.shape[1]
    drift = features @ theta[:nbase]
    phi = np.zeros(len(gaps))
    if model['kind'] == 'markov':
        group = pattern_groups(person['all_anchors'], len(gaps)+1, model['edges'])[1:]
        drift += theta[nbase + group]
        phi = np.clip(theta[nbase+args.markov_strata+group], -.95, .05)
    persistence = np.exp(phi*gaps)
    scale = gaps.copy()
    nz = np.abs(phi) >= 1e-8
    scale[nz] = np.expm1(phi[nz]*gaps[nz])/phi[nz]
    return drift*scale, persistence, args.transition_sd**2*gaps


def smooth_development(person, model, args):
    """Complete-record fitting restricted by the caller to development patients."""
    n = len(person['daily']); rows = []; targets = []
    def add(idx, weight, target, variance):
        row = np.zeros(n); sd = np.sqrt(max(variance, 1e-10))
        row[idx] = np.asarray(weight)/sd; rows.append(row); targets.append(target/sd)
    add([0], [1.], 0., args.prior_sd**2)
    for r in person['anchors'].itertuples(index=False):
        if not r.observed: continue
        idx = np.arange(max(0, int(r.window_start)), min(n-1, int(r.window_end), int(r.t))+1)
        add(idx, np.full(len(idx), r.loading/len(idx)),
            r.value-getattr(r, 'intercept', 0.), r.observation_sd**2)
    for k in np.flatnonzero(person['daily_observed']):
        add([k], [1.], person['daily'][k], person['daily_variance'])
    drift, persistence, variance = transition_arrays(person, model, args)
    for k in range(1, n):
        add([k-1, k], [-persistence[k-1], 1.], drift[k-1], variance[k-1])
    return np.linalg.lstsq(np.asarray(rows), np.asarray(targets), rcond=None)[0]


def fit_models(data, E, args=None):
    args = E._dc_args('linear') if args is None else args
    if args.s0_basis != 'linear':
        raise ValueError('The finalized prospective empirical comparators use the linear basis.')
    train_ids = sorted(data.individuals.loc[data.individuals.split.eq('train'), 'id'].astype(int))
    models = {}
    for channel in ['Y1', 'Y2']:
        readout = E._dc_fit_daily_readout(data, set(train_ids), args.daily_ridge, anchor_name=channel)
        people = {pid: person_arrays(data, pid, readout, channel, E) for pid in train_ids}
        nbase = next(iter(people.values()))['features'].shape[1]
        counts = np.concatenate([prior_counts(p['all_anchors'], len(p['daily'])) for p in people.values()])
        edges = np.quantile(counts, np.arange(1, args.markov_strata)/args.markov_strata)
        lg = dict(kind='linear_gaussian', theta=np.zeros(nbase), readout=readout, edges=edges)
        for _ in range(args.iters):
            xblocks = []; yblocks = []
            for person in people.values():
                z = smooth_development(person, lg, args); ok = person['gaps'] > 0
                if ok.any():
                    xblocks.append(person['features'][ok]); yblocks.append(np.diff(z)[ok]/person['gaps'][ok])
            x = np.vstack(xblocks); y = np.concatenate(yblocks)
            penalty = args.beta_ridge*np.eye(nbase); penalty[0, 0] = 0.
            lg['theta'] = np.linalg.solve(x.T@x+penalty, x.T@y)
        markov = dict(kind='markov', theta=np.r_[lg['theta'], np.zeros(2*args.markov_strata)],
                      readout=readout, edges=edges)
        for _ in range(args.markov_iters):
            xblocks = []; yblocks = []
            for person in people.values():
                z = smooth_development(person, markov, args); ok = person['gaps'] > 0
                groups = pattern_groups(person['all_anchors'], len(z), edges)[1:]
                onehot = np.eye(args.markov_strata)[groups]
                xblocks.append(np.column_stack([person['features'], onehot, onehot*z[:-1, None]])[ok])
                yblocks.append(np.diff(z)[ok]/person['gaps'][ok])
            x = np.vstack(xblocks); y = np.concatenate(yblocks)
            penalty = args.markov_ridge*np.eye(x.shape[1]); penalty[0, 0] = 0.
            theta = np.linalg.solve(x.T@x+penalty, x.T@y)
            theta[nbase+args.markov_strata:] = np.clip(theta[nbase+args.markov_strata:], -.95, .05)
            markov['theta'] = theta
        models[channel] = dict(lg=lg, markov=markov)
    return models, args, train_ids


def predict_patient(data, pid, model, channel, args, E):
    person = person_arrays(data, pid, model['readout'], channel, E)
    drift, persistence, variance = transition_arrays(person, model, args)
    return forward_filter(drift, persistence, variance, person['daily'],
        person['daily_observed'], person['daily_variance'], person['anchors'], args.prior_sd**2)
