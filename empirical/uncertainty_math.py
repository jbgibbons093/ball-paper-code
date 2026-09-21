"""Moment propagation for a recall mean under a frozen deep ensemble."""
import numpy as np

def recall_moments(means,variances,features,weight_covariance,bias_variance):
    means=np.asarray(means,dtype=float)
    variances=np.asarray(variances,dtype=float)
    features=np.asarray(features,dtype=float)
    covariance=np.asarray(weight_covariance,dtype=float)
    bias=np.asarray(bias_variance,dtype=float)
    if means.ndim!=2 or variances.shape!=means.shape or features.shape[:2]!=means.shape:
        raise ValueError('Expected member-by-session means and variances and member-by-session-by-feature encodings.')
    if means.shape[1]==0 or not all(np.isfinite(x).all() for x in [means,variances,features,covariance,bias]):
        raise ValueError('Recall moments require finite, nonempty arrays.')
    if np.any(variances<0) or np.any(covariance<0) or np.any(bias<0):
        raise ValueError('Variance components must be nonnegative.')
    n=means.shape[1]
    member_means=means.mean(axis=1)
    averaged_features=features.mean(axis=1)
    within=variances.sum(axis=1)/n**2+(averaged_features**2*covariance).sum(axis=1)+bias
    return float(member_means.mean()),float(within.mean()+member_means.var())

def expected_fractional_variance_reduction(latent_variance,loading,observation_sd):
    if latent_variance<0 or observation_sd<=0:raise ValueError('Invalid latent or observation variance.')
    signal=loading**2*latent_variance
    return signal/(signal+observation_sd**2)

def causal_window_positions(days,start,end,target_position):
    days=np.asarray(days)
    if not 0<=target_position<len(days):raise ValueError('Target position is outside the recorded sequence.')
    return np.flatnonzero((days>=start)&(days<=end)&(np.arange(len(days))<=target_position))
