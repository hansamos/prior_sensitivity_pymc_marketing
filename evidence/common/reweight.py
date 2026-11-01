"""PSIS reweighting of posterior draws to a changed prior and/or likelihood (ArviZ 1.x).

For draws theta_s from the reference posterior, the posterior under a changed model has weights

    log w_s = sum_i [log p_new_i(theta_s) - log p_ref_i(theta_s)]

over the terms that changed (prior factors and/or log-likelihood terms). Terms that did not change
cancel and must not be included. The ratio is smoothed with Pareto-smoothed importance sampling
(Vehtari et al., 2024); the Pareto k-hat and the effective sample size say when to trust it.

ArviZ 1.x `psislw` is written for `loo`: it returns weights proportional to exp(-x) for the
input x (x = log-likelihood). A log importance ratio r therefore goes in as `-r`.
"""
from __future__ import annotations

import dataclasses

import numpy as np
from arviz_stats.base import array_stats
from scipy.special import logsumexp

CONSTANT_RATIO_ATOL = 1e-10  # absolute spread below which the log ratio counts as constant


@dataclasses.dataclass(frozen=True)
class Reweighting:
    log_ratio: np.ndarray  # (S,) raw log ratio
    weights: np.ndarray  # (S,) normalized; PSIS-smoothed unless status is "identical"/"bounded"
    pareto_k: float  # nan when PSIS was not applied
    ess: float
    ess_fraction: float  # ess / S
    n_draws: int
    n_zero_weight: int  # draws with density exactly 0 under the new model
    status: str  # "identical" | "bounded" | "good" | "warning" | "fail"
    k_threshold: float


def good_k_threshold(n_draws: int) -> float:
    """min(1 - 1/log10(S), 0.7): the PSIS "good" bound (Vehtari et al., 2024)."""
    return float(min(1.0 - 1.0 / np.log10(max(n_draws, 2)), 0.7))


def reweight(log_ratio, r_eff: float = 1.0) -> Reweighting:
    """Smoothed weights and diagnostics for a flat (S,) log importance ratio.

    Draws with log ratio -inf (zero density under the new model) get weight 0 and are left out
    of the smoothing, which is an adaptation, not part of the published method.
    """
    lr = np.asarray(log_ratio, dtype=np.float64).ravel()
    n = lr.size
    if n < 2:
        raise ValueError("Need at least 2 draws.")
    if np.any(np.isnan(lr)) or np.any(np.isposinf(lr)):
        raise ValueError("Log ratio contains NaN or +inf.")
    finite = np.isfinite(lr)
    if not finite.any():
        raise ValueError("New model has zero density on every reference draw.")
    n_zero, lr_f = int(n - finite.sum()), lr[finite]
    k_thr = good_k_threshold(lr_f.size)

    if np.ptp(lr_f) <= CONSTANT_RATIO_ATOL:  # identical, or a pure truncation: nothing to smooth
        w = np.where(finite, 1.0, 0.0)
        w /= w.sum()
        ess = float(1.0 / np.sum(w**2))
        return Reweighting(lr, w, float("nan"), ess, ess / n, n, n_zero,
                           "identical" if n_zero == 0 else "bounded", k_thr)

    lw_f, k = array_stats.psislw(-lr_f, r_eff=r_eff)  # sign: see module docstring
    lw = np.full(n, -np.inf)
    lw[finite] = np.asarray(lw_f, dtype=np.float64)
    if np.any(np.isnan(lw)) or np.any(np.isposinf(lw)):
        raise ValueError("PSIS produced NaN or +inf log weights.")
    w = np.exp(lw - logsumexp(lw))
    ess = float(1.0 / np.sum(w**2))
    k = float(np.asarray(k))
    status = "fail" if (not np.isfinite(k) or k >= 1.0) else "warning" if k > k_thr else "good"
    return Reweighting(lr, w, k, ess, ess / n, n, n_zero, status, k_thr)


def weighted_mean(x, w) -> np.ndarray:
    """Posterior mean under the weights; x has the draw axis first."""
    x = np.asarray(x, dtype=np.float64)
    return np.tensordot(w, x, axes=(0, 0))


def weighted_sd(x, w) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    m = weighted_mean(x, w)
    return np.sqrt(np.tensordot(w, (x - m) ** 2, axes=(0, 0)))
