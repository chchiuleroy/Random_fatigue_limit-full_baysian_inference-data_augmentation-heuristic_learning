#!/usr/bin/env python3
"""
rfl_self_adaptive_mcmc.py
Full Bayesian RFL -- Self-Regulating Adaptive MCMC (SEV + LogNormal)

wiki concept integration:
  [[concept_adaptive_mcmc]]             -- AM (Haario 2001), Diminishing Adaptation
  [[concept_self_regulating_annealing]] -- alpha(d^2) = sqrt((nu+d^2)/(nu+dim-2)), HTDM
  [[concept_bayesian_rfl_inference]]    -- DA + NCP + NUTS (baseline to compare)

Algorithm: 4-block Gibbs sweep
  A: (b0, b1, log sigma)  -- Self-Regulating Adaptive Metropolis (d=3)
  B: mu_Delta             -- exact conjugate Gibbs (Normal posterior)
  C: sigma_Delta          -- exact conjugate Gibbs (Scaled-Inv-chi^2 posterior)
  D: log Delta_i each     -- vectorised Self-Adaptive MH (parallel per obs)

Self-Regulating alpha (HTDM analogy applied to MCMC proposals):
  alpha(d^2, dim) = sqrt((nu + d^2) / (nu + dim - 2))  [dim >= 2]
                  = sqrt((nu + d^2) / (nu - 1))          [dim = 1]
  d^2 = Mahalanobis distance^2 from current point to running mean

  Far from mean (large d^2) -> alpha > 1 -> proposal expanded  (high-T: explore)
  Near mean (small d^2)     -> alpha < 1 -> proposal contracted (low-T: refine)

Key insight: by using exact Gibbs for (mu_Delta, sigma_Delta), we avoid
the MH acceptance collapse that occurs when moving the hyperparameters
shifts all 75 Delta_i prior terms simultaneously.
"""

import numpy as np
from scipy import stats
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

# ======================================================================
# DATA  (P&M 1999, n=75)
# ======================================================================
_raw = {
    0.675: [102.95, 280.32, 339.83, 366.9, 485.62, 658.96, 896.33,
            1241.76, 1250.2, 1329.78, 1399.83, 1459.14, 3249.82, 11748.1, 11748.1],
    0.750: [6.71, 9.93, 12.6, 15.58, 16.19, 17.28, 18.62,
            20.3, 24.9, 26.26, 27.94, 36.35, 48.42, 50.09, 67.34],
    0.825: [1.246, 1.258, 1.46, 1.492, 2.4, 2.41, 2.59,
            2.903, 3.33, 3.59, 3.847, 4.11, 4.82, 5.56, 5.598],
    0.900: [0.201, 0.216, 0.226, 0.252, 0.257, 0.295, 0.311,
            0.342, 0.356, 0.451, 0.457, 0.509, 0.54, 0.68, 1.129],
    0.950: [0.037, 0.072, 0.074, 0.076, 0.083, 0.085, 0.105,
            0.109, 0.12, 0.123, 0.143, 0.203, 0.206, 0.217, 0.257],
}
S_list, Y_list, ev_list = [], [], []
for s, vals in _raw.items():
    mx = max(vals); nm = vals.count(mx)
    for v in vals:
        S_list.append(s); Y_list.append(np.log(v))
        ev_list.append(0 if (v == mx and nm > 1) else 1)

S_OBS    = np.array(S_list, dtype=float)
Y_OBS    = np.array(Y_list, dtype=float)
EVENT    = np.array(ev_list, dtype=int)
N_OBS    = len(S_OBS)
MIN_S    = float(S_OBS.min())
LOG_S    = np.log(S_OBS)

fail_idx = np.where(EVENT == 1)[0]   # 73
cens_idx = np.where(EVENT == 0)[0]   # 2


# ======================================================================
# LOG-POSTERIOR COMPONENTS
# ======================================================================

def log_prior_reg(b0, b1, ls):
    """Flat Uniform priors for regression block (b0, b1, log sigma)."""
    if not (-50 < b0 < 50):                    return -np.inf
    if not (-30 < b1 < 0):                     return -np.inf
    if not (np.log(0.01) < ls < np.log(5.0)): return -np.inf
    return 0.0


def log_lik_sev_all(b0, b1, ls, log_delta):
    """SEV conditional log-lik, scalar sum, given log Delta (N_OBS,)."""
    if np.any(log_delta >= LOG_S):
        return -np.inf
    sigma  = np.exp(ls)
    excess = np.maximum(S_OBS - np.exp(log_delta), 1e-10)
    mu_c   = b0 + b1 * np.log(excess)
    z      = (Y_OBS - mu_c) / sigma
    ll     = np.empty(N_OBS)
    ll[fail_idx] = -ls + z[fail_idx] - np.exp(np.clip(z[fail_idx], -500, 20))
    ll[cens_idx] = -np.exp(np.clip(z[cens_idx], -500, 20))
    return float(np.sum(ll))


def _ll_vec(b0, b1, ls, log_delta):
    """SEV log-lik per obs, returns (N_OBS,) array."""
    sigma  = np.exp(ls)
    excess = np.maximum(S_OBS - np.exp(log_delta), 1e-10)
    mu_c   = b0 + b1 * np.log(excess)
    z      = (Y_OBS - mu_c) / sigma
    ll     = np.empty(N_OBS)
    ll[fail_idx] = -ls + z[fail_idx] - np.exp(np.clip(z[fail_idx], -500, 20))
    ll[cens_idx] = -np.exp(np.clip(z[cens_idx], -500, 20))
    return ll


def _reg_hessian_cov(b0, b1, ls, log_delta, eps=5e-3):
    """
    Numerical inverse-Hessian of log-posterior for (b0, b1, ls).

    Used to warm-start the Welford covariance in Block A.  With this,
    the AM proposal already captures the b0-b1 correlation (~+0.96 for
    P&M data) from iteration 1, instead of discovering it after hundreds
    of slow-mixing samples.

    Returns the 3x3 posterior covariance matrix, or None if numerical issues.
    """
    x = np.array([b0, b1, ls], float)
    if not np.isfinite(log_lik_sev_all(*x, log_delta)):
        return None
    n = 3
    H = np.zeros((n, n))
    f0 = log_lik_sev_all(*x, log_delta)
    for i in range(n):
        ei = np.zeros(n); ei[i] = eps
        fp = log_lik_sev_all(*(x + ei), log_delta)
        fm = log_lik_sev_all(*(x - ei), log_delta)
        H[i, i] = (fp - 2 * f0 + fm) / eps**2
        for j in range(i + 1, n):
            ej = np.zeros(n); ej[j] = eps
            fpp = log_lik_sev_all(*(x + ei + ej), log_delta)
            fpm = log_lik_sev_all(*(x + ei - ej), log_delta)
            fmp = log_lik_sev_all(*(x - ei + ej), log_delta)
            fmm = log_lik_sev_all(*(x - ei - ej), log_delta)
            H[i, j] = H[j, i] = (fpp - fpm - fmp + fmm) / (4 * eps**2)
    H_neg = -(H + H.T) / 2
    lam_min = np.linalg.eigvalsh(H_neg).min()
    if lam_min <= 0:
        H_neg += (abs(lam_min) + 1e-6) * np.eye(n)
    try:
        C = np.linalg.inv(H_neg)
        if np.all(np.linalg.eigvalsh(C) > 0):
            return C
    except np.linalg.LinAlgError:
        pass
    return None


# ======================================================================
# EXACT CONJUGATE GIBBS for hyperparameters
# ======================================================================

def _gibbs_mu_d(log_delta, sigma_d, rng):
    """
    Exact draw: mu_Delta | sigma_Delta, log_Delta
    Prior: flat on (-5, log(MIN_S) - 0.05)
    Posterior: N(xbar, sigma_d^2 / n), truncated to prior bounds.
    """
    n        = len(log_delta)
    xbar     = float(log_delta.mean())
    sd_post  = sigma_d / np.sqrt(n)
    upper_mu = np.log(MIN_S) - 0.05
    for _ in range(30):
        mu_new = rng.normal(xbar, sd_post)
        if -5.0 < mu_new < upper_mu:
            return mu_new
    return float(np.clip(xbar, -4.9, upper_mu - 1e-4))


def _gibbs_sigma_d(log_delta, mu_d, rng):
    """
    Exact draw: sigma_Delta | mu_Delta, log_Delta
    Prior: log-Uniform = Jeffreys scale prior p(sigma) ~ 1/sigma
    Posterior: sigma_Delta ~ sqrt(RSS / chi^2(n))  [Scaled-Inv-chi^2(n, RSS/n)]
    """
    n   = len(log_delta)
    RSS = float(np.sum((log_delta - mu_d)**2))
    RSS = max(RSS, 1e-10)
    for _ in range(30):
        chi2   = rng.chisquare(n)
        sd_new = np.sqrt(RSS / chi2)
        if 0.001 < sd_new < 2.0:
            return sd_new
    return float(np.sqrt(RSS / n))


# ======================================================================
# SELF-REGULATING alpha  (HTDM analogy)
# ======================================================================

def self_reg_alpha(x, mean, cov_inv, nu, dim):
    """
    Normalised Self-Regulating alpha:

        alpha(d^2) = sqrt((nu + d^2) / (nu + dim))

    Normalisation: at the expected equilibrium Mahalanobis distance
    E[d^2] = dim (for a Gaussian chain at stationarity), alpha = 1.0
    -- exactly reproducing standard Adaptive Metropolis.

    The deviation from 1.0 encodes the HTDM self-regulating idea:
      d^2 > dim  (far from running mean, exploring tail)  -> alpha > 1
      d^2 < dim  (near running mean, converging)          -> alpha < 1

    This applies to both the regression block (dim=3) and each
    Delta_i (dim=1), giving state-dependent but calibrated scaling.

    Clipped to [0.1, 3.0] for numerical stability.
    """
    diff = np.asarray(x, float) - np.asarray(mean, float)
    if dim == 1:
        d2 = float(diff.ravel()[0]**2) * float(np.asarray(cov_inv).ravel()[0])
    else:
        d2 = float(diff @ np.asarray(cov_inv) @ diff)
    return float(np.clip(np.sqrt((nu + d2) / (nu + dim)), 0.1, 3.0))


# ======================================================================
# WELFORD ONLINE STATISTICS
# ======================================================================

class WelfordCov:
    def __init__(self, dim):
        self.n = 0; self.mean = np.zeros(dim); self.M2 = np.zeros((dim, dim))

    def update(self, x):
        self.n += 1
        d1 = x - self.mean; self.mean += d1 / self.n
        self.M2 += np.outer(d1, x - self.mean)

    @property
    def cov(self):
        return self.M2 / max(self.n - 1, 1)


class WelfordVarVec:
    """Vectorised Welford for N_OBS scalar series simultaneously."""
    def __init__(self, n):
        self.count = 0; self.mean = np.zeros(n); self.M2 = np.zeros(n)

    def update(self, x):
        self.count += 1
        d1 = x - self.mean; self.mean += d1 / self.count
        self.M2 += d1 * (x - self.mean)

    @property
    def var(self):
        return self.M2 / max(self.count - 1, 1) + 1e-12

    @property
    def std(self):
        return np.sqrt(self.var)


# ======================================================================
# DIAGNOSTICS
# ======================================================================

def rhat(chains):
    """Gelman-Rubin Rhat (classic). chains: (n_chains, n_draws, n_dim).
    For consistency with the rest of this project (Section 5.1 uses arviz's
    rank-normalized Rhat), prefer arviz.rhat() for reported numbers -- this
    function historically divided B by M instead of M-1 (underestimating
    between-chain variance); fixed 2026-08-11."""
    M, N, D = chains.shape
    cm = chains.mean(axis=1); gm = cm.mean(axis=0)
    B  = N / (M - 1) * ((cm - gm)**2).sum(axis=0)
    W  = chains.var(axis=1, ddof=1).mean(axis=0)
    v  = (N - 1) / N * W + B / N
    return np.sqrt(v / np.maximum(W, 1e-15))


def ess_bulk(chains):
    """Bulk ESS via FFT autocorrelation. chains: (n_chains, n_draws, n_dim)"""
    M, N, D = chains.shape
    out = []
    for d in range(D):
        x   = chains[:, :, d].flatten() - chains[:, :, d].mean()
        n   = len(x)
        fft = np.fft.rfft(x, n=2*n)
        ac  = np.fft.irfft(fft * np.conj(fft))[:n]
        ac  = ac / (np.arange(n, 0, -1) * (x.var() + 1e-15))
        rho = ac[1:]
        pairs = rho[:-1] + rho[1:]
        cut   = next((i for i, p in enumerate(pairs) if p < 0), len(pairs))
        tau   = 1.0 + 2.0 * rho[:cut+1].sum()
        out.append(n / max(tau, 1.0))
    return np.array(out)


# ======================================================================
# SINGLE-CHAIN SAMPLER
# ======================================================================

def _run_chain(seed, n_warmup, n_draws, nu, verbose):
    rng = np.random.default_rng(seed)
    D_R = 3   # regression block dimension (b0, b1, log_sigma)

    # -- Initial state near MLE, perturbed per chain --------------------
    # Perturbation is 1-2 CONDITIONAL posterior SD, NOT marginal SD.
    # Marginal SD (b0≈0.30, b1≈0.36) is ~7× the conditional SD (b0≈0.022).
    # Starting chains at 7 conditional SDs apart causes each chain to adapt
    # its Δ_i to an inconsistent (b0,b1) configuration, trapping it on a
    # different part of the (mu_d, b1) identifiability ridge.
    b0    = -9.37  + rng.normal(0, 0.025)   # ~1 cond. SD
    b1    = -8.53  + rng.normal(0, 0.020)   # ~1 cond. SD
    ls    = np.log(0.19) + rng.normal(0, 0.055)   # ~1 cond. SD
    mu_d  = -0.644 + rng.normal(0, 0.010)
    sd_d  = 0.036  * np.exp(rng.normal(0, 0.05))   # sigma_Delta > 0

    reg = np.array([b0, b1, ls])
    log_delta = np.full(N_OBS, mu_d) + rng.normal(0, sd_d, N_OBS)
    log_delta = np.clip(log_delta, -5.0, LOG_S - 0.001)

    # -- Welford trackers -----------------------------------------------
    wc_reg = WelfordCov(D_R)
    wvv_dl = WelfordVarVec(N_OBS)

    # -- Pilot run: warm log_delta before computing Hessian -------------
    # The Hessian of log p(b0,b1,ls | log_delta, Y) at the RANDOM initial
    # log_delta is ill-conditioned (condition ~1e12): exp(z_i) is huge for
    # high-N censored observations (z ≈ 13), making H extremely negative in
    # some directions while nearly flat in others.
    # Fix: run 80 iterations of Blocks B/C/D alone to bring log_delta close
    # to the conditional MAP; then the Hessian is well-conditioned (~300).
    _min_s = 0.04
    _wvv_p = WelfordVarVec(N_OBS)
    for _ in range(200):
        mu_d = _gibbs_mu_d(log_delta, sd_d, rng)
        sd_d = _gibbs_sigma_d(log_delta, mu_d, rng)
        _wvv_p.update(log_delta)
        _step = np.maximum(2.38 * _wvv_p.std, _min_s)
        _ldp  = log_delta + rng.normal(0.0, _step)
        _vld  = _ldp < LOG_S
        _lc   = _ll_vec(b0, b1, ls, log_delta)
        _lp   = np.where(_vld, _ll_vec(b0,b1,ls,np.where(_vld,_ldp,log_delta)), -np.inf)
        _pc   = stats.norm.logpdf(log_delta, mu_d, sd_d)
        _pp   = np.where(_vld, stats.norm.logpdf(_ldp, mu_d, sd_d), -np.inf)
        _lr   = (_pp + _lp) - (_pc + _lc)
        _acc  = _vld & (np.log(rng.uniform(size=N_OBS)) < _lr)
        log_delta = np.where(_acc, _ldp, log_delta)

    # -- Warm-start AM covariance from Hessian at warmed log_delta ------
    # Fall-back: hand-coded covariance based on theoretical structure
    #   Corr(b0,b1|Δ) ≈ +0.96, SD(b0) ≈ 0.083, SD(b1) ≈ 0.062
    sd0c, sd1c, sdlsc = 0.083, 0.062, 0.080
    C_hand = np.array([
        [sd0c**2,          0.96 * sd0c * sd1c, 0.0],
        [0.96 * sd0c * sd1c,  sd1c**2,         0.0],
        [0.0,              0.0,                 sdlsc**2],
    ])
    C0 = _reg_hessian_cov(b0, b1, ls, log_delta)
    if C0 is not None:
        eig0 = np.linalg.eigvalsh(C0)
        diag0 = np.sqrt(np.diag(C0))
        cond0 = eig0.max() / max(eig0.min(), 1e-30)
        if cond0 < 1e4 and np.all(diag0 < 1.0):
            C_init = C0          # well-conditioned Hessian
        else:
            C_init = C_hand      # fall back
    else:
        C_init = C_hand

    # Warm-start Welford: inject n_fake virtual samples AT the current starting
    # point with covariance C_init.  Setting mean=reg (not some arbitrary
    # reference) keeps d^2 from the running mean near dim at stationarity.
    n_fake = 100
    wc_reg.n    = n_fake
    wc_reg.mean = reg.copy()   # mean = starting point, not fake origin
    wc_reg.M2   = (n_fake - 1) * C_init

    # -- Dynamic Hessian covariance (used during warmup) ----------------
    # The conditional posterior of (b0,b1,ls) given log_delta changes every
    # iteration as log_delta evolves.  A fixed warm-start covariance quickly
    # becomes stale.  During warmup we recompute the Hessian every 50 steps
    # to track the current conditional curvature.  During sampling we switch
    # to the Welford covariance (which by then has accumulated enough samples
    # from the ~23%-acceptance warmup chain).
    C_dyn = C_init.copy()   # starts as pilot-run Hessian or C_hand fallback

    h_dl        = 2.38    # Delta step: max(h_dl * running_std * alpha_i, min_step)
    min_step    = 0.04    # floor ~= sigma_Delta (avoids vanishing step early on)
    N_DL_SWEEPS = 1       # Block D sweeps per Gibbs iteration

    # NOTE: Block E (translation chain) was removed.  Uniform shifts of all
    # log_delta_i bias the chain toward larger Δ when individual log_delta_i
    # are near their conditional posteriors, causing systematic drift in mu_d
    # and b1.  The standard 4-block Gibbs is used; the identifiability ridge
    # is traversed by the AM warmup over enough iterations.

    # -- Counters & storage ---------------------------------------------
    n_total         = n_warmup + n_draws
    acc_reg         = 0
    acc_dl          = np.zeros(N_OBS, dtype=int)
    acc_dl_recent   = np.zeros(N_OBS, dtype=int)  # reset every 100 iters for h_dl adapt

    th_out = np.zeros((n_draws, 5))   # (b0, b1, log_sigma, mu_d, log_sigma_d)
    dl_out = np.zeros((n_draws, N_OBS))

    alpha_reg_log = []
    t0 = time.time()

    for it in range(n_total):
        warmup = it < n_warmup
        b0, b1, ls = reg

        # ==============================================================
        # BLOCK A: (b0, b1, log sigma) via Self-Regulating AM (d=3)
        #
        # Warmup:   use C_dyn (Hessian refreshed every 50 iters) so the
        #           proposal tracks the CURRENT conditional posterior.
        # Sampling: use wc_reg.cov (Welford accumulated during warmup).
        #
        # Self-Regulating alpha always uses the Welford mean as reference
        # point so the scale-up/down reflects the chain's exploration state.
        # ==============================================================

        # Refresh Hessian every 50 warmup iterations
        if warmup and it % 50 == 0:
            _C_new = _reg_hessian_cov(b0, b1, ls, log_delta)
            if _C_new is not None:
                _eig = np.linalg.eigvalsh(_C_new)
                # Accept only if well-conditioned and SDs are < 0.5
                if _eig.min() > 0 and np.all(np.sqrt(np.diag(_C_new)) < 0.5):
                    C_dyn = _C_new

        wc_reg.update(reg)

        C_use = C_dyn if warmup else wc_reg.cov
        try:
            C_inv = np.linalg.inv(C_use + 1e-8 * np.eye(D_R))
            # Self-reg alpha only during sampling: warmup uses pure AM (alpha=1)
            # so the biased running mean doesn't inflate proposals.
            if warmup:
                alpha_r = 1.0
            else:
                alpha_r = self_reg_alpha(reg, wc_reg.mean, C_inv, nu, D_R)
        except np.linalg.LinAlgError:
            alpha_r = 1.0
        C_prop = (2.38**2 / D_R) * alpha_r**2 * C_use
        try:
            L_r = np.linalg.cholesky(C_prop + 1e-8 * np.eye(D_R))
        except np.linalg.LinAlgError:
            L_r = np.diag(np.sqrt(np.diag(C_use)) * alpha_r
                          * np.sqrt(2.38**2 / D_R))

        alpha_reg_log.append(alpha_r)
        reg_p = reg + L_r @ rng.standard_normal(D_R)

        lp_curr = log_prior_reg(*reg)   + log_lik_sev_all(*reg,   log_delta)
        lp_prop = log_prior_reg(*reg_p) + log_lik_sev_all(*reg_p, log_delta)

        if np.log(rng.uniform()) < lp_prop - lp_curr:
            reg = reg_p
            b0, b1, ls = reg
            acc_reg += 1

        # ==============================================================
        # BLOCK B: mu_Delta -- exact conjugate Gibbs
        # Posterior: N(xbar_Delta, sigma_d^2 / n)
        # No MH rejection -- exact and zero-cost.
        # ==============================================================
        mu_d = _gibbs_mu_d(log_delta, sd_d, rng)

        # ==============================================================
        # BLOCK C: sigma_Delta -- exact conjugate Gibbs
        # Posterior: sqrt(RSS / chi^2(n)), Jeffreys scale prior.
        # ==============================================================
        sd_d = _gibbs_sigma_d(log_delta, mu_d, rng)

        # ==============================================================
        # BLOCK D: log Delta_i -- vectorised Self-Regulating MH
        #
        # N_DL_SWEEPS passes per Gibbs step: each Delta_i is a separate
        # chain conditionally on (b0,b1,ls,mu_d,sd_d).  Multiple sweeps
        # per Block-A move are necessary to break the slow-mixing
        # correlation inherent in centered DA parameterisation -- the
        # Gibbs collapses to a tight conditional mode when Delta_i and
        # (b0,b1,ls) are updated only once each per sweep.
        # ==============================================================
        wvv_dl.update(log_delta)        # Welford updated once per main iteration
        var_inv_i = 1.0 / wvv_dl.var

        for _sw in range(N_DL_SWEEPS):
            d2_i    = (log_delta - wvv_dl.mean)**2 * var_inv_i
            alpha_i = np.clip(np.sqrt((nu + d2_i) / (nu + 1.0)), 0.1, 4.0)
            step_i  = np.maximum(h_dl * wvv_dl.std * alpha_i, min_step)
            ld_prop = log_delta + rng.normal(0.0, step_i)
            valid   = ld_prop < LOG_S

            ll_curr = _ll_vec(b0, b1, ls, log_delta)
            ll_prop = np.where(
                valid,
                _ll_vec(b0, b1, ls, np.where(valid, ld_prop, log_delta)),
                -np.inf
            )
            lp_prior_curr = stats.norm.logpdf(log_delta, mu_d, sd_d)
            lp_prior_prop = np.where(
                valid, stats.norm.logpdf(ld_prop, mu_d, sd_d), -np.inf
            )
            log_r  = (lp_prior_prop + ll_prop) - (lp_prior_curr + ll_curr)
            accept = valid & (np.log(rng.uniform(size=N_OBS)) < log_r)
            log_delta        = np.where(accept, ld_prop, log_delta)
            acc_dl          += accept.astype(int)
            acc_dl_recent   += accept.astype(int)

        if warmup and (it + 1) % 100 == 0 and it > 0:
            # Adaptation uses RECENT window (not cumulative), so h_dl
            # responds to current mixing, not the slow initial period.
            avg_recent = acc_dl_recent.sum() / (100 * N_OBS * N_DL_SWEEPS)
            if   avg_recent < 0.25: h_dl *= 0.85
            elif avg_recent > 0.55: h_dl *= 1.2
            h_dl = float(np.clip(h_dl, 0.1, 20.0))
            acc_dl_recent[:] = 0   # reset window

        # -- Store draws ------------------------------------------------
        if not warmup:
            k = it - n_warmup
            th_out[k] = [b0, b1, ls, mu_d, np.log(sd_d)]
            dl_out[k] = np.exp(log_delta)

        if verbose and (it + 1) % 1000 == 0:
            phase = "warmup" if warmup else "sample"
            print(f"  [seed={seed} {phase}] {it+1}/{n_total}  "
                  f"reg-acc={acc_reg/(it+1):.2f}  "
                  f"Delta-acc={acc_dl.mean()/(it+1)/N_DL_SWEEPS:.2f}  "
                  f"alpha(last100)={np.mean(alpha_reg_log[-100:]):.3f}  "
                  f"b0={b0:.3f} b1={b1:.3f} sigma={np.exp(ls):.4f}  "
                  f"mu_d={mu_d:.4f} sd_d={sd_d:.4f}")

    return th_out, dl_out, acc_reg / n_total, acc_dl / n_total, alpha_reg_log


# ======================================================================
# MAIN
# ======================================================================
if __name__ == "__main__":
    N_CHAINS = 4
    N_WARMUP = 2000
    N_DRAWS  = 3000
    NU       = 5.0   # key knob: smaller -> more state-dependent; try 3, 5, 10, 50

    PNAMES = ["b0", "b1", "log_sigma", "mu_d", "log_sigma_d"]

    # MLE reference (SEV + LogNormal, from wiki concept_bayesian_rfl_inference)
    MLE = np.array([-9.370, -8.534, np.log(0.190), -0.644, np.log(0.036)])

    print(f"Self-Regulating Adaptive MCMC -- RFL (SEV + LogNormal)")
    print(f"nu={NU}  chains={N_CHAINS}  warmup={N_WARMUP}  draws={N_DRAWS}")
    print(f"n={N_OBS}  failures={len(fail_idx)}  censored={len(cens_idx)}")
    print(f"Blocks: A=(b0,b1,log_sigma) Self-AM  |  B=mu_d exact  "
          f"|  C=sigma_d exact  |  D=Delta_i Self-MH")
    print("=" * 70)

    seeds  = [42, 137, 271, 999]
    chains = []
    n_workers = min(N_CHAINS, os.cpu_count() or 4)

    print(f"Parallel execution: {n_workers} workers on {os.cpu_count()} cores")
    print("=" * 70)

    t_total = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_run_chain, seed, N_WARMUP, N_DRAWS, NU, False): (ch, seed)
            for ch, seed in enumerate(seeds[:N_CHAINS])
        }
        results = [None] * N_CHAINS
        for future in as_completed(futures):
            ch, seed = futures[future]
            th, dl, acc_r, acc_dl, alpha_log = future.result()
            results[ch] = (th, dl, acc_r, acc_dl, alpha_log)
            alpha_arr = np.array(alpha_log)
            print(f"  Chain {ch+1} (seed={seed}) done: "
                  f"reg-acc={acc_r:.2f}  Delta-acc={acc_dl.mean():.2f}  "
                  f"alpha_reg: mean={alpha_arr.mean():.3f} "
                  f"p5={np.percentile(alpha_arr,5):.3f} "
                  f"p95={np.percentile(alpha_arr,95):.3f}")

    for ch in range(N_CHAINS):
        th, dl, acc_r, acc_dl, alpha_log = results[ch]
        chains.append(th)

    elapsed = time.time() - t_total
    chains_arr = np.stack(chains)   # (N_CHAINS, N_DRAWS, 5)
    all_draws  = chains_arr.reshape(-1, 5)

    rh = rhat(chains_arr)
    es = ess_bulk(chains_arr)
    pm = all_draws.mean(axis=0)
    ps = all_draws.std(axis=0)
    lo = np.percentile(all_draws, 2.5,  axis=0)
    hi = np.percentile(all_draws, 97.5, axis=0)

    print(f"\n{'=' * 70}")
    print(f"POSTERIOR SUMMARY  (Self-Reg AM + conjugate Gibbs, nu={NU})")
    print(f"{'=' * 70}")
    print(f"  {'Param':<12} {'Mean':>9} {'SD':>8} {'2.5%':>9} {'97.5%':>9}"
          f" {'MLE':>9} {'Rhat':>7} {'ESS':>7}")
    print(f"  {'-' * 67}")
    for i, pn in enumerate(PNAMES):
        print(f"  {pn:<12} {pm[i]:>9.4f} {ps[i]:>8.4f} "
              f"{lo[i]:>9.4f} {hi[i]:>9.4f} "
              f"{MLE[i]:>9.4f} {rh[i]:>7.4f} {es[i]:>7.0f}")

    sig_draws  = np.exp(all_draws[:, 2])
    sigd_draws = np.exp(all_draws[:, 4])
    print(f"\n  sigma   (natural): mean={sig_draws.mean():.4f}  "
          f"SD={sig_draws.std():.4f}  "
          f"95%CI=[{np.percentile(sig_draws,2.5):.4f},{np.percentile(sig_draws,97.5):.4f}]"
          f"  MLE=0.190")
    print(f"  sigma_d (natural): mean={sigd_draws.mean():.4f}  "
          f"SD={sigd_draws.std():.4f}  "
          f"95%CI=[{np.percentile(sigd_draws,2.5):.4f},{np.percentile(sigd_draws,97.5):.4f}]"
          f"  MLE=0.036")

    alpha_last = np.array(results[-1][4])
    print(f"\n  --- Self-Regulating alpha_reg summary (last chain) ---")
    print(f"  Warmup mean: {np.mean(alpha_last[:N_WARMUP]):.4f}  "
          f"Sample mean: {np.mean(alpha_last[N_WARMUP:]):.4f}")
    print(f"  nu={NU}: far from mean -> alpha>1 (explore), "
          f"near mean -> alpha<1 (refine)")
    print(f"  nu=inf would collapse to standard AM (alpha==1 always)")

    print(f"\n  --- COMPARISON ---")
    ref = {
        "MLE (SEV+INLA)":       [-9.370, -8.534, 0.190, -0.644, 0.036],
        "PyMC NUTS (post mean)": [-9.29,  -8.75,  0.195, -0.660, 0.038],
        f"Self-AM nu={NU:.0f} (this)": [pm[0], pm[1], sig_draws.mean(),
                                         pm[3], sigd_draws.mean()],
    }
    print(f"  {'Method':<28} {'b0':>7} {'b1':>7} {'sigma':>7} "
          f"{'mu_d':>8} {'sigma_d':>8}")
    print(f"  {'-' * 60}")
    for mname, vals in ref.items():
        print(f"  {mname:<28} {vals[0]:>7.3f} {vals[1]:>7.3f} "
              f"{vals[2]:>7.4f} {vals[3]:>8.4f} {vals[4]:>8.4f}")

    print(f"\n  Total elapsed: {elapsed:.1f}s  "
          f"| Rhat max={rh.max():.4f}  "
          f"| ESS min={es.min():.0f}")
    print("Done.")
