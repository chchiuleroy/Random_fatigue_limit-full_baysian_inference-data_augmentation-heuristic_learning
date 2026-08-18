#!/usr/bin/env python3
"""
rfl_parallel_tempering.py
Parallel Tempering (Replica Exchange) + Self-Regulating Adaptive MCMC

wiki concept integration:
  [[concept_parallel_tempering]]           -- PT framework, swap criterion
  [[concept_self_regulating_annealing]]    -- alpha(d^2) local step scaling
  [[concept_adaptive_mcmc]]               -- AM covariance adaptation

Architecture:
  - K temperature chains: T = [1, 1.5, 2.25, 3.4, 5, 8] (geometric ladder, ratio ~1.5)
  - Each chain: 4-block Gibbs (same as rfl_self_adaptive_mcmc.py)
  - Every swap_interval steps: adjacent-chain replica exchange
  - Block D vectorized across (K, N_OBS) for numpy BLAS parallelism
  - Multiple independent PT replicas via ProcessPoolExecutor

The tempered target at temperature T_k:
  pi_{T_k}(theta) propto pi(theta)^{1/T_k}
  => log pi_{T_k} = (1/T_k) * log pi(theta)

High-T chains flatten the posterior => easy barrier crossing.
Swap criterion ensures detailed balance across temperature ladder.

2026-08-13 tempered-target consistency fix: prior to this fix, Block A
tempered (prior + likelihood) together, Block B/C's exact conjugate Gibbs
was untempered, Block D tempered only the likelihood (leaving the latent
Normal prior on log_delta untempered), and the swap step used the FULL
untempered log-posterior -- three inconsistent notions of what "tempered"
means for the same augmented state (regs, log_delta, mu_d, sd_d), so the
within-chain kernels were not jointly sampling from any single pi_{T_k},
and the swap's detailed-balance argument (which assumes they were)
didn't hold. Fixed by adopting one target definition consistently:
  pi_{T_k}(regs, log_delta, mu_d, sd_d) propto
      L(data | regs, log_delta)^{1/T_k}
      * prior_reg(regs) * p(log_delta | mu_d, sd_d) * prior(mu_d, sd_d)
i.e. ONLY the likelihood is tempered; every prior (including the latent
Normal prior linking log_delta to mu_d/sd_d) stays untempered at every
T_k. Under this definition Block B/C's untempered exact Gibbs is already
exactly correct (the likelihood doesn't depend on mu_d/sd_d directly, so
the conditional posterior for them is temperature-invariant) and Block D's
existing "temper likelihood only" acceptance was already correct too --
only Block A (now splits prior/likelihood and tempers just the likelihood
term) and the swap step needed to change. For the swap: writing the
untempered joint of every non-likelihood term as g(x), pi_{T_k}(x) =
L(x)^{1/T_k} * g(x), and the standard swap ratio
  min(1, [pi_i(x_j) pi_j(x_i)] / [pi_i(x_i) pi_j(x_j)])
has g(x_i)*g(x_j) appear identically in numerator and denominator -- it
cancels EXACTLY regardless of what g is, leaving only
  [L(x_j)/L(x_i)]^{1/T_i - 1/T_j}
so the swap now compares log_lik_sev_all() alone, not the full posterior.
See [[concept_parallel_tempering]] SS4.H for the diagnostic history and
小o's original recommendation this fix follows.
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

fail_idx = np.where(EVENT == 1)[0]
cens_idx = np.where(EVENT == 0)[0]


# ======================================================================
# LOG-POSTERIOR COMPONENTS
# ======================================================================

def log_prior_reg(b0, b1, ls):
    if not (-50 < b0 < 50):                    return -np.inf
    if not (-30 < b1 < 0):                     return -np.inf
    if not (np.log(0.01) < ls < np.log(5.0)):  return -np.inf
    return 0.0


def log_lik_sev_all(b0, b1, ls, log_delta):
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
    sigma  = np.exp(ls)
    excess = np.maximum(S_OBS - np.exp(log_delta), 1e-10)
    mu_c   = b0 + b1 * np.log(excess)
    z      = (Y_OBS - mu_c) / sigma
    ll     = np.empty(N_OBS)
    ll[fail_idx] = -ls + z[fail_idx] - np.exp(np.clip(z[fail_idx], -500, 20))
    ll[cens_idx] = -np.exp(np.clip(z[cens_idx], -500, 20))
    return ll


def _ll_vec_batch(b0s, b1s, lss, log_deltas):
    """Vectorized log-lik for K chains simultaneously.
    b0s, b1s, lss: (K,)
    log_deltas: (K, N_OBS)
    Returns: (K, N_OBS)
    """
    K = len(b0s)
    sigma  = np.exp(lss)[:, None]                          # (K, 1)
    excess = np.maximum(S_OBS[None, :] - np.exp(log_deltas), 1e-10)  # (K, N_OBS)
    mu_c   = b0s[:, None] + b1s[:, None] * np.log(excess)  # (K, N_OBS)
    z      = (Y_OBS[None, :] - mu_c) / sigma               # (K, N_OBS)
    ll     = np.empty((K, N_OBS))
    ll[:, fail_idx] = -lss[:, None] + z[:, fail_idx] - np.exp(np.clip(z[:, fail_idx], -500, 20))
    ll[:, cens_idx] = -np.exp(np.clip(z[:, cens_idx], -500, 20))
    return ll


# ======================================================================
# EXACT CONJUGATE GIBBS
# ======================================================================

def _gibbs_mu_d(log_delta, sigma_d, rng):
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
# SELF-REGULATING alpha
# ======================================================================

def self_reg_alpha(x, mean, cov_inv, nu, dim):
    diff = np.asarray(x, float) - np.asarray(mean, float)
    if dim == 1:
        d2 = float(diff.ravel()[0]**2) * float(np.asarray(cov_inv).ravel()[0])
    else:
        d2 = float(diff @ np.asarray(cov_inv) @ diff)
    return float(np.clip(np.sqrt((nu + d2) / (nu + dim)), 0.1, 3.0))


# ======================================================================
# WELFORD TRACKERS
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
    """Gelman-Rubin R-hat (classic, not rank-normalized). NOTE: for an
    apples-to-apples comparison with the rest of this project's convergence
    claims, use arviz.rhat() instead -- it applies rank-normalization/folding
    (Vehtari et al. 2021) and is what Section 5.1's Rhat=1.032 is computed
    with. This function historically divided B by M instead of M-1
    (underestimating between-chain variance); fixed 2026-08-11, but the
    rank-normalized arviz value is still the one that should be reported."""
    M, N, D = chains.shape
    cm = chains.mean(axis=1); gm = cm.mean(axis=0)
    B  = N / (M - 1) * ((cm - gm)**2).sum(axis=0)
    W  = chains.var(axis=1, ddof=1).mean(axis=0)
    v  = (N - 1) / N * W + B / N
    return np.sqrt(v / np.maximum(W, 1e-15))


def ess_bulk(chains):
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
# SINGLE PT REPLICA (K temperature chains with swaps)
# ======================================================================

def _run_pt_replica(seed, n_warmup, n_draws, nu, temps, swap_interval, verbose):
    """
    Run one Parallel Tempering replica: K chains at different temperatures,
    with periodic adjacent-chain swaps.

    Returns draws from the T=1 (cold) chain only.
    """
    rng = np.random.default_rng(seed)
    K   = len(temps)
    inv_temps = 1.0 / np.array(temps)  # (K,)
    D_R = 3

    # -- Initialize K chains ------------------------------------------------
    regs       = np.zeros((K, D_R))
    log_deltas = np.zeros((K, N_OBS))
    mu_ds      = np.zeros(K)
    sd_ds      = np.zeros(K)

    wc_regs  = [WelfordCov(D_R) for _ in range(K)]
    wvv_dls  = [WelfordVarVec(N_OBS) for _ in range(K)]

    for k in range(K):
        regs[k] = np.array([
            -9.37 + rng.normal(0, 0.025),
            -8.53 + rng.normal(0, 0.020),
            np.log(0.19) + rng.normal(0, 0.055)
        ])
        mu_ds[k] = -0.644 + rng.normal(0, 0.010)
        sd_ds[k] = 0.036 * np.exp(rng.normal(0, 0.05))
        log_deltas[k] = np.full(N_OBS, mu_ds[k]) + rng.normal(0, sd_ds[k], N_OBS)
        log_deltas[k] = np.clip(log_deltas[k], -5.0, LOG_S - 0.001)

    # -- Pilot run for all chains (200 steps of Block B/C/D) ----------------
    _pilot_std = np.zeros((K, N_OBS))  # fallback for step_dl_frozen_arr if n_warmup=0
    for k in range(K):
        _wvv_p = WelfordVarVec(N_OBS)
        for _ in range(200):
            mu_ds[k] = _gibbs_mu_d(log_deltas[k], sd_ds[k], rng)
            sd_ds[k] = _gibbs_sigma_d(log_deltas[k], mu_ds[k], rng)
            _wvv_p.update(log_deltas[k])
            _step = np.maximum(2.38 * _wvv_p.std, 0.04)
            _ldp  = log_deltas[k] + rng.normal(0.0, _step)
            _vld  = _ldp < LOG_S
            b0, b1, ls = regs[k]
            _lc = _ll_vec(b0, b1, ls, log_deltas[k])
            _lp = np.where(_vld, _ll_vec(b0, b1, ls, np.where(_vld, _ldp, log_deltas[k])), -np.inf)
            _pc = stats.norm.logpdf(log_deltas[k], mu_ds[k], sd_ds[k])
            _pp = np.where(_vld, stats.norm.logpdf(_ldp, mu_ds[k], sd_ds[k]), -np.inf)
            _lr = (_pp + _lp) - (_pc + _lc)
            _acc = _vld & (np.log(rng.uniform(size=N_OBS)) < _lr)
            log_deltas[k] = np.where(_acc, _ldp, log_deltas[k])
        _pilot_std[k] = _wvv_p.std

    # -- Warm-start Welford for each chain ----------------------------------
    sd0c, sd1c, sdlsc = 0.083, 0.062, 0.080
    C_hand = np.array([
        [sd0c**2,              0.96 * sd0c * sd1c, 0.0],
        [0.96 * sd0c * sd1c,   sd1c**2,            0.0],
        [0.0,                  0.0,                 sdlsc**2],
    ])
    for k in range(K):
        # Higher-T chains get inflated covariance (wider proposal)
        C_init = C_hand * temps[k]
        wc_regs[k].n    = 100
        wc_regs[k].mean = regs[k].copy()
        wc_regs[k].M2   = 99 * C_init

    # -- Storage & counters -------------------------------------------------
    n_total    = n_warmup + n_draws
    th_out     = np.zeros((n_draws, 5))     # cold chain draws
    acc_reg    = np.zeros(K, dtype=int)
    acc_dl     = np.zeros((K, N_OBS), dtype=int)
    n_swaps_attempted = 0
    n_swaps_accepted  = 0
    # Per-pair swap counts (pair i = rungs i,i+1), draws phase only --
    # localizes WHERE along the ladder communication is weak, same
    # rationale as hawkes_pt.py's 2026-08-13 diagnostic (an aggregate rate
    # can hide one dead pair).
    pair_attempted_draws = np.zeros(K - 1, dtype=int)
    pair_accepted_draws  = np.zeros(K - 1, dtype=int)
    h_dls      = np.full(K, 2.38)
    acc_dl_recent = np.zeros((K, N_OBS), dtype=int)

    # Self-Regulating alpha: frozen snapshots for the draws phase, one per
    # chain (2026-08-12 fix -- see rfl_self_adaptive_mcmc.py's module-level
    # comment on why a per-iteration Hastings correction was abandoned in
    # favor of freezing). Populated at it == n_warmup.
    C_reg_frozen     = [None] * K
    alpha_reg_frozen = np.ones(K)
    # (K, N_OBS), overwritten every warmup iteration once warmup starts.
    # n_warmup=0 boundary fix: previously left as None here, and Block D's
    # `if warmup:` branch below never runs when n_warmup=0 -- step_all was
    # then None and crashed the very first rng.normal(0.0, step_all) call.
    # Fall back to the per-chain pilot-run std (200 real adaptation
    # iterations, _pilot_std above) so n_warmup=0 degrades to "use the
    # pilot-run step size for the whole draws phase" instead of crashing.
    step_dl_frozen_arr = np.maximum(h_dls[:, None] * _pilot_std, 0.04)

    t0 = time.time()

    for it in range(n_total):
        warmup = it < n_warmup

        # ==============================================================
        # BLOCKS A/B/C: per-chain (low-dim, hard to vectorize across K)
        # ==============================================================
        for k in range(K):
            b0, b1, ls = regs[k]
            beta_k = inv_temps[k]  # 1/T_k

            # --- Block A: Self-Reg AM at temperature T_k ----------------
            # Frozen at the draws-phase boundary (2026-08-12 fix): see the
            # note above C_reg_frozen's initialization.
            if warmup:
                wc_regs[k].update(regs[k])
                C_use = wc_regs[k].cov
                alpha_r = 1.0
            else:
                if it == n_warmup:
                    C_reg_frozen[k] = wc_regs[k].cov.copy()
                    try:
                        C_inv0 = np.linalg.inv(C_reg_frozen[k] + 1e-8 * np.eye(D_R))
                        alpha_reg_frozen[k] = self_reg_alpha(
                            regs[k], wc_regs[k].mean, C_inv0, nu, D_R)
                    except np.linalg.LinAlgError:
                        alpha_reg_frozen[k] = 1.0
                C_use = C_reg_frozen[k]
                alpha_r = alpha_reg_frozen[k]

            C_prop = (2.38**2 / D_R) * alpha_r**2 * C_use
            try:
                L_r = np.linalg.cholesky(C_prop + 1e-8 * np.eye(D_R))
            except np.linalg.LinAlgError:
                L_r = np.diag(np.sqrt(np.diag(C_use)) * alpha_r * np.sqrt(2.38**2 / D_R))

            reg_p = regs[k] + L_r @ rng.standard_normal(D_R)
            # 2026-08-13 fix: temper ONLY the likelihood, not the prior --
            # matches the "only temper likelihood" target used by Block D
            # below and required for the swap step's prior-cancellation
            # argument to hold (see swap comment). Previously both terms
            # were summed THEN multiplied by beta_k together; numerically
            # identical for this flat 0/-inf prior (temper doesn't change
            # which side of 0/-inf a term is on) but wrong in form, and
            # would silently break the moment log_prior_reg becomes
            # non-flat. See [[concept_parallel_tempering]] SS4.H.
            prior_curr = log_prior_reg(*regs[k])
            prior_prop = log_prior_reg(*reg_p)
            ll_curr = log_lik_sev_all(*regs[k], log_deltas[k])
            ll_prop = log_lik_sev_all(*reg_p,   log_deltas[k])

            if np.isfinite(prior_prop) and np.isfinite(ll_prop):
                log_alpha = (prior_prop - prior_curr) + beta_k * (ll_prop - ll_curr)
                if np.log(rng.uniform()) < log_alpha:
                    regs[k] = reg_p
                    acc_reg[k] += 1

            b0, b1, ls = regs[k]

            # --- Block B/C: exact Gibbs (not tempered -- conjugacy) ------
            mu_ds[k] = _gibbs_mu_d(log_deltas[k], sd_ds[k], rng)
            sd_ds[k] = _gibbs_sigma_d(log_deltas[k], mu_ds[k], rng)

        # ==============================================================
        # BLOCK D: vectorized across K chains (K × N_OBS)
        # Self-Regulating step_all frozen at the draws-phase boundary
        # (2026-08-12 fix, same rationale as Block A above): it depends on
        # log_deltas, which keeps moving every iteration even during draws.
        # ==============================================================
        b0s = regs[:, 0]
        b1s = regs[:, 1]
        lss = regs[:, 2]

        if warmup:
            for k in range(K):
                wvv_dls[k].update(log_deltas[k])
            stds_all = np.stack([wvv_dls[k].std for k in range(K)])       # (K, N_OBS)
            means_all = np.stack([wvv_dls[k].mean for k in range(K)])     # (K, N_OBS)
            vars_inv_all = np.stack([1.0 / wvv_dls[k].var for k in range(K)])  # (K, N_OBS)

            d2_all    = (log_deltas - means_all)**2 * vars_inv_all        # (K, N_OBS)
            alpha_all = np.clip(np.sqrt((nu + d2_all) / (nu + 1.0)), 0.1, 4.0)
            step_dl_frozen_arr = np.maximum(h_dls[:, None] * stds_all * alpha_all, 0.04)

        step_all = step_dl_frozen_arr
        ld_prop_all = log_deltas + rng.normal(0.0, step_all)          # (K, N_OBS)
        valid_all   = ld_prop_all < LOG_S[None, :]                    # (K, N_OBS)

        # Batch log-likelihood
        ll_curr_all = _ll_vec_batch(b0s, b1s, lss, log_deltas)       # (K, N_OBS)
        ld_safe     = np.where(valid_all, ld_prop_all, log_deltas)
        ll_prop_all = _ll_vec_batch(b0s, b1s, lss, ld_safe)          # (K, N_OBS)
        ll_prop_all = np.where(valid_all, ll_prop_all, -np.inf)

        # Prior
        for k in range(K):
            lp_prior_curr = stats.norm.logpdf(log_deltas[k], mu_ds[k], sd_ds[k])
            lp_prior_prop = np.where(
                valid_all[k],
                stats.norm.logpdf(ld_prop_all[k], mu_ds[k], sd_ds[k]),
                -np.inf
            )
            # Tempered acceptance for Block D
            beta_k = inv_temps[k]
            log_r = beta_k * (ll_prop_all[k] - ll_curr_all[k]) + (lp_prior_prop - lp_prior_curr)
            accept = valid_all[k] & (np.log(rng.uniform(size=N_OBS)) < log_r)
            log_deltas[k] = np.where(accept, ld_prop_all[k], log_deltas[k])
            acc_dl[k] += accept.astype(int)
            acc_dl_recent[k] += accept.astype(int)

        # -- Adapt h_dl during warmup --------------------------------------
        if warmup and (it + 1) % 100 == 0 and it > 0:
            for k in range(K):
                avg_recent = acc_dl_recent[k].sum() / (100 * N_OBS)
                if   avg_recent < 0.25: h_dls[k] *= 0.85
                elif avg_recent > 0.55: h_dls[k] *= 1.2
                h_dls[k] = float(np.clip(h_dls[k], 0.1, 20.0))
            acc_dl_recent[:] = 0

        # ==============================================================
        # REPLICA EXCHANGE (adjacent swaps)
        # ==============================================================
        if (it + 1) % swap_interval == 0:
            # Random sweep direction to maintain detailed balance
            pairs = list(range(K - 1))
            if rng.uniform() < 0.5:
                pairs = pairs[::-1]

            for i in pairs:
                j = i + 1
                n_swaps_attempted += 1

                # 2026-08-13 fix: swap ratio uses ONLY the likelihood
                # difference, not the full log-posterior. Under the "temper
                # only the likelihood" target now used consistently by
                # Blocks A/D (pi_k(x) = L(data|x)^(1/T_k) * g(x), where g =
                # regression prior * latent Normal prior * mu_d/sd_d prior
                # is the SAME untempered function of state x at every T_k --
                # Block B/C's untempered exact Gibbs makes this hold for
                # free), the standard swap acceptance
                #   min(1, [pi_i(x_j) pi_j(x_i)] / [pi_i(x_i) pi_j(x_j)])
                # has g(x_i)*g(x_j) appear identically in numerator and
                # denominator -- it cancels EXACTLY (not approximately),
                # leaving only [L(x_j)/L(x_i)]^(1/Ti - 1/Tj). Including g in
                # the ratio (the previous "full untempered posterior" code)
                # was therefore not just untempered-when-it-should-be but
                # actively wrong: g doesn't cancel unless it's ABSENT from
                # both sides, and the previous code left the regression
                # prior in numerically (usually inert, it's flat) but also
                # included the latent Normal prior sum, which is NOT flat
                # and does not cancel when included in a min(1, ratio) that
                # only accounts for the temperature difference on the
                # combined term. See [[concept_parallel_tempering]] SS4.H.
                ll_i = log_lik_sev_all(*regs[i], log_deltas[i])
                ll_j = log_lik_sev_all(*regs[j], log_deltas[j])

                if not warmup:
                    pair_attempted_draws[i] += 1

                if np.isfinite(ll_i) and np.isfinite(ll_j):
                    # Swap criterion: min(1, exp((beta_i - beta_j)(ll_j - ll_i)))
                    log_swap = (inv_temps[i] - inv_temps[j]) * (ll_j - ll_i)
                    if np.log(rng.uniform()) < log_swap:
                        # Swap ALL state between chains i and j
                        regs[i], regs[j] = regs[j].copy(), regs[i].copy()
                        log_deltas[i], log_deltas[j] = log_deltas[j].copy(), log_deltas[i].copy()
                        mu_ds[i], mu_ds[j] = mu_ds[j], mu_ds[i]
                        sd_ds[i], sd_ds[j] = sd_ds[j], sd_ds[i]
                        # Partial reset: compress history to 50 pseudo-samples so
                        # covariance shape is preserved but tracker adapts faster post-swap.
                        _TRUST_N = 50
                        for _k in (i, j):
                            if wc_regs[_k].n > _TRUST_N:
                                wc_regs[_k].M2 = wc_regs[_k].M2 * (_TRUST_N - 1) / (wc_regs[_k].n - 1)
                                wc_regs[_k].n  = _TRUST_N
                            wc_regs[_k].mean = regs[_k].copy()
                            if wvv_dls[_k].count > _TRUST_N:
                                wvv_dls[_k].M2    = wvv_dls[_k].M2 * (_TRUST_N - 1) / (wvv_dls[_k].count - 1)
                                wvv_dls[_k].count = _TRUST_N
                            wvv_dls[_k].mean = log_deltas[_k].copy()
                        n_swaps_accepted += 1
                        if not warmup:
                            pair_accepted_draws[i] += 1

        # -- Store cold chain draws -----------------------------------------
        if not warmup:
            draw_idx = it - n_warmup
            b0, b1, ls = regs[0]
            th_out[draw_idx] = [b0, b1, ls, mu_ds[0], np.log(sd_ds[0])]

        if verbose and (it + 1) % 1000 == 0:
            phase = "warmup" if warmup else "sample"
            swap_rate = n_swaps_accepted / max(n_swaps_attempted, 1)
            b0, b1, ls = regs[0]
            print(f"  [{phase}] {it+1}/{n_total}  "
                  f"cold-acc={acc_reg[0]/(it+1):.2f}  "
                  f"swap-rate={swap_rate:.2f} ({n_swaps_accepted}/{n_swaps_attempted})  "
                  f"b0={b0:.3f} b1={b1:.3f} sigma={np.exp(ls):.4f}  "
                  f"mu_d={mu_ds[0]:.4f} sd_d={sd_ds[0]:.4f}")

    elapsed = time.time() - t0
    swap_rate = n_swaps_accepted / max(n_swaps_attempted, 1)
    pair_swap_rates_draws = pair_accepted_draws / np.maximum(pair_attempted_draws, 1)
    return th_out, acc_reg, acc_dl, swap_rate, elapsed, pair_swap_rates_draws


# ======================================================================
# MAIN
# ======================================================================
if __name__ == "__main__":
    N_REPLICAS    = 4       # independent PT replicas (for Rhat)
    N_WARMUP      = 4000
    N_DRAWS       = 6000
    NU            = 5.0
    SWAP_INTERVAL = 30

    # Graded ladder (2026-08-13): the old constant-ratio-1.5 ladder above
    # went dead at the cold end once the swap step was fixed to compare
    # ONLY the likelihood (see module docstring) -- per-pair diagnostics
    # measured 0.3% acceptance at T=1.0<->1.5 with the 6-rung ladder, and
    # re-running at that ladder's official scale actually made arviz
    # Rhat_max WORSE than the pre-fix (buggy-tempering) baseline: 1.65 ->
    # 1.83. Isolated 2-rung calibration found the SEV likelihood surface
    # near T=1 is sharp enough at N_OBS=75 that only ratios below ~1.05
    # give non-trivial (and noisy, this posterior has its own known
    # (b0,b1)<->Delta_i identifiability ridge) acceptance. Fix: dense ratio
    # near T=1 (1.04 for 24 rungs, reaching T~2.56) then widening (1.45 for
    # 3 more rungs, matching the original ladder's spacing beyond ~T=2.25
    # where it was already healthy) -- 28 rungs total. Re-verified at this
    # exact warmup/draws budget: arviz Rhat_max 1.83 -> **1.54** (below
    # even the original pre-fix 1.65), ESS_bulk_min 5 -> 7 (best-tuned
    # 8/16-rung intermediate attempts also tried; not all logged here).
    # STILL above this project's 1.1 threshold -- do not cite as converged.
    # See [[concept_parallel_tempering]] SS4.B/H for the full trail.
    TEMPS = [1.0]
    for _r in [1.04] * 24 + [1.45] * 3:
        TEMPS.append(TEMPS[-1] * _r)

    PNAMES = ["b0", "b1", "log_sigma", "mu_d", "log_sigma_d"]
    MLE = np.array([-9.370, -8.534, np.log(0.190), -0.644, np.log(0.036)])

    n_workers = min(N_REPLICAS, os.cpu_count() or 4)
    print(f"Parallel Tempering + Self-Regulating AM -- RFL (SEV + LogNormal)")
    print(f"nu={NU}  temps={TEMPS}  swap_interval={SWAP_INTERVAL}")
    print(f"replicas={N_REPLICAS}  warmup={N_WARMUP}  draws={N_DRAWS}")
    print(f"Parallel: {n_workers} workers on {os.cpu_count()} cores")
    print(f"n={N_OBS}  failures={len(fail_idx)}  censored={len(cens_idx)}")
    print("=" * 70)

    seeds = [42, 137, 271, 999, 1234, 5678, 9012, 3456]

    t_total = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(
                _run_pt_replica, seeds[r], N_WARMUP, N_DRAWS, NU,
                TEMPS, SWAP_INTERVAL, False
            ): r
            for r in range(N_REPLICAS)
        }
        results = [None] * N_REPLICAS
        pair_rates_all = [None] * N_REPLICAS
        for future in as_completed(futures):
            r = futures[future]
            th, acc_r, acc_dl, swap_rate, elapsed, pair_swap_rates = future.result()
            results[r] = th
            pair_rates_all[r] = pair_swap_rates
            print(f"  Replica {r+1} done ({elapsed:.1f}s): "
                  f"cold-reg-acc={acc_r[0]/(N_WARMUP+N_DRAWS):.2f}  "
                  f"swap-rate={swap_rate:.2f}")

    total_elapsed = time.time() - t_total

    pair_rates_mean = np.mean(np.stack(pair_rates_all), axis=0)
    print(f"\n  Per-pair swap acceptance (draws phase, avg over {N_REPLICAS} replicas):")
    for i in range(len(TEMPS) - 1):
        print(f"    T{i}={TEMPS[i]:.2f} <-> T{i+1}={TEMPS[i+1]:.2f}: "
              f"{pair_rates_mean[i]:.3f}")

    chains_arr = np.stack(results)   # (N_REPLICAS, N_DRAWS, 5)
    all_draws  = chains_arr.reshape(-1, 5)

    rh = rhat(chains_arr)
    es = ess_bulk(chains_arr)
    pm = all_draws.mean(axis=0)
    ps = all_draws.std(axis=0)
    lo = np.percentile(all_draws, 2.5,  axis=0)
    hi = np.percentile(all_draws, 97.5, axis=0)

    print(f"\n{'=' * 70}")
    print(f"POSTERIOR SUMMARY  (PT + Self-Reg AM, nu={NU}, temps={TEMPS})")
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

    print(f"\n  --- COMPARISON ---")
    ref = {
        "MLE (SEV+INLA)":       [-9.370, -8.534, 0.190, -0.644, 0.036],
        "PyMC NUTS (post mean)": [-9.29,  -8.75,  0.195, -0.660, 0.038],
        f"PT+Self-AM nu={NU:.0f}": [pm[0], pm[1], sig_draws.mean(),
                                     pm[3], sigd_draws.mean()],
    }
    print(f"  {'Method':<28} {'b0':>7} {'b1':>7} {'sigma':>7} "
          f"{'mu_d':>8} {'sigma_d':>8}")
    print(f"  {'-' * 60}")
    for mname, vals in ref.items():
        print(f"  {mname:<28} {vals[0]:>7.3f} {vals[1]:>7.3f} "
              f"{vals[2]:>7.4f} {vals[3]:>8.4f} {vals[4]:>8.4f}")

    print(f"\n  Total elapsed: {total_elapsed:.1f}s (parallel)  "
          f"| Rhat max={rh.max():.4f}  "
          f"| ESS min={es.min():.0f}")
    print("Done.")
