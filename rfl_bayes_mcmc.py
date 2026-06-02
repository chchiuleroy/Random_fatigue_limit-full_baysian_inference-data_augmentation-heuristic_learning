#!/usr/bin/env python3
"""
rfl_bayes_mcmc.py
Full Bayesian RFL — Uniform Priors, 4 Model Variants
(Normal | SEV) × (LogNormal | Gamma) g(Δ)

MCMC: Adaptive Random-Walk Metropolis-Hastings (pure numpy/scipy)
Parallel: Python multiprocessing, one process per chain
"""

import numpy as np
from scipy import stats, special
from multiprocessing import Pool, cpu_count
import warnings, time
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════
# 1. DATA — P&M (1999) n=75
# ═══════════════════════════════════════════════════════════════════════
_raw = {
    0.675: [102.95, 280.32, 339.83, 366.9, 485.62, 658.96, 896.33,
            1241.76, 1250.2, 1329.78, 1399.83, 1459.14, 3249.82,
            11748.1, 11748.1],
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
    max_v = max(vals)
    n_max = vals.count(max_v)
    for v in vals:
        S_list.append(s)
        Y_list.append(np.log(v))
        ev_list.append(0 if (v == max_v and n_max > 1) else 1)

S_OBS = np.array(S_list); Y_OBS = np.array(Y_list)
EVENT = np.array(ev_list, dtype=int)
N_OBS = len(S_OBS); MIN_S = S_OBS.min()

_DATA_INFO = f"Data: n={N_OBS}  failures={EVENT.sum()}  censored={N_OBS-EVENT.sum()}"


# ═══════════════════════════════════════════════════════════════════════
# 2. QUADRATURE
# ═══════════════════════════════════════════════════════════════════════
N_QUAD = 24
_GL_X, _GL_W = np.polynomial.legendre.leggauss(N_QUAD)

# Precompute quadrature nodes per obs (independent of params)
# d_pts: (N_OBS, N_QUAD)
_D_PTS = S_OBS[:, None] / 2.0 * (_GL_X[None, :] + 1.0)
_JAC   = S_OBS / 2.0          # (N_OBS,)


# ═══════════════════════════════════════════════════════════════════════
# 3. LOG-POSTERIOR
# ═══════════════════════════════════════════════════════════════════════

def log_posterior(theta, error, g_type):
    """
    Uniform prior → log p(θ|data) = log_likelihood within bounds, -inf outside.
    theta = [b0, b1, sig, p1, p2]
    """
    b0, b1, sig, p1, p2 = theta

    # ── Prior bounds check ──────────────────────────────────────────
    upper_mu_d = np.log(MIN_S) - 0.05
    if not (-50 < b0 < 50):          return -np.inf
    if not (-30 < b1 < 0):           return -np.inf
    if not (0.01 < sig < 5):         return -np.inf
    if g_type == "lognormal":
        if not (-5 < p1 < upper_mu_d):  return -np.inf
        if not (0.001 < p2 < 2):        return -np.inf
    else:  # gamma
        if not (0.1 < p1 < 50):         return -np.inf
        if not (0.1 < p2 < 200):        return -np.inf

    # ── Marginal log-likelihood via GL quadrature ──────────────────
    # mu_c: (N_OBS, N_QUAD)
    mu_c = b0 + b1 * np.log(np.maximum(S_OBS[:, None] - _D_PTS, 1e-10))

    # log f(y|Δ) or log S(y|Δ)
    if error == "normal":
        z       = (Y_OBS[:, None] - mu_c) / sig
        lf_fail = -0.5 * np.log(2*np.pi) - np.log(sig) - 0.5 * z**2
        lf_cens = special.log_ndtr(-z)
    else:  # SEV
        z       = (Y_OBS[:, None] - mu_c) / sig
        ez      = np.exp(np.clip(z, -500, 20))
        lf_fail = -np.log(sig) + z - ez
        lf_cens = -ez

    fail = (EVENT == 1)[:, None]
    lf   = np.where(fail, lf_fail, lf_cens)   # (N_OBS, N_QUAD)

    # log g(Δ)
    d = _D_PTS
    if g_type == "lognormal":
        log_d = np.log(np.maximum(d, 1e-300))
        lg = (-0.5*np.log(2*np.pi) - np.log(p2)
              - (log_d - p1)**2 / (2*p2**2) - log_d)
    else:
        lg = ((p1-1)*np.log(np.maximum(d, 1e-300))
              + p1*np.log(p2) - p2*d - special.gammaln(p1))

    # Integrate over Δ: (N_OBS,)
    integrand = np.exp(lf + lg) * _JAC[:, None]
    integral  = (_GL_W[None, :] * integrand).sum(axis=1)   # (N_OBS,)

    # Add P(Δ > S_i) for censored
    if g_type == "lognormal":
        p_above = 0.5 * special.erfc((np.log(S_OBS) - p1) / (np.sqrt(2)*p2))
    else:
        p_above = special.gammaincc(p1, p2 * S_OBS)

    integral[EVENT == 0] += p_above[EVENT == 0]
    integral = np.maximum(integral, 1e-300)

    return float(np.sum(np.log(integral)))


# ═══════════════════════════════════════════════════════════════════════
# 4. ADAPTIVE METROPOLIS-HASTINGS (single chain)
# ═══════════════════════════════════════════════════════════════════════

def _run_chain(args):
    """Single AMRWM chain — runs in subprocess for parallelism."""
    seed, error, g_type, init, n_tune, n_draws = args
    rng    = np.random.default_rng(seed)
    n_dim  = len(init)

    # Proposal scale (diagonal, tuned during warmup)
    scale  = np.abs(init) * 0.1 + 0.01

    theta  = np.array(init, dtype=float)
    lp     = log_posterior(theta, error, g_type)

    # Storage
    samples    = np.zeros((n_draws, n_dim))
    accepted   = 0

    # ── Tuning phase ────────────────────────────────────────────────
    tune_samples = []
    acc_tune     = 0
    for t in range(n_tune):
        proposal = theta + rng.normal(0, scale, n_dim)
        lp_prop  = log_posterior(proposal, error, g_type)
        if np.log(rng.uniform()) < lp_prop - lp:
            theta = proposal; lp = lp_prop; acc_tune += 1
        tune_samples.append(theta.copy())

        # Adapt every 100 steps
        if (t + 1) % 100 == 0:
            rate = acc_tune / 100
            if rate < 0.15:   scale *= 0.7
            elif rate > 0.50: scale *= 1.3
            acc_tune = 0

    # Use empirical covariance from last 50% of tuning
    tune_arr = np.array(tune_samples[n_tune//2:])
    cov_emp  = np.cov(tune_arr.T) if tune_arr.shape[0] > n_dim + 1 else np.diag(scale**2)
    cov_emp  = cov_emp * 2.38**2 / n_dim   # Gelman's optimal scaling

    # ── Sampling phase ───────────────────────────────────────────────
    try:
        L = np.linalg.cholesky(cov_emp + 1e-8 * np.eye(n_dim))
    except np.linalg.LinAlgError:
        L = np.diag(scale)

    for i in range(n_draws):
        proposal = theta + L @ rng.standard_normal(n_dim)
        lp_prop  = log_posterior(proposal, error, g_type)
        if np.log(rng.uniform()) < lp_prop - lp:
            theta = proposal; lp = lp_prop; accepted += 1
        samples[i] = theta

    acc_rate = accepted / n_draws
    return samples, acc_rate


# ═══════════════════════════════════════════════════════════════════════
# 5. STARTING VALUES
# ═══════════════════════════════════════════════════════════════════════

def _start_theta(g_type, rng):
    """Random starting point within prior bounds."""
    b0  = rng.uniform(-15, -5)
    b1  = rng.uniform(-12, -5)
    sig = rng.uniform(0.1,  0.5)
    if g_type == "lognormal":
        p1  = rng.uniform(-1.5, np.log(MIN_S) - 0.1)
        p2  = rng.uniform(0.01, 0.5)
    else:
        p1  = rng.uniform(1.0,  10.0)
        p2  = rng.uniform(10.0, 100.0)
    return np.array([b0, b1, sig, p1, p2])


# ═══════════════════════════════════════════════════════════════════════
# 6. DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════

def rhat(chains):
    """Gelman-Rubin R̂ for each parameter. chains: (n_chains, n_draws, n_dim)"""
    M, N, _ = chains.shape
    chain_mean = chains.mean(axis=1)          # (M, n_dim)
    grand_mean = chain_mean.mean(axis=0)      # (n_dim,)
    B = N * ((chain_mean - grand_mean)**2).mean(axis=0)
    W = chains.var(axis=1, ddof=1).mean(axis=0)
    var_est = (N-1)/N * W + B/N
    return np.sqrt(var_est / np.maximum(W, 1e-15))

def ess(chains):
    """Bulk ESS (simplified via autocorrelation). chains: (n_chains, n_draws, n_dim)"""
    M, N, D = chains.shape
    ess_vals = []
    for d in range(D):
        combined = chains[:, :, d].flatten()
        # Autocorrelation via FFT
        x   = combined - combined.mean()
        n   = len(x)
        fft = np.fft.rfft(x, n=2*n)
        acf_full = np.fft.irfft(fft * np.conj(fft))[:n] / (np.arange(n, 0, -1) * x.var() + 1e-15)
        # Find first negative pair
        rho = acf_full[1:]
        pairs = rho[:-1] + rho[1:]
        cutoff = next((i for i, p in enumerate(pairs) if p < 0), len(pairs))
        tau = 1 + 2 * rho[:cutoff+1].sum()
        ess_vals.append(n / max(tau, 1))
    return np.array(ess_vals)


# ═══════════════════════════════════════════════════════════════════════
# 7. VECTORISED PER-OBS LOG-LIKELIHOOD  (for LOO)
# ═══════════════════════════════════════════════════════════════════════

def compute_loglik(chains, error, g_type):
    """
    Fully vectorised per-obs log-likelihood.
    chains: (n_chains, n_draws, 5)
    returns: (n_chains, n_draws, N_OBS)
    """
    M, N, _ = chains.shape

    def e4(x): return x[:, :, np.newaxis, np.newaxis]  # (M,N,1,1)
    b0_ = e4(chains[:,:,0]); b1_  = e4(chains[:,:,1])
    sig_= e4(chains[:,:,2]); p1_  = e4(chains[:,:,3])
    p2_ = e4(chains[:,:,4])

    D4  = _D_PTS[None, None, :, :]          # (1,1,N_OBS,N_QUAD)
    S4  = S_OBS[None, None, :, None]        # (1,1,N_OBS,1)
    Y4  = Y_OBS[None, None, :, None]
    W4  = _GL_W[None, None, None, :]
    jac3= _JAC[None, None, :]

    mu_c = b0_ + b1_ * np.log(np.maximum(S4 - D4, 1e-10))

    if error == "normal":
        z       = (Y4 - mu_c) / sig_
        lf_fail = -0.5*np.log(2*np.pi) - np.log(sig_) - 0.5*z**2
        lf_cens = special.log_ndtr(-z)
    else:
        z       = (Y4 - mu_c) / sig_
        ez      = np.exp(np.clip(z, -500, 20))
        lf_fail = -np.log(sig_) + z - ez
        lf_cens = -ez

    fail4 = (EVENT == 1)[None, None, :, None]
    lf    = np.where(fail4, lf_fail, lf_cens)

    d = D4
    if g_type == "lognormal":
        log_d = np.log(np.maximum(d, 1e-300))
        lg = (-0.5*np.log(2*np.pi) - np.log(p2_)
              - (log_d - p1_)**2 / (2*p2_**2) - log_d)
    else:
        lg = ((p1_-1)*np.log(np.maximum(d,1e-300))
              + p1_*np.log(p2_) - p2_*d - special.gammaln(p1_))

    integrand = np.exp(lf + lg) * jac3[:,:,:,None]
    integral  = (W4 * integrand).sum(axis=-1)   # (M,N,N_OBS)

    S3 = S_OBS[None, None, :]; p1_3 = chains[:,:,3,None]; p2_3 = chains[:,:,4,None]
    if g_type == "lognormal":
        p_above = 0.5 * special.erfc((np.log(S3)-p1_3) / (np.sqrt(2)*p2_3))
    else:
        p_above = special.gammaincc(p1_3, p2_3*S3)

    cens3 = (EVENT == 0)[None, None, :]
    integral += np.where(cens3, p_above, 0.0)
    return np.log(np.maximum(integral, 1e-300))   # (M,N,N_OBS)


# ═══════════════════════════════════════════════════════════════════════
# 8. MAIN
# ═══════════════════════════════════════════════════════════════════════

VARIANTS = [
    ("normal", "lognormal"),
    ("normal", "gamma"),
    ("sev",    "lognormal"),
    ("sev",    "gamma"),
]

N_CHAINS = 4
N_TUNE   = 2000
N_DRAWS  = 3000
N_CORES  = min(cpu_count(), N_CHAINS)

PARAM_NAMES = {
    "lognormal": ["b0", "b1", "sigma", "mu_d",   "sigma_d"],
    "gamma":     ["b0", "b1", "sigma", "alpha_d", "beta_r"],
}

if __name__ == "__main__":   # Windows multiprocessing guard
 print(_DATA_INFO)
 all_results = {}

 for error, g_type in VARIANTS:
    name   = f"{error}_{g_type}"
    pnames = PARAM_NAMES[g_type]
    print(f"\n{'═'*65}")
    print(f"  [{name}]  chains={N_CHAINS}  tune={N_TUNE}  draws={N_DRAWS}  cores={N_CORES}")
    print(f"{'═'*65}")

    rng0  = np.random.default_rng(42)
    inits = [_start_theta(g_type, rng0) for _ in range(N_CHAINS)]
    for k, init in enumerate(inits):
        if not np.isfinite(log_posterior(init, error, g_type)):
            inits[k] = (np.array([-9.0,-8.0,0.2,np.log(MIN_S)-0.5,0.05])
                        if g_type=="lognormal"
                        else np.array([-9.0,-8.0,0.2,5.0,50.0]))

    t0   = time.time()
    args = [(42+k, error, g_type, inits[k], N_TUNE, N_DRAWS)
            for k in range(N_CHAINS)]

    with Pool(N_CORES) as pool:
        results = pool.map(_run_chain, args)

    elapsed   = time.time() - t0
    chains    = np.stack([r[0] for r in results])   # (n_chains, n_draws, 5)
    acc_rates = [r[1] for r in results]
    print(f"  Elapsed: {elapsed:.1f}s  |  Acceptance: {[f'{a:.2f}' for a in acc_rates]}")

    rh       = rhat(chains)
    es       = ess(chains)
    post_mean= chains.mean(axis=(0,1))
    post_std = chains.std(axis=(0,1))
    ci_lo    = np.percentile(chains, 2.5,  axis=(0,1))
    ci_hi    = np.percentile(chains, 97.5, axis=(0,1))

    print(f"\n  {'Param':<8} {'Mean':>9} {'Std':>8} {'2.5%':>9} {'97.5%':>9} {'Rhat':>7} {'ESS':>7}")
    print(f"  {'-'*60}")
    for i, pn in enumerate(pnames):
        print(f"  {pn:<8} {post_mean[i]:>9.4f} {post_std[i]:>8.4f} "
              f"{ci_lo[i]:>9.4f} {ci_hi[i]:>9.4f} "
              f"{rh[i]:>7.4f} {es[i]:>7.0f}")

    print(f"\n  Computing per-obs log-likelihood (vectorised)…")
    t1  = time.time()
    ll  = compute_loglik(chains, error, g_type)
    print(f"  Done in {time.time()-t1:.1f}s")

    mean_ll      = ll.mean(axis=(0,1))
    elpd_approx  = mean_ll.sum()
    in_sample_ll = ll.mean()
    print(f"\n  Mean log-lik/obs (in-sample) : {in_sample_ll:.4f}")
    print(f"  ELPD approx (mean LL sum)    : {elpd_approx:.2f}")
    print(f"  Rhat max = {rh.max():.4f}  |  ESS min = {es.min():.0f}")

    all_results[name] = dict(chains=chains, rhat=rh, ess=es,
                             post_mean=post_mean, post_std=post_std,
                             ci_lo=ci_lo, ci_hi=ci_hi,
                             elpd=elpd_approx, ll=ll, pnames=pnames)

 # ── Comparison table ────────────────────────────────────────────────
 print(f"\n\n{'═'*65}")
 print("  MODEL COMPARISON SUMMARY")
 print(f"{'═'*65}")
 print(f"  {'Model':<22} {'b0':>8} {'b1':>8} {'sigma':>7} {'ELPD':>10} {'Rhat':>7}")
 print(f"  {'-'*63}")
 for name, res in all_results.items():
     pm = res["post_mean"]
     print(f"  {name:<22} {pm[0]:>8.3f} {pm[1]:>8.3f} {pm[2]:>7.4f} "
           f"{res['elpd']:>10.2f} {res['rhat'].max():>7.4f}")
 print("\nDone.")
