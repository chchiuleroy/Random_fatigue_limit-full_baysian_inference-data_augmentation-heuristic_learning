#!/usr/bin/env python3
"""
rfl_bayes_uniform.py
Full Bayesian RFL — Uniform Priors, 4 Model Variants
(Normal | SEV) × (LogNormal | Gamma)  g(Δ)

Prior philosophy:
  - Uniform with physical bounds = 真正從無知出發，不假設中心
  - β₁ < 0 是唯一硬約束（超額應力大→壽命短）
  - σ, σ_Δ 下限避免數值退化

Parallelism:
  pm.sample(cores=N)          → N 條 chains 平行跑
  add_log_likelihood_fast()   → 完全向量化，無 Python for loop
"""

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
from scipy import stats, special
import arviz as az
import multiprocessing
import warnings
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════
# 1. DATA — Pascual & Meeker (1999) n=75
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
    max_v  = max(vals)
    n_max  = vals.count(max_v)
    for v in vals:
        S_list.append(s)
        Y_list.append(np.log(v))          # ln(kilocycles)
        # duplicated max → Type I runout (censored at that time)
        ev_list.append(0 if (v == max_v and n_max > 1) else 1)

S_OBS  = np.array(S_list,  dtype=float)
Y_OBS  = np.array(Y_list,  dtype=float)
EVENT  = np.array(ev_list, dtype=int)    # 1=failure, 0=censored
N_OBS  = len(S_OBS)
MIN_S  = S_OBS.min()                     # 0.675

print(f"Data: n={N_OBS}  failures={EVENT.sum()}  censored={N_OBS-EVENT.sum()}")


# ═══════════════════════════════════════════════════════════════════════
# 2. GAUSS-LEGENDRE QUADRATURE  (integrating out Δ)
# ═══════════════════════════════════════════════════════════════════════
N_QUAD        = 24
_GL_X, _GL_W  = np.polynomial.legendre.leggauss(N_QUAD)


# ═══════════════════════════════════════════════════════════════════════
# 3. DISTRIBUTION HELPERS
# ═══════════════════════════════════════════════════════════════════════

def _norm_lf(y, mu, sig, cens):
    return stats.norm.logsf(y, mu, sig) if cens else stats.norm.logpdf(y, mu, sig)

def _sev_lf(y, mu, sig, cens):
    z = (y - mu) / sig
    if cens:
        return -np.exp(z)                    # log S(y)
    return -np.log(sig) + z - np.exp(z)     # log f(y)

def _lnorm_g(d, mu_d, sd_d):
    return stats.lognorm.logpdf(d, s=sd_d, scale=np.exp(mu_d))

def _lnorm_pabove(s, mu_d, sd_d):
    return stats.lognorm.sf(s, s=sd_d, scale=np.exp(mu_d))

def _gamma_g(d, alpha, beta_r):           # beta_r = rate (1/scale)
    return stats.gamma.logpdf(d, a=alpha, scale=1.0/beta_r)

def _gamma_pabove(s, alpha, beta_r):
    return stats.gamma.sf(s, a=alpha, scale=1.0/beta_r)


# ═══════════════════════════════════════════════════════════════════════
# 4. MARGINAL LOG-LIKELIHOOD  ∫ f(y|Δ) g(Δ) dΔ
# ═══════════════════════════════════════════════════════════════════════

def marginal_loglik(b0, b1, sig, p1, p2, error, g_type):
    """
    Marginal log-likelihood over all 75 observations.
    Gauss-Legendre quadrature on (0, S_i) for the Δ integral.

    Failure   i : ∫₀^{S_i} f(y_i|Δ) g(Δ) dΔ
    Censored  i : P(Δ>S_i) + ∫₀^{S_i} P(Y>y_i|Δ) g(Δ) dΔ
    """
    lf_fn     = _sev_lf    if error  == 'sev'      else _norm_lf
    g_fn      = _gamma_g   if g_type == 'gamma'    else _lnorm_g
    pabove_fn = _gamma_pabove if g_type == 'gamma' else _lnorm_pabove

    total = 0.0
    for i in range(N_OBS):
        s_i, y_i, ev_i = S_OBS[i], Y_OBS[i], EVENT[i]

        # GL nodes on (0, s_i)
        d_pts = s_i / 2.0 * (_GL_X + 1.0)    # (N_QUAD,)
        jac   = s_i / 2.0
        mu_c  = b0 + b1 * np.log(s_i - d_pts)  # conditional mean of ln N

        cens   = (ev_i == 0)
        lf     = lf_fn(y_i, mu_c, sig, cens)   # (N_QUAD,)
        lg     = g_fn(d_pts, p1, p2)            # (N_QUAD,)
        integ  = float(np.dot(_GL_W, np.exp(lf + lg) * jac))

        if cens:
            integ += float(pabove_fn(s_i, p1, p2))

        total += np.log(max(integ, 1e-300))

    return np.float64(total)


# ═══════════════════════════════════════════════════════════════════════
# 5. PYTENSOR Op  (blackbox; uses Metropolis, no gradient needed)
# ═══════════════════════════════════════════════════════════════════════

class _RFLLogLikOp(pytensor.graph.Op):
    """Wraps marginal_loglik as a PyTensor Op."""
    __props__ = ("error", "g_type")

    def __init__(self, error, g_type):
        self.error  = error
        self.g_type = g_type

    def make_node(self, b0, b1, sig, p1, p2):
        ins  = [pt.as_tensor_variable(x).astype("float64")
                for x in (b0, b1, sig, p1, p2)]
        outs = [pt.dscalar()]
        return pytensor.graph.basic.Apply(self, ins, outs)

    def perform(self, node, inputs, output_storage):
        ll = marginal_loglik(*[float(x) for x in inputs],
                             error=self.error, g_type=self.g_type)
        output_storage[0][0] = np.float64(ll)


# ═══════════════════════════════════════════════════════════════════════
# 6. MODEL FACTORY
# ═══════════════════════════════════════════════════════════════════════

def build_model(error="normal", g_type="lognormal"):
    """
    PyMC model with Uniform priors.

    Priors (物理約束決定上下限，不假設任何中心值):
      β₀  ~ Uniform(-50, 50)       截距，尺度由資料決定
      β₁  ~ Uniform(-30, 0)        必須為負 (Basquin 物理約束)
      σ   ~ Uniform(0.01, 5)       條件散布，正值
      ── LogNormal g(Δ) ──
      μ_Δ ~ Uniform(-5, ln(S_min)) Δ 中位數必須小於最小應力
      σ_Δ ~ Uniform(0.001, 2)      允許退化到高度異質
      ── Gamma g(Δ) ──
      α   ~ Uniform(0.1, 50)       shape 參數
      β_r ~ Uniform(0.1, 200)      rate 參數 (mean = α/β_r)
    """
    ll_op = _RFLLogLikOp(error=error, g_type=g_type)
    upper_mu_d = float(np.log(MIN_S) - 0.05)   # ln(Δ) 必須 < ln(S_min)

    with pm.Model() as model:
        beta0 = pm.Uniform("beta0", lower=-50.0, upper=50.0)
        beta1 = pm.Uniform("beta1", lower=-30.0, upper=0.0)
        sigma = pm.Uniform("sigma", lower=0.01,  upper=5.0)

        if g_type == "lognormal":
            p1 = pm.Uniform("mu_d",   lower=-5.0,  upper=upper_mu_d)
            p2 = pm.Uniform("sigma_d",lower=0.001, upper=2.0)
        else:
            p1 = pm.Uniform("alpha_d",lower=0.1,   upper=50.0)
            p2 = pm.Uniform("beta_r", lower=0.1,   upper=200.0)

        pm.Potential("loglik", ll_op(beta0, beta1, sigma, p1, p2))

    return model


# ═══════════════════════════════════════════════════════════════════════
# 7. PER-OBS LOG-LIKELIHOOD  (for LOO comparison)
# ═══════════════════════════════════════════════════════════════════════

def add_log_likelihood(idata, error, g_type):
    """
    Attach per-observation log-likelihood — fully vectorised (no Python loops).

    Broadcast shapes:
      params   : (n_c, n_d,    1,      1   )   ← posterior draws
      S/Y/EVENT: (  1,   1,  N_OBS,    1   )   ← data
      d_pts    : (  1,   1,  N_OBS,  N_QUAD)   ← quadrature nodes
      GL_W     : (  1,   1,    1,   N_QUAD)

    Peak memory ≈ 4 × (n_c × n_d × N_OBS × N_QUAD) × 8 B
                = 4 × (4 × 3000 × 75 × 24) × 8 B ≈ 690 MB
    """
    import xarray as xr

    post    = idata.posterior
    p1_name = "alpha_d" if g_type == "gamma" else "mu_d"
    p2_name = "beta_r"  if g_type == "gamma" else "sigma_d"

    # ── Posterior arrays: (n_c, n_d) ──────────────────────────────────
    b0  = post["beta0"].values   # (n_c, n_d)
    b1  = post["beta1"].values
    sig = post["sigma"].values
    p1  = post[p1_name].values
    p2  = post[p2_name].values
    n_chains, n_draws = b0.shape

    # ── Expand to (n_c, n_d, 1, 1) ────────────────────────────────────
    def e(x): return x[:, :, np.newaxis, np.newaxis]
    b0_, b1_, sig_, p1_, p2_ = e(b0), e(b1), e(sig), e(p1), e(p2)

    # ── Quadrature nodes (independent of params) ──────────────────────
    # d_pts: (N_OBS, N_QUAD)
    d_pts = S_OBS[:, None] / 2.0 * (_GL_X[None, :] + 1.0)
    jac   = S_OBS / 2.0           # (N_OBS,)

    # Broadcast-ready views
    d_4d  = d_pts[None, None, :, :]          # (1, 1, N_OBS, N_QUAD)
    S_4d  = S_OBS[None, None, :, None]       # (1, 1, N_OBS,    1  )
    Y_4d  = Y_OBS[None, None, :, None]       # (1, 1, N_OBS,    1  )
    W_4d  = _GL_W[None, None, None, :]       # (1, 1,   1,   N_QUAD)
    jac_3d = jac[None, None, :]              # (1, 1, N_OBS)

    # ── Conditional mean: (n_c, n_d, N_OBS, N_QUAD) ───────────────────
    mu_c = b0_ + b1_ * np.log(np.maximum(S_4d - d_4d, 1e-10))

    # ── log f(y|Δ) or log S(y|Δ) ──────────────────────────────────────
    if error == "normal":
        z       = (Y_4d - mu_c) / sig_                   # (n_c, n_d, N_OBS, N_QUAD)
        lf_fail = -0.5 * np.log(2 * np.pi) - np.log(sig_) - 0.5 * z ** 2
        lf_cens = special.log_ndtr(-z)                    # log Φ(−z) = log(1−Φ(z))
    else:                                                  # SEV
        z       = (Y_4d - mu_c) / sig_
        ez      = np.exp(np.clip(z, -500, 500))
        lf_fail = -np.log(sig_) + z - ez
        lf_cens = -ez

    # Select by event indicator (broadcast over chains/draws/quad)
    fail_4d = (EVENT == 1)[None, None, :, None]           # (1, 1, N_OBS, 1)
    lf      = np.where(fail_4d, lf_fail, lf_cens)         # (n_c, n_d, N_OBS, N_QUAD)

    # ── log g(Δ) ──────────────────────────────────────────────────────
    if g_type == "lognormal":
        log_d = np.log(np.maximum(d_4d, 1e-300))
        lg    = (-0.5 * np.log(2 * np.pi) - np.log(p2_)
                 - (log_d - p1_) ** 2 / (2.0 * p2_ ** 2) - log_d)
    else:                                                  # gamma  (rate param)
        lg    = ((p1_ - 1.0) * np.log(np.maximum(d_4d, 1e-300))
                 + p1_ * np.log(p2_) - p2_ * d_4d
                 - special.gammaln(p1_))

    # ── Gauss-Legendre integral: (n_c, n_d, N_OBS) ────────────────────
    integrand = np.exp(lf + lg) * jac_3d[:, :, :, np.newaxis]
    integral  = np.sum(W_4d * integrand, axis=-1)          # (n_c, n_d, N_OBS)

    # ── Add P(Δ > S_i) for right-censored observations ────────────────
    # p_above: (n_c, n_d, N_OBS)
    S_3d  = S_OBS[None, None, :]
    p1_3d = p1[:, :, None]
    p2_3d = p2[:, :, None]

    if g_type == "lognormal":
        # P(Δ>s) = Φ(-(log s - μ_Δ)/σ_Δ)
        z_s     = (np.log(S_3d) - p1_3d) / p2_3d
        p_above = 0.5 * special.erfc(z_s / np.sqrt(2))
    else:
        # P(Δ>s; α, rate) = Γ(α, rate·s)/Γ(α) = gammaincc(α, rate·s)
        p_above = special.gammaincc(p1_3d, p2_3d * S_3d)

    cens_3d  = (EVENT == 0)[None, None, :]                 # (1, 1, N_OBS)
    integral += np.where(cens_3d, p_above, 0.0)

    # ── Final log-likelihood ───────────────────────────────────────────
    log_lik = np.log(np.maximum(integral, 1e-300))         # (n_c, n_d, N_OBS)

    idata.add_groups(
        log_likelihood=xr.Dataset(
            {"obs": xr.DataArray(log_lik, dims=["chain", "draw", "obs_id"])}
        )
    )
    return idata


# ═══════════════════════════════════════════════════════════════════════
# 8. MAIN SAMPLING LOOP
# ═══════════════════════════════════════════════════════════════════════

VARIANTS  = [
    ("normal", "lognormal"),
    ("normal", "gamma"),
    ("sev",    "lognormal"),
    ("sev",    "gamma"),
]

N_CORES   = min(multiprocessing.cpu_count(), 4)
N_CHAINS  = 4
N_DRAWS   = 3000
N_TUNE    = 1500

all_traces = {}

for error, g_type in VARIANTS:
    name = f"{error}_{g_type}"
    print(f"\n{'═'*60}")
    print(f"  [{name}]  error={error}  g(Δ)={g_type}")
    print(f"  chains={N_CHAINS}  draws={N_DRAWS}  cores={N_CORES}")
    print(f"{'═'*60}")

    model = build_model(error=error, g_type=g_type)
    with model:
        step  = pm.Metropolis()
        idata = pm.sample(
            draws       = N_DRAWS,
            tune        = N_TUNE,
            chains      = N_CHAINS,
            cores       = N_CORES,      # 平行鏈
            step        = step,
            progressbar = True,
            return_inferencedata = True,
            random_seed = list(range(N_CHAINS)),
        )

    # Attach per-obs log-likelihood (for LOO)
    print(f"  Computing per-obs log-likelihood …")
    idata = add_log_likelihood(idata, error, g_type)

    all_traces[name] = idata

    # Convergence & posterior summary
    p1n = "alpha_d" if g_type == "gamma" else "mu_d"
    p2n = "beta_r"  if g_type == "gamma" else "sigma_d"
    vnames = ["beta0", "beta1", "sigma", p1n, p2n]
    summ = az.summary(idata, var_names=vnames, round_to=4)
    print(summ.to_string())
    print(f"\n  R̂ max = {summ['r_hat'].max():.4f}  |  ESS min = {summ['ess_bulk'].min():.0f}")


# ═══════════════════════════════════════════════════════════════════════
# 9. MODEL COMPARISON  (LOO-CV)
# ═══════════════════════════════════════════════════════════════════════

print(f"\n\n{'═'*60}")
print("  MODEL COMPARISON — LOO (PSIS)")
print(f"{'═'*60}")

loo_results = {}
for name, idata in all_traces.items():
    try:
        loo = az.loo(idata, var_name="obs")
        loo_results[name] = loo
        print(f"\n{name}:")
        print(f"  elpd_loo = {loo.elpd_loo:.2f}  (se={loo.se:.2f})")
        print(f"  p_loo    = {loo.p_loo:.2f}")
    except Exception as e:
        print(f"\n{name}: LOO failed — {e}")

if loo_results:
    print("\n── LOO Comparison ──")
    comp = az.compare(loo_results, ic="loo")
    print(comp.to_string())


# ═══════════════════════════════════════════════════════════════════════
# 10. POSTERIOR MEANS SUMMARY
# ═══════════════════════════════════════════════════════════════════════

print(f"\n\n{'═'*60}")
print("  POSTERIOR MEANS")
print(f"{'═'*60}")
print(f"  {'Model':<22} {'β₀':>8} {'β₁':>8} {'σ':>7} {'p1':>10} {'p2':>10}")
print(f"  {'-'*67}")
for name, idata in all_traces.items():
    post  = idata.posterior
    b0_m  = float(post["beta0"].mean())
    b1_m  = float(post["beta1"].mean())
    sig_m = float(post["sigma"].mean())
    g_type = name.split("_")[1]
    p1n   = "alpha_d" if g_type == "gamma" else "mu_d"
    p2n   = "beta_r"  if g_type == "gamma" else "sigma_d"
    p1_m  = float(post[p1n].mean())
    p2_m  = float(post[p2n].mean())
    print(f"  {name:<22} {b0_m:>8.3f} {b1_m:>8.3f} {sig_m:>7.4f} {p1_m:>10.4f} {p2_m:>10.4f}")

print("\nDone.")
