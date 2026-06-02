#!/usr/bin/env python3
"""
rfl_pymc_ncp.py — Full Bayesian RFL, Non-Centered Parameterization
===================================================================
NCP 策略：
  1. σ_Δ, σ  → 在 log 空間取樣（scale 參數的標準 NCP）
  2. LogNormal g(Δ) 積分 → 改用 Gauss-Hermite in z-space
       z = (log Δ - μ_Δ) / σ_Δ，節點固定 → 後驗曲面更平滑
       Δ_j = exp(μ_Δ + σ_Δ * z_j)  （NCP 代換）

效果：解決 σ_Δ 漏斗問題，提升 Metropolis 混合
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pymc as pm
from pytensor.compile.ops import as_op
import pytensor.tensor as pt
from scipy import special
import arviz as az
import time

# ═══════════════════════════════════════════════════════════════════════
# 1. DATA
# ═══════════════════════════════════════════════════════════════════════
_raw = {
    0.675: [102.95,280.32,339.83,366.9,485.62,658.96,896.33,
            1241.76,1250.2,1329.78,1399.83,1459.14,3249.82,11748.1,11748.1],
    0.750: [6.71,9.93,12.6,15.58,16.19,17.28,18.62,
            20.3,24.9,26.26,27.94,36.35,48.42,50.09,67.34],
    0.825: [1.246,1.258,1.46,1.492,2.4,2.41,2.59,
            2.903,3.33,3.59,3.847,4.11,4.82,5.56,5.598],
    0.900: [0.201,0.216,0.226,0.252,0.257,0.295,0.311,
            0.342,0.356,0.451,0.457,0.509,0.54,0.68,1.129],
    0.950: [0.037,0.072,0.074,0.076,0.083,0.085,0.105,
            0.109,0.12,0.123,0.143,0.203,0.206,0.217,0.257],
}
S_list, Y_list, ev_list = [], [], []
for s, vals in _raw.items():
    mx = max(vals); nm = vals.count(mx)
    for v in vals:
        S_list.append(s); Y_list.append(np.log(v))
        ev_list.append(0 if (v == mx and nm > 1) else 1)
S_OBS  = np.array(S_list); Y_OBS = np.array(Y_list)
EVENT  = np.array(ev_list, dtype=int)
N_OBS  = len(S_OBS); MIN_S = S_OBS.min()

# ── GL quadrature (Gamma g(Δ) 用) ────────────────────────────────────
_GL_X, _GL_W = np.polynomial.legendre.leggauss(24)
_D_PTS = S_OBS[:, None] / 2.0 * (_GL_X[None, :] + 1.0)
_JAC   = S_OBS / 2.0

# ── GH quadrature in z-space (LogNormal NCP 用) ──────────────────────
# 積分：∫ f(y|Δ) g(Δ;μ,σ) dΔ = (1/√2π) ∫ f(y|e^{μ+σz}) e^{-z²/2} dz
#      ≈ (1/√2π) Σⱼ wⱼ f(y|Δⱼ),  Δⱼ = exp(μ + σ*zⱼ)
# probabilist's hermegauss: ∫ f(x) e^{-x²/2} dx ≈ Σ wⱼ f(xⱼ)
N_GH = 20
_GH_Z, _GH_W = np.polynomial.hermite_e.hermegauss(N_GH)
_INV_SQRT2PI  = 1.0 / np.sqrt(2 * np.pi)


# ═══════════════════════════════════════════════════════════════════════
# 2. MARGINAL LOG-LIKELIHOOD (NCP 版)
# ═══════════════════════════════════════════════════════════════════════

def _lf_normal(y4, mu_c, sig):
    z = (y4 - mu_c) / sig
    return -0.5*np.log(2*np.pi) - np.log(sig) - 0.5*z**2, special.log_ndtr(-z)

def _lf_sev(y4, mu_c, sig):
    z = (y4 - mu_c) / sig
    ez = np.exp(np.clip(z, -500, 20))
    return -np.log(sig) + z - ez, -ez


def loglik_lognormal_ncp(b0, b1, log_sig, mu_d, log_sd_d, error):
    """
    LogNormal g(Δ) — NCP: Gauss-Hermite in z-space + log(σ_Δ), log(σ).

    z_j   : 固定 GH 節點
    Δ_j   = exp(μ_Δ + σ_Δ * z_j)  ← 隨 (μ_Δ, σ_Δ) 移動，但節點本身不變
    積分  ≈ (1/√2π) Σⱼ wⱼ f(y|Δⱼ) × I(Δⱼ < S_i)
    """
    sig  = np.exp(log_sig)
    sd_d = np.exp(log_sd_d)

    # Δ at GH nodes: (N_OBS, N_GH)  — broadcast over obs
    delta = np.exp(mu_d + sd_d * _GH_Z[None, :])  # (1,N_GH) broadcast → (N_OBS,N_GH)
    valid = delta < S_OBS[:, None]                 # constraint Δ < S_i
    delta_safe = np.where(valid, delta, 1e-10)

    mu_c = b0 + b1 * np.log(np.maximum(S_OBS[:, None] - delta_safe, 1e-10))
    Y4   = Y_OBS[:, None]

    lf_fail, lf_cens = (_lf_normal if error == "normal" else _lf_sev)(Y4, mu_c, sig)
    fail = (EVENT == 1)[:, None]
    lf   = np.where(fail, lf_fail, lf_cens)
    lf   = np.where(valid, lf, -700.0)        # invalid nodes → 0 contribution

    # GH integral: (N_OBS,)
    # (1/√2π) Σⱼ wⱼ exp(lf_j)
    integrand = np.exp(lf)                     # (N_OBS, N_GH)
    integral  = _INV_SQRT2PI * np.dot(integrand, _GH_W)   # (N_OBS,)

    # P(Δ > S_i) for censored obs
    p_above = 0.5 * special.erfc((np.log(S_OBS) - mu_d) / (np.sqrt(2) * sd_d))
    integral = integral.copy()
    integral[EVENT == 0] += p_above[EVENT == 0]

    return np.array(np.sum(np.log(np.maximum(integral, 1e-300))), dtype=np.float64)


def loglik_gamma_ncp(b0, b1, log_sig, log_alpha, log_beta_r, error):
    """
    Gamma g(Δ) — NCP: log(α), log(β_r) + Gauss-Legendre on (0, S_i).
    """
    sig    = np.exp(log_sig)
    alpha  = np.exp(log_alpha)
    beta_r = np.exp(log_beta_r)

    mu_c = b0 + b1 * np.log(np.maximum(S_OBS[:, None] - _D_PTS, 1e-10))
    Y4   = Y_OBS[:, None]

    lf_fail, lf_cens = (_lf_normal if error == "normal" else _lf_sev)(Y4, mu_c, sig)
    fail = (EVENT == 1)[:, None]
    lf   = np.where(fail, lf_fail, lf_cens)

    lg = ((alpha-1)*np.log(np.maximum(_D_PTS, 1e-300))
          + alpha*np.log(beta_r) - beta_r*_D_PTS - special.gammaln(alpha))

    integral = (_GL_W[None, :] * np.exp(lf + lg) * _JAC[:, None]).sum(axis=1)
    p_above  = special.gammaincc(alpha, beta_r * S_OBS)
    integral = integral.copy()
    integral[EVENT == 0] += p_above[EVENT == 0]

    return np.array(np.sum(np.log(np.maximum(integral, 1e-300))), dtype=np.float64)


# ═══════════════════════════════════════════════════════════════════════
# 3. PYMC MODEL FACTORY (NCP 版)
# ═══════════════════════════════════════════════════════════════════════

def build_model_ncp(error, g_type):
    upper_mu_d   = float(np.log(MIN_S) - 0.05)
    log_sig_lo   = np.log(0.01);  log_sig_hi  = np.log(5.0)
    log_sdd_lo   = np.log(0.001); log_sdd_hi  = np.log(2.0)
    log_alpha_lo = np.log(0.1);   log_alpha_hi = np.log(50.0)
    log_betar_lo = np.log(0.1);   log_betar_hi = np.log(200.0)

    if g_type == "lognormal":
        @as_op(itypes=[pt.dscalar]*5, otypes=[pt.dscalar])
        def ll_op(b0, b1, log_sig, mu_d, log_sd_d):
            return loglik_lognormal_ncp(
                float(b0), float(b1), float(log_sig),
                float(mu_d), float(log_sd_d), error)
    else:
        @as_op(itypes=[pt.dscalar]*5, otypes=[pt.dscalar])
        def ll_op(b0, b1, log_sig, log_alpha, log_beta_r):
            return loglik_gamma_ncp(
                float(b0), float(b1), float(log_sig),
                float(log_alpha), float(log_beta_r), error)

    with pm.Model() as model:
        b0      = pm.Uniform("beta0",    lower=-50.0,     upper=50.0)
        b1      = pm.Uniform("beta1",    lower=-30.0,     upper=0.0)
        # NCP: sample log(σ) → σ = exp(log_σ)
        log_sig = pm.Uniform("log_sigma",lower=log_sig_lo,upper=log_sig_hi)
        sigma   = pm.Deterministic("sigma", pt.exp(log_sig))

        if g_type == "lognormal":
            mu_d    = pm.Uniform("mu_d",    lower=-5.0,     upper=upper_mu_d)
            # NCP: sample log(σ_Δ) → σ_Δ = exp(log_σ_Δ)
            log_sdd = pm.Uniform("log_sigma_d", lower=log_sdd_lo, upper=log_sdd_hi)
            sigma_d = pm.Deterministic("sigma_d", pt.exp(log_sdd))
            pm.Potential("loglik", ll_op(b0, b1, log_sig, mu_d, log_sdd))
        else:
            log_alpha = pm.Uniform("log_alpha_d", lower=log_alpha_lo, upper=log_alpha_hi)
            log_betar = pm.Uniform("log_beta_r",  lower=log_betar_lo, upper=log_betar_hi)
            alpha_d   = pm.Deterministic("alpha_d", pt.exp(log_alpha))
            beta_r    = pm.Deterministic("beta_r",  pt.exp(log_betar))
            pm.Potential("loglik", ll_op(b0, b1, log_sig, log_alpha, log_betar))

    return model


# ═══════════════════════════════════════════════════════════════════════
# 4. INIT VALUES (NCP space)
# ═══════════════════════════════════════════════════════════════════════

INIT_NCP = {
    ("normal","lognormal"): dict(
        beta0=-9.2, beta1=-8.1,
        log_sigma=np.log(0.60), mu_d=-0.65, log_sigma_d=np.log(0.04)),
    ("normal","gamma"): dict(
        beta0=-9.2, beta1=-8.1,
        log_sigma=np.log(0.60), log_alpha_d=np.log(8.0), log_beta_r=np.log(80.0)),
    ("sev","lognormal"): dict(
        beta0=-9.4, beta1=-8.5,
        log_sigma=np.log(0.19), mu_d=-0.65, log_sigma_d=np.log(0.04)),
    ("sev","gamma"): dict(
        beta0=-9.4, beta1=-8.5,
        log_sigma=np.log(0.19), log_alpha_d=np.log(8.0), log_beta_r=np.log(80.0)),
}


# ═══════════════════════════════════════════════════════════════════════
# 5. VECTORISED PER-OBS LOG-LIKELIHOOD
# ═══════════════════════════════════════════════════════════════════════

def compute_loglik(idata, error, g_type):
    post = idata.posterior
    def e(x): return x[:,:,None,None]

    b0_  = e(post["beta0"].values)
    b1_  = e(post["beta1"].values)
    sig_ = e(post["sigma"].values)   # already back-transformed

    if g_type == "lognormal":
        mu_d_  = e(post["mu_d"].values)
        sd_d_  = e(post["sigma_d"].values)

        # GH nodes: Δ = exp(μ_Δ + σ_Δ * z_j)
        D4     = np.exp(mu_d_ + sd_d_ * _GH_Z[None,None,None,:])  # (nc,nd,1,N_GH)
        S4     = S_OBS[None,None,:,None]
        Y4     = Y_OBS[None,None,:,None]
        W4     = _GH_W[None,None,None,:]
        valid4 = D4 < S4
        D_safe = np.where(valid4, D4, 1e-10)

        mu_c   = b0_ + b1_ * np.log(np.maximum(S4 - D_safe, 1e-10))
        z_r    = (Y4 - mu_c) / sig_
        if error == "normal":
            lf_f = -0.5*np.log(2*np.pi) - np.log(sig_) - 0.5*z_r**2
            lf_c = special.log_ndtr(-z_r)
        else:
            ez   = np.exp(np.clip(z_r,-500,20))
            lf_f = -np.log(sig_) + z_r - ez
            lf_c = -ez

        fail4 = (EVENT==1)[None,None,:,None]
        lf    = np.where(fail4, lf_f, lf_c)
        lf    = np.where(valid4, lf, -700.0)

        integ = _INV_SQRT2PI * (W4 * np.exp(lf)).sum(axis=-1)   # (nc,nd,N_OBS)

        mu_d3  = post["mu_d"].values[:,:,None]
        sd_d3  = post["sigma_d"].values[:,:,None]
        p_ab   = 0.5*special.erfc((np.log(S_OBS[None,None,:])-mu_d3) / (np.sqrt(2)*sd_d3))

    else:  # gamma
        alpha_ = e(post["alpha_d"].values)
        betar_ = e(post["beta_r"].values)
        D4     = _D_PTS[None,None,:,:]
        S4     = S_OBS[None,None,:,None]
        Y4     = Y_OBS[None,None,:,None]
        W4     = _GL_W[None,None,None,:]
        jac4   = _JAC[None,None,:,None]

        mu_c = b0_ + b1_ * np.log(np.maximum(S4-D4,1e-10))
        z_r  = (Y4-mu_c)/sig_
        if error == "normal":
            lf_f = -0.5*np.log(2*np.pi)-np.log(sig_)-0.5*z_r**2
            lf_c = special.log_ndtr(-z_r)
        else:
            ez   = np.exp(np.clip(z_r,-500,20))
            lf_f = -np.log(sig_)+z_r-ez; lf_c=-ez

        fail4 = (EVENT==1)[None,None,:,None]
        lf    = np.where(fail4, lf_f, lf_c)
        lg    = ((alpha_-1)*np.log(np.maximum(D4,1e-300))
                 +alpha_*np.log(betar_)-betar_*D4-special.gammaln(alpha_))
        integ = (W4*np.exp(lf+lg)*jac4).sum(axis=-1)

        al3   = post["alpha_d"].values[:,:,None]
        br3   = post["beta_r"].values[:,:,None]
        p_ab  = special.gammaincc(al3, br3*S_OBS[None,None,:])

    cens3 = (EVENT==0)[None,None,:]
    integ += np.where(cens3, p_ab, 0.0)
    return np.log(np.maximum(integ, 1e-300))


# ═══════════════════════════════════════════════════════════════════════
# 6. MAIN
# ═══════════════════════════════════════════════════════════════════════

VARIANTS = [
    ("normal","lognormal"),
    ("normal","gamma"),
    ("sev",   "lognormal"),
    ("sev",   "gamma"),
]
N_CHAINS = 4
N_TUNE   = 3000
N_DRAWS  = 2000
N_CORES  = 1   # as_op can't pickle → sequential

if __name__ == "__main__":
    print(f"n={N_OBS}  failures={EVENT.sum()}  censored={N_OBS-EVENT.sum()}")
    print("NCP: log(sigma), log(sigma_d) + Gauss-Hermite in z-space\n")
    all_res = {}

    for error, g_type in VARIANTS:
        name   = f"{error}_{g_type}"
        init   = INIT_NCP[(error, g_type)]
        p1n    = "mu_d"    if g_type == "lognormal" else "alpha_d"
        p2n    = "sigma_d" if g_type == "lognormal" else "beta_r"

        print(f"\n{'='*65}")
        print(f"  [{name}]  tune={N_TUNE}  draws={N_DRAWS}  chains={N_CHAINS}")
        print(f"{'='*65}")

        model = build_model_ncp(error, g_type)
        t0    = time.time()

        with model:
            step  = pm.Metropolis()
            idata = pm.sample(
                draws=N_DRAWS, tune=N_TUNE, chains=N_CHAINS,
                cores=N_CORES, step=step, initvals=init,
                progressbar=True, random_seed=list(range(N_CHAINS)),
            )

        elapsed = time.time() - t0
        print(f"\n  Elapsed: {elapsed:.1f}s")

        vnames = ["beta0","beta1","sigma", p1n, p2n]
        summ   = az.summary(idata, var_names=vnames, round_to=4)
        print(summ[["mean","sd","eti89_lb","eti89_ub","r_hat","ess_bulk"]].to_string())

        print("\n  Computing per-obs log-likelihood...")
        t1  = time.time()
        ll  = compute_loglik(idata, error, g_type)
        print(f"  Done in {time.time()-t1:.1f}s")

        import xarray as xr
        ll_ds = xr.Dataset({"obs": xr.DataArray(ll, dims=["chain","draw","obs_id"])})
        try:    idata["log_likelihood"] = ll_ds
        except: idata.log_likelihood = ll_ds

        try:
            loo = az.loo(idata, var_name="obs")
            elpd_val = getattr(loo, "elpd_loo", None) or getattr(loo, "elpd", None)
            print(f"\n  LOO elpd = {elpd_val:.2f}")
        except Exception as e:
            elpd_val = float(ll.mean(axis=(0,1)).sum())
            print(f"\n  ELPD proxy = {elpd_val:.2f}")
            loo = None

        all_res[name] = dict(summ=summ, elpd=elpd_val, rhat=summ["r_hat"].max())

    # ── Comparison ───────────────────────────────────────────────────
    print(f"\n\n{'='*65}")
    print("  FINAL COMPARISON (NCP)")
    print(f"{'='*65}")
    print(f"  {'Model':<22} {'b0':>8} {'b1':>8} {'sigma':>7} {'Rhat_max':>9} {'ELPD':>10}")
    print(f"  {'-'*67}")
    for name, res in all_res.items():
        pm_ = res["summ"]["mean"].values
        print(f"  {name:<22} {pm_[0]:>8.3f} {pm_[1]:>8.3f} {pm_[2]:>7.4f} "
              f"{res['rhat']:>9.4f} {res['elpd']:>10.2f}")
    print("\nDone.")
