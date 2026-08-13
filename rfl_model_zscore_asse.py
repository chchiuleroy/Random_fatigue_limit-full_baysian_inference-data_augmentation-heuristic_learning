#!/usr/bin/env python3
"""
rfl_model_zscore_asse.py
Roy 新想法驗證：模型 z-score ASSE
======================================

原始 Chiu 樣本 z-score（有偏）：
  omega_ij = (y_ij - y_bar_j) / s_j      <- 15 個樣本，含 Delta 和 sigma 混在一起

Roy 新想法（模型 z-score）：
  E(lnY | S_j)  = beta0 - sigma*gamma_E + beta1 * E[ln(S-Delta)]  [GL 積分]
  V(lnY | S_j)  = beta1^2 * Var[ln(S-Delta)] + pi^2*sigma^2/6     [異方差！]
  z_ij^model    = (y_ij - E(lnY|S_j)) / sqrt(V(lnY|S_j))
  p_ij          = Phi(z_ij)                                         [percentile]
  Delta_hat_ij  = exp(mu_d + sigma_d * z_ij)                       [逆推 Delta]
  y_hat_ij      = beta0 + beta1*ln(S_j - Delta_hat) - sigma*gamma_E
  ASSE          = sum|y_ij - y_hat_ij|

比較：
  A. 樣本 z-score ASSE (Chiu)
  B. 模型 z-score ASSE (Roy 新想法, plugin 後驗均值)
  C. 模型 z-score ASSE (完整後驗分佈，E[ASSE])
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import arviz as az
from scipy import stats, special
import time

EULER_GAMMA = 0.5772156649015329
PI2_OVER_6  = np.pi**2 / 6.0

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
S_list, Y_list, ev_list, grp_list = [], [], [], []
for j, (s, vals) in enumerate(_raw.items()):
    mx = max(vals); nm = vals.count(mx)
    for v in vals:
        S_list.append(s); Y_list.append(np.log(v))
        ev_list.append(0 if (v == mx and nm > 1) else 1)
        grp_list.append(j)

S_OBS  = np.array(S_list,  dtype=float)
Y_OBS  = np.array(Y_list,  dtype=float)
EVENT  = np.array(ev_list, dtype=int)
GRP    = np.array(grp_list, dtype=int)
N_OBS  = len(S_OBS)
MIN_S  = S_OBS.min()
LOG_S  = np.log(S_OBS)
S_UNIQ = np.array(sorted(set(S_OBS)))

fail_idx = np.where(EVENT == 1)[0]
cens_idx = np.where(EVENT == 0)[0]
Y_fail   = Y_OBS[fail_idx]
Y_cens   = Y_OBS[cens_idx]

# ═══════════════════════════════════════════════════════════════════════
# 2. GL 積分工具
# ═══════════════════════════════════════════════════════════════════════
N_QUAD  = 32
GL_X, GL_W = np.polynomial.legendre.leggauss(N_QUAD)

def _gl_moments(mu_d, sd_d, S_j):
    """
    GL 積分算 E[ln(S-Δ)] 和 Var[ln(S-Δ)] (條件 Δ<S)
    回傳 (E1, Var_logSD, norm)
    """
    d_pts  = S_j / 2.0 * (GL_X + 1.0)
    jac    = S_j / 2.0
    log_d  = np.log(np.maximum(d_pts, 1e-300))
    log_Sd = np.log(np.maximum(S_j - d_pts, 1e-300))
    log_g  = (-0.5*np.log(2*np.pi) - np.log(sd_d)
              - (log_d - mu_d)**2 / (2*sd_d**2) - log_d)
    w_g    = GL_W * np.exp(log_g) * jac     # (N_QUAD,)
    norm   = w_g.sum()
    if norm < 1e-10:
        return None, None, norm
    E1         = np.dot(w_g, log_Sd) / norm
    E2         = np.dot(w_g, log_Sd**2) / norm
    Var_logSD  = max(E2 - E1**2, 0.0)
    return E1, Var_logSD, norm


def compute_E_V(b0, b1, sig, mu_d, sd_d, S_j, error='sev'):
    """
    E(lnY | S_j) 和 V(lnY | S_j) — LogNormal g(Delta)

    SEV  : E = b0 - sigma*gamma_E + b1*E1
           V = b1^2*Var[ln(S-Δ)] + pi^2*sigma^2/6
    Normal: E = b0 + b1*E1
            V = b1^2*Var[ln(S-Δ)] + sigma^2
    """
    E1, Var_logSD, norm = _gl_moments(mu_d, sd_d, S_j)
    if E1 is None:
        if error == 'sev':
            return b0 - sig * EULER_GAMMA, PI2_OVER_6 * sig**2
        else:
            return b0, sig**2

    if error == 'sev':
        E_lnY = b0 - sig * EULER_GAMMA + b1 * E1
        V_lnY = b1**2 * Var_logSD + PI2_OVER_6 * sig**2
    else:  # normal
        E_lnY = b0 + b1 * E1
        V_lnY = b1**2 * Var_logSD + sig**2

    return E_lnY, max(V_lnY, 1e-10)


def model_zscore_asse(b0, b1, sig, mu_d, sd_d, error='sev', verbose=False):
    """
    Roy 新想法：模型 z-score ASSE
    error : 'sev' | 'normal'
    """
    euler_corr = EULER_GAMMA if error == 'sev' else 0.0
    asse_parts = []
    info = []

    for s_j in S_UNIQ:
        idx_j = np.where((S_OBS == s_j) & (EVENT == 1))[0]
        if len(idx_j) == 0:
            continue

        E_j, V_j = compute_E_V(b0, b1, sig, mu_d, sd_d, s_j, error=error)
        sd_j  = np.sqrt(V_j)
        y_j   = Y_OBS[idx_j]
        z_j   = (y_j - E_j) / sd_j
        D_hat = np.exp(mu_d + sd_d * z_j)
        D_hat = np.clip(D_hat, 1e-6, s_j * 0.999)
        y_hat = b0 + b1 * np.log(s_j - D_hat) - sig * euler_corr
        asse_parts.append(np.abs(y_j - y_hat).sum())
        if verbose:
            info.append((s_j, E_j, sd_j, z_j.mean(), D_hat.mean()))

    asse = sum(asse_parts)
    return (asse, info) if verbose else asse


# ═══════════════════════════════════════════════════════════════════════
# 3. 樣本 z-score ASSE (Chiu baseline)
# ═══════════════════════════════════════════════════════════════════════

def sample_zscore_asse(b0, b1, sig, mu_d, sd_d, error='sev'):
    """Chiu (2005) 樣本 z-score ASSE — 用組內樣本均值/標準差。"""
    euler_corr = EULER_GAMMA if error == 'sev' else 0.0
    asse_parts = []
    for s_j in S_UNIQ:
        idx_j = np.where((S_OBS == s_j) & (EVENT == 1))[0]
        if len(idx_j) == 0:
            continue
        y_j   = Y_OBS[idx_j]
        z_j   = (y_j - y_j.mean()) / y_j.std(ddof=1)   # sample z-score
        D_hat = np.exp(mu_d + sd_d * z_j)
        D_hat = np.clip(D_hat, 1e-6, s_j * 0.999)
        y_hat = b0 + b1 * np.log(s_j - D_hat) - sig * euler_corr
        asse_parts.append(np.abs(y_j - y_hat).sum())
    return sum(asse_parts)


# ═══════════════════════════════════════════════════════════════════════
# 4. PyMC DA+NCP 模型（SEV+LogNormal）
# ═══════════════════════════════════════════════════════════════════════

def build_model(error='sev'):
    upper_mu_d = float(np.log(MIN_S) - 0.05)
    with pm.Model() as m:
        b0      = pm.Uniform("beta0",      lower=-50.,         upper=50.)
        b1      = pm.Uniform("beta1",      lower=-30.,         upper=0.)
        log_sig = pm.Uniform("log_sigma",  lower=np.log(0.01), upper=np.log(5.))
        sigma   = pm.Deterministic("sigma", pt.exp(log_sig))
        mu_d    = pm.Uniform("mu_d",       lower=-5.,          upper=upper_mu_d)
        log_sdd = pm.Uniform("log_sigma_d",lower=np.log(0.001),upper=np.log(2.))
        sigma_d = pm.Deterministic("sigma_d", pt.exp(log_sdd))
        z_delta = pm.Normal("z_delta", mu=0., sigma=1., shape=N_OBS)
        log_delta = mu_d + sigma_d * z_delta
        delta     = pm.Deterministic("delta", pt.exp(log_delta))
        mu_cond   = b0 + b1 * pt.log(pt.maximum(S_OBS - delta, 1e-8))
        if error == 'normal':
            # Failures: ln N_i ~ Normal(mu_cond, sigma)
            pm.Normal("obs_fail", mu=mu_cond[fail_idx], sigma=sigma, observed=Y_fail)
            # Censored: log(1 - Phi(z)) = log(0.5*erfc(z/sqrt(2)))
            z_c   = (Y_cens - mu_cond[cens_idx]) / sigma
            log_s = pt.log(0.5 * pt.erfc(z_c / pt.sqrt(2.)))
            pm.Potential("obs_cens", pt.sum(log_s))
        else:  # SEV: log f = -log(sigma) + z - exp(z), log S = -exp(z)
            z_f = (Y_fail - mu_cond[fail_idx]) / sigma
            pm.Potential("obs_fail",
                pt.sum(-pt.log(sigma) + z_f - pt.exp(pt.clip(z_f,-500,20))))
            z_c = (Y_cens - mu_cond[cens_idx]) / sigma
            pm.Potential("obs_cens",
                pt.sum(-pt.exp(pt.clip(z_c,-500,20))))
    return m


# ═══════════════════════════════════════════════════════════════════════
# 5. MAIN
# ═══════════════════════════════════════════════════════════════════════

def run_one(error, init_sig):
    """Run DA+NCP for one error type, return (idata, pm_dict)."""
    model = build_model(error=error)
    with model:
        idata = pm.sample(
            draws=2000, tune=2000, chains=4, cores=4,
            initvals=dict(beta0=-9.4 if error=='sev' else -9.2,
                          beta1=-8.5 if error=='sev' else -8.1,
                          log_sigma=np.log(init_sig),
                          mu_d=-0.65, log_sigma_d=np.log(0.04),
                          z_delta=np.zeros(N_OBS)),
            target_accept=0.95,  # raised from 0.85 (2026-08-12), see README §5.5
            progressbar=True,
            random_seed=[0,1,2,3],
        )
    pm_ = {v: float(idata.posterior[v].mean())
           for v in ["beta0","beta1","sigma","mu_d","sigma_d"]}
    return idata, pm_


def post_asse(idata, error, n_samp=500):
    """Posterior ASSE distribution (sample ~n_samp draws)."""
    post  = idata.posterior
    b0_s  = post["beta0"].values.flatten()
    b1_s  = post["beta1"].values.flatten()
    sig_s = post["sigma"].values.flatten()
    mud_s = post["mu_d"].values.flatten()
    sdd_s = post["sigma_d"].values.flatten()
    n_s   = len(b0_s)
    stride = max(1, n_s // n_samp)
    idx   = np.arange(0, n_s, stride)
    model_z = [model_zscore_asse(b0_s[k],b1_s[k],sig_s[k],mud_s[k],sdd_s[k],error=error)
               for k in idx]
    samp_z  = [sample_zscore_asse(b0_s[k],b1_s[k],sig_s[k],mud_s[k],sdd_s[k],error=error)
               for k in idx]
    return np.array(model_z), np.array(samp_z)


if __name__ == "__main__":
    print("="*70)
    print("  Roy Model z-score ASSE — SEV+LogNormal vs Normal+LogNormal")
    print("="*70)

    VARIANTS = [
        ("sev",    0.19, "SEV   + LogNormal"),
        ("normal", 0.60, "Normal + LogNormal"),
    ]

    results = {}

    for error, init_sig, label in VARIANTS:
        print(f"\n{'='*70}")
        print(f"  [{label}]  Running DA+NCP+NUTS ...")
        print(f"{'='*70}")

        idata, pm_ = run_one(error, init_sig)
        b0p, b1p, sigp = pm_["beta0"], pm_["beta1"], pm_["sigma"]
        mudp, sddp = pm_["mu_d"], pm_["sigma_d"]

        summ = az.summary(idata,
                          var_names=["beta0","beta1","sigma","mu_d","sigma_d"],
                          round_to=4)
        print("\nPosterior means:")
        print(summ[["mean","sd","r_hat","ess_bulk"]].to_string())

        # Plugin ASSE
        asse_samp = sample_zscore_asse(b0p, b1p, sigp, mudp, sddp, error=error)
        asse_model, ev_info = model_zscore_asse(
            b0p, b1p, sigp, mudp, sddp, error=error, verbose=True)

        print(f"\n  Plugin ASSE:")
        print(f"    Sample z-score : {asse_samp:.4f}")
        print(f"    Model  z-score : {asse_model:.4f}  <-- Roy idea")

        # Per-group E, SD
        print(f"\n  E(lnY|S) and SD(lnY|S) per stress level:")
        print(f"  {'S':>7}  {'E':>10}  {'SD':>10}  {'z_mean':>8}  {'D_hat_mean':>12}")
        print(f"  {'-'*55}")
        for s_j, E_j, sd_j, z_m, D_m in ev_info:
            print(f"  {s_j:>7.3f}  {E_j:>10.4f}  {sd_j:>10.4f}  {z_m:>8.3f}  {D_m:>12.4f}")

        # Full posterior ASSE
        print(f"\n  Computing posterior ASSE distribution (~500 draws)...")
        mz_post, sz_post = post_asse(idata, error, n_samp=500)
        print(f"    Model z-score: mean={mz_post.mean():.4f} "
              f"median={np.median(mz_post):.4f} "
              f"95%CI=[{np.percentile(mz_post,2.5):.3f},{np.percentile(mz_post,97.5):.3f}]")
        print(f"    Samp  z-score: mean={sz_post.mean():.4f} "
              f"median={np.median(sz_post):.4f} "
              f"95%CI=[{np.percentile(sz_post,2.5):.3f},{np.percentile(sz_post,97.5):.3f}]")

        results[label] = dict(
            pm=pm_, plugin_samp=asse_samp, plugin_model=asse_model,
            post_model=mz_post, post_samp=sz_post, error=error
        )

    # ── 最終比較表 ─────────────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print("  FINAL COMPARISON  (y-ASSE, ln(kilocycle), failures only)")
    print(f"{'='*70}")
    print(f"  {'Model variant':<25} {'Method':<22} {'Plugin':>8} {'Post.mean':>10} {'Post.median':>12}")
    print(f"  {'-'*80}")
    for label, res in results.items():
        print(f"  {label:<25} {'Sample z-score (Chiu)':<22} "
              f"{res['plugin_samp']:>8.4f} {res['post_samp'].mean():>10.4f} "
              f"{np.median(res['post_samp']):>12.4f}")
        print(f"  {'':<25} {'Model  z-score (Roy)':<22} "
              f"{res['plugin_model']:>8.4f} {res['post_model'].mean():>10.4f} "
              f"{np.median(res['post_model']):>12.4f}")
        print(f"  {'-'*80}")

    print(f"\n  Context (different scale — not directly comparable):")
    print(f"    Chiu (2005) EIV z-ASSE     = 10.80")
    print(f"    Method C-2 LAD (Roy) z-ASSE=  9.94")
    print("\nDone.")
