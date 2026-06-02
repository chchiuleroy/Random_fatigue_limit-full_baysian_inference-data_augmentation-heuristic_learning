#!/usr/bin/env python3
"""
rfl_pymc.py — Full Bayesian RFL with PyMC 6 + Metropolis
4 variants: (Normal|SEV) x (LogNormal|Gamma)
Uniform priors, parallel chains, arviz diagnostics
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

_GL_X, _GL_W = np.polynomial.legendre.leggauss(24)
_D_PTS = S_OBS[:, None] / 2.0 * (_GL_X[None, :] + 1.0)
_JAC   = S_OBS / 2.0

# ═══════════════════════════════════════════════════════════════════════
# 2. LOG-LIKELIHOOD FACTORY
# ═══════════════════════════════════════════════════════════════════════

def _make_loglik_op(error, g_type):
    """Return an as_op-decorated function for the given model variant."""

    def _ll(b0, b1, sig, p1, p2):
        mu_c = b0 + b1 * np.log(np.maximum(S_OBS[:, None] - _D_PTS, 1e-10))
        z    = (Y_OBS[:, None] - mu_c) / sig
        if error == "normal":
            lf = np.where((EVENT == 1)[:, None],
                          -0.5*np.log(2*np.pi) - np.log(sig) - 0.5*z**2,
                          special.log_ndtr(-z))
        else:  # sev
            ez = np.exp(np.clip(z, -500, 20))
            lf = np.where((EVENT == 1)[:, None],
                          -np.log(sig) + z - ez, -ez)

        if g_type == "lognormal":
            log_d = np.log(np.maximum(_D_PTS, 1e-300))
            lg = (-0.5*np.log(2*np.pi) - np.log(p2)
                  - (log_d - p1)**2 / (2*p2**2) - log_d)
        else:  # gamma
            lg = ((p1-1)*np.log(np.maximum(_D_PTS, 1e-300))
                  + p1*np.log(p2) - p2*_D_PTS - special.gammaln(p1))

        integral = (_GL_W[None, :] * np.exp(lf + lg) * _JAC[:, None]).sum(axis=1)

        if g_type == "lognormal":
            p_ab = 0.5 * special.erfc((np.log(S_OBS) - p1) / (np.sqrt(2)*p2))
        else:
            p_ab = special.gammaincc(p1, p2 * S_OBS)

        integral[EVENT == 0] += p_ab[EVENT == 0]
        return np.array(np.sum(np.log(np.maximum(integral, 1e-300))),
                        dtype=np.float64)

    # Wrap as pytensor Op (no gradient -> Metropolis)
    @as_op(itypes=[pt.dscalar]*5, otypes=[pt.dscalar])
    def op(b0, b1, sig, p1, p2):
        return _ll(float(b0), float(b1), float(sig), float(p1), float(p2))

    return op


# ═══════════════════════════════════════════════════════════════════════
# 3. MODEL FACTORY
# ═══════════════════════════════════════════════════════════════════════

# Good starting values near known MLE region (Normal + LogNormal)
_INIT_NORMAL_LN = dict(beta0=-9.2,  beta1=-8.1,  sigma=0.60,
                        mu_d=-0.65,  sigma_d=0.04)
_INIT_NORMAL_GA = dict(beta0=-9.2,  beta1=-8.1,  sigma=0.60,
                        alpha_d=8.0, beta_r=80.0)
_INIT_SEV_LN    = dict(beta0=-9.4,  beta1=-8.5,  sigma=0.19,
                        mu_d=-0.65,  sigma_d=0.04)
_INIT_SEV_GA    = dict(beta0=-9.4,  beta1=-8.5,  sigma=0.19,
                        alpha_d=8.0, beta_r=80.0)

INIT_MAP = {
    ("normal","lognormal"): _INIT_NORMAL_LN,
    ("normal","gamma"):     _INIT_NORMAL_GA,
    ("sev",   "lognormal"): _INIT_SEV_LN,
    ("sev",   "gamma"):     _INIT_SEV_GA,
}

def build_model(error, g_type):
    ll_op = _make_loglik_op(error, g_type)
    upper_mu_d = float(np.log(MIN_S) - 0.05)

    with pm.Model() as model:
        b0  = pm.Uniform("beta0",  lower=-50.0, upper=50.0)
        b1  = pm.Uniform("beta1",  lower=-30.0, upper=0.0)
        sig = pm.Uniform("sigma",  lower=0.01,  upper=5.0)

        if g_type == "lognormal":
            p1 = pm.Uniform("mu_d",   lower=-5.0,  upper=upper_mu_d)
            p2 = pm.Uniform("sigma_d",lower=0.001, upper=2.0)
        else:
            p1 = pm.Uniform("alpha_d",lower=0.1,   upper=50.0)
            p2 = pm.Uniform("beta_r", lower=0.1,   upper=200.0)

        pm.Potential("loglik", ll_op(b0, b1, sig, p1, p2))
    return model


# ═══════════════════════════════════════════════════════════════════════
# 4. VECTORISED PER-OBS LOG-LIKELIHOOD  (for LOO)
# ═══════════════════════════════════════════════════════════════════════

def compute_loglik(idata, error, g_type):
    post = idata.posterior
    p1n  = "alpha_d" if g_type == "gamma" else "mu_d"
    p2n  = "beta_r"  if g_type == "gamma" else "sigma_d"

    def e(x): return x[:, :, None, None]
    b0_  = e(post["beta0"].values);  b1_ = e(post["beta1"].values)
    sig_ = e(post["sigma"].values);  p1_ = e(post[p1n].values)
    p2_  = e(post[p2n].values)

    D4  = _D_PTS[None, None];  S4  = S_OBS[None, None, :, None]
    Y4  = Y_OBS[None, None, :, None]; W4 = _GL_W[None, None, None, :]
    jac3= _JAC[None, None, :, None]

    mu_c = b0_ + b1_ * np.log(np.maximum(S4 - D4, 1e-10))
    z    = (Y4 - mu_c) / sig_

    if error == "normal":
        lf_f = -0.5*np.log(2*np.pi) - np.log(sig_) - 0.5*z**2
        lf_c = special.log_ndtr(-z)
    else:
        ez   = np.exp(np.clip(z, -500, 20))
        lf_f = -np.log(sig_) + z - ez
        lf_c = -ez

    fail4 = (EVENT == 1)[None, None, :, None]
    lf    = np.where(fail4, lf_f, lf_c)

    if g_type == "lognormal":
        log_d = np.log(np.maximum(D4, 1e-300))
        lg    = (-0.5*np.log(2*np.pi) - np.log(p2_)
                 - (log_d - p1_)**2 / (2*p2_**2) - log_d)
    else:
        lg    = ((p1_-1)*np.log(np.maximum(D4, 1e-300))
                 + p1_*np.log(p2_) - p2_*D4 - special.gammaln(p1_))

    integ = (W4 * np.exp(lf + lg) * jac3).sum(axis=-1)

    S3 = S_OBS[None, None, :]; p1_3 = post[p1n].values[:, :, None]
    p2_3 = post[p2n].values[:, :, None]
    if g_type == "lognormal":
        p_ab = 0.5 * special.erfc((np.log(S3)-p1_3) / (np.sqrt(2)*p2_3))
    else:
        p_ab = special.gammaincc(p1_3, p2_3*S3)

    integ += np.where((EVENT == 0)[None, None, :], p_ab, 0.0)
    return np.log(np.maximum(integ, 1e-300))   # (n_c, n_d, N_OBS)


# ═══════════════════════════════════════════════════════════════════════
# 5. MAIN
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
N_CORES  = 4   # parallel chains (Windows: needs __main__ guard)

if __name__ == "__main__":
    print(f"n={N_OBS}  failures={EVENT.sum()}  censored={N_OBS-EVENT.sum()}")
    all_res = {}

    for error, g_type in VARIANTS:
        name = f"{error}_{g_type}"
        init = INIT_MAP[(error, g_type)]
        p1n  = "alpha_d" if g_type == "gamma" else "mu_d"
        p2n  = "beta_r"  if g_type == "gamma" else "sigma_d"

        print(f"\n{'='*65}")
        print(f"  [{name}]  tune={N_TUNE}  draws={N_DRAWS}  chains={N_CHAINS}")
        print(f"{'='*65}")

        model = build_model(error, g_type)
        t0    = time.time()

        with model:
            step  = pm.Metropolis()
            idata = pm.sample(
                draws       = N_DRAWS,
                tune        = N_TUNE,
                chains      = N_CHAINS,
                cores       = N_CORES,
                step        = step,
                initvals    = init,
                progressbar = True,
                random_seed = list(range(N_CHAINS)),
            )

        elapsed = time.time() - t0
        print(f"\n  Elapsed: {elapsed:.1f}s")

        # Summary
        vnames = ["beta0","beta1","sigma", p1n, p2n]
        summ   = az.summary(idata, var_names=vnames, round_to=4)
        # Print whichever columns are available
        print(summ[["mean","sd","eti89_lb","eti89_ub","r_hat","ess_bulk"]].to_string())

        # Per-obs LL
        print("\n  Computing per-obs log-likelihood...")
        t1  = time.time()
        ll  = compute_loglik(idata, error, g_type)   # (n_c, n_d, N_OBS)
        print(f"  Done in {time.time()-t1:.1f}s")

        import xarray as xr
        # arviz 1.1.0: InferenceData is DataTree — assign group directly
        ll_ds = xr.Dataset(
            {"obs": xr.DataArray(ll, dims=["chain","draw","obs_id"])}
        )
        try:
            idata["log_likelihood"] = ll_ds          # DataTree assignment
        except Exception:
            idata.log_likelihood = ll_ds             # older arviz fallback

        try:
            loo = az.loo(idata, var_name="obs")
            print(f"\n  LOO elpd={loo.elpd_loo:.2f}  (se={loo.se:.2f})  p_loo={loo.p_loo:.2f}")
        except Exception as e:
            # Fallback: report mean LL × N as ELPD proxy
            elpd_proxy = float(ll.mean(axis=(0,1)).sum())
            print(f"\n  LOO failed ({e})")
            print(f"  ELPD proxy (mean LL sum) = {elpd_proxy:.2f}")
            loo = type("L", (), {"elpd_loo": elpd_proxy, "se": float("nan"),
                                  "p_loo": float("nan")})()

        all_res[name] = dict(idata=idata, summ=summ, loo=loo)

    # ── Final comparison ──────────────────────────────────────────────
    print(f"\n\n{'='*65}")
    print("  FINAL COMPARISON")
    print(f"{'='*65}")
    print(f"  {'Model':<22} {'b0':>8} {'b1':>8} {'sigma':>7} {'Rhat_max':>9} {'ELPD':>10}")
    print(f"  {'-'*67}")
    for name, res in all_res.items():
        s  = res["summ"]
        pm_ = s["mean"].values
        rh  = s["r_hat"].max()
        elpd = res["loo"].elpd_loo if res["loo"] else float("nan")
        print(f"  {name:<22} {pm_[0]:>8.3f} {pm_[1]:>8.3f} {pm_[2]:>7.4f} {rh:>9.4f} {elpd:>10.2f}")

    print("\nDone.")
