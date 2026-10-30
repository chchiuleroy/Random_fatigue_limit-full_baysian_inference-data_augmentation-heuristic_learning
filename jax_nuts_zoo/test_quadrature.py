"""Quadrature-accuracy checks behind the numerical-accuracy notes in README sections 3.2 (Step 3) and 5.5.
Run from this folder:  python test_quadrature.py   (needs results/full__{sev,normal}__lognormal.npz for part 4)."""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
from scipy import integrate, stats
import jax.numpy as jnp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rfl_zoo as Z
import yasse as Y

# ---- (1) Weibull-limit log-marginal vs adaptive quadrature ------------------------------------------------------
S = np.array([0.675, 0.75, 0.9, 0.95]); Yv = np.array([6.5, 3.1, -1.0, -2.6]); EV = np.ones(4, int)
worst = 0.0
for kind in ["normal", "sev", "t"]:
    for k, lam in [(25.0, 0.55), (8.0, 0.5), (60.0, 0.6), (3.0, 0.4)]:
        p = dict(beta0=-9.3, beta1=-8.4, log_sigma=np.log(0.3), log_k_d=np.log(k), log_lam_d=np.log(lam))
        if kind == "t":
            p["log_nu"] = np.log(5.0)
        mine = np.asarray(Z.loglik_obs({a: jnp.asarray(b) for a, b in p.items()}, jnp.asarray(S), jnp.asarray(Yv), jnp.asarray(EV), kind, "weibull", False))
        for s, y, m in zip(S, Yv, mine):
            def f(d):
                z = (y - (p["beta0"] + p["beta1"] * np.log(s - d))) / np.exp(p["log_sigma"]); sg = np.exp(p["log_sigma"])
                dens = {"normal": lambda: stats.norm.pdf(z) / sg, "sev": lambda: np.exp(z - np.exp(z)) / sg, "t": lambda: stats.t.pdf(z, 5.0) / sg}[kind]()
                return dens * stats.weibull_min.pdf(d, k, scale=lam)
            ref = np.log(max(integrate.quad(f, 0, s - 1e-12, limit=400, points=[min(lam, s * 0.99)])[0], 1e-300))
            worst = max(worst, abs(m - ref))
print(f"(1) Weibull-limit log-marginal vs scipy.quad, worst abs diff (3 errors x 4 shapes x 4 stresses): {worst:.2e}")

# ---- (2) Gamma-limit Gauss-Legendre: 24 nodes (rfl_pymc_ncp.py) vs 400 nodes vs quad ----------------------------------
for alpha, beta_r in [(50.0, 100.0), (40.0, 60.0), (150.0, 250.0)]:
    p = dict(beta0=-9.3, beta1=-8.4, log_sigma=np.log(0.3), log_alpha_d=np.log(alpha), log_beta_r=np.log(beta_r))
    s, y = 0.9, -1.0
    def f(d):
        z = (y - (p["beta0"] + p["beta1"] * np.log(s - d))) / 0.3
        return np.exp(z - np.exp(z)) / 0.3 * stats.gamma.pdf(d, alpha, scale=1 / beta_r)
    ref = np.log(integrate.quad(f, 0, s - 1e-12, limit=400, points=[alpha / beta_r])[0])
    out = {}
    for n in (24, 400):
        Z._GL_X, Z._GL_W = np.polynomial.legendre.leggauss(n)
        out[n] = float(Z.loglik_obs({a: jnp.asarray(b) for a, b in p.items()}, jnp.asarray([s]), jnp.asarray([y]), jnp.asarray([1]), "sev", "gamma", False)[0])
    print(f"(2) Gamma(alpha={alpha:g}, rate={beta_r:g}) mean {alpha/beta_r:.3f}, sd {np.sqrt(alpha)/beta_r:.3f}: log-marginal error GL-24 {abs(out[24]-ref):.2e}, GL-400 {abs(out[400]-ref):.2e}")
Z._GL_X, Z._GL_W = np.polynomial.legendre.leggauss(Z.N_GL)

# ---- (3) LogNormal-limit marginal CDF: Gauss-Hermite node count vs adaptive quadrature -------------------------------
b0, b1, sig, muD, sdD = -9.074687, -7.602654, np.sqrt(0.0312110845352731), -0.5952965, 0.0345517
def exact(om, s):
    f = lambda z: stats.norm.pdf(z) * stats.norm.cdf((om - (b0 + b1 * np.log(s - np.exp(muD + sdD * z)))) / sig) if np.exp(muD + sdD * z) < s else 0.0
    return integrate.quad(f, -12, min(12, (np.log(s) - muD) / sdD), limit=400, epsabs=1e-12)[0]
def gh(om, s, n):
    z, w = np.polynomial.hermite_e.hermegauss(n); d = np.exp(muD + sdD * z); v = d < s
    return float((w[v] / np.sqrt(2 * np.pi) * stats.norm.cdf((om - (b0 + b1 * np.log(s - d[v]))) / sig)).sum())
for s, om in [(0.675, 7.0), (0.75, 2.5)]:
    ex = exact(om, s)
    print(f"(3) S={s}, omega={om}: CDF error GH-20 {abs(gh(om,s,20)-ex):.1e} | GH-40 {abs(gh(om,s,40)-ex):.1e} | GH-80 {abs(gh(om,s,80)-ex):.1e} | GH-200 {abs(gh(om,s,200)-ex):.1e}")

# ---- (4) retired H0 / H1 (Delta-space Gauss-Legendre, as in the removed rfl_hl_asse.py) vs node count ------------------
# Self-contained re-implementation: moments and the exact marginal CDF are computed by n-node Gauss-Legendre on (0, S)
# with the (untruncated) LogNormal density as weight, normalised by its sum -- the construction of the retired scripts.
# 73 non-tied failures, posterior-mean theta from results/full__{sev,normal}__lognormal.npz (censored-maxima protocol).
from scipy import special
def retired_asse(th, kind, n, heuristic):
    X, W = np.polynomial.legendre.leggauss(n)
    tot = 0.0
    for s, y in Y.failure_data(all_failures=False):
        d = s / 2 * (X + 1); g = stats.norm.pdf(np.log(d), th["mu_d"], th["sd_d"]) / d
        w = W * g * (s / 2); w = w / w.sum()
        L = np.log(s - d); E1 = (w * L).sum(); V1 = max((w * L * L).sum() - E1 ** 2, 0.0)
        em, ev = Y.err_moments(kind, th)
        E = th["b0"] + th["b1"] * E1 + em; V = th["b1"] ** 2 * V1 + ev
        if heuristic == "H0":
            z = (y - E) / np.sqrt(V)
        else:  # H1: exact marginal CDF -> standard-normal quantile -> Delta_hat
            mu = th["b0"] + th["b1"] * L
            zz = (y[:, None] - mu[None, :]) / th["sigma"]
            c = 1 - np.exp(-np.exp(np.clip(zz, -500, 20))) if kind == "sev" else special.ndtr(zz)
            z = special.ndtri(np.clip((c * w[None, :]).sum(1), 1e-6, 1 - 1e-6))
        dh = np.exp(th["mu_d"] + th["sd_d"] * np.clip(z, -6, 6))
        tot += np.abs(y - Y.yhat_from_delta(kind, th, s, dh)).sum()
    return tot
for kind in ["sev", "normal"]:
    f = os.path.join(HERE, "results", f"full__{kind}__lognormal.npz")
    if not os.path.exists(f):
        print("(4) skipped, missing", f); continue
    th = Y.plugin_from_npz(np.load(f), kind, "lognormal")
    print(f"(4) {kind}+LogNormal (H0, H1) by Gauss-Legendre node count: " +
          ", ".join(f"{n}: ({retired_asse(th, kind, n, 'H0'):.3f}, {retired_asse(th, kind, n, 'H1'):.3f})" for n in (32, 100, 400)))
