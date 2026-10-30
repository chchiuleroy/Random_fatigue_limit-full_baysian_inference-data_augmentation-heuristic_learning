"""Verify the JAX RFL zoo against the numpy reference likelihood in ../rfl_pymc_ncp.py and check gradients.
Run from this folder:  python test_zoo.py   (needs PyMC only because the reference module imports it)."""
import sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import rfl_pymc_ncp as R                      # reference likelihood (numpy), repo root
sys.path.insert(0, HERE)
import rfl_zoo as Z
import jax, jax.numpy as jnp
# Roy's reference uses 20 Gauss-Hermite / 24 Gauss-Legendre nodes; compare like with like (the zoo's defaults are 200/400)
_GH_BAK = (Z._GH_Z, Z._GH_W, Z._GL_X, Z._GL_W)
Z._GH_Z, Z._GH_W = np.polynomial.hermite_e.hermegauss(20)
Z._GL_X, Z._GL_W = np.polynomial.legendre.leggauss(24)
from scipy import stats

S, Y, EV = R.S_OBS, R.Y_OBS, R.EVENT
print("n =", len(S), " censored =", int((EV == 0).sum()))
rng = np.random.default_rng(0)
fails = 0


def total(p, kind, g):
    return float(Z.loglik_obs({k: jnp.asarray(v) for k, v in p.items()}, jnp.asarray(S), jnp.asarray(Y),
                              jnp.asarray(EV), kind, g).sum())


for kind in ["normal", "sev"]:
    for g in ["lognormal", "gamma"]:
        diffs = []
        for _ in range(6):
            b0, b1 = rng.uniform(-12, -7), rng.uniform(-10, -6)
            ls = np.log(rng.uniform(0.15, 0.7))
            if g == "lognormal":
                mu_d, lsd = rng.uniform(-1.2, -0.5), np.log(rng.uniform(0.02, 0.4))
                ref = float(R.loglik_lognormal_ncp(b0, b1, ls, mu_d, lsd, kind))
                mine = total(dict(beta0=b0, beta1=b1, log_sigma=ls, mu_d=mu_d, log_sigma_d=lsd), kind, g)
            else:
                la, lb = np.log(rng.uniform(2, 20)), np.log(rng.uniform(20, 150))
                ref = float(R.loglik_gamma_ncp(b0, b1, ls, la, lb, kind))
                mine = total(dict(beta0=b0, beta1=b1, log_sigma=ls, log_alpha_d=la, log_beta_r=lb), kind, g)
            diffs.append(abs(ref - mine))
        ok = max(diffs) < 1e-6
        fails += (not ok)
        print(f"{kind:6s} x {g:9s}: max|ref - mine| over 6 draws = {max(diffs):.2e}  {'PASS' if ok else 'FAIL'}")

# Student-t sanity: sf matches scipy; nu -> large approaches normal; gradient vs finite difference
z = np.array([-3.0, -0.5, 0.0, 0.7, 2.5]); nu = 4.3
print("t_sf vs scipy max err:", float(np.abs(np.asarray(Z.t_sf(jnp.asarray(z), nu)) - stats.t.sf(z, nu)).max()))
p0 = dict(beta0=-9.0, beta1=-8.0, log_sigma=np.log(0.4), mu_d=-0.7, log_sigma_d=np.log(0.1), log_nu=np.log(1e5))
d_t = total(p0, "t", "lognormal"); d_n = total({k: v for k, v in p0.items() if k != "log_nu"}, "normal", "lognormal")
print(f"t(nu=1e5) vs normal total loglik: {d_t:.6f} vs {d_n:.6f}  diff={abs(d_t-d_n):.2e}")
for g, base in [("lognormal", dict(beta0=-9.0, beta1=-8.0, log_sigma=np.log(0.4), mu_d=-0.7, log_sigma_d=np.log(0.1), log_nu=np.log(5.0))),
                ("gamma", dict(beta0=-9.0, beta1=-8.0, log_sigma=np.log(0.4), log_alpha_d=np.log(8.0), log_beta_r=np.log(80.0), log_nu=np.log(5.0)))]:
    f = lambda pv: Z.loglik_obs(dict(zip(base.keys(), pv)), jnp.asarray(S), jnp.asarray(Y), jnp.asarray(EV), "t", g).sum()
    v0 = jnp.asarray(list(base.values()))
    grad = np.asarray(jax.grad(f)(v0))
    fd = np.array([(float(f(v0.at[i].add(1e-5))) - float(f(v0.at[i].add(-1e-5)))) / 2e-5 for i in range(len(v0))])
    rel = np.abs(grad - fd) / np.maximum(1.0, np.abs(fd))
    print(f"grad check t x {g}: max rel err {rel.max():.2e}  (incl. log_nu through censored sf)")
    fails += rel.max() > 1e-3
# gamma-limit censored term gradient wrt alpha
base = dict(beta0=-9.0, beta1=-8.0, log_sigma=np.log(0.4), log_alpha_d=np.log(8.0), log_beta_r=np.log(80.0))
f = lambda pv: Z.loglik_obs(dict(zip(base.keys(), pv)), jnp.asarray(S), jnp.asarray(Y), jnp.asarray(EV), "sev", "gamma").sum()
v0 = jnp.asarray(list(base.values())); grad = np.asarray(jax.grad(f)(v0))
fd = np.array([(float(f(v0.at[i].add(1e-5))) - float(f(v0.at[i].add(-1e-5)))) / 2e-5 for i in range(len(v0))])
print("grad check sev x gamma (censored term via gammaincc):", f"max rel err {(np.abs(grad-fd)/np.maximum(1,np.abs(fd))).max():.2e}")
# has_cens=False path equals the general path when nothing is censored
ones = jnp.ones_like(jnp.asarray(EV))
pp = dict(beta0=-9.0, beta1=-8.0, log_sigma=np.log(0.4), mu_d=-0.7, log_sigma_d=np.log(0.1), log_nu=np.log(5.0))
for kind in ["normal", "sev", "t"]:
    pk = {k: jnp.asarray(v) for k, v in pp.items() if kind == "t" or k != "log_nu"}
    a = float(Z.loglik_obs(pk, jnp.asarray(S), jnp.asarray(Y), ones, kind, "lognormal", True).sum())
    b = float(Z.loglik_obs(pk, jnp.asarray(S), jnp.asarray(Y), ones, kind, "lognormal", False).sum())
    print(f"has_cens skip, {kind}: {a:.9f} vs {b:.9f}  diff {abs(a-b):.1e}")
    fails += abs(a - b) > 1e-9
print("\nFAILS:", fails)
