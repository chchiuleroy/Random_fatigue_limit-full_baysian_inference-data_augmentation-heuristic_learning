"""Does y-ASSE (H0/H1) keep falling as quadrature nodes grow?  Two implementations, same theta_hat (Roy's DA+NCP SEV+LogNormal
fit, README 5.4):  (a) 'retired': Delta-space Gauss-Legendre on (0,S) with the (untruncated) LogNormal density as weight,
the construction of the removed rfl_hl_asse.py;  (b) 'current': probability-integral-transform nodes of yasse.py.
    python node_convergence.py
"""
import os, json, numpy as np
from scipy import stats, special
import yasse as Y

HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, "results", "roy_da_thetas.json")))["SEV"]["theta"]   # theta_hat of rfl_asse_full_sample.py (unchanged), see README 5.4
th = dict(b0=d["beta0"], b1=d["beta1"], sigma=d["sigma"], mu_d=d["mu_d"], sd_d=d["sigma_d"]); kind = "sev"
data = Y.failure_data(True)


def retired(n, h):
    X, W = np.polynomial.legendre.leggauss(n); tot = 0.0
    for s, y in data:
        dd = s / 2 * (X + 1); g = stats.norm.pdf(np.log(dd), th["mu_d"], th["sd_d"]) / dd
        w = W * g * (s / 2); wsum = w.sum(); w = w / wsum
        L = np.log(s - dd); E1 = (w * L).sum(); V1 = max((w * L * L).sum() - E1 ** 2, 0.0)
        em, ev = Y.err_moments(kind, th)
        E = th["b0"] + th["b1"] * E1 + em; V = th["b1"] ** 2 * V1 + ev
        if h == "H0":
            z = (y - E) / np.sqrt(V)
        else:
            mu = th["b0"] + th["b1"] * L
            c = 1 - np.exp(-np.exp(np.clip((y[:, None] - mu[None, :]) / th["sigma"], -500, 20)))
            z = special.ndtri(np.clip((c * w[None, :]).sum(1), 1e-6, 1 - 1e-6))
        dh = np.exp(th["mu_d"] + th["sd_d"] * np.clip(z, -6, 6))
        tot += np.abs(y - Y.yhat_from_delta(kind, th, s, dh)).sum()
    return tot, wsum


def current(n, h):
    X, W = np.polynomial.legendre.leggauss(n); Y._X, Y._W = X, W / 2.0
    return Y.asse(kind, "lognormal", th, data, h)[0]


print(f"{'nodes':>6s} | retired H0   H1   (sum of weights at S=0.95 before normalising) | current H0   H1")
for n in [16, 32, 64, 100, 200, 400, 800, 1600, 3200]:
    r0, ws = retired(n, "H0"); r1, _ = retired(n, "H1")
    print(f"{n:6d} | {r0:9.3f} {r1:9.3f}   {ws:10.3e} | {current(n,'H0'):9.3f} {current(n,'H1'):9.3f}")
