"""Roy's y-ASSE ("model z-score ASSE", rfl_model_zscore_asse.py / rfl_hl_asse.py), generalised to the 9-model zoo.

For plug-in parameters theta_hat (posterior means of a full-sample fit), per stress level S_j and failure y_ij:
    E(lnY|S_j), V(lnY|S_j)  from the model  (Delta ~ g truncated to (0,S_j);  error moments added)
    z_ij  = (y_ij - E) / sqrt(V)                      model z-score
    H0:   p = Phi(z);   Delta_hat = G^{-1}(p)  [= exp(mu_d + sd_d z) for LogNormal];  y_hat = b0 + b1 ln(S_j - Delta_hat) + e_shift
    ASSE  = sum_{i,j} | y_ij - y_hat_ij |             (n = 75 with all_failures=True, the README section 5.4 protocol; n = 73 excludes the two tied maxima)
Heuristics of the HL loop implemented for every model: H0 (Phi(z)), H1 (exact marginal CDF), H3 (per-group z calibration,
sd floor 0.5). H2 (Cornish-Fisher) is SEV-specific and not implemented; H4 (posterior-mean Delta_i given y_i) was dropped
because it reconstructs each y_i from its own value (in-sample).
Error moments: normal (0, s^2); sev (-s*gamma_E, s^2 pi^2/6); t (0, s^2 nu/(nu-2); undefined for nu <= 2, raises).
"""
import numpy as np
from scipy import stats, special

EULER = 0.5772156649015329
_X, _W = np.polynomial.legendre.leggauss(200)
_W = _W / 2.0


def g_cdf(gt, th, d):
    if gt == "lognormal":
        return stats.norm.cdf((np.log(np.maximum(d, 1e-300)) - th["mu_d"]) / th["sd_d"])
    if gt == "gamma":
        return stats.gamma.cdf(d, th["alpha"], scale=1.0 / th["beta"])
    return -np.expm1(-(d / th["lam"]) ** th["k"])


def g_ppf(gt, th, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    if gt == "lognormal":
        return np.exp(th["mu_d"] + th["sd_d"] * special.ndtri(p))
    if gt == "gamma":
        return stats.gamma.ppf(p, th["alpha"], scale=1.0 / th["beta"])
    return th["lam"] * (-np.log1p(-p)) ** (1.0 / th["k"])


def err_moments(kind, th):
    s = th["sigma"]
    if kind == "normal":
        return 0.0, s * s
    if kind == "sev":
        return -s * EULER, s * s * np.pi ** 2 / 6.0
    nu = th["nu"]
    if nu <= 2.0:
        raise ValueError("Student-t variance is infinite for nu <= 2: plug-in nu = %.3f" % nu)
    return 0.0, s * s * nu / (nu - 2.0)


def nodes(gt, th, s):
    """PIT nodes on the truncated limit distribution: Delta_k, equal weights (sum 1)."""
    uS = float(g_cdf(gt, th, s))
    u = uS * (_X + 1.0) / 2.0
    return g_ppf(gt, th, u), _W.copy()


def EV(kind, gt, th, s):
    d, w = nodes(gt, th, s)
    L = np.log(np.maximum(s - d, 1e-12))
    E1 = float((w * L).sum()); Var = max(float((w * L * L).sum()) - E1 * E1, 0.0)
    em, ev = err_moments(kind, th)
    return th["b0"] + th["b1"] * E1 + em, max(th["b1"] ** 2 * Var + ev, 1e-10)


def yhat_from_delta(kind, th, s, dhat):
    em, _ = err_moments(kind, th)
    shift = em if kind == "sev" else 0.0
    return th["b0"] + th["b1"] * np.log(s - np.clip(dhat, 1e-6, 0.999 * s)) + shift


def marg_cdf_at(kind, gt, th, s, y):
    d, w = nodes(gt, th, s)
    mu = th["b0"] + th["b1"] * np.log(np.maximum(s - d, 1e-12))
    z = (np.asarray(y)[:, None] - mu[None, :]) / th["sigma"]
    if kind == "normal":
        c = special.ndtr(z)
    elif kind == "sev":
        c = 1.0 - np.exp(-np.exp(np.clip(z, -500, 20)))
    else:
        c = special.stdtr(th["nu"], z)
    return (c * w[None, :]).sum(1)


def asse(kind, gt, th, data, heuristic="H0"):
    """data: list of (s, y_array_of_failures)."""
    tot, per = 0.0, {}
    for s, y in data:
        E, V = EV(kind, gt, th, s)
        z = (y - E) / np.sqrt(V)
        if heuristic == "H0":
            dh = g_ppf(gt, th, special.ndtr(np.clip(z, -6, 6)))
        elif heuristic == "H1":
            dh = g_ppf(gt, th, marg_cdf_at(kind, gt, th, s, y))
        elif heuristic == "H3":
            sd = max(z.std(ddof=1), 0.5)
            dh = g_ppf(gt, th, special.ndtr(np.clip((z - z.mean()) / sd, -6, 6)))
        else:
            raise ValueError(heuristic)
        a = float(np.abs(y - yhat_from_delta(kind, th, s, dh)).sum())
        per[s] = a; tot += a
    return tot, per


def plugin_from_npz(npz, kind, gt):
    p = {k: np.asarray(npz[k]) for k in npz.files}
    th = {"b0": float(p["beta0"].mean()), "b1": float(p["beta1"].mean()), "sigma": float(np.exp(p["log_sigma"]).mean())}
    if gt == "lognormal":
        th.update(mu_d=float(p["mu_d"].mean()), sd_d=float(np.exp(p["log_sigma_d"]).mean()))
    elif gt == "gamma":
        th.update(alpha=float(np.exp(p["log_alpha_d"]).mean()), beta=float(np.exp(p["log_beta_r"]).mean()))
    else:
        th.update(k=float(np.exp(p["log_k_d"]).mean()), lam=float(np.exp(p["log_lam_d"]).mean()))
    if kind == "t":
        th["nu"] = float(np.exp(p["log_nu"]).mean())
    return th


def failure_data(all_failures=True):
    """[(s, ln N)] used by the ASSE sums; all_failures=True is the README section 5.4 protocol (n = 75)."""
    import rfl_data
    return rfl_data.groups(all_failures)
