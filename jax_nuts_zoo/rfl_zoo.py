"""RFL model zoo in JAX / NumPyro (full Bayes, NUTS).

Random-fatigue-limit family (Pascual & Meeker 1999 structure, same likelihood layout as rfl_pymc_ncp.py):
    ln Y_i | Delta_i = beta0 + beta1 * ln(S_i - Delta_i) + sigma * eps_i ,   Delta_i < S_i
    error eps      : "normal" | "sev" (smallest extreme value) | "t" (Student-t, nu estimated)
    fatigue limit  : "lognormal" (Gauss-Hermite in z-space, 200 nodes) | "gamma" (Gauss-Legendre on (0,S_i), 400 nodes)
Failures contribute the Delta-marginal density; right-censored runouts contribute the Delta-marginal survival
plus P(Delta >= S_i) (never fails), exactly as rfl_pymc_ncp.py. Priors: the same Uniform bounds as that file
(log_nu ~ U(log 1.5, log 100) for the t error). Gradients of betainc / gammaincc w.r.t. the shape arguments are not
available in JAX, so those two (used only for censored observations) get a central-difference custom JVP.
"""
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp, log_ndtr, gammaln, betainc, gammaincc
import numpyro
import numpyro.distributions as dist

jax.config.update("jax_enable_x64", True)

N_GH, N_GL = 200, 400   # fixed-node rules; 20/24 (rfl_pymc_ncp.py) are too coarse: see test_quadrature.py
_GH_Z, _GH_W = np.polynomial.hermite_e.hermegauss(N_GH)
_GL_X, _GL_W = np.polynomial.legendre.leggauss(N_GL)
LOG_SQRT2PI = 0.5 * np.log(2 * np.pi)


# ---- special functions with finite-difference JVP for the shape argument ---------------------------------
def _fd_jvp(fun, arg_idx_fd, h_rel=1e-4):
    f = jax.custom_jvp(fun)

    @f.defjvp
    def _jvp(primals, tangents):
        out = fun(*primals)
        tan = jnp.zeros_like(out)
        for i, (p, t) in enumerate(zip(primals, tangents)):
            if i in arg_idx_fd:
                h = h_rel * jnp.maximum(jnp.abs(p), 1.0)
                d = (fun(*[q + h if j == i else q for j, q in enumerate(primals)])
                     - fun(*[q - h if j == i else q for j, q in enumerate(primals)])) / (2 * h)
                tan = tan + d * t
            else:  # exact derivative w.r.t. the evaluation point handled by autodiff of the primal fun
                _, dt = jax.jvp(lambda z: fun(*[z if j == i else q for j, q in enumerate(primals)]), (p,), (t,))
                tan = tan + dt
        return out, tan
    return f


def _gammaincc_raw(a, x):
    return gammaincc(a, x)


gammaincc_fd = _fd_jvp(_gammaincc_raw, {0})


def _t_sf_raw(z, nu):
    x = nu / (nu + z * z)
    half = 0.5 * betainc(0.5 * nu, 0.5, x)
    return jnp.where(z >= 0, half, 1.0 - half)


def _t_sf_jvp_wrapper():
    f = jax.custom_jvp(_t_sf_raw)

    @f.defjvp
    def _jvp(primals, tangents):
        z, nu = primals
        dz, dnu = tangents
        out = _t_sf_raw(z, nu)
        pdf = jnp.exp(gammaln(0.5 * (nu + 1)) - gammaln(0.5 * nu) - 0.5 * jnp.log(nu * jnp.pi)
                      - 0.5 * (nu + 1) * jnp.log1p(z * z / nu))
        h = 1e-4 * jnp.maximum(nu, 1.0)
        dsf_dnu = (_t_sf_raw(z, nu + h) - _t_sf_raw(z, nu - h)) / (2 * h)
        return out, -pdf * dz + dsf_dnu * dnu
    return f


t_sf = _t_sf_jvp_wrapper()


# ---- error models: (log pdf, log survival) of the standardised residual ----------------------------------
def err_logs(kind, z, sig, nu=None):
    if kind == "normal":
        return -LOG_SQRT2PI - jnp.log(sig) - 0.5 * z * z, log_ndtr(-z)
    if kind == "sev":
        ez = jnp.exp(jnp.clip(z, -500.0, 20.0))
        return -jnp.log(sig) + z - ez, -ez
    if kind == "t":
        lp = (gammaln(0.5 * (nu + 1)) - gammaln(0.5 * nu) - 0.5 * jnp.log(nu * jnp.pi)
              - 0.5 * (nu + 1) * jnp.log1p(z * z / nu) - jnp.log(sig))
        return lp, jnp.log(jnp.maximum(t_sf(z, nu), 1e-300))
    raise ValueError(kind)


def _lf(kind, z, sig, nu, ev, has_cens):
    """Failure density, or survival for censored rows; the survival branch is skipped when nothing is censored."""
    lf_fail, lf_cens = err_logs(kind, z, sig, nu) if has_cens else (err_logs(kind, z, sig, nu)[0], None)
    if not has_cens:
        return lf_fail
    return jnp.where((ev == 1)[:, None], lf_fail, lf_cens)


def loglik_obs(p, S, Y, ev, kind, gtype, has_cens=True):
    """Per-observation log marginal likelihood. p: dict of unconstrained-bounded parameters."""
    sig = jnp.exp(p["log_sigma"])
    nu = jnp.exp(p["log_nu"]) if kind == "t" else None
    S2, Y2 = S[:, None], Y[:, None]
    if gtype == "lognormal":
        mu_d, sd_d = p["mu_d"], jnp.exp(p["log_sigma_d"])
        delta = jnp.exp(mu_d + sd_d * jnp.asarray(_GH_Z)[None, :])
        valid = delta < S2
        d_safe = jnp.where(valid, delta, 1e-10)
        mu_c = p["beta0"] + p["beta1"] * jnp.log(jnp.maximum(S2 - d_safe, 1e-10))
        lf = _lf(kind, (Y2 - mu_c) / sig, sig, nu, ev, has_cens)
        lterm = jnp.where(valid, lf + jnp.log(jnp.asarray(_GH_W))[None, :], -jnp.inf)
        lint = logsumexp(lterm, axis=1) - LOG_SQRT2PI
        # P(Delta >= S): 0.5 erfc((ln S - mu_d)/(sqrt2 sd_d))
        p_above = 0.5 * jax.scipy.special.erfc((jnp.log(S) - mu_d) / (jnp.sqrt(2.0) * sd_d))
    elif gtype == "weibull":
        # Delta ~ Weibull(shape k, scale lam). PIT quadrature: int_0^S f(y|D) g(D) dD = int_0^{u_S} f(y|G^-1(u)) du,
        # u = G(D); GL nodes on (0,u_S) -> accuracy does not depend on how tight g is.
        k, lam = jnp.exp(p["log_k_d"]), jnp.exp(p["log_lam_d"])
        uS = -jnp.expm1(-(S / lam) ** k)                       # P(Delta < S), (n,)
        u = uS[:, None] / 2.0 * (jnp.asarray(_GL_X)[None, :] + 1.0)
        delta = lam * (-jnp.log1p(-u)) ** (1.0 / k)
        mu_c = p["beta0"] + p["beta1"] * jnp.log(jnp.maximum(S2 - delta, 1e-10))
        lf = _lf(kind, (Y2 - mu_c) / sig, sig, nu, ev, has_cens)
        lint = logsumexp(lf + jnp.log(jnp.asarray(_GL_W))[None, :], axis=1) + jnp.log(uS / 2.0)
        p_above = jnp.exp(-(S / lam) ** k)
    else:
        alpha, beta_r = jnp.exp(p["log_alpha_d"]), jnp.exp(p["log_beta_r"])
        D = S2 / 2.0 * (jnp.asarray(_GL_X)[None, :] + 1.0)
        jac = (S / 2.0)[:, None]
        mu_c = p["beta0"] + p["beta1"] * jnp.log(jnp.maximum(S2 - D, 1e-10))
        lf = _lf(kind, (Y2 - mu_c) / sig, sig, nu, ev, has_cens)
        lg = ((alpha - 1) * jnp.log(jnp.maximum(D, 1e-300)) + alpha * jnp.log(beta_r) - beta_r * D - gammaln(alpha))
        lint = logsumexp(lf + lg + jnp.log(jnp.asarray(_GL_W))[None, :] + jnp.log(jac), axis=1)
        p_above = gammaincc_fd(alpha, beta_r * S)
    if not has_cens:
        return lint
    lint_c = jnp.logaddexp(lint, jnp.log(jnp.maximum(p_above, 1e-300)))
    return jnp.where(ev == 1, lint, lint_c)


PARAM_NAMES = {"lognormal": ["beta0", "beta1", "log_sigma", "mu_d", "log_sigma_d"],
               "gamma": ["beta0", "beta1", "log_sigma", "log_alpha_d", "log_beta_r"],
               "weibull": ["beta0", "beta1", "log_sigma", "log_k_d", "log_lam_d"]}


def make_model(S, Y, ev, kind, gtype, upper_mu_d):
    has_cens = bool(np.any(np.asarray(ev) == 0))
    S, Y, ev = jnp.asarray(S), jnp.asarray(Y), jnp.asarray(ev)

    def model():
        p = {"beta0": numpyro.sample("beta0", dist.Uniform(-50.0, 50.0)),
             "beta1": numpyro.sample("beta1", dist.Uniform(-30.0, 0.0)),
             "log_sigma": numpyro.sample("log_sigma", dist.Uniform(np.log(0.01), np.log(5.0)))}
        if gtype == "lognormal":
            p["mu_d"] = numpyro.sample("mu_d", dist.Uniform(-5.0, upper_mu_d))
            p["log_sigma_d"] = numpyro.sample("log_sigma_d", dist.Uniform(np.log(0.001), np.log(2.0)))
        elif gtype == "weibull":
            p["log_k_d"] = numpyro.sample("log_k_d", dist.Uniform(np.log(1.0), np.log(100.0)))
            p["log_lam_d"] = numpyro.sample("log_lam_d", dist.Uniform(np.log(0.05), np.log(2.0)))
        else:
            p["log_alpha_d"] = numpyro.sample("log_alpha_d", dist.Uniform(np.log(0.1), np.log(50.0)))
            p["log_beta_r"] = numpyro.sample("log_beta_r", dist.Uniform(np.log(0.1), np.log(200.0)))
        if kind == "t":
            p["log_nu"] = numpyro.sample("log_nu", dist.Uniform(np.log(1.5), np.log(100.0)))
        numpyro.factor("loglik", loglik_obs(p, S, Y, ev, kind, gtype, has_cens).sum())
    return model


# ---- posterior predictive for a NEW specimen at stress s (conditional on failure) ------------------------
def _err_cdf(kind, z, nu):
    from scipy import special
    if kind == "normal":
        return special.ndtr(z)
    if kind == "sev":
        return 1.0 - np.exp(-np.exp(np.clip(z, -500, 20)))
    return special.stdtr(nu, z)


def _err_logpdf(kind, z, sig, nu):
    from scipy import special
    if kind == "normal":
        return -LOG_SQRT2PI - np.log(sig) - 0.5 * z * z
    if kind == "sev":
        return -np.log(sig) + z - np.exp(np.clip(z, -500, 20))
    return (special.gammaln(0.5 * (nu + 1)) - special.gammaln(0.5 * nu) - 0.5 * np.log(nu * np.pi)
            - 0.5 * (nu + 1) * np.log1p(z * z / nu) - np.log(sig))


def predictive_components(post, s, kind, gtype):
    """Return weights (M,J), locations mu (M,J), sig (M,), nu (M,) for M posterior draws."""
    from scipy import special
    b0, b1, sig = post["beta0"], post["beta1"], np.exp(post["log_sigma"])
    nu = np.exp(post["log_nu"]) if kind == "t" else np.full_like(b0, np.inf)
    if gtype == "lognormal":
        mu_d, sd_d = post["mu_d"], np.exp(post["log_sigma_d"])
        delta = np.exp(mu_d[:, None] + sd_d[:, None] * _GH_Z[None, :])
        w = np.where(delta < s, _GH_W[None, :], 0.0)
    elif gtype == "weibull":
        k, lam = np.exp(post["log_k_d"]), np.exp(post["log_lam_d"])
        uS = -np.expm1(-(s / lam) ** k)
        u = uS[:, None] / 2.0 * (_GL_X[None, :] + 1.0)
        delta = lam[:, None] * (-np.log1p(-u)) ** (1.0 / k[:, None])
        w = _GL_W[None, :] * np.ones_like(delta)
    else:
        alpha, beta_r = np.exp(post["log_alpha_d"]), np.exp(post["log_beta_r"])
        delta = s / 2.0 * (_GL_X[None, :] + 1.0) * np.ones_like(alpha)[:, None]
        lg = ((alpha[:, None] - 1) * np.log(np.maximum(delta, 1e-300)) + alpha[:, None] * np.log(beta_r[:, None])
              - beta_r[:, None] * delta - special.gammaln(alpha[:, None]))
        w = _GL_W[None, :] * np.exp(lg)
    w = w / np.maximum(w.sum(1, keepdims=True), 1e-300)
    mu = b0[:, None] + b1[:, None] * np.log(np.maximum(s - np.minimum(delta, s - 1e-10), 1e-10))
    return w, mu, sig, nu


def predictive_logpdf(post, s, y, kind, gtype):
    w, mu, sig, nu = predictive_components(post, s, kind, gtype)
    y = np.atleast_1d(y)
    z = (y[None, :, None] - mu[:, None, :]) / sig[:, None, None]
    lp = _err_logpdf(kind, z, sig[:, None, None], nu[:, None, None]) + np.log(np.maximum(w, 1e-300))[:, None, :]
    mx = lp.max(2, keepdims=True)
    per_draw = (mx[..., 0] + np.log(np.exp(lp - mx).sum(2)))           # (M, n_y)
    m2 = per_draw.max(0, keepdims=True)
    return m2[0] + np.log(np.exp(per_draw - m2).mean(0))               # log mean over draws


def predictive_cdf(post, s, grid, kind, gtype):
    w, mu, sig, nu = predictive_components(post, s, kind, gtype)
    z = (grid[None, :, None] - mu[:, None, :]) / sig[:, None, None]
    c = _err_cdf(kind, z, nu[:, None, None]) if kind == "t" else _err_cdf(kind, z, None)
    return (c * w[:, None, :]).sum(2).mean(0)
