"""Simulation check of what y-ASSE (H0 / H1 / H3) rewards.  Data are simulated from a KNOWN structure with parameters
taken from a real-data fit, all 9 structures are refitted by maximum likelihood (box-constrained to the NUTS prior
ranges, so MLE = MAP under those flat priors) and scored.

    python sim_study.py select <truth_error> <truth_limit> [reps]      # test A: which structure does each score pick?
    python sim_study.py noise  <truth_error> <truth_limit> [reps]      # test B: correct structure, sigma multiplied by 0.5/1/2

Test A output: results/sim_select_<truth>.jsonl, one line per replication with, for each of the 9 fitted structures,
max log-likelihood, AIC and y-ASSE H0/H1/H3 (plug-in = MLE).  Test B: results/sim_noise_<truth>.jsonl.
MLE rather than posterior means keeps 9 structures x many replications affordable; the question is about the
behaviour of the score, not about Bayesian estimation.
"""
import os, sys, json, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import jax, jax.numpy as jnp
from scipy.optimize import minimize
from scipy.special import ndtri

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rfl_zoo as Z, rfl_data, yasse as Y

KINDS, GTYPES = ["normal", "sev", "t"], ["lognormal", "gamma", "weibull"]
S_OBS, _, EV1 = rfl_data.arrays(True)
LEVELS = sorted(set(S_OBS))
BOUNDS = {"beta0": (-50, 50), "beta1": (-30, 0), "log_sigma": (np.log(0.01), np.log(5.0)), "mu_d": (-5, -0.45),
          "log_sigma_d": (np.log(0.001), np.log(2.0)), "log_k_d": (0.0, np.log(100.0)), "log_lam_d": (np.log(0.05), np.log(2.0)),
          "log_alpha_d": (np.log(0.1), np.log(50.0)), "log_beta_r": (np.log(0.1), np.log(200.0)), "log_nu": (np.log(2.5), np.log(100.0))}


def names(kind, gt):
    return Z.PARAM_NAMES[gt] + (["log_nu"] if kind == "t" else [])


def real_init(kind, gt):
    z = np.load(os.path.join(HERE, "results", f"full75__{kind}__{gt}.npz"))
    return {k: float(np.clip(z[k].mean(), *BOUNDS[k])) for k in names(kind, gt)}


_fns = {}


def nll_fn(kind, gt):
    if (kind, gt) not in _fns:
        nm = names(kind, gt)
        S, EV = jnp.asarray(S_OBS), jnp.asarray(EV1)

        def nll(v, Yv):
            return -Z.loglik_obs({k: v[i] for i, k in enumerate(nm)}, S, Yv, EV, kind, gt, False).sum()
        _fns[(kind, gt)] = jax.jit(jax.value_and_grad(nll))
    return _fns[(kind, gt)]


def fit_mle(kind, gt, Yv, rng, n_start=3):
    nm = names(kind, gt); f = nll_fn(kind, gt); Yj = jnp.asarray(Yv)
    base = real_init(kind, gt); best = None
    for s in range(n_start):
        x0 = np.array([base[k] for k in nm])
        if s > 0:
            lo, hi = np.array([BOUNDS[k][0] for k in nm]), np.array([BOUNDS[k][1] for k in nm])
            x0 = np.clip(x0 + rng.normal(0, 0.15, len(nm)) * np.maximum(1.0, np.abs(x0)), lo, hi)
        r = minimize(lambda v: tuple(np.asarray(a, dtype=float) for a in f(jnp.asarray(v), Yj)), x0, jac=True,
                     method="L-BFGS-B", bounds=[BOUNDS[k] for k in nm], options=dict(maxiter=400))
        if best is None or r.fun < best.fun:
            best = r
    return dict(zip(nm, best.x)), -float(best.fun)


def theta(kind, gt, p):
    th = {"b0": p["beta0"], "b1": p["beta1"], "sigma": float(np.exp(p["log_sigma"]))}
    if gt == "lognormal":
        th.update(mu_d=p["mu_d"], sd_d=float(np.exp(p["log_sigma_d"])))
    elif gt == "gamma":
        th.update(alpha=float(np.exp(p["log_alpha_d"])), beta=float(np.exp(p["log_beta_r"])))
    else:
        th.update(k=float(np.exp(p["log_k_d"])), lam=float(np.exp(p["log_lam_d"])))
    if kind == "t":
        th["nu"] = float(np.exp(p["log_nu"]))
    return th


def simulate(kind, gt, th, rng, sigma_mult=1.0):
    """One data set on the real stress design (15 failures per level): Delta ~ G truncated to (0,S), then the error."""
    out = []
    for s in S_OBS:
        while True:
            u = rng.uniform()
            d = float(Y.g_ppf(gt, th, u))
            if d < s:
                break
        if kind == "normal":
            e = rng.normal()
        elif kind == "sev":
            e = np.log(-np.log1p(-rng.uniform()))
        else:
            e = rng.standard_t(th["nu"])
        out.append(th["b0"] + th["b1"] * np.log(s - d) + sigma_mult * th["sigma"] * e)
    return np.array(out)


def groups(Yv):
    return [(s, np.sort(Yv[S_OBS == s])) for s in LEVELS]


def truth_theta(kind, gt):
    return theta(kind, gt, real_init(kind, gt))


def score_all(Yv, rng, only=None):
    res = {}
    gdata = groups(Yv)
    for kind in KINDS:
        for gt in GTYPES:
            if only and (kind, gt) not in only:
                continue
            try:
                p, ll = fit_mle(kind, gt, Yv, rng)
                th = theta(kind, gt, p)
                row = dict(ll=ll, aic=2 * len(p) - 2 * ll, sigma=th["sigma"])
                for h in ["H0", "H1", "H3"]:
                    row[h] = float(Y.asse(kind, gt, th, gdata, h)[0])
            except Exception as e:  # noqa
                row = dict(error=repr(e)[:120])
            res[f"{kind}/{gt}"] = row
    return res


if __name__ == "__main__":
    mode, tk, tg = sys.argv[1], sys.argv[2], sys.argv[3]
    reps = int(sys.argv[4]) if len(sys.argv) > 4 else 30
    th0 = truth_theta(tk, tg)
    path = os.path.join(HERE, "results", f"sim_{mode}_{tk}_{tg}.jsonl")
    done = sum(1 for _ in open(path, encoding="utf-8")) if os.path.exists(path) else 0
    print(f"{mode} truth {tk}/{tg} true theta {({k: round(v, 4) for k, v in th0.items()})}  resuming at rep {done}", flush=True)
    for r in range(done, reps):
        rng = np.random.default_rng(7000 + r)   # seed = replication index
        t0 = time.time()
        if mode == "select":
            Yv = simulate(tk, tg, th0, rng)
            rec = dict(rep=r, fits=score_all(Yv, rng))
        else:
            rec = dict(rep=r, mult={})
            for m in [0.5, 1.0, 2.0]:
                Yv = simulate(tk, tg, th0, rng, m)
                rec["mult"][str(m)] = score_all(Yv, rng, only={(tk, tg)})[f"{tk}/{tg}"]
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"rep {r} {time.time() - t0:.0f}s", flush=True)
