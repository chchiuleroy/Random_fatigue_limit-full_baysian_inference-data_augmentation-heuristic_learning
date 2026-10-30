"""ADR-RFL tool: what an Agentic-Distributional-Regression agent may see when it chooses the RFL error and
fatigue-limit distributions.  It returns fit diagnostics and WAIC -- never an ASSE.

    python adr_rfl.py list                    # the allowed structures
    python adr_rfl.py data                    # data summary per stress level
    python adr_rfl.py fit <error> <limit>     # full-Bayes (NUTS) fit summary of one structure

error in {normal, sev, t};  limit in {lognormal, gamma, weibull}.
`fit` reads the NUTS fit that `runner.py full75` wrote to results/ (one NUTS run, one seed, per structure; a refit would give
statistically equivalent but not identical draws); it fails if that structure has not been fitted.
WAIC uses the posterior draws and the exact marginal likelihood of rfl_zoo.loglik_obs (Delta integrated out).
"""
import os, sys, json
os.environ.setdefault("OMP_NUM_THREADS", "1")
import warnings
warnings.filterwarnings("ignore")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
KINDS, GTYPES = ["normal", "sev", "t"], ["lognormal", "gamma", "weibull"]


def waic(kind, gt, post):
    import jax, jax.numpy as jnp
    import rfl_zoo as Z, rfl_data
    S, Y, EV = rfl_data.arrays(all_failures=True)
    f = lambda p: Z.loglik_obs(p, jnp.asarray(S), jnp.asarray(Y), jnp.asarray(EV), kind, gt, False)
    ll = np.asarray(jax.vmap(f)({k: jnp.asarray(v) for k, v in post.items()}))          # (draws, n)
    lppd_i = np.log(np.mean(np.exp(ll - ll.max(0)), 0)) + ll.max(0)
    p_i = ll.var(0, ddof=1)
    e_i = lppd_i - p_i
    return dict(elpd_waic=float(e_i.sum()), se=float(np.sqrt(len(e_i) * e_i.var(ddof=1))), p_waic=float(p_i.sum()),
                elpd_by_stress={f"{s:g}": round(float(e_i[S == s].sum()), 3) for s in sorted(set(S))})


def fit_summary(kind, gt):
    path = os.path.join(HERE, "results", f"full75__{kind}__{gt}")
    if not os.path.exists(path + ".json"):
        raise SystemExit(f"no fit for {kind} x {gt}")
    r = json.load(open(path + ".json"))
    z = np.load(path + ".npz"); post = {k: z[k] for k in z.files}
    natural = {}
    for k, v in post.items():
        name, val = (k[4:] if k.startswith("log_") else k), (np.exp(v) if k.startswith("log_") else v)
        natural[name] = dict(mean=round(float(val.mean()), 5), sd=round(float(val.std()), 5))
    out = dict(error=kind, limit=gt, n=r["n"], posterior=natural,
               diagnostics=dict(rhat_max=round(r["rhat_max"], 3), ess_bulk_min=round(r["ess_bulk_min"]),
                                divergences=r["divergences"], chains="2 x 1000 draws, NUTS dense mass, target_accept 0.9"))
    out["waic"] = waic(kind, gt, post)
    return out


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "list"
    if cmd == "list":
        print(json.dumps(dict(error=KINDS, limit=GTYPES), indent=1))
    elif cmd == "data":
        import rfl_data
        S, Y, EV = rfl_data.arrays(True)
        print("model: ln N = beta0 + beta1 * ln(S - Delta) + sigma * eps,  Delta ~ fatigue-limit distribution, Delta < S for a failure")
        for s in sorted(set(S)):
            y = Y[S == s]
            print(f"S={s:g}: n={len(y)}  ln N mean {y.mean():.3f}  sd {y.std(ddof=1):.3f}  min {y.min():.3f}  max {y.max():.3f}")
    elif cmd == "fit":
        print(json.dumps(fit_summary(sys.argv[2], sys.argv[3]), indent=1))
    else:
        raise SystemExit(__doc__)
