"""Does the simulation conclusion about y-ASSE survive when the structures are fitted with the repo's own DA+NCP+NUTS code?

    python da_sim_study.py <first_rep> <last_rep_exclusive> [ln]   # truth: SEV + LogNormal (parameters of the real-data fit)
    with "ln" only the two LogNormal structures are fitted (DA) and each is also fitted by MLE on the same data set.
    (A smoke test of the repo's Gamma DA model on simulated data gave Rhat 2.76, ESS 4, 708 divergences and a log-likelihood
    far below the marginal fit's, so the Gamma DA fits are not usable for a selection test.)

Per replication: simulate one data set on the real design (sim_study.simulate), fit four structures with the repo's DA models
{SEV, Normal} x {LogNormal, Gamma} -- LogNormal via rfl_asse_full_sample.build_model, Gamma via rfl_pymc_da.build_da_model
(both unchanged; only their module-level data are replaced by the simulated y, all 75 treated as failures) -- take the
posterior means as theta_hat, and score with y-ASSE H0 / H1 / H3 (yasse.py) and with the marginal log-likelihood at theta_hat
(pseudo-AIC = 2k - 2 loglik; zoo likelihood).  Weibull / Student-t have no DA model in the repo and are not included.
Sampler: 4 chains x (1000 tune + 1000 draws), target_accept 0.95, PyMC NUTS; the DA models do not converge on this data
(README 5.4), so every row carries its n_divergences / Rhat / ESS.  Output: results/da_sim_sev_lognormal.jsonl.
"""
import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
STRUCTS = [("sev", "lognormal"), ("normal", "lognormal"), ("sev", "gamma"), ("normal", "gamma")]


def fit_da(kind, gt, Yv):
    import pymc as pm, arviz as az
    sys.path.insert(0, REPO)
    import rfl_asse_full_sample as A, rfl_pymc_da as D
    if gt == "lognormal":
        A.Y_OBS = Yv
        model, init = A.build_model(kind), dict(beta0=-9.4 if kind == "sev" else -9.2, beta1=-8.5 if kind == "sev" else -8.1,
                                                log_sigma=np.log(0.19 if kind == "sev" else 0.6), mu_d=-0.65,
                                                log_sigma_d=np.log(0.04), z_delta=np.zeros(A.N_OBS))
        names = ["beta0", "beta1", "sigma", "mu_d", "sigma_d"]
    else:
        D.Y_OBS = Yv; D.EVENT = np.ones(len(Yv), int)
        D.fail_idx = np.arange(len(Yv)); D.cens_idx = np.array([], int)
        D.Y_fail = Yv; D.Y_cens = Yv[:0]; D.S_fail = D.S_OBS; D.S_cens = D.S_OBS[:0]
        model, init = D.build_da_model(kind, "gamma"), D.make_init(kind, "gamma")
        names = ["beta0", "beta1", "sigma", "alpha_d", "beta_r"]
    with model:
        idata = pm.sample(draws=1000, tune=1000, chains=4, cores=4, initvals=init, target_accept=0.95, progressbar=False,
                          random_seed=[0, 1, 2, 3], nuts_sampler="pymc")
    th = {v: float(idata.posterior[v].mean()) for v in names}
    summ = az.summary(idata, var_names=names)
    return th, dict(n_div=int(idata.sample_stats["diverging"].sum()), rhat_max=float(summ["r_hat"].max()), ess_min=float(summ["ess_bulk"].min()))


def score(kind, gt, th_da, Yv):
    import sim_study as SS, yasse as Y
    if gt == "lognormal":
        th = dict(b0=th_da["beta0"], b1=th_da["beta1"], sigma=th_da["sigma"], mu_d=th_da["mu_d"], sd_d=th_da["sigma_d"])
        p = dict(beta0=th["b0"], beta1=th["b1"], log_sigma=np.log(th["sigma"]), mu_d=th["mu_d"], log_sigma_d=np.log(th["sd_d"]))
    else:
        th = dict(b0=th_da["beta0"], b1=th_da["beta1"], sigma=th_da["sigma"], alpha=th_da["alpha_d"], beta=th_da["beta_r"])
        p = dict(beta0=th["b0"], beta1=th["b1"], log_sigma=np.log(th["sigma"]), log_alpha_d=np.log(th["alpha"]), log_beta_r=np.log(th["beta"]))
    import jax.numpy as jnp
    nm = SS.names(kind, gt)
    ll = -float(SS.nll_fn(kind, gt)(jnp.asarray([p[k] for k in nm]), jnp.asarray(Yv))[0])
    gd = SS.groups(Yv)
    row = dict(ll=ll, aic=2 * len(nm) - 2 * ll, sigma=th["sigma"])
    for h in ["H0", "H1", "H3"]:
        row[h] = float(Y.asse(kind, gt, th, gd, h)[0])
    return row


if __name__ == "__main__":
    a, b = int(sys.argv[1]), int(sys.argv[2])
    ln_only = len(sys.argv) > 3 and sys.argv[3] == "ln"
    sys.path.insert(0, HERE)
    import sim_study as SS
    th0 = SS.truth_theta("sev", "lognormal")
    structs = [x for x in STRUCTS if x[1] == "lognormal"] if ln_only else STRUCTS
    path = os.path.join(HERE, "results", "da_sim_ln_sev_lognormal.jsonl" if ln_only else "da_sim_sev_lognormal.jsonl")
    for r in range(a, b):
        t0 = time.time()
        Yv = SS.simulate("sev", "lognormal", th0, np.random.default_rng(7000 + r))
        rec = dict(rep=r, fits={})
        for kind, gt in structs:
            th_da, diag = fit_da(kind, gt, Yv)
            rec["fits"][f"{kind}/{gt}"] = {**score(kind, gt, th_da, Yv), **diag, "theta_da": th_da}
            if ln_only:
                pm_, ll = SS.fit_mle(kind, gt, Yv, np.random.default_rng(7000 + r))
                th = SS.theta(kind, gt, pm_)
                rec["fits"][f"{kind}/{gt}"]["mle"] = dict(ll=ll, theta=th, **{h: float(__import__("yasse").asse(kind, gt, th, SS.groups(Yv), h)[0]) for h in ["H0", "H1", "H3"]})
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"rep {r} done {time.time() - t0:.0f}s", flush=True)
