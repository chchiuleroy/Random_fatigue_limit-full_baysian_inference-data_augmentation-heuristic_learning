"""Fit the 9 full-Bayes RFL models (error x fatigue-limit) with NUTS (NumPyro / JAX).

    python runner.py full75 [WORKERS]    # README section 5.4 protocol: all 75 points are failures, no censoring
    python runner.py full   [WORKERS]    # the two tied maxima at S=0.675 right-censored (older protocol)

One job = one model; results are written to results/<stage>__<error>__<limit>.json (diagnostics, posterior medians)
and .npz (thinned posterior draws), and existing results are skipped, so an interrupted run can simply be restarted.
Sampler: 2 chains x (1000 warm-up + 1000 draws), dense mass matrix, target_accept 0.9, init near the published MLEs.
Note: results published in this repo were produced with seeds derived from Python's per-process hash(); the seed rule
below (CRC32 of the job id) is deterministic and was adopted afterwards, so re-running gives statistically equivalent
but not bit-identical draws.
"""
import os, sys, json, time, itertools, zlib
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import warnings
warnings.filterwarnings("ignore")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
KINDS, GTYPES = ["normal", "sev", "t"], ["lognormal", "gamma", "weibull"]


def job_id(stage, kind, gtype):
    return f"{stage}__{kind}__{gtype}"


def run_job(job):
    stage, kind, gtype = job
    jid = job_id(*job)
    os.makedirs(RES, exist_ok=True)
    path = os.path.join(RES, jid + ".json")
    if os.path.exists(path):
        return jid, "skip"
    sys.path.insert(0, HERE)
    import jax, arviz as az, pandas as pd
    from numpyro.infer import MCMC, NUTS, init_to_value
    import rfl_zoo as Z, rfl_data
    S, Y, EV = rfl_data.arrays(all_failures=(stage == "full75"))
    upper = float(np.log(S.min()) - 0.05)
    init = {"normal": dict(beta0=-9.2, beta1=-8.1, log_sigma=np.log(0.4)), "sev": dict(beta0=-9.4, beta1=-8.5, log_sigma=np.log(0.19)),
            "t": dict(beta0=-9.2, beta1=-8.1, log_sigma=np.log(0.3), log_nu=np.log(8.0))}[kind]
    init.update({"lognormal": dict(mu_d=min(-0.65, upper - 0.05), log_sigma_d=np.log(0.04)),
                 "gamma": dict(log_alpha_d=np.log(30.0), log_beta_r=np.log(60.0)),
                 "weibull": dict(log_k_d=np.log(15.0), log_lam_d=np.log(0.55))}[gtype])
    model = Z.make_model(S, Y, EV, kind, gtype, upper)
    t0 = time.time()
    mcmc = MCMC(NUTS(model, init_strategy=init_to_value(values=init), target_accept_prob=0.9, dense_mass=True),
                num_warmup=1000, num_samples=1000, num_chains=2, chain_method="sequential", progress_bar=False)
    mcmc.run(jax.random.PRNGKey(zlib.crc32(jid.encode()) % (2 ** 31)))
    dt = time.time() - t0
    samples = {k: np.asarray(v) for k, v in mcmc.get_samples().items()}
    ex = mcmc.get_extra_fields()
    summ = az.summary(az.from_numpyro(mcmc), kind="diagnostics").apply(pd.to_numeric, errors="coerce")
    rec = dict(job=jid, stage=stage, kind=kind, gtype=gtype, n=int(len(S)), seconds=round(dt, 1),
               rhat_max=float(summ["r_hat"].max()), ess_bulk_min=float(summ["ess_bulk"].min()),
               divergences=int(np.asarray(ex["diverging"]).sum()) if "diverging" in ex else None,
               post_median={k: float(np.median(v)) for k, v in samples.items()})
    np.savez_compressed(os.path.join(RES, jid + ".npz"), **{k: v[::max(1, len(v) // 400)] for k, v in samples.items()})
    json.dump(rec, open(path, "w"), indent=1)
    return jid, f"{dt:.0f}s rhat={rec['rhat_max']:.3f} ess={rec['ess_bulk_min']:.0f} div={rec['divergences']}"


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "full75"
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    jobs = [(stage, k, g) for k, g in itertools.product(KINDS, GTYPES)]
    print(f"{stage}: {len(jobs)} jobs, {workers} workers", flush=True)
    import concurrent.futures as cf, multiprocessing as mp
    t0 = time.time()
    with cf.ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
        futs = {ex.submit(run_job, j): j for j in jobs}
        for n, f in enumerate(cf.as_completed(futs), 1):
            try:
                jid, msg = f.result()
            except Exception as e:  # noqa
                jid, msg = job_id(*futs[f]), f"ERROR {e!r}"
            print(f"[{n}/{len(jobs)}] {time.time()-t0:6.0f}s {jid}: {msg}", flush=True)
    print("DONE", flush=True)
