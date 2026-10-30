"""Summary of da_sim_study.py ln runs (DA+NCP fits vs MLE on the same simulated data): python analyze_da_sim.py"""
import os, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
recs = [json.loads(l) for l in open(os.path.join(HERE, "results", "da_sim_ln_sev_lognormal.jsonl"), encoding="utf-8")]
out = dict(n_reps=len(recs), per_structure={}, picks_sev_lognormal={})
for k in ["sev/lognormal", "normal/lognormal"]:
    rows = [r["fits"][k] for r in recs]
    m = lambda g: float(np.mean([g(x) for x in rows]))
    d = dict(da_rhat_max=m(lambda x: x["rhat_max"]), da_ess_min=m(lambda x: x["ess_min"]), da_divergences=m(lambda x: x["n_div"]),
             sigma_da=m(lambda x: x["sigma"]), sigma_mle=m(lambda x: x["mle"]["theta"]["sigma"]), ll_da=m(lambda x: x["ll"]), ll_mle=m(lambda x: x["mle"]["ll"]))
    for h in ["H0", "H1", "H3"]:
        d[h + "_da"] = m(lambda x: x[h]); d[h + "_mle"] = m(lambda x: x["mle"][h])
    out["per_structure"][k] = d
for src in ["da", "mle"]:
    for h in ["H0", "H1", "H3", "aic"]:
        picks = 0
        for r in recs:
            if src == "da":
                vals = {k: r["fits"][k][h] for k in r["fits"]}
            else:
                vals = {k: (r["fits"][k]["mle"][h] if h != "aic" else 10 - 2 * r["fits"][k]["mle"]["ll"]) for k in r["fits"]}
            picks += min(vals, key=vals.get) == "sev/lognormal"
        out["picks_sev_lognormal"][f"{src}_{h}"] = picks
json.dump(out, open(os.path.join(HERE, "results", "da_sim_summary.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
