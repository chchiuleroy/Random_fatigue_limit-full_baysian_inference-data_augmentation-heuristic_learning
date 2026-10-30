"""Summaries of the sim_study.py runs: python analyze_sim.py  (reads results/sim_*.jsonl, writes results/sim_summary.json)."""
import os, glob, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCORES = ["aic", "H0", "H1", "H3"]
out = {}
print("TEST A -- which structure does each score pick?  (chance = 1/9 = 11%)")
for f in sorted(glob.glob(os.path.join(HERE, "results", "sim_select_*.jsonl"))):
    truth = os.path.basename(f)[len("sim_select_"):-len(".jsonl")].replace("_", "/", 1)
    recs = [json.loads(l) for l in open(f, encoding="utf-8")]
    print(f"\ntruth {truth}, {len(recs)} replications")
    print(f"{'score':6s}{'picks truth':>12s}{'picks truth limit':>19s}{'picks Gamma':>13s}{'mean rank of truth':>20s}{'mean sigma of pick':>20s}")
    out[truth] = {}
    for sc in SCORES:
        hit = hitlim = gam = 0; ranks = []; sig = []
        for r in recs:
            fits = {k: v for k, v in r["fits"].items() if sc in v}
            ordered = sorted(fits, key=lambda k: fits[k][sc])
            best = ordered[0]
            hit += best == truth; hitlim += best.split("/")[1] == truth.split("/")[1]; gam += best.endswith("gamma")
            ranks.append(ordered.index(truth) + 1 if truth in ordered else np.nan); sig.append(fits[best]["sigma"])
        n = len(recs)
        out[truth][sc] = dict(picks_truth=hit / n, picks_truth_limit=hitlim / n, picks_gamma=gam / n, mean_rank=float(np.nanmean(ranks)), mean_sigma=float(np.mean(sig)))
        print(f"{sc:6s}{100 * hit / n:11.0f}%{100 * hitlim / n:18.0f}%{100 * gam / n:12.0f}%{np.nanmean(ranks):20.2f}{np.mean(sig):20.3f}")
    # sigma of the fitted structures vs score, pooled over replications: Spearman between sigma and each score within a rep
    from scipy.stats import spearmanr
    for sc in ["H0", "H1", "H3"]:
        rho = [spearmanr([v["sigma"] for v in r["fits"].values() if sc in v], [v[sc] for v in r["fits"].values() if sc in v])[0] for r in recs]
        out[truth][f"spearman_sigma_vs_{sc}"] = float(np.nanmean(rho))
    print("mean within-replication Spearman(fitted sigma, score) over the 9 structures: " +
          ", ".join(f"{sc} {out[truth]['spearman_sigma_vs_' + sc]:+.2f}" for sc in ["H0", "H1", "H3"]))

print("\nTEST B -- correct structure, true sigma multiplied (mean score of the correctly specified fit)")
for f in sorted(glob.glob(os.path.join(HERE, "results", "sim_noise_*.jsonl"))):
    truth = os.path.basename(f)[len("sim_noise_"):-len(".jsonl")].replace("_", "/", 1)
    recs = [json.loads(l) for l in open(f, encoding="utf-8")]
    print(f"truth {truth}, {len(recs)} replications")
    print(f"{'sigma x':8s}{'fitted sigma':>14s}{'H0':>9s}{'H1':>9s}{'H3':>9s}")
    out["noise_" + truth] = {}
    for m in ["0.5", "1.0", "2.0"]:
        rows = [r["mult"][m] for r in recs if "H1" in r["mult"][m]]
        out["noise_" + truth][m] = {k: float(np.mean([x[k] for x in rows])) for k in ["sigma", "H0", "H1", "H3"]}
        print(f"{m:8s}{np.mean([x['sigma'] for x in rows]):14.3f}{np.mean([x['H0'] for x in rows]):9.3f}{np.mean([x['H1'] for x in rows]):9.3f}{np.mean([x['H3'] for x in rows]):9.3f}")
json.dump(out, open(os.path.join(HERE, "results", "sim_summary.json"), "w"), indent=1)
