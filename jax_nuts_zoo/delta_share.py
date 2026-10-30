"""Share of the conditional variance of ln N (given S) that comes from the latent fatigue limit Delta, per model,
averaged over the five stress levels:  b1^2 Var(ln(S-Delta)) / V,  plug-in = posterior means (same moments as yasse.EV).
    python delta_share.py        # writes results/delta_share.csv and prints it with WAIC (needs results/waic_full75.csv)
"""
import os, sys, csv, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import yasse as Y

data = Y.failure_data(True)
yas = {(r["error"], r["limit"]): r for r in csv.DictReader(open(os.path.join(HERE, "results", "yasse_full75.csv"), encoding="utf-8"))}
rows = []
for kind in ["normal", "sev", "t"]:
    for gt in ["lognormal", "gamma", "weibull"]:
        th = Y.plugin_from_npz(np.load(os.path.join(HERE, "results", f"full75__{kind}__{gt}.npz")), kind, gt)
        sh = []
        for s, y in data:
            d, w = Y.nodes(gt, th, s); L = np.log(np.maximum(s - d, 1e-12)); E1 = (w * L).sum(); V1 = (w * L * L).sum() - E1 ** 2
            _, ev = Y.err_moments(kind, th); vd = th["b1"] ** 2 * V1; sh.append(vd / (vd + ev))
        rows.append(dict(error=kind, limit=gt, delta_share=round(float(np.mean(sh)), 3), sigma=round(th["sigma"], 3), b1=round(th["b1"], 2),
                         H0=yas[(kind, gt)]["H0"], H1=yas[(kind, gt)]["H1"], H3=yas[(kind, gt)]["H3"]))
with open(os.path.join(HERE, "results", "delta_share.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
from scipy.stats import spearmanr
for h in ["H0", "H1", "H3"]:
    print(h, "Spearman(delta_share, y-ASSE) =", round(spearmanr([r["delta_share"] for r in rows], [float(r[h]) for r in rows])[0], 3))
