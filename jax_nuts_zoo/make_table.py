"""y-ASSE (model z-score ASSE, README section 3.2) for every fitted model, plug-in theta_hat = posterior means.

    python make_table.py [stage]      # stage = full75 (default; README section 5.4 protocol, n = 75)

Heuristics (rfl_hl_asse.py names): H0 = Phi(z) percentile mapping (the README pipeline); H1 = exact marginal CDF;
H3 = per-group z calibration (sd floor 0.5). Writes results/yasse_<stage>.csv.
"""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import yasse as Y

stage = sys.argv[1] if len(sys.argv) > 1 else "full75"
data = Y.failure_data(all_failures=(stage == "full75"))
n = sum(len(y) for _, y in data)
rows = []
for kind in ["normal", "sev", "t"]:
    for gt in ["lognormal", "gamma", "weibull"]:
        base = os.path.join(HERE, "results", f"{stage}__{kind}__{gt}")
        if not os.path.exists(base + ".json"):
            continue
        r = json.load(open(base + ".json"))
        th = Y.plugin_from_npz(np.load(base + ".npz"), kind, gt)
        vals = {h: Y.asse(kind, gt, th, data, h)[0] for h in ["H0", "H1", "H3"]}
        rows.append(dict(error=kind, limit=gt, rhat_max=round(r["rhat_max"], 3), ess_bulk_min=round(r["ess_bulk_min"]),
                         divergences=r["divergences"], **{h: round(v, 4) for h, v in vals.items()},
                         **{k: round(v, 5) for k, v in th.items()}))
import csv
cols = sorted({k for r in rows for k in r}, key=lambda c: (c not in ("error", "limit"), c))
head = ["error", "limit", "rhat_max", "ess_bulk_min", "divergences", "H0", "H1", "H3"]
cols = head + [c for c in cols if c not in head]
with open(os.path.join(HERE, "results", f"yasse_{stage}.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
    for r in rows:
        w.writerow({c: r.get(c, "") for c in cols})
print(f"y-ASSE, stage {stage}, n = {n}")
print(f"{'error':7s}{'limit':10s}{'Rhat':>6s}{'ESS':>6s}{'div':>5s}{'H0':>9s}{'H1':>9s}{'H3':>9s}")
for r in rows:
    print(f"{r['error']:7s}{r['limit']:10s}{r['rhat_max']:6.3f}{r['ess_bulk_min']:6.0f}{r['divergences']:5d}{r['H0']:9.3f}{r['H1']:9.3f}{r['H3']:9.3f}")
