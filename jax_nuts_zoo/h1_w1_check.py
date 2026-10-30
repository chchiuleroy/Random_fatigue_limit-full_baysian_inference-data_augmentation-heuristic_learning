"""H1 y-ASSE as a Wasserstein-type distance.
H1 maps y -> p = F(y|S) (exact marginal CDF) -> Delta_hat = G^-1(p) -> y_hat = mu(Delta_hat) [+ SEV shift], where
mu(D) = b0 + b1 ln(S - D) is the error-free (structural) part.  So y_hat = Q_mu(F_y(y)): the quantile-quantile map from the
law of y|S to the law of the structural part mu(Delta).  Summing |y - y_hat| over the n_j observations of a level therefore
estimates n_j * W1(law of y|S, law of mu(Delta)), W1 = integral over u of |Q_y(u) - Q_mu(u)|.  This script computes the
theoretical right-hand side from the fitted plug-in (no data) and prints it next to the sample H1 ASSE of make_table.py.
    python h1_w1_check.py
"""
import os, sys, csv
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import yasse as Y

data = Y.failure_data(True)
tab = {(r["error"], r["limit"]): float(r["H1"]) for r in csv.DictReader(open(os.path.join(HERE, "results", "yasse_full75.csv"), encoding="utf-8"))}
u = (np.arange(4000) + 0.5) / 4000
print(f"{'model':18s}{'H1 sample':>10s}{'n*W1 theory':>13s}")
for kind in ["normal", "sev", "t"]:
    for gt in ["lognormal", "gamma", "weibull"]:
        th = Y.plugin_from_npz(np.load(os.path.join(HERE, "results", f"full75__{kind}__{gt}.npz")), kind, gt)
        tot = 0.0
        for s, y in data:
            ys = np.linspace(y.min() - 8 * th["sigma"] - 3, y.max() + 8 * th["sigma"] + 3, 6000)
            qy = np.interp(u, Y.marg_cdf_at(kind, gt, th, s, ys), ys)
            qm = Y.yhat_from_delta(kind, th, s, Y.g_ppf(gt, th, u))
            tot += len(y) * np.mean(np.abs(qy - qm))
        print(f"{kind:7s}{gt:11s}{tab[(kind, gt)]:10.3f}{tot:13.3f}")
