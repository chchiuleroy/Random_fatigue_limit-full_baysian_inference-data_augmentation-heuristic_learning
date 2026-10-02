#!/usr/bin/env python3
"""
mcmc_convergence_graph.py
MCMC convergence graph (Gu, Lakatos & Hamis 2026, arXiv:2609.37352)

wiki: [[paper_gu_2026_mcmc_convergence_graph]], [[concept_parallel_tempering]]

Idea: compute Rhat for every PAIR of chains; draw an edge i--j when
Rhat_ij < rho. Connected components of size >= 2 (K of them) are groups of
chains that agree; chains agreeing with nobody are isolated (I of them).
  K=1, I=0  global agreement
  K>=2, I=0 local agreement  -> several modes, or non-identifiability
  I>=1      isolated chains  -> a lone mode OR a chain that failed to mix
I/n close to 1 = widespread isolation = poor mixing, not multimodality.

Pairwise statistic: arviz.rhat(method="rank") = max(z_scale, folded), i.e.
the rank-normalised split-Rhat that Stan reports. This is NOT the authors' R
package (mcmcConvergenceGraph); results are not guaranteed bit-identical.

Limits (see the wiki page): needs disjoint modes (a continuous ridge breaks
the meaning of K); n >= 2M chains for M modes (so n=4 can hardly show K);
component size is NOT posterior mode weight; default rho=1.05 can be too
tight for diffuse targets (paper Appendix B2) -- sweep rho.

Usage:
  python mcmc_convergence_graph.py chains.npy [--rho 1.05 1.1] [--names a b c]
  python mcmc_convergence_graph.py --selftest
chains.npy shape = (n_chains, n_draws, n_params).
"""

import argparse
import itertools

import arviz as az
import numpy as np


def pairwise_rhat(draws):
    """draws (n, N, d) -> R (d, n, n) symmetric, diagonal 1.0."""
    n, _, d = draws.shape
    R = np.ones((d, n, n))
    for s in range(d):
        for i, j in itertools.combinations(range(n), 2):
            R[s, i, j] = R[s, j, i] = float(az.rhat(draws[[i, j], :, s], method="rank"))
    return R


def components(adj):
    """Connected components of a boolean adjacency matrix -> list of lists."""
    n = len(adj)
    seen, out = [False] * n, []
    for r in range(n):
        if seen[r]:
            continue
        stack, comp = [r], []
        seen[r] = True
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in range(n):
                if adj[u][v] and not seen[v]:
                    seen[v] = True
                    stack.append(v)
        out.append(sorted(comp))
    return sorted(out, key=lambda c: (-len(c), c))


def graph_summary(adj):
    """-> dict(K, I, components); K = components with >= 2 chains, I = singletons."""
    comps = components(adj)
    return {"K": sum(len(c) > 1 for c in comps),
            "I": sum(len(c) == 1 for c in comps),
            "components": comps}


def convergence_graph(R, rho=1.05):
    """Per-parameter graphs plus intersection / union. NaN Rhat -> no edge
    (a failed statistic must not count as agreement)."""
    d, n, _ = R.shape
    adj = (R < rho) & ~np.eye(n, dtype=bool)
    adj = adj & ~np.isnan(R)
    return {"per_param": [graph_summary(adj[s]) for s in range(d)],
            "intersection": graph_summary(adj.all(axis=0)),
            "union": graph_summary(adj.any(axis=0))}


def report(draws, rhos=(1.05,), names=None):
    n, N, d = draws.shape
    if n < 2:
        raise ValueError("need >= 2 chains: the graph is built from chain pairs")
    names = names or [f"p{s}" for s in range(d)]
    if len(names) != d:
        raise ValueError(f"got {len(names)} names for {d} parameters")
    R = pairwise_rhat(draws)
    print(f"chains n={n}, draws N={N}, params d={d}")
    if n < 4:
        print("WARNING: n < 4 chains; the graph says almost nothing.")
    for s, nm in enumerate(names):
        iu = np.triu_indices(n, 1)
        full = float(az.rhat(draws[:, :, s], method="rank"))
        print(f"{nm}: all-chain rank-Rhat={full:.3f}  pairwise "
              f"[{np.nanmin(R[s][iu]):.3f}, {np.nanmax(R[s][iu]):.3f}]")
    for rho in rhos:
        g = convergence_graph(R, rho)
        print(f"-- rho={rho}")
        for nm, gs in zip(names, g["per_param"]):
            print(f"   {nm}: K={gs['K']} I={gs['I']} (I/n={gs['I'] / n:.2f}) {gs['components']}")
        for key in ("intersection", "union"):
            gs = g[key]
            print(f"   {key}: K={gs['K']} I={gs['I']} {gs['components']}")
    return R


def _selftest():
    """Each case encodes WHY the diagnostic exists, not just that it runs."""
    rng = np.random.default_rng(0)
    n, N = 8, 2000

    # same target, mixed well -> one component, nobody isolated
    iid = rng.normal(size=(n, N, 1))
    g = convergence_graph(pairwise_rhat(iid), 1.05)["per_param"][0]
    assert g["K"] == 1 and g["I"] == 0, g

    # two well-separated modes, 5 vs 3 chains: plain Rhat would just say "bad";
    # the graph must recover the 5/3 split (the paper's bimodal case)
    bi = rng.normal(0, 0.1, size=(n, N, 1))
    bi[:5] += 0.5
    bi[5:] -= 0.5
    g = convergence_graph(pairwise_rhat(bi), 1.05)["per_param"][0]
    assert g["K"] == 2 and g["I"] == 0, g
    assert [len(c) for c in g["components"]] == [5, 3], g

    # one chain stuck elsewhere (drifting) -> isolated, not a "mode"
    st = rng.normal(size=(n, N, 1))
    st[0, :, 0] = np.cumsum(rng.normal(0, 0.05, size=N)) + 3.0
    g = convergence_graph(pairwise_rhat(st), 1.05)["per_param"][0]
    assert g["I"] >= 1 and [0] in g["components"], g

    # modes differ only in x1: intersection splits, union stays connected
    two = rng.normal(0, 0.1, size=(n, N, 2))
    two[:4, :, 0] += 0.5
    two[4:, :, 0] -= 0.5
    gg = convergence_graph(pairwise_rhat(two), 1.05)
    assert gg["intersection"]["K"] == 2 and gg["union"]["K"] == 1, gg

    # NaN Rhat must not be read as agreement
    R = np.full((1, 3, 3), np.nan)
    assert convergence_graph(R, 1.05)["per_param"][0]["I"] == 3
    # bad input must fail loudly, not silently drop parameters / crash in numpy
    for bad, kw in ((iid[:1], {}), (two, {"names": ["only_one"]})):
        try:
            report(bad, **kw)
        except ValueError:
            continue
        raise AssertionError("bad input was accepted")
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("chains", nargs="?", help=".npy, shape (n_chains, n_draws, n_params)")
    ap.add_argument("--rho", type=float, nargs="+", default=[1.05])
    ap.add_argument("--names", nargs="+")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if not a.chains:
        ap.error("need chains.npy or --selftest")
    x = np.load(a.chains)
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3, "expected (n_chains, n_draws, n_params)"
    report(x, a.rho, a.names)


if __name__ == "__main__":
    main()
