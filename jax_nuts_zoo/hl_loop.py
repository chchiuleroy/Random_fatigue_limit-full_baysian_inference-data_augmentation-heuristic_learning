"""Heuristic-Learning (HL) loop over the RFL structure space, with y-ASSE as the reward.

    python hl_loop.py            # writes results/hl_loop.json

State  = (error, fatigue-limit distribution, percentile heuristic) -- 3 x 3 x 3 = 27 states.
Reward = -y-ASSE of the state (plug-in posterior means of the full-sample fit, README section 3.2 Step 3).
Policy = pick one coordinate to change (error / limit / heuristic) with probability proportional to a weight, move to a
         random other value on that coordinate, accept if y-ASSE falls (stricter = keep the better).
Update = a coordinate whose move was accepted gets weight x1.5, a rejected one x0.8 (floor 0.2): the learned weights
         are the "heuristic" HL returns, next to the final state.
The loop starts from every one of the 27 states, 40 proposals each, seeded.  Variants:
  fixed-H0 : heuristic frozen at H0 (the README pipeline)                      -> searches the 9 structures only
  free     : heuristic may change too
  gated    : as 'free', but structures failing Rhat_max <= 1.05 and ESS_min >= 30 are not allowed
There is no held-out data: the reward is the same quantity that is reported, so the winner is a selection among
27 numbers on n = 75, not a validated improvement.
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import yasse as Y

KINDS, GTYPES, HEUR = ["normal", "sev", "t"], ["lognormal", "gamma", "weibull"], ["H0", "H1", "H3"]
DATA = Y.failure_data(True)
_cache, _fit = {}, {}


def fit_info(k, g):
    if (k, g) not in _fit:
        base = os.path.join(HERE, "results", f"full75__{k}__{g}")
        r = json.load(open(base + ".json"))
        _fit[(k, g)] = (Y.plugin_from_npz(np.load(base + ".npz"), k, g), r["rhat_max"], r["ess_bulk_min"])
    return _fit[(k, g)]


def reward(state):
    if state not in _cache:
        k, g, h = state
        _cache[state] = Y.asse(k, g, fit_info(k, g)[0], DATA, h)[0]
    return _cache[state]


def allowed(state, gated):
    if not gated:
        return True
    _, rhat, ess = fit_info(state[0], state[1])
    return rhat <= 1.05 and ess >= 30


def run(start, variant, rng, n_prop=40):
    gated, frozen = variant == "gated", variant == "fixed-H0"
    w = np.array([1.0, 1.0, 0.0 if frozen else 1.0])
    state = start
    if not allowed(state, gated):
        return None
    cur = reward(state)
    for _ in range(n_prop):
        if w.sum() == 0:
            break
        c = rng.choice(3, p=w / w.sum())
        new = list(state)
        new[c] = rng.choice([v for v in ([KINDS, GTYPES, HEUR][c]) if v != state[c]])
        new = tuple(new)
        if not allowed(new, gated):
            w[c] = max(0.2, w[c] * 0.8); continue
        r = reward(new)
        if r < cur - 1e-9:
            state, cur = new, r; w[c] *= 1.5
        else:
            w[c] = max(0.2, w[c] * 0.8)
    return state, cur, w / w.sum()


if __name__ == "__main__":
    out = {}
    for variant in ["fixed-H0", "free", "gated"]:
        rng = np.random.default_rng(7)
        _cache.clear()   # evaluation count is per variant
        ends, ws = [], []
        for k in KINDS:
            for g in GTYPES:
                for h in (["H0"] if variant == "fixed-H0" else HEUR):
                    res = run((k, g, h), variant, rng)
                    if res:
                        ends.append(res[:2]); ws.append(res[2])
        best = min(ends, key=lambda e: e[1])
        freq = {}
        for s, _ in ends:
            freq[s] = freq.get(s, 0) + 1
        top = sorted(freq.items(), key=lambda kv: -kv[1])[:3]
        # exhaustive reference over the same allowed set
        allst = [(k, g, h) for k in KINDS for g in GTYPES for h in (["H0"] if variant == "fixed-H0" else HEUR) if allowed((k, g, h), variant == "gated")]
        ex = min(allst, key=reward)
        out[variant] = dict(best_state=list(best[0]), best_asse=round(best[1], 4), exhaustive_best=list(ex), exhaustive_asse=round(reward(ex), 4),
                            n_starts=len(ends), end_state_counts={"/".join(s): c for s, c in top},
                            mean_final_weights=dict(zip(["error", "limit", "heuristic"], np.round(np.mean(ws, 0), 3).tolist())),
                            distinct_asse_evaluations=len(_cache))
        print(variant, json.dumps(out[variant]))
    json.dump(out, open(os.path.join(HERE, "results", "hl_loop.json"), "w"), indent=1)
