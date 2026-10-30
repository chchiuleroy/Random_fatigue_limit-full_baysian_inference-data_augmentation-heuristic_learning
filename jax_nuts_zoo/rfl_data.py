"""Pascual & Meeker (1999) fatigue data (n = 75, 5 stress levels, cycles to failure) -- same numbers as
rfl_pymc_ncp.py / rfl_asse_full_sample.py."""
import numpy as np

RAW = {
    0.675: [102.95, 280.32, 339.83, 366.9, 485.62, 658.96, 896.33, 1241.76, 1250.2, 1329.78, 1399.83, 1459.14, 3249.82, 11748.1, 11748.1],
    0.75: [6.71, 9.93, 12.6, 15.58, 16.19, 17.28, 18.62, 20.3, 24.9, 26.26, 27.94, 36.35, 48.42, 50.09, 67.34],
    0.825: [1.246, 1.258, 1.46, 1.492, 2.4, 2.41, 2.59, 2.903, 3.33, 3.59, 3.847, 4.11, 4.82, 5.56, 5.598],
    0.9: [0.201, 0.216, 0.226, 0.252, 0.257, 0.295, 0.311, 0.342, 0.356, 0.451, 0.457, 0.509, 0.54, 0.68, 1.129],
    0.95: [0.037, 0.072, 0.074, 0.076, 0.083, 0.085, 0.105, 0.109, 0.12, 0.123, 0.143, 0.203, 0.206, 0.217, 0.257],
}
LEVELS = sorted(RAW)


def arrays(all_failures=True):
    """(S, ln N, event). all_failures=True is the README section 5.4 protocol (all 75 points are failures);
    False right-censors the two tied maxima at S=0.675 (the older 73-failure protocol)."""
    S, Y, EV = [], [], []
    for s in LEVELS:
        vals = sorted(RAW[s]); mx = max(vals); nmx = vals.count(mx)
        for v in vals:
            S.append(s); Y.append(np.log(v)); EV.append(1 if (all_failures or not (v == mx and nmx > 1)) else 0)
    return np.array(S), np.array(Y), np.array(EV, int)


def groups(all_failures=True):
    """[(s, sorted ln N of the points counted as failures)] for the ASSE computation."""
    S, Y, EV = arrays(all_failures)
    return [(s, np.sort(Y[(S == s) & (EV == 1)])) for s in LEVELS]
