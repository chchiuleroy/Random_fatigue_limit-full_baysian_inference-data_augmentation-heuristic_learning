# Full Bayesian RFL Model: Model Z-Score Prediction

> **Roy (2026-06-02, updated 2026-08-20)**
> Full Bayesian Inference to improve individual lifetime prediction for the Random Fatigue Limit (RFL) model.
> Proposes replacing sample statistics with model-theoretic moments (Model Z-Score): fix $\hat\theta$ from a full-sample fit, standardize each observation's residual against the model's marginal moments, and invert that percentile through $\Delta$'s LogNormal quantile function (see §3.2–§3.3 and §5.4's provenance note). NUTS sampler configuration (`target_accept`) is chosen via a Heuristic Learning candidate-search over a fixed policy space (§3.5), not a hand-picked default. Full n=75 sample, no synthetic censoring; **the sampler does not currently converge** (§5.4).

---

## Table of Contents

1. [Background](#1-background)
2. [Problem Definition and Motivation](#2-problem-definition-and-motivation)
3. [Methods](#3-methods)
4. [Data](#4-data)
5. [Results](#5-results)
6. [Comparison with Prior Work](#6-comparison-with-prior-work)
7. [Theoretical Contributions](#7-theoretical-contributions)
8. [Code Description](#8-code-description)
9. [How to Run](#9-how-to-run)
10. [References](#10-references)

---

## Abbreviations

### Core Models and Methods

| Full Name | Description |
|-----------|-------------|
| Random Fatigue Limit→(RFL) | Random fatigue limit model; the primary subject of this study |
| Smallest Extreme Value→(SEV) | Log-Weibull distribution; the lifetime failure distribution family |
| Data Augmentation→(DA) | Explicitly sample the latent variable $\Delta_i$ in MCMC to avoid numerical integration |
| Non-Centered Parameterization→(NCP) | Decouples prior from hyperparameters to eliminate Neal's Funnel |
| No-U-Turn Sampler→(NUTS) | Adaptive HMC algorithm (Hoffman & Gelman 2014) |
| Hamiltonian Monte Carlo→(HMC) | Underlying algorithm of NUTS |
| Markov Chain Monte Carlo→(MCMC) | General posterior sampling framework |
| Heuristic Learning→(HL) | Weng (2026)'s policy-search pattern (state→policy→feedback→loop); used here for NUTS sampler-configuration selection — see §3.5 for the methods-level statement and §5.4 for the actual per-candidate diagnostics. **Not** used for choosing among prediction pipelines |
| LogNormal→(LN) | Log-normal distribution; the prior family for $\Delta_i$ |

### Prediction Error Metrics

| Full Name | Description |
|-----------|-------------|
| Absolute Sum of Standardized-score Errors→(ASSE) | $\sum_i\|y_i - \hat{y}_i\|$; three variants: z-ASSE (z-score space), y-ASSE (ln-lifetime space), rank-ASSE (rank space) |

### Statistical Inference Methods

| Full Name | Description |
|-----------|-------------|
| Maximum Likelihood Estimation→(MLE) | Frequentist baseline |
| Integrated Nested Laplace Approximation→(INLA) | Integrated nested Laplace approximation |
| Errors-In-Variables→(EIV) | Measurement-error model; the framework of Chiu (2005) z-score |
| Ordinary Least Squares→(OLS) | Basis for Roy Method C-1 |
| Least Absolute Deviations→(LAD) | Basis for Roy Method C-2 |
| Leave-One-Out Expected Log Pointwise Predictive Density→(LOO-ELPD) | LOO cross-validation metric for model comparison |
| Pareto-Smoothed Importance Sampling Leave-One-Out→(PSIS-LOO) | Pareto-smoothed IS-LOO |
| Effective Sample Size→(ESS) | Measures MCMC mixing quality |
| Credible Interval→(CI) | Bayesian credible interval |
| Standard Deviation→(SD) | Standard deviation |

### Numerical Methods

| Full Name | Description |
|-----------|-------------|
| Gauss-Legendre→(GL) | Gauss-Legendre quadrature; used to compute theoretical moments $E(\ln N \mid S_j)$ |
| Gauss-Hermite→(GH) | Gauss-Hermite quadrature |
| Cumulative Distribution Function→(CDF) | Cumulative distribution function |
| Expectation-Maximization→(EM) | EM algorithm; used for mixture model estimation |
| Gaussian Mixture Model→(GMM) | Gaussian mixture model |

---

## 1. Background

The **Random Fatigue Limit (RFL) model** was proposed by Pascual & Meeker (1999) for analyzing fatigue test data. The core assumption is that each specimen has a random fatigue limit $\Delta_i$; fatigue failure occurs only when the applied stress exceeds $\Delta_i$.

### Original Model (P&M 1999)

$$\ln N_i \mid \Delta_i \sim \text{SEV}(\beta_0 + \beta_1 \ln(S_i - \Delta_i), \sigma)$$

$$\Delta_i \sim \text{LogNormal}(\mu_\Delta, \sigma_\Delta)$$

where SEV is the log-Weibull distribution, $N_i$ is the number of cycles to failure, and $S_i$ is the applied stress.

**Limitations of prior work:**
- P&M (1999) used Laplace approximation for inference, limiting prediction accuracy
- Chiu (2005) estimated individual $\Delta_i$ via within-group sample z-scores, introducing substantial error
- Standard MLE/INLA methods cannot quantify parameter uncertainty

---

## 2. Problem Definition and Motivation

### 2.1 Three Deficiencies of the Chiu (2005) Sample Z-Score

Chiu's thesis estimated each specimen's fatigue limit using within-group sample statistics:

$$\omega_{ij} = \frac{y_{ij} - \bar{y}_j}{s_j}$$

**Deficiency 1: Insufficient sample size.**
Each stress level has only $n_j = 15$ observations, making $\bar{y}_j$ and $s_j$ unstable (large standard errors).

**Deficiency 2: Inseparable variance components.**

$$s_j^2 \approx \underbrace{\beta_1^2 \cdot \text{Var}[\ln(S_j - \Delta)]}_{\text{individual }\Delta} + \underbrace{\frac{\pi^2 \sigma^2}{6}}_{\text{SEV residual}}$$

The sample $s_j^2$ conflates two sources; individual $\Delta_i$ uncertainty cannot be separated from the residual $\sigma$.

**Deficiency 3: Ignoring cross-stress heteroscedasticity.**

$s_j$ does not vary with stress level $S_j$, but theoretically $V(\ln N \mid S_j)$ varies substantially with $S_j$ through $\text{Var}[\ln(S_j - \Delta)]$ (at low stress, $S_j - \Delta$ is small and unstable, yielding larger variance). This **heteroscedastic structure** is invisible to Chiu's method.

### 2.2 Core Idea of This Work

> **Replace sample statistics with theoretical moments derived from the full Bayesian posterior.**

Full Bayesian inference yields the posterior

$$\hat{\theta} = (\hat{\beta}_0, \hat{\beta}_1, \hat{\sigma}, \hat{\mu}_\Delta, \hat{\sigma}_\Delta)$$

from which precise theoretical means and variances at each stress level can be computed, correctly separating the contribution of $\Delta$ variation from the residual $\sigma$.

---

## 3. Methods

### 3.1 Full Bayesian Inference: DA + NCP + NUTS

#### Joint Posterior

$$p(\theta, \Delta_{1:n} \mid \mathbf{y}) \propto \prod_{i=1}^{n} f(y_i \mid \Delta_i, \theta) \cdot \prod_{i=1}^{n} g(\Delta_i \mid \theta) \cdot p(\theta)$$

**Data Augmentation (DA):** Explicitly sample $\Delta_i$ as a latent variable, avoiding numerical integration. Each MCMC step has the current value of $\Delta_i$; the likelihood is evaluated directly without GL/GH quadrature.

#### Non-Centered Parameterization (NCP)

Direct sampling of $\log \Delta_i \sim \mathcal{N}(\mu_\Delta, \sigma_\Delta)$ produces Neal's Funnel when $\sigma_\Delta \approx 0.033$ (near-degenerate), causing many divergences.

**Solution:**

$$z_{\Delta_i} \sim \mathcal{N}(0, 1), \qquad \log \Delta_i = \mu_\Delta + \sigma_\Delta \cdot z_{\Delta_i}$$

Decoupling the prior from hyperparameters $(\mu_\Delta, \sigma_\Delta)$ flattens the posterior geometry and enables efficient NUTS exploration.

#### Prior Specification (Uniform + Jeffreys)

| Parameter | Prior | Rationale |
|-----------|-------|-----------|
| $\beta_0$ | $\text{Uniform}(-50, 50)$ | Flat; no prior information |
| $\beta_1$ | $\text{Uniform}(-30, 0)$ | Hard constraint: higher excess stress → shorter life |
| $\log\sigma$ | $\text{Uniform}(\log 0.01, \log 5)$ | Jeffreys scale prior |
| $\mu_\Delta$ | $\text{Uniform}(-5, \log S_{\min})$ | Ensures $\Delta$ median $< S_{\min}$ |
| $\log\sigma_\Delta$ | $\text{Uniform}(\log 0.001, \log 2)$ | Jeffreys scale prior |

#### Algorithm

**PyMC**'s **No-U-Turn Sampler (NUTS)** (Hoffman & Gelman 2014) with gradients computed by pytensor automatic differentiation.
4 chains in parallel, each with 1000 warm-up + 1000 sampling steps, yielding 4000 posterior draws.

---

### 3.2 Novel Model Z-Score Method (Roy)

#### Step 1: Compute Theoretical Moments per Stress Level

**Theoretical mean** (32-point Gauss-Legendre integration):

$$E(\ln N \mid S_j) = \hat{\beta}_0 - \hat{\sigma}\gamma_E + \hat{\beta}_1 \cdot \frac{\displaystyle\int_0^{S_j} \ln(S_j - \Delta) \cdot g(\Delta \mid \hat{\mu}_\Delta, \hat{\sigma}_\Delta) \thinspace d\Delta}{F_\Delta(S_j)}, \qquad F_\Delta(S_j) = \int_0^{S_j} g(\Delta \mid \hat{\mu}_\Delta, \hat{\sigma}_\Delta) \thinspace d\Delta$$

where $\gamma_E = 0.5772$ (Euler-Mascheroni constant). $g(\Delta\mid\cdot)$ is the (untruncated) LogNormal density, so the raw integral over $[0,S_j]$ is not itself a conditional expectation — dividing by $F_\Delta(S_j)$ (the LogNormal CDF at $S_j$, i.e. $P(\Delta<S_j)$) is what makes this $E[\ln(S_j-\Delta)\mid\Delta<S_j]$. `gl_moments()` in `rfl_asse_full_sample.py` (originally `_gl_moments()` in the now-removed `rfl_model_zscore_asse.py`) computes this correctly (the quadrature weight sum `norm` *is* the discretized $F_\Delta(S_j)$, and both `E1`/`E2` divide by it) — this formula previously omitted the denominator, which the code has always included (2026-08-13 fix, precision-only, does not change any reported number).

**Theoretical variance** (correctly separating two sources; $\text{Var}[\cdot]$ here means the same $\Delta<S_j$-conditional variance as above, i.e. computed from the same $F_\Delta(S_j)$-normalized moments):

$$V(\ln N \mid S_j) = \hat{\beta}_1^2 \cdot \underbrace{\text{Var}[\ln(S_j - \Delta) \mid \Delta<S_j]}_{\Delta\text{ contribution}} + \underbrace{\frac{\pi^2 \hat{\sigma}^2}{6}}_{\text{SEV term}} \quad \left(\text{Normal: } \hat{\sigma}^2\right)$$

#### Step 2: Complete Prediction Pipeline

```
1. Model z-score:   z_ij = (y_ij - E(lnN|S_j)) / sqrt(V(lnN|S_j))
2. Percentile:      p_ij = Phi(z_ij)    [Normal CDF]
3. Invert for Delta:Delta_ij = exp(mu_Delta + sigma_Delta * z_ij)
4. Predict:         y_ij = beta0 + beta1 * ln(S_j - Delta_ij) - sigma*gamma_E  (SEV)
5. ASSE:            sum|y_ij - y_hat_ij|
```

---

### 3.3 Prediction Pipeline (current)

The percentile method (steps above) is the only prediction pipeline reported in this README. Earlier work compared it against several alternative constructions of $\hat\Delta_i$ (exact marginal CDF instead of the Normal approximation $\Phi(z)$, a skewness correction, per-group recalibration, a DA-posterior plug-in) — those comparisons, and the numbers behind them, were computed on the retired 73-observation synthetic-censoring pipeline and are not restated here; see `git log`, or the author's local `todo.md` (a task-tracking file kept outside this repo, not included here — not resolvable by GitHub readers) if that history is needed.

---

### 3.4 Extension to Censored Data

The P&M (1999) dataset used here contains $n=75$ complete failure observations with no censoring. For datasets with censored observations, the likelihood contribution must be modified.

#### Setup

Let $\delta_i \in \lbrace 0, 1\rbrace$ be the failure indicator: $\delta_i = 1$ denotes an observed failure ($y_i = \ln N_i$ known), $\delta_i = 0$ denotes censoring at $c_i$ ($\ln N_i > c_i$, true value unknown).

#### Modified Joint Likelihood

$$p(\mathbf{y},\boldsymbol{\delta} \mid \boldsymbol{\Delta}, \theta) = \prod_{i:\delta_i=1} f(y_i \mid \Delta_i, \theta) \cdot \prod_{i:\delta_i=0} S(c_i \mid \Delta_i, \theta)$$

Survival function by model distribution:

$$S_{\text{SEV}}(c_i \mid \Delta_i, \theta) = \exp\left(-\exp\left(\frac{c_i - \mu_i}{\sigma}\right)\right), \qquad \mu_i = \beta_0 + \beta_1\ln(S_j - \Delta_i)$$

$$S_{\text{Normal}}(c_i \mid \Delta_i, \theta) = 1 - \Phi\left(\frac{c_i - \mu_i}{\sigma}\right)$$

#### Four Censoring Types

| Censoring Type | Description | Likelihood Contribution |
|----------------|-------------|------------------------|
| **Type I (time censoring)** | Experiment stops at fixed time $c$; non-failed units censored | $S(c \mid \Delta_i, \theta)$ |
| **Type II (failure-count censoring)** | Stops after the $r$-th failure; remaining units censored | $S(y_{(r)} \mid \Delta_i, \theta)$ (censoring time equals $r$-th failure time) |
| **Random censoring** | Each specimen has its own independent censoring time $c_i$ | $S(c_i \mid \Delta_i, \theta)$ |
| **Interval censoring** | Failure known only to lie within $[L_i, U_i]$ | $F(U_i \mid \Delta_i, \theta) - F(L_i \mid \Delta_i, \theta)$ |

#### Advantages of the DA Framework

Censored observations are handled naturally in the DA framework: $\Delta_i$ is still explicitly sampled; simply replace the likelihood contribution from $f(y_i \mid \cdot)$ with $S(c_i \mid \cdot)$. **No additional numerical integration per censored observation is needed**, and the NCP and NUTS structure remains unchanged.

#### PyMC Implementation

```python
# delta: shape (n,), 1=failure, 0=censored; c: censoring time vector
import pytensor.tensor as pt

loglik = pt.switch(
    delta,
    dist.logp(y),                               # failure: log pdf
    pt.log1p(-pt.exp(dist.logcdf(c)))           # censored: log survival = log(1 - CDF)
)
pm.Potential('likelihood', loglik.sum())
```

#### Handling ASSE with Censoring

The true $y_i$ is unknown for censored observations. Two principled approaches exist:

1. **Exclude censored observations:** Compute ASSE using only $\delta_i = 1$ failures (most common)
2. **Conditional expectation imputation:** Substitute $E[\ln N \mid \ln N > c_i, \Delta_i, \theta]$ for $y_i$, incorporating censored observations into ASSE

`rfl_asse_full_sample.py`, the current script (§5.4), implements neither — it has no censored-observation handling at all, by design (§4: no trustworthy censoring metadata for this dataset). The now-removed `rfl_bayes_asse.py` (see §8) did implement approach 1 (excluding censored rows via an `only_fail` flag) for the synthetic-censoring test case described in §4's note; approach 2 (conditional expectation imputation) was never implemented anywhere in this repo. See `rfl_asse_full_sample.py`'s module docstring for a fuller write-up of the tradeoffs between the two approaches if real censored data is added later.

---

### 3.5 Heuristic Learning (HL) for NUTS Sampler Configuration

**Scope note.** HL is used here **only** for choosing the NUTS sampler configuration, not for choosing among prediction pipelines. The prediction pipeline itself is fixed by §3.2–§3.3 (the percentile method, applied to θ̂ from a full-sample fit) per Roy's 2026-08-20 decision; nothing about HL selects, ranks, or scores that method against alternatives.

**Motivation.** The RFL model's persistent NUTS divergences (see §5.1 callout and §5.4) require an explicit choice of `target_accept` (and, potentially, `draws`/`tune`). Two failure modes were observed:

1. **Escalating chain (tried first, failed).** Raising `target_accept`/`draws`/`tune` together in response to the previous round's diagnostics made divergences *monotonically worse* across rounds (3165→3108→4973→7116). Each escalation reacted to the previous round's specific failure mode rather than sweeping the actual decision space.
2. **Fixed default (also failed).** Any single hand-picked `target_accept` — 0.85, 0.90, 0.95, 0.99 — leaves either divergences or Rhat at unacceptable levels (see §5.4 tables). No one setting dominates.

**HL policy-search formulation** (following [[concept_heuristic_learning]]):

| HL component | Concrete instantiation |
|---|---|
| **State** | Convergence diagnostics of the current NUTS fit: `n_divergences`, `Rhat_max`, `ESS_min` |
| **Policy space** | A **fixed, finite** set of named candidate configs: `target_accept ∈ {0.85, 0.90, 0.95, 0.99}`, `draws=tune=2000` fixed across candidates |
| **Feedback signal** | Per-candidate `(Rhat_max, n_divergences)` after running that candidate to completion (not partial-run heuristics) |
| **Loop / selection rule** | Run **all** candidates independently, then pick `argmin(Rhat_max, n_divergences)` lexicographically. **Not** a single escalating chain conditioned on the previous round's output — that was the failure mode |

**Thresholds** (from `rfl_asse_full_sample.py`, applied inclusively — `n_divergences ≤ DIVERGENCE_OK`, `Rhat_max ≤ RHAT_OK`, `ESS_min ≥ ESS_OK`): `DIVERGENCE_OK=50`, `RHAT_OK=1.05`, `ESS_OK=200`. These are reported honestly against each candidate but do not gate selection — with no candidate satisfying the script's combined acceptance check on the current model, gating would return no configuration.

**Result on this repo (current run).** Candidate search selected `target_accept=0.85` for SEV and `target_accept=0.95` for Normal (§5.4). This is the search finding the least-bad configuration in the tested policy space, not a validated fit — `target_accept=0.99` drove SEV divergences to zero at the cost of Rhat=1.50 and all 4 chains hitting `max_treedepth`. That pattern is *consistent with* a geometry problem (e.g., the un-fixed `Δ<S` soft floor) rather than a step-size tuning problem, but as noted in §5.1's callout `max_treedepth` alone is not a step-size-independent proof. What HL here does establish is a stronger and narrower claim: **no acceptable configuration exists inside this specific policy space**. That is a *finding*, not something to paper over — see §5.4 for full per-candidate diagnostics and the reparameterization work tracked in `todo.md`.

---

## 4. Data

> **⚠️ 2026-08-20: source attribution below is now suspected wrong, not independently confirmed either way.** Review found that neither of P&M (1999)'s own two datasets (laminate panel, $n=125$ with 10 right-censored; nickel-base superalloy, $246\to115$ observations after trimming, 32 unique strain levels) matches this repo's 75-observation, 5-stress-level×15-replicate structure. What *does* match that structure is the **concrete-fatigue dataset** used in the authors' *Response* to discussants (§2.2, in `roy_km/_raw/pascual_meeker_1999_response.pdf` — the author's separate local knowledge-base repo, not included here and not resolvable by GitHub readers): "five stress levels with 15 measurements each." That is a structural match only — the actual numeric values here (102.95, 280.32, …, 11748.1 at $S=0.675$) have **not** been cross-checked against Castillo & Hadi (1995)'s original concrete data (not available in this repo), so this is not yet a confirmed re-attribution, just a strong reason to distrust the "aluminum alloy, R.R. Moore" label below. Tracked in `todo.md` pending that verification.

**Source (as previously documented, credibility now in question — see callout above):** Pascual & Meeker (1999) *Technometrics* — aluminum alloy (R.R. Moore rotating bending fatigue test)

| Data Characteristic | Value |
|---------------------|-------|
| Total observations | 75 |
| Failures per raw data (no censoring column) | 75 |
| Stress levels | 5 (S = 0.675, 0.750, 0.825, 0.900, 0.950 ksi) |
| Specimens per stress level | 15 |
| Response variable | $\ln N$ (log number of cycles to failure) |

Some (now-removed) scripts additionally applied a synthetic censoring label — see note below — splitting these into 73 "failures" + 2 "censored"; that split is an artifact of those scripts, not a property of the raw data.

> **Note (2026-08-11, revised 2026-08-20):** This repo has no trustworthy censoring-indicator metadata for this dataset — the CSV/hardcoded arrays carry no source-verified `event`/censoring column (see the source-attribution warning above). `rfl_pymc_da.py` (and, historically, three now-removed scripts — `rfl_model_zscore_asse.py`, `rfl_hl_asse.py`, `rfl_bayes_asse.py`; see §8) synthesize 2 censored observations as a live test case for the [censored-data extension](#34-extension-to-censored-data): within each stress-level group, if the maximum value is tied (appears more than once), all copies of that tied maximum are flagged `event=0` (censored) instead of `event=1` (failure), giving 73 failures + 2 censored. **`rfl_asse_full_sample.py` does not do this** — it treats all 75 observations as exact failures (§5.4). This heuristic is fragile — any genuine failure value that happens to tie for the group maximum would also be mislabeled — but for this specific dataset it only affects those 2 points.

---

## 5. Results

### 5.1 MCMC Convergence (SEV + LogNormal, Best Model)

> **⚠️ 2026-08-12 — "Divergences = 0" does not currently reproduce; see the full diagnostic in [[concept_model_zscore_bayes_hl]] (Obsidian-style wikilink into the author's separate local `roy_km` knowledge base, not this repo — not resolvable by GitHub readers; independently reviewed by 小o).** Re-running `rfl_pymc_da.py`'s exact SEV+LogNormal model (same code, same `initvals`, same `random_seed=[0,1,2,3]`) in the current environment gives **2214 divergences**, not 0 — confirmed **not** a script-specific error: all three RFL PyMC scripts run that day (this one and the two now-removed scripts `rfl_model_zscore_asse.py`, `rfl_hl_asse.py`; see §8) gave the identical divergence count when run side-by-side. Why the original "0" no longer reproduces is unconfirmed (a missing `g++`/different PyTensor compilation path is one plausible factor, not a verified cause).

| Metric | Value (⚠️ historical run, not reproducible in the current environment — see callout above) |
|--------|-------|
| Max $\hat{R}$ | 1.032 |
| Min ESS | 117 |
| Divergences | 0 |
| LOO-ELPD | −76.39 (better than Normal+LN = −79.38) |

#### Metric Interpretation

**$\hat{R}$ (Potential Scale Reduction Factor)**

The Gelman-Rubin convergence diagnostic compares within-chain variance to between-chain variance:

$$\hat{R} = \sqrt{\frac{\text{between-chain var} + \text{within-chain var (weighted)}}{\text{within-chain var}}}$$

- $\hat{R} = 1$: all chains have fully converged to the same distribution
- $\hat{R} < 1.01$: strict criterion; good convergence
- $\hat{R} < 1.05$: lenient criterion; generally acceptable
- $\hat{R} \geq 1.1$: insufficient convergence; extend sampling or redesign priors

**This study (historical, unreproducible run — see callout above): max $\hat{R}$ = 1.032** — would be within the lenient threshold if reproducible, but re-running the identical code/seed today gives 2214 divergences instead of 0, so this $\hat R$ shouldn't be read as a current, trustworthy convergence statement. The worst parameter is likely $\sigma_\Delta$ (true value ≈ 0.038, near-degenerate, harder to sample).

---

**ESS (Effective Sample Size)**

MCMC chains exhibit autocorrelation; the nominal 4000 samples carry information equivalent to ESS independent samples:

$$\text{ESS} = \frac{4000}{1 + 2\sum_{k=1}^{\infty}\rho_k}$$

where $\rho_k$ is the lag-$k$ autocorrelation. Higher autocorrelation yields smaller ESS.

- ESS > 400: reliable estimation of means and variances
- ESS > 100: minimum acceptable threshold
- ESS < 100: increase sampling count

**This study (historical, unreproducible run — see callout above): min ESS = 117** — barely above threshold, indicating moderate autocorrelation for some parameter (likely $\sigma_\Delta$). For more precise posterior quantiles, increase warm-up or sample count.

---

**Divergences**

During HMC/NUTS numerical integration, if energy conservation is severely violated (typically in regions of sharp posterior curvature, such as Neal's Funnel), the step "diverges." Posterior regions near divergent points are not correctly explored, biasing posterior estimates.

- Divergences = 0: HMC operating normally; flat posterior geometry
- Divergences > 0: reduce step size (increase `target_accept`) or switch to NCP

**This study: Divergences = 0 as originally reported, but not reproducible as of 2026-08-12** — see the callout at the top of §5.1 and the full diagnostic in [[concept_model_zscore_bayes_hl]] (roy_km wiki, independently reviewed by 小o, `codex exec`, delegation-marked). Confirmed: not a script-to-script difference — in that 2026-08-12 re-run, `rfl_pymc_da.py` and the two other RFL scripts run that day (`rfl_model_zscore_asse.py`, `rfl_hl_asse.py`; both since removed, see §8) all gave the identical 2214-divergence count under identical settings. What's *not* confirmed: an initial mean-based comparison suggested divergent draws correlate with larger $\sigma_\Delta$ rather than proximity to the hard $S_j - \Delta_i$ boundary — 小o's review flagged real gaps in that comparison (autocorrelated draws treated as independent samples, mean instead of tail quantiles, a stored divergent draw being the trajectory endpoint rather than necessarily where integration failed), so this NCP/CP-regime story is a **hypothesis, not a finding**. A sharper, independently-flagged lead: the code's docstring claims a truncated LogNormal for $\Delta_i$, but the implementation is an *untruncated* LogNormal plus a `pt.maximum(...)` soft penalty in the likelihood — a real model/implementation mismatch that's a more likely direct source of difficult geometry, and the better starting point for future work. Raising `target_accept` from 0.85 (2214 divergences) through 0.90/0.95/0.99 (1638/932/388) reduces but does not eliminate them, and 0.99 introduces `max_treedepth` warnings on all 4 chains (step-size sensitivity, not proof of a step-size-independent problem). Reparameterization (targeting the truncation mismatch above) is real modeling work, not implemented here; Roy's 2026-08-12 decision was to document this and move on rather than implement it in this pass.

---

**LOO-ELPD (Leave-One-Out Expected Log Pointwise Predictive Density)**

Leave one observation out each time, train on the remaining 74, compute the log predictive density for the held-out point, and sum:

$$\text{LOO-ELPD} = \sum_{i=1}^{75} \log p(y_i \mid \mathbf{y}_{-i})$$

- Larger values (closer to 0) indicate better predictive performance
- A difference $\Delta\text{ELPD} > 2$ is generally considered meaningful

**This study (historical, unreproducible run — see callout above):** SEV+LN = −76.39, Normal+LN = −79.38, $\Delta = 2.99$. SEV's heavier-tailed behavior better captures fatigue life data.

### 5.2 Posterior Parameter Estimates (full-sample, current)

Posterior means (θ̂ plug-in) from the HL-selected `target_accept` candidate for each branch — same numbers used to compute §5.4's full-sample ASSE, from `rfl_asse_full_sample.py`:

| Parameter | SEV + LogNormal (target_accept=0.85) | Normal + LogNormal (target_accept=0.95) |
|-----------|:-----------------------------------:|:---------------------------------------:|
| $\hat\beta_0$ | −9.3723 | −9.4315 |
| $\hat\beta_1$ | −9.5951 | −10.1187 |
| $\hat\sigma$ | 0.1893 | 0.2219 |
| $\hat\mu_\Delta$ | −0.7204 | −0.7686 |
| $\hat\sigma_\Delta$ | 0.0407 | 0.0453 |

> ⚠️ Same non-convergence caveat as §5.4 applies to these θ̂: neither branch met the script's combined acceptance check (see §3.5 for the exact `DIVERGENCE_OK`/`RHAT_OK`/`ESS_OK` thresholds and their comparators). Posterior SDs are not tabulated here because the un-converged chains do not give a trustworthy posterior spread; the posterior means above are used only as plug-in point estimates in §5.4's percentile method.

<details>
<summary>Prior (2026-06) SEV+LogNormal posterior means — historical, not reproducible in the current environment (see §5.1 callout)</summary>

| Parameter | Posterior Mean | Posterior SD | MLE Reference |
|-----------|:--------------:|:------------:|:-------------:|
| $\beta_0$ | −9.286 | 0.427 | −9.370 |
| $\beta_1$ | −8.746 | 1.355 | −8.534 |
| $\sigma$ | 0.195 | 0.072 | 0.190 |
| $\mu_\Delta$ | −0.660 | 0.082 | −0.644 |
| $\sigma_\Delta$ | 0.038 | 0.006 | 0.036 |

Historical run's posterior means nearly reproduced MLE; kept here only as an audit trail against which the current-environment estimates (both branches, above) can be compared.

</details>

### 5.3 Heteroscedastic Structure Across Stress Levels

> ⚠️ Same historical, unreproducible run as §5.1 (see callout there) — not regenerated in this pass.

| $S_j$ | $E(\ln Y \mid S_j)$ | SD (SEV) | SD (Normal) |
|-------|:-------------------:|:--------:|:-----------:|
| 0.675 | 6.82 | **1.158** | 1.127 |
| 0.750 | 3.38 | 0.808 | 0.794 |
| 0.825 | 0.91 | 0.599 | 0.586 |
| 0.900 | −0.99 | 0.470 | 0.471 |
| 0.950 | −2.10 | 0.525 | 0.508 |

The SD at low stress (S=0.675) is **2.5x** that at high stress (S=0.9), revealing a pronounced heteroscedastic structure. (SD (Normal) column corrected 2026-08-11 — both SEV and Normal columns now consistently tick back up at S=0.950.)

### 5.4 Full-Sample ASSE (n=75, no synthetic censoring) — `rfl_asse_full_sample.py`

> [!warning] Provisional — sampler has **not converged**, numbers below are the best available, not a validated result
>
> **Why this table exists.** Earlier versions of this README computed ASSE over 73 "failures" only — the 2 tied-max observations at $S=0.675$ (both 11748.1 cycles) were treated as synthetically right-censored. §4 notes that no code path in this repo has a trustworthy record of genuine censoring metadata for this dataset — some earlier scripts synthesize 2 censored observations from the tied-max heuristic described there (`rfl_asse_full_sample.py` does not: it treats all 75 as exact). `rfl_asse_full_sample.py` (new, 2026-08-19) computes ASSE the "model z-score" way (θ̂ plug-in, percentile → $\hat\Delta_i$ → $\hat y_i$) over the full 75-observation sample with **no synthetic censoring at all**; the DA/NCP latent-$z_\Delta$-posterior reconstruction previously used elsewhere in this repo was rejected as in-sample circular reconstruction — it uses each $y_i$'s own value, via the joint likelihood, to infer that same observation's $\Delta_i$, then reconstructs $\hat y_i$ from it and compares back to $y_i$.
>
> **What in the percentile method is and isn't grounded in P&M (1999) — precisely scoped, not an argument invented for this repo, but also not a full endorsement.** Section 4.5 of the original paper ("Residual Analysis") defines, for a fixed ML estimate $\hat\theta$, the standardized residual $e_i^{*} = [\log(y_i) - \hat\mu(x_i)] / \hat\sigma(x_i)$, where $\hat\mu(x_i)$/$\hat\sigma(x_i)$ are the ML-estimated mean/SD of $\log$ life at that stress level **conditional on the specimen failing** (i.e., $\Delta<S$) — this is the same *form* as $E(Y\mid S_j)$/$\sqrt{V(Y\mid S_j)}$ in this repo's notation, and $e_i^*$ the same form as the $z_i$ used here, though not the same *values*: P&M plugs in the ML estimate $\hat\theta$, this repo plugs in a Bayesian posterior-mean $\hat\theta$ instead. P&M (1999) uses $e_i^*$ **only** to build residual-vs-stress diagnostic plots (Figs. 9/16 of the paper) — it never inverts $e_i^*$ back through $\Delta$'s distribution, never reconstructs $\hat y_i$ from it, and never sums it into a score. Section 4.4 is a **separate** procedure — an EDF/probability-integral-transform goodness-of-fit test using $z_i=F_W(w_i;x_i,\hat\theta)$ (the full marginal CDF, not the two-moment Normal approximation) — used only for a Kolmogorov–Smirnov test, not the same construction as §4.5's $e_i^*$ and not something this README should describe as "a sharper version" of it. So: **the standardized-residual formula ($z_i$) is P&M's own established technique; the further steps this repo adds on top of it — inverting $z_i$ through $\Delta$'s LogNormal quantile to get $\hat\Delta_i$, reconstructing $\hat y_i$, and summing $|y_i-\hat y_i|$ into an ASSE-style score — are this repo's own construction, not verified or endorsed by P&M (1999).** It is meaningfully different from the DA/NCP latent-posterior approach in one specific, checkable sense: it does not sample or compute each observation's conditional posterior $p(\Delta_i\mid y_i,\theta)$; it applies one fixed, global set of plug-in moments $(\hat\theta)$ to every observation and maps each residual to a quantile deterministically — see §7.2 for the precise (corrected) statement of this distinction.
>
> The response to discussants (§2.2) computes a separate, summed absolute-error criterion, $E=\sum_j\sum_i|\log(y_{ij})-\log(\hat y_{ij})|$, benchmarking the RFL model against Castillo & Hadi (1995) and **five** other published S-N models (Little & Ekvall twice, Spindel & Haibach, Bastenaire, Castillo et al. 1985) — but there $\hat y_{ij}$ comes from inverting the model's marginal CDF at a **fixed rank-based plotting position** $p_i=(i-.5)/15$ (determined by $y_{ij}$'s rank among the 15 replicates at that stress level, not by its numeric magnitude, though the rank itself still comes from sorting the same observed data, and $\hat\theta$ is still fit on the full sample) — this is this README's "rank-ASSE" (§6.1), a related but distinct construction from the $z_i$-based percentile method used here, and it is also an in-sample construction by this README's own standard, not something exempted from the removals below. Whether the "sum standardized residuals into one score" convention this repo has historically attributed to Chiu (2005) actually originates there has been **partially verified in this pass** — Chiu's thesis Table 3 (p. 23) is a published ASSE-style comparison across seven S-N models on the P&M dataset, reproduced verbatim in §6.2 — but the exact per-observation matching Chiu used for competitor models (rank plot position vs sample-based) is only inferred from P&M's Response §2.2 characterisation, not directly re-derived from Chiu's chapter text.
>
> **Two real bugs found and fixed in the process** (independent code review, `codex exec`, delegation-marked): (1) the 32-point Gauss-Legendre quadrature's own weighted sum, used as the $P(\Delta<S_j)$ normalization, had up to ~2.8% error (some stress levels came out **above 1**, which is impossible for a probability). $N_\text{QUAD}$ was raised to 128 (validated convergent near the current posterior region), and — since $\Delta$ is LogNormal so $P(\Delta<S_j)=\Phi((\ln S_j-\mu_\Delta)/\sigma_\Delta)$ has a closed form — that analytic value is now computed alongside the quadrature sum purely as a **cross-check**: `gl_moments()` still divides the numerator ($E_1$, which has no closed form) by the quadrature-based sum, but raises an error if the two disagree by more than 1%, rather than silently returning a moment computed from a possibly-wrong normalization. (2) The SEV branch's hand-written likelihood used `pt.clip(z_f, -500, 20)` — once a residual's $z$-score exceeded the clip ceiling, the likelihood's exponential term stopped changing but the linear term kept growing, giving the **wrong** log-density and gradient direction beyond that point, not just numerical protection. Fixed by switching to PyMC's built-in `pm.Gumbel` (max-type) with the standard sign-flip trick for the min-type SEV/Gumbel we need ($Y\sim\text{SEV}(\mu,\sigma) \Leftrightarrow -Y\sim\text{Gumbel}_\max(-\mu,\sigma)$), which needs no manual clipping.
>
> **A real, unfixed problem: the `Δ<S` support is still a soft floor, not a hard constraint.** `pt.maximum(S_OBS-delta, 1e-8)` lets $\Delta_i \geq S_i$ samples get a finite (clipped) likelihood instead of being properly excluded from the model's support, creating a kink/flat-gradient region right where the sampler needs to be well-behaved. This is a **pre-existing pattern throughout this repo** (`rfl_pymc_da.py` and, historically, the now-removed `rfl_bayes_asse.py`), not something new in this file — but it appears to be the actual root cause of this model's persistent divergence problems (see next paragraph), and the correct fix is a genuine reparameterization, not another patch. **Not fixed in this pass** — tracked in `todo.md`.
>
> **HL-loop candidate search (2026-08-19): 8 independent NUTS fits, none converged — and the pattern rules out "just tune harder."** Per §3.5's HL formulation (state=convergence diagnostics, policy=`target_accept` ∈ {0.85, 0.90, 0.95, 0.99} with `draws=tune=2000` fixed, feedback=per-candidate `(Rhat_max, n_divergences)`, selection=`argmin` lexicographic), both SEV and Normal branches:
>
> | SEV target_accept | divergences | Rhat_max | ESS_min |
> |:-:|:-:|:-:|:-:|
> | 0.85 | 3165 | 1.18 | 16.0 |
> | 0.90 | 3262 | 1.19 | 15.0 |
> | 0.95 | 696 | 1.53 | 7.0 |
> | 0.99 | **0** | **1.50** | 7.0 |
>
> | Normal target_accept | divergences | Rhat_max | ESS_min |
> |:-:|:-:|:-:|:-:|
> | 0.85 | 1442 | 1.15 | 20.0 |
> | 0.90 | 1791 | 1.23 | 30.0 |
> | 0.95 | 859 | 1.14 | 20.0 |
> | 0.99 | 51 | 1.58 | 7.0 |
>
> SEV at `target_accept=0.99` hit **zero divergences** — and still has Rhat=1.50, because all 4 chains exhausted `max_treedepth`: the sampler took safely tiny steps and never actually traversed/mixed across the posterior. Zero divergences with terrible Rhat is the textbook signature of a geometry problem HMC/NUTS step-size tuning cannot fix — consistent with the un-fixed `Δ<S` soft-floor being the real blocker, not sampler configuration.
>
> **Result: neither branch met the script's combined acceptance check (see §3.5 for the exact thresholds and comparators).** Per Roy's 2026-08-19 decision, the numbers below are recorded as the current best-effort (lowest Rhat_max among the 4 candidates each), explicitly **not validated**, pending the reparameterization fix (tracked in `todo.md`).

| | SEV (target_accept=0.85) | Normal (target_accept=0.95) |
|---|:-:|:-:|
| $\hat\theta$ | $\beta_0 = -9.3723$, $\beta_1 = -9.5951$, $\sigma = 0.1893$, $\mu_\Delta = -0.7204$, $\sigma_\Delta = 0.0407$ | $\beta_0 = -9.4315$, $\beta_1 = -10.1187$, $\sigma = 0.2219$, $\mu_\Delta = -0.7686$, $\sigma_\Delta = 0.0453$ |
| **Full-sample (n=75) ASSE** | **4.7503** | **4.4573** |
| Divergences | 3165 | 859 |
| Rhat_max | 1.18 | 1.14 |
| ESS_min | 16.0 | 20.0 |

This table exists to answer "what's the full-sample ASSE" honestly, not to declare a new best method.

---

## 6. Comparison with Prior Work

### 6.1 ASSE Metric Notes

> **Important:** Different studies use different prediction evaluation spaces; values are not directly comparable across spaces.

| Space | Definition | Used by |
|-------|------------|---------|
| **ASSE** (z-score space, **z-ASSE**) | $\sum_i \|\omega_{ij} - \hat{\omega}_{ij}\|$ | Chiu (2005), Roy C-2, SEV+INLA z-score |
| **ASSE** (ln-lifetime space, **y-ASSE**) | $\sum_i \|y_{ij} - \hat{y}_{ij}\|$ | This work (model z-score method) |
| **ASSE** (rank space, **rank-ASSE**) | $E = \sum_i \|y_{ij} - \hat{y}_{(ij)}\|$ (rank-matched) | P&M (1999) |

### 6.2 Historical S-N Model Comparison (Chiu 2005, Thesis Table 3) with this Repo's Result Appended

Chiu (2005) benchmarked the RFL model (Nor-Nor, Normal residual + Normal $\Delta$) against six other published S-N models plus Pascual & Meeker (1999)'s own RFL fit on the same $n=75$ P&M dataset. The comparison metric is $\text{ASSE} = \sum_j \sum_i |\omega_{ij} - \hat\omega_{ij}|$ where $\omega_{ij} = \ln y_{ij}$; for models without per-observation $\Delta$ estimates, $\hat\omega_{ij}$ is matched via inverting the model's marginal CDF at the rank plotting position $p_i = (i-0.5)/15$ (this is what P&M's Response paper §2.2 later formalised).

**Chiu (2005) Table 3, verbatim (thesis p. 23):**

| Model | Parameters | ASSE |
|-------|:----------:|:----:|
| Little and Ekvall (1981), variant 1 | 3 | 41.13 |
| Little and Ekvall (1981), variant 2 | 3 | 31.17 |
| Spindel and Haibach (1981) | 6 | 17.35 |
| Bastenaire (1972) | 5 | 20.52 |
| Castillo et al. (1985) | 4 | 20.27 |
| Castillo and Hadi (1995) | 5 | 18.12 |
| Pascual and Meeker (1999), Nor-Nor | 5 | 12.84 |
| Chiu (2005), Nor-Nor | 5 | **10.80** |

**This repo's current-run full-sample ASSE (§5.4, appended for reference):**

| Model | Parameters | ASSE | Notes |
|-------|:----------:|:----:|:-----:|
| This repo, Normal + LogNormal (percentile method, §5.4) | 5 | **4.4573** | ⚠️ see caveat below |
| This repo, SEV + LogNormal (percentile method, §5.4) | 5 | **4.7503** | ⚠️ see caveat below |

> **⚠️ These two rows are NOT directly comparable to Chiu's numbers in absolute value.** Chiu (2005)'s ASSE aligns $\hat\omega_{ij}$ to observed $\omega_{ij}$ via a rank-based plot position ($p_i = (i-0.5)/15$), so every model — including those without per-observation $\Delta$ — is scored on the same convention. This repo's §5.4 numbers use the **percentile method** described in §3.2/§3.3: $\hat y_i$ comes from inverting $\Delta$'s LogNormal quantile at $\Phi(z_i)$ where $z_i$ is the individual standardized residual (not a rank plot position). Both are y-space $|\omega - \hat\omega|$ sums, but the two constructions can differ by a factor of several. The rows are placed side-by-side because Roy asked for them here for context — not because "4.4573 beats 10.80". A common-convention re-computation of this repo's method against Chiu's rank-based scoring would need to be redone from scratch and has not been performed in this pass. Additionally: this repo's numbers come from a **non-converged** sampler (§5.4).

The other in-sample summary tables that used to live here (Chiu (2005) and Roy Methods A/B/C-1/C-2 in z-score space; `SEV+INLA`/`Burr+INLA`/`Burr+EM-GMM` and retired 73-observation model z-score / sample z-score / DA-plugin numbers in ln-lifetime space; the rank-matched $E$ criterion in rank space) were removed 2026-08-20 per Roy's decision — see `git log` if that history is needed.

### 6.3 Method Comparison Summary

| Method | Parameter Uncertainty | Heteroscedasticity | Separable Delta vs sigma | Computational Cost |
|--------|:---------------------:|:------------------:|:------------------------:|:-----------------:|
| Chiu (2005) z-score | ✗ | ✗ | ✗ | Low |
| MLE + INLA | ✗ | Partial | ✗ | Medium |
| Bayes DA + sample z-score | ✓ | ✗ | ✗ | High |
| **Bayes DA + model z-score (this work)** | ✓ | ✓ | ✓ | High |

---

## 7. Theoretical Contributions

### 7.1 Correctness of the Model Moment Substitution

The core substitution of this method:

$$\underbrace{\omega_{ij} = \frac{y_{ij} - \bar{y}_j}{s_j}}_{\text{Chiu: sample statistic, noise mixed in}} \longrightarrow \underbrace{z_{ij} = \frac{y_{ij} - E(\ln N \mid S_j)}{\sqrt{V(\ln N \mid S_j)}}}_{\text{This work: model statistic, noise separated}}$$

In $V(\ln N \mid S_j) = \beta_1^2 \cdot \text{Var}[\ln(S_j - \Delta)] + \pi^2\sigma^2/6$, the first term is the contribution of $\Delta$ and the second is the SEV residual — the two sources are cleanly separated by full Bayesian inference.

### 7.2 Relation to the Conditional Posterior (corrected 2026-08-20)

An earlier version of this section claimed the model z-score pipeline is "essentially computing an approximation to the conditional posterior of each individual $\Delta_i$," $p(\Delta_i\mid y_i,\theta)\propto f(y_i\mid\Delta_i,\theta)\cdot g(\Delta_i\mid\theta)$ — flagged by review (2026-08-20) as contradicting §5.4's claim that the percentile method is *not* the same operation as DA/NCP's per-observation posterior. The precise, non-contradictory statement: it does **not** sample or evaluate $p(\Delta_i\mid y_i,\theta)$ for any $i$. It fixes one global $\hat\theta$ from the full-sample fit, computes the marginal (Δ-integrated-out) moments $E(Y\mid S_j)/V(Y\mid S_j)$ once per stress level, and applies a deterministic residual-to-quantile mapping to each $y_i$ — a moment-matching heuristic for the inverse problem $y_i\to\Delta_i$, not a Bayesian posterior computation. Whether this heuristic is a good approximation to the true per-observation posterior is an open question this repo has not verified either way.

### 7.3 Normal Approximation vs. Exact Marginal CDF

Using the two-moment Normal approximation $\Phi(z)$ to map a residual to a percentile is a simplification of the exact marginal CDF conditional on failure,

$$F_W(w \mid S_j, \hat\theta) = \frac{\displaystyle\int_0^{S_j} F_{\text{SEV}}(w \mid \beta_0 + \beta_1 \ln(S_j - \Delta), \sigma) \cdot g(\Delta \mid \hat{\mu}_\Delta, \hat{\sigma}_\Delta) \thinspace d\Delta}{F_\Delta(S_j)} \neq \Phi(z).$$

$\Phi(z)$ only matches the first two moments of this true marginal; it does not generally over- or under-state percentiles in a fixed direction — that depends on the true marginal's skewness and tail weight relative to Normal, which is itself a function of $\hat\sigma_\Delta$ at each $S_j$, not a universal property. This repo does not currently have a validated claim about which direction the approximation error goes for this model/data — an earlier draft asserted one direction without support and has been removed.

---

## 8. Code Description

### Core Code (This Work)

| File | Function | Priority |
|------|----------|:--------:|
| `rfl_pymc_da.py` | **Main inference:** DA+NCP+NUTS (SEV+LN and Normal+LN models) | ⭐⭐⭐ |
| `rfl_asse_full_sample.py` | **Model z-score ASSE (current):** full n=75 sample, no synthetic censoring, θ̂ plug-in percentile method, HL candidate search over `target_accept` — see §5.4 | ⭐⭐⭐ |

### Removed (2026-08-20)

`rfl_model_zscore_asse.py`, `rfl_hl_asse.py`, `rfl_bayes_asse.py` — superseded by `rfl_asse_full_sample.py` and deleted from this repo; see §5.4 and `git log` for their history.

### Auxiliary Code (Development)

| File | Function | Notes |
|------|----------|-------|
| `rfl_bayes_mcmc.py` | Pure NumPy Metropolis MCMC | No PyMC dependency; used for verification |
| `rfl_bayes_uniform.py` | PyMC + `as_op` + GL integration | Marginal likelihood version; slower |
| `rfl_pymc.py` | PyMC + numerical integration (4 variants) | Early attempt |
| `rfl_pymc_ncp.py` | PyMC + NCP (log space + GH in z-space) | NCP development version |

---

## 9. How to Run

### Requirements

```bash
pip install pymc numpy scipy arviz
```

Recommended PyMC version >= 5.0, using pytensor for automatic differentiation.

### Main Inference (approx. 5–7 minutes, 4 chains)

```bash
# Full Bayesian inference via DA+NCP+NUTS, running both SEV+LN and Normal+LN
python rfl_pymc_da.py
```

Output: posterior samples, Rhat, ESS. (Does not compute LOO-ELPD or ASSE — the §5.1 LOO-ELPD numbers were produced by a separate, ad hoc analysis, not by this script's current main block; for ASSE see `rfl_asse_full_sample.py` / §5.4.)

### Full-Sample Model Z-Score ASSE (approx. 20–30 minutes — runs an HL candidate search, 4 NUTS fits per error model)

```bash
# Full n=75 sample, no synthetic censoring, theta_hat plug-in percentile method,
# HL loop searches target_accept in {0.85,0.90,0.95,0.99} and keeps the best-Rhat candidate
python rfl_asse_full_sample.py
```

Output: per-candidate convergence diagnostics, chosen $\hat\theta$ for SEV and Normal, full-sample ASSE. **Neither branch currently converges** (see §5.4) — this is documented, expected behavior pending the `Δ<S` reparameterization fix tracked in `todo.md`, not a bug in this script.

---

## 10. References

### Key References

1. **Pascual, F. G., & Meeker, W. Q. (1999).** Estimating fatigue curves with the random fatigue-limit model. *Technometrics*, 41(4), 277–289.
   *Foundational RFL model paper. The n=75 dataset used here does not match either of this paper's own two datasets — see §4's provenance caveat.*

2. **Chiu, C.-H. (2005).** *二元分佈族及其統計推論 (A Family of Bivariate Distributions With Some Applications to Statistical Inferences).* Master's Thesis, Graduate Institute of Management Science, Tamkang University, June 2005. Advisor: Wen-Tao Huang. Local copy: `C:/Users/roy/MD thesis.pdf`.
   *Chapter 2 applies the Nor-Nor variant of the RFL family to the Pascual & Meeker (1999) fatigue dataset; Table 3 (thesis p. 23) is the published ASSE comparison against six other S-N models and is reproduced verbatim in §6.2.*

3. **Hoffman, M. D., & Gelman, A. (2014).** The No-U-Turn Sampler: Adaptively setting path lengths in Hamiltonian Monte Carlo. *Journal of Machine Learning Research*, 15(1), 1593–1623.
   *The NUTS algorithm; the core MCMC engine used in this study.*

4. **Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2013).** *Bayesian Data Analysis* (3rd ed.). CRC Press.
   *Methodological basis for Non-Centered Parameterization (the Neal's Funnel fix).*

5. **Salvatier, J., Wiecki, T. V., & Fonnesbeck, C. (2016).** Probabilistic programming in Python using PyMC3. *PeerJ Computer Science*, 2, e55.
   *The PyMC probabilistic programming framework.*

### Methodological References

6. **Tanner, M. A., & Wong, W. H. (1987).** The calculation of posterior distributions by data augmentation. *Journal of the American Statistical Association*, 82(398), 528–540.
   *Theoretical foundation for Data Augmentation.*

7. **Neal, R. M. (2003).** Slice sampling. *Annals of Statistics*, 31(3), 705–767.
   *Neal's Funnel was first described by Neal; NCP is its standard fix.*

8. **Weng, J. (2026).** Learning beyond gradients: Heuristic learning for sequential decision problems. Preprint.
   *The original Heuristic Learning paper; this study applies the HL policy-search pattern to NUTS sampler configuration selection (§3.5 for the methods-level statement, §5.4 for the actual per-candidate diagnostics), **not** to choosing among prediction pipelines. §3.5's inline `[[concept_heuristic_learning]]` is an Obsidian wikilink into the author's separate local knowledge base and is not resolvable by GitHub readers — the primary citable source is this reference.*

9. **Murphy, S. A., & van der Vaart, A. W. (2000).** On profile likelihood. *Journal of the American Statistical Association*, 95(450), 449–465.
   *Theoretical basis for Profile Likelihood; relevant to Roy's semiparametric RFL work.*

10. **Meeker, W. Q., & Escobar, L. A. (1998).** *Statistical Methods for Reliability Data*. John Wiley & Sons.
    *Standard reference for SEV distribution (log-Weibull) and statistical methods for fatigue analysis.*

### Bayesian Computation References

11. **Vehtari, A., Gelman, A., & Gabry, J. (2017).** Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC. *Statistics and Computing*, 27(5), 1413–1432.
    *Methodological basis for PSIS-LOO; used in this study for model comparison (LOO-ELPD).*

12. **Betancourt, M. (2017).** A conceptual introduction to Hamiltonian Monte Carlo. *arXiv preprint arXiv:1701.02434*.
    *In-depth introduction to HMC and NUTS, including the geometric interpretation of NCP.*
