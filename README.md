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

#### Step 3: How ASSE Is Computed — Derivation and Worked Example (added 2026-10-29)

This step spells out *why* each line of the pipeline is what it is, and walks one stress level through it numerically. Everything below uses the plug-in $\hat\theta$ (posterior means) and the n=75 full sample of §5.4, with all 75 points treated as failures.

**(a) Mean and variance of $\ln N$ come from the *marginal* distribution of $\ln N$ (the mixture over $\Delta$).** Conditional on $\Delta$, $\ln N = \mu(S,\Delta) + \sigma\varepsilon$ with $\mu(S,\Delta)=\beta_0+\beta_1\ln(S-\Delta)$ and $\varepsilon \perp \Delta$. Only specimens with $\Delta<S$ fail, so $\Delta$ is restricted to $(0,S_j)$ (hence the normalization by $F_\Delta(S_j)$). With $e_\varepsilon$, $v_\varepsilon$ the mean and variance of the standardized error ($-\gamma_E$ and $\pi^2/6$ for SEV; $0$ and $1$ for Normal), the laws of total expectation and total variance give

$$E(\ln N\mid S_j)=\hat\beta_0+\hat\beta_1\,E_\Delta[\ln(S_j-\Delta)]+\hat\sigma e_\varepsilon,\qquad V(\ln N\mid S_j)=\hat\beta_1^2\,\mathrm{Var}_\Delta[\ln(S_j-\Delta)]+\hat\sigma^2 v_\varepsilon ,$$

where $E_\Delta$ and $\mathrm{Var}_\Delta$ are taken under $g(\Delta)$ truncated to $(0,S_j)$. The first variance term is the between-specimen ($\Delta$) part, the second is the within-specimen (residual) part; the cross term vanishes because $\varepsilon\perp\Delta$.

**(b) The percentile of $y_{ij}$ is matched to the same percentile of $\Delta$.** The model z-score $z_{ij}=(y_{ij}-E)/\sqrt{V}$ places the observation inside the marginal distribution of $\ln N$ at that stress; $p_{ij}=\Phi(z_{ij})$ is its percentile (a Normal approximation to the marginal CDF). Because $\beta_1<0$, a larger $\Delta$ shortens $S-\Delta$ and *lengthens* life, so life and $\Delta$ are monotonically related, and the observation is assigned the $\Delta$ at the **same percentile** of $g$:

$$\hat\Delta_{ij}=G^{-1}(p_{ij}) \;=\; \exp\!\big(\hat\mu_\Delta+\hat\sigma_\Delta\,\Phi^{-1}(p_{ij})\big)=\exp\!\big(\hat\mu_\Delta+\hat\sigma_\Delta\,z_{ij}\big)\quad\text{(LogNormal; } \Phi^{-1}(\Phi(z))=z\text{)} .$$

**(c) The prediction is the structural part at $\hat\Delta_{ij}$**, not a simulated draw: $\hat y_{ij}=\hat\beta_0+\hat\beta_1\ln(S_j-\hat\Delta_{ij})+\hat\sigma e_\varepsilon$ (the error is replaced by its mean), and $\text{ASSE}=\sum_{j}\sum_i|y_{ij}-\hat y_{ij}|$.

**Implementation details** (`rfl_model_zscore_asse.py`, `rfl_hl_asse.py`, `rfl_asse_full_sample.py`): the moments are computed by 32-point Gauss–Legendre on $(0,S_j)$ divided by $F_\Delta(S_j)$; $z$ is clipped to $[-6,6]$ in the H0 heuristic; $\hat\Delta$ is clipped to $[10^{-6},\,0.999\,S_j]$; the older 3.4766 / 3.6156 figures sum over the 73 non-tied observations, the current §5.4 figures over all 75.

**Worked example — SEV, $S_1=0.675$** with the §5.4 estimate $\hat\beta_0=-9.3723,\ \hat\beta_1=-9.5951,\ \hat\sigma=0.1893,\ \hat\mu_\Delta=-0.7204,\ \hat\sigma_\Delta=0.0407$:

| quantity | value |
|---|---|
| $E_\Delta[\ln(S_1-\Delta)]$, $\mathrm{Var}_\Delta[\ln(S_1-\Delta)]$ | $-1.6768$, $0.01160$ |
| $E(\ln N\mid S_1)=-9.3723+(-9.5951)(-1.6768)+(-0.1893\gamma_E)$ | $6.6076$ |
| $V(\ln N\mid S_1)=9.5951^2(0.01160)+\pi^2(0.1893)^2/6$ | $1.1272$ (sd $1.0617$) |

| obs (sorted) | $y$ | $z$ | $p=\Phi(z)$ | $\hat\Delta=e^{\hat\mu_\Delta+\hat\sigma_\Delta z}$ | $\hat y$ | $\lvert y-\hat y\rvert$ |
|---|---|---|---|---|---|---|
| 1 | 4.634 | −1.859 | 0.0315 | 0.45111 | 4.878 | 0.244 |
| 8 | 7.124 | 0.487 | 0.6867 | 0.49629 | 7.041 | 0.083 |
| 15 | 9.371 | 2.603 | 0.9954 | 0.54094 | 9.799 | 0.428 |

Summing all 15 observations of this level gives its ASSE contribution (1.752 for SEV); the five levels together give the totals below.

| error | per-level contributions ($S$ = 0.675, 0.75, 0.825, 0.9, 0.95) | total (independent re-implementation) | §5.4 |
|---|---|---|---|
| SEV | 1.752, 0.718, 0.700, 0.627, 0.955 | 4.7515 | 4.7503 |
| Normal | 2.002, 0.628, 0.593, 0.480, 0.754 | 4.4564 | 4.4573 |

The small differences come from rounding $\hat\theta$ to four decimals and from 32-point vs. high-order quadrature in the moments.

**Numerical-accuracy note (checked 2026-10-29).** The integrand is narrow (the $\Delta$ spread is small relative to $S-\Delta$), so fixed-node quadrature matters. For the H0 heuristic the effect is small (3.636 with 32 nodes vs. 3.603 with 400, SEV+LogNormal, 73 non-tied observations, $\hat\theta$ from an independent NUTS fit). For the exact-marginal-CDF heuristic H1, which evaluates the marginal CDF by the same quadrature, the effect is large: 12.917 with 32 nodes, 2.241 with 100 and 2.135 with 400 nodes, i.e. the earlier H1 > H0 ranking looks quadrature-limited. Those retired 73-observation comparisons were not otherwise re-run.

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

> **⚠️ 2026-08-20 → updated 2026-09-22: "aluminum alloy" label is very likely wrong; re-attribution to concrete-fatigue data is now strongly (but not fully) supported.** Neither of P&M (1999)'s own two datasets (laminate panel, $n=125$ with 10 right-censored; nickel-base superalloy, $246\to115$ observations after trimming, 32 unique strain levels) matches this repo's 75-observation, 5-stress-level×15-replicate structure. What does match is the **concrete-fatigue dataset** from Castillo & Hadi (1995), used in P&M's *Response* to discussants (§2.2, full text at `roy_km/_raw/pascual_meeker_1999_response.pdf` — the author's separate local knowledge-base repo, not included here and not resolvable by GitHub readers): "The data consisted of **five stress levels with 15 measurements each**." 2026-09-22 re-read of that Response in full cross-checked every reported summary statistic against this repo's derived wiki notes and found no discrepancy: log-likelihoods/AIC for all four RFL distribution combinations (Table 1: SEV-SEV −75.583/161.166, Normal-Normal −72.883/155.766, SEV-Normal −72.878/155.757, Normal-SEV −74.746/159.491) and the absolute-prediction-error comparison (Table 3: Normal-Normal RFL $E=12.84$, Castillo–Hadi's own model $E=18.12$) all match. **What is still unconfirmed**: the underlying Castillo & Hadi (1995) *JASA* paper (paywalled; no free full text found across a multi-round search including their 2009 book *A Unified Statistical Methodology for Modeling Fatigue Damage*) has not been obtained, so the 75 raw values here (102.95, 280.32, …, 11748.1 at $S=0.675$) have **not** been checked digit-for-digit against C&H's original table. Treat "concrete fatigue, Castillo & Hadi (1995)" as the working attribution (structural + all cross-checkable summary statistics match) but not a fully closed verification. Tracked in `todo.md`.

**Source (working attribution per callout above — structural and summary-statistic match confirmed; digit-level match against original C&H (1995) table not obtainable, paper paywalled):** Castillo & Hadi (1995), *Journal of the American Statistical Association* 90(431), 1041–1054 — concrete fatigue data, as used and re-analyzed in Pascual & Meeker (1999) *Technometrics* Response to discussants §2.2. (Previously mislabeled as "aluminum alloy, R.R. Moore rotating bending fatigue test" — that label does not match either of P&M's own two datasets and is now believed to be a stale carryover error.)

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
> The response to discussants (§2.2) computes a separate, summed absolute-error criterion, $E=\sum_j\sum_i|\log(y_{ij})-\log(\hat y_{ij})|$, benchmarking the RFL model against Castillo & Hadi (1995) and **five** other published S-N models (Little & Ekvall twice, Spindel & Haibach, Bastenaire, Castillo et al. 1985) — but there $\hat y_{ij}$ comes from inverting the model's marginal CDF at a **fixed rank-based plotting position** $p_i=(i-.5)/15$ (determined by $y_{ij}$'s rank among the 15 replicates at that stress level, not by its numeric magnitude, though the rank itself still comes from sorting the same observed data, and $\hat\theta$ is still fit on the full sample) — this is this README's "rank-ASSE" (§6.1), a related but distinct construction from the $z_i$-based percentile method used here; it is the same plot-position alignment that Chiu (2005) Table 3 uses (§6.2). Whether the "sum standardized residuals into one score" convention this repo has historically attributed to Chiu (2005) actually originates there has been **partially verified in this pass** — Chiu's thesis Table 3 (p. 23) is a published ASSE-style comparison across seven S-N models on the P&M dataset, reproduced verbatim in §6.2 — but the exact per-observation matching Chiu used for competitor models (rank plot position vs sample-based) is only inferred from P&M's Response §2.2 characterisation, not directly re-derived from Chiu's chapter text.
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

### 5.5 Error × Fatigue-Limit Model Zoo, JAX/NumPyro NUTS — `jax_nuts_zoo/` (added 2026-10-29)

> [!note] Scope
> Same data (n=75, all failures, §5.4 protocol), same y-ASSE pipeline as §3.2 Step 3 — only the model family is widened: error $\in$ {Normal, SEV, Student-t} $\times$ fatigue limit $\Delta \in$ {LogNormal, Gamma, Weibull} = 9 models, each fit by full Bayes and scored with the **same** plug-in y-ASSE. Single-seed point estimates; no uncertainty on the ASSE values. The ADR (agent-chosen distribution) task and the HL loop over structures that motivated this are **not** part of this commit.

**How it differs from §5.4's sampler.** `jax_nuts_zoo/rfl_zoo.py` integrates $\Delta$ out of the likelihood (Gauss–Hermite in $z$-space for LogNormal, Gauss–Legendre on $(0,S)$ for Gamma, probability-integral-transform quadrature for Weibull), so there is no soft floor on $S-\Delta$ and no latent $\Delta_i$ to sample. The sampler is NumPyro NUTS with a dense mass matrix, 2 chains × (1000 warm-up + 1000 draws), `target_accept` 0.9. Shape gradients of `betainc`/`gammaincc` (missing in JAX) use a finite-difference `custom_jvp`. Equivalence with the repo's numpy likelihood (`rfl_pymc_ncp.py`) is checked in `test_zoo.py` (max difference $\le 1.1\times10^{-12}$ at matched node counts; gradients vs finite differences).

**Why 200/400 quadrature nodes.** The integrand in $\Delta$ is narrow, so fixed small rules are inaccurate: with Gauss–Legendre 24 nodes the Gamma-limit log-marginal is off by 0.31–0.37 (vs $\sim10^{-14}$ at 400 nodes), and the old 32-node construction moves H0 from 3.636 to 3.603 and H1 from 12.917 to 2.135 as nodes go 32 → 400 (`test_quadrature.py`, SEV + LogNormal on the censored-maxima fit). The zoo uses 200 Gauss–Hermite and 400 Gauss–Legendre nodes. Weibull-limit log-marginals agree with adaptive `scipy.integrate.quad` to $3.4\times10^{-3}$ (worst case over 48 parameter/stress combinations).

**y-ASSE (plug-in = posterior means), n = 75** (`jax_nuts_zoo/results/yasse_full75.csv`, produced by `make_table.py full75`). H0 is the pipeline of §3.2 (Φ(z) percentile → $\hat\Delta$ → $\hat y$); H1 uses the exact marginal CDF instead of Φ(z); H3 is per-stress-level z calibration (sd floor 0.5); (A fourth heuristic, $E[\Delta_i\mid y_i,\hat\theta]$, uses the conditional posterior of each observation's own $\Delta_i$ given $y_i$, the circular reconstruction §5.4 rejects, and was dropped. H0, H1 and H3 are different: with $\hat\theta$ fixed they map $y_i$ through the marginal law at its stress level, a monotone transform, and never infer $\Delta_i$ from its own likelihood; they still use $\hat\theta$ fitted on the same 75 points, as the §5.4 pipeline does.)

| Error | $\Delta$ | Rhat max | ESS min | Divergences | **H0** | H1 | H3 |
|---|---|:-:|:-:|:-:|:-:|:-:|:-:|
| Normal | LogNormal | 1.020 | 147 | 5 | **4.332** | 2.337 | 12.738 |
| Normal | Gamma | 1.020 | 197 | 107 | 7.781 | 1.093 | 33.165 |
| Normal | Weibull | 1.030 | 127 | 82 | 5.957 | 2.780 | 12.107 |
| SEV | LogNormal | 1.030 | 99 | 1 | **4.383** | 2.377 | 12.310 |
| SEV | Gamma | 1.040 | 33 | 153 | 7.667 | 0.949 | 32.543 |
| SEV | Weibull | 1.020 | 107 | 109 | 5.433 | 1.718 | 10.864 |
| Student-t | LogNormal | 1.030 | 161 | 5 | **4.132** | 2.038 | 12.583 |
| Student-t | Gamma | **1.630** | **3** | 207 | 7.461 | 0.420 | 28.067 |
| Student-t | Weibull | 1.000 | 226 | 11 | 6.118 | 2.933 | 12.078 |

**What can and cannot be read from this table.**

- **Student-t × Gamma did not converge (Rhat 1.63, ESS 3): its row is not usable.** The other Gamma fits and the Normal/SEV Weibull fits have 82–153 divergences (ESS as low as 33), so those rows are provisional too; Student-t × Weibull has 11. Only the LogNormal fits (1–5 divergences, Rhat $\le$ 1.03) are comparable in quality to each other. ESS min of 99–161 is modest even there.
- Under H0 (the pipeline this README is about) LogNormal $\Delta$ gives the lowest ASSE for every error family (4.13–4.38), Weibull is next (5.43–6.12), Gamma worst (7.46–7.78). The spread among the three LogNormal rows (4.132 / 4.332 / 4.383) is small, from one seed and a plug-in point estimate, so the error-family ordering within LogNormal is **not** established by this run.
- **Student-t plug-in.** $\nu$ enters the plug-in as its posterior mean like every other parameter. Its posterior is very skewed (mean 19–25, median 5.6–13, 5th percentile 2.1–2.4, prior upper bound 100), so the Student-t rows depend on this choice: with the posterior median instead, Student-t+LogNormal H0 was 4.244 rather than 4.132 (H1 2.189 vs 2.038). The ranking in the table did not change. `err_moments` raises if the plug-in $\nu \le 2$ (infinite variance) instead of substituting another value.
- H1 ranks Gamma best (0.42–1.09) while H3 puts Gamma at 28–33, an order of magnitude worse than the other models. Such a disagreement between heuristics, together with the divergences, points to a degenerate percentile/CDF mapping in the poorly sampled Gamma posteriors rather than a better model; do not read H1 as evidence for Gamma.
- **These H0 values (4.332 Normal, 4.383 SEV, LogNormal) are lower than the §5.4 table (4.4573 / 4.7503).** The $\hat\theta$ differ: §5.4 used the unconverged chains (Rhat 1.14–1.18, hundreds to thousands of divergences); here $\hat\theta$ comes from marginalised-likelihood fits with Rhat $\le$ 1.03 (e.g. Normal: $\beta_0=-9.506$, $\beta_1=-9.644$, $\sigma=0.215$, $\mu_\Delta=-0.722$, $\sigma_\Delta=0.041$). That the marginalisation (no soft floor) is why it converges is a plausible explanation consistent with §5.4's diagnosis, not something this run isolates. The ASSE computation itself reproduces §5.4 when given §5.4's $\hat\theta$: 4.7515 vs 4.7503 (SEV) and 4.4564 vs 4.4573 (Normal), the residual differences are the 73-vs-75-point and node-count details noted in §3.2 Step 3 (not isolated individually).

**Seeds.** The published `results/*.npz` were fitted with seeds derived from Python's per-process `hash()`; `runner.py` now uses a deterministic CRC32 seed, so a re-run gives statistically equivalent but not bit-identical draws. The two `results/full__*` files are the older censored-maxima protocol fits, kept only because `test_quadrature.py` part 4 reads them. The `results/*.json` files were trimmed to the fields `runner.py` writes (an earlier runner version had also stored rank-based ASSE fields; they are not used anywhere here).

### 5.6 ADR-RFL and an HL Loop on the Same 9 Structures (added 2026-10-29)

Question: if an agent (ADR, Tedesco 2026, arXiv 2610.37611) chooses the error and fatigue-limit distributions **without seeing ASSE**, which structure does it pick and how does that pick score in y-ASSE? And what does an HL loop that uses y-ASSE as its reward pick?

**ADR-RFL** (`adr_rfl.py`; record in `results/adr_rfl_agents.json`). Three independent Sonnet subagents, each in a sandbox holding only the tool, the likelihood code and the 9 fits (no `yasse.py`, no ASSE numbers, no README), told that a hidden metric would score the choice. The tool returns posterior summaries, Rhat/ESS/divergences and WAIC (`results/waic_full75.csv`). The agents did not trigger new NUTS fits: all 9 structures were pre-fitted (one NUTS run each, one seed), and the agents read those published fits, so what they saw is the same evidence a refit would approximate (a re-run would give statistically equivalent, not identical, draws); their search space was exactly those 9 structures. The "blind" property rests on the sandbox and the agents' self-reported commands.

- All three chose **Normal + LogNormal** (top 3 identical: Normal, Student-t, SEV, all LogNormal). WAIC elpd: Normal+LN −77.94, Student-t+LN −78.07, SEV+LN −78.39, Weibull −80.8 to −81.4, Gamma −85.4 to −85.7; the standard errors are about 7, so the three LogNormal structures are statistically tied (gaps 0.1–0.5) and the error family is **not identified** by this data (Student-t ν posterior about 25 ± 26). The LogNormal-vs-Weibull gap (about 3 elpd) is also within the SE.
- Agreement of three agents is **not three independent confirmations**: they used the same tool and the same WAIC table, and effectively took its argmax (breaking the tie by parsimony). Their own caveats: nobody compared the LogNormal fatigue-limit mean with the stress range beyond the lowest level (one agent noted Δ's median ≈ 0.49 lies below the lowest stress 0.675, and that the Weibull fit's shape ≈ 30 is near-degenerate, so the data cannot tell a fixed limit from a very tight random one); σ is weakly identified (e.g. 0.21 ± 0.12); SEV+LogNormal ESS is only 99.
- **Score of the ADR choice (Normal + LogNormal): H0 4.332 (2nd of 9, best is Student-t+LogNormal 4.132), H1 2.337 (6th of 9), H3 12.738 (6th of 9).** In the H0 pipeline the agents' pick is within 0.20 of the best; under H1 and H3 it is mid-table.

**HL loop** (`hl_loop.py`, `results/hl_loop.json`): state = (error, limit, heuristic ∈ {H0,H1,H3}), reward = −y-ASSE, proposals change one coordinate with learned coordinate weights, 40 proposals from each of the 27 starts. There is no held-out data, so the loop selects among 27 reported numbers, not a validated improvement; its learned weights stay near uniform and carry no information, and the exhaustive minimum over the 27 states is the same as the loop's result.

| Variant | Winner | y-ASSE | Note |
|---|---|:-:|---|
| heuristic fixed to H0 (the README pipeline) | Student-t + LogNormal | 4.132 | all 9 starts end here |
| heuristic free | Student-t + Gamma + H1 | 0.420 | that fit did not converge (Rhat 1.63, ESS 3) |
| free, non-converged fits excluded (Rhat ≤ 1.05, ESS ≥ 30) | SEV + Gamma + H1 | 0.949 | 16 of 24 starts; the other 8 stop at Student-t + LogNormal + H1 |

**What H1 measures (derived, then checked numerically on the 9 fits with `h1_w1_check.py`).** H1 maps $y \mapsto p=F(y\mid S) \mapsto \hat\Delta=G^{-1}(p) \mapsto \hat y=\mu(\hat\Delta)$ with $\mu(\Delta)=\beta_0+\beta_1\ln(S-\Delta)$ the error-free part, i.e. $\hat y = Q_\mu(F_y(y))$, the quantile-to-quantile map from the law of $y\mid S$ to the law of $\mu(\Delta)$ (plus the constant SEV mean shift $-\sigma\gamma$ when the error is SEV, which `yhat_from_delta` adds). So $\sum_i|y_i-\hat y_i|$ estimates $\sum_j n_j W_1(\mathcal L(y\mid S_j),\mathcal L(\mu(\Delta)+\text{shift}))$: it shrinks when the dispersion the error term adds is small *relative to the dispersion of the structural part* (not when $\sigma$ alone is small), whether or not the structure fits the data. H1 maps $p$ through the untruncated $G^{-1}$ and clips $\hat\Delta$ at $0.999S$ as the §3.2 pipeline does; $G(0.675)\ge 1-2\times10^{-15}$ for all nine real-data fits, so truncation and the clip have no numerical effect here. The theoretical $\sum_j n_j W_1$ computed from the fitted parameters alone agrees with the sample H1 ASSE (e.g. Normal+LogNormal 2.322 vs 2.337, SEV+LogNormal 2.380 vs 2.377, Student-t+Gamma 0.421 vs 0.420; largest gap 0.2, Student-t+Weibull 2.735 vs 2.933); `h1_w1_check.py` uses the same quantile maps as `yasse.py`, so this confirms the algebra against the code, not the code path independently. H1 is therefore a noise-size measure more than a goodness-of-fit measure; a correctly specified model with large $\sigma$ scores worse than a misspecified one with small $\sigma$. (I have not shown the same identity for H0, which uses the moment-matched $\Phi(z)$ instead of $F$.)

**Simulation check (`sim_study.py`, `analyze_sim.py`, `results/sim_*.jsonl`, `results/sim_summary.json`).** Data were simulated from a known structure on the real design (5 stress levels × 15 failures), with parameters taken from the real-data fit of that structure; all 9 structures were then refitted by maximum likelihood (NUTS priors as box bounds, 3 starts; MLE instead of NUTS so that 4 × 100 × 9 fits are affordable) and scored with the MLE plug-in. 100 replications per truth. "Picks truth" = the structure with the lowest score (AIC for the first row) is the true one; chance is 11%.

| True structure | Score | Picks truth | Picks truth's limit family | Picks a Gamma limit | Mean rank of truth (1 = best) |
|---|---|:-:|:-:|:-:|:-:|
| SEV + LogNormal | AIC | 52% | 86% | 5% | 1.91 |
| | H0 | 44% | 83% | 11% | 2.56 |
| | H1 | **2%** | 2% | **82%** | 5.29 |
| | H3 | 27% | 73% | 0% | 3.26 |
| Normal + LogNormal | AIC | 24% | 89% | 3% | 2.27 |
| | H0 | 15% | 86% | 5% | 3.57 |
| | H1 | **1%** | 2% | **87%** | 5.49 |
| | H3 | 31% | 75% | 0% | 2.65 |
| SEV + Weibull | AIC | 53% | 76% | 0% | 1.96 |
| | H0 | 7% | 37% | 12% | 4.72 |
| | H1 | **2%** | 38% | **61%** | 4.76 |
| | H3 | 20% | 74% | 0% | 2.77 |
| Normal + Gamma | AIC | 11% | 45% | 45% | 3.97 |
| | H0 | 0% | 3% | 3% | 6.36 |
| | H1 | 24% | 50% | 50% | 3.66 |
| | H3 | 2% | 23% | 23% | 5.15 |

Within a replication, the Spearman correlation between the fitted $\hat\sigma$ of the 9 structures and their score averages +0.80 to +0.86 for H1 (+0.35 to +0.46 for H0; −0.36 to +0.52 for H3). A second test (true structure SEV + LogNormal, 60 replications, only the correct structure fitted, true $\sigma$ multiplied by 0.5 / 1 / 2) gives mean scores H0 3.105 / 3.822 / 7.802 and H1 0.986 / 1.544 / 5.580: a correctly specified model scores worse as the noise grows. (The fitted $\hat\sigma$ is 0.071 / 0.084 / 0.201 against true 0.075 / 0.151 / 0.302. Why the fitted $\hat\sigma$ falls below the true one at multiplier 1 — for example the MLE assigning part of the scatter to $\Delta$ — is a hypothesis: this run stored neither the $\Delta$ parameters nor a variance decomposition, and the three multipliers used different, unpaired random draws.)

Reading (what was observed with this implementation, these four truths and MLE fits; not a general property of the heuristics): **H1 did not pick the true structure** — with LogNormal or Weibull truth it picks the true structure in 1–2% of replications (below chance) and picks a Gamma limit (the smallest fitted $\sigma$) in 61–87%, where AIC picks the truth in 24–53% and the right limit family in 76–89%. H0 does better for LogNormal truths (44% and 15% for SEV and Normal; error family poorly identified) but is worse than AIC and fails for Weibull (7%) and Gamma (0%) truths. H3 is comparable to AIC for LogNormal truths (31% vs 24% for Normal + LogNormal, a difference within the ±5% sampling error of 100 replications) and weaker elsewhere. With a Gamma truth H1 scores best of the four (24% vs AIC 11%), as expected from a score that favours small fitted $\sigma$ and Gamma fits (half of H1's picks are Gamma there, whatever the error family). Limits: four truths only, parameters from fits of one data set, MLE rather than posterior means, Student-t ν bounded to [2.5, 100]; the mean fitted $\sigma$ of the structures H1 picks is 0.017–0.043, against 0.069–0.115 for AIC. No fit raised an exception, but optimizer success flags, gradient norms and bound hits were not stored, so convergence of each MLE is not verified (3 starts, all in the neighbourhood of the real-data fit). The simulated data were drawn from $\Delta$ truncated to $(0,S)$ while the fitted likelihood does not divide by $G(S)$; $G(0.675)=1$ to within $2\times10^{-15}$ for the nine real-data fits that supply the truth parameters, so this mismatch has no numerical effect there.

**Check with the repo's DA+NCP fits (`da_sim_study.py`, `analyze_da_sim.py`, `results/da_sim_summary.json`).** The same simulation (truth SEV + LogNormal, 6 replications only) was refitted with `rfl_asse_full_sample.build_model` (unchanged; only its data replaced), next to an MLE fit of the same data. The DA fits did not converge: SEV + LogNormal mean Rhat_max 1.67, ESS_min 20, about 756 divergences; Normal + LogNormal 1.82, 7, about 617. Their plug-in log-likelihood is far below the MLE's (−92.7 vs −67.1; −82.7 vs −67.7), and the scores differ accordingly (H0 5.639 vs 3.642 and 4.256 vs 3.682; H1 1.212 vs 1.522 and 1.095 vs 1.574; H3 33.4 vs 11.0 and 26.0 vs 10.7). Out of 6 replications the SEV + LogNormal structure was picked by DA-H0 1, DA-H1 2, DA-H3 4, MLE-H0 3, MLE-H1 2, MLE-H3 2 and AIC 4 times — too few to support any selection claim. A one-off smoke test of the repo's Gamma DA model (`rfl_pymc_da.build_da_model`) on the first simulated data set gave Rhat_max 2.76, ESS_min 4, 708 divergences and a log-likelihood of −116.9, so a DA-based test of the Gamma preference is not possible with that code. The MLE-based simulation above is therefore the only selection evidence, and it has not been reproduced with the §3.1 DA sampler.

**What the free-heuristic loop's Gamma + H1 result means (an association over the 9 real fits, now supported by the simulation above).** y-ASSE replaces the error term by its mean, so a model that puts more of the within-stress scatter into $\Delta$ has less left to miss; but this is not a necessary condition (the quantile map, $\beta_1$ and the scale of $\Delta$ can compensate each other). `delta_share.py` computes the share of $\mathrm{Var}(\ln N\mid S)$ due to $\Delta$ (mean over the five stress levels, plug-in θ̂, `results/delta_share.csv`): Gamma 0.96–0.98, LogNormal 0.87–0.88, Weibull 0.85–0.91. The H1 y-ASSE is **perfectly monotone** in this share across the 9 models (Spearman −1.0); H0 and H3 are not (Spearman +0.45 and +0.67, i.e. the opposite direction, mainly because Gamma scores badly there). WAIC ranks the same Gamma fits last (about 7.5 elpd below LogNormal) and those fits need β₁ ≈ −22 against ≈ −9.6, with 107–207 divergences. So a loop allowed to choose both heuristic and structure on y-ASSE alone picks fits that WAIC and the sampler diagnostics rank last. That is the usual symptom of over-optimising an internal metric (the internal score improves while an independent criterion degrades), **not a demonstration of reward hacking**: WAIC and y-ASSE measure different things, and H0 and H3 do not show the pattern. This section therefore reports the fixed-H0 result as the HL outcome and does not treat the free-heuristic winner as a finding about the models. Testing the hypothesis would need, for example, simulated data from a known structure (does y-ASSE-H1 still prefer a Gamma refit?).

**Scope and limits.** One data set (n = 75), one seed per fit, plug-in point estimates; the H0 gaps among the three LogNormal structures (4.132, 4.332, 4.383) are not tested for significance. The agents choose among 9 pre-fitted structures, not open-ended distributions as in ADR proper. y-ASSE and WAIC answer different questions (reconstruction of $y$ through the quantile map vs. predictive density); here they agree on LogNormal under H0 and disagree under H1.

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

> **Comparison with Chiu (2005).** Table 3 above and the two rows appended here report the same metric — ASSE $=\sum|\omega - \hat\omega|$ in y (ln-lifetime) space — on the same $n=75$ dataset, so they sit on one scale and Chiu (2005)'s Nor-Nor value of **10.80** is used as the reference. Relative to it, this repo's SEV fit (4.7503) is lower by **6.0497 (−56.0%)** and the Normal fit (4.4573) by **6.3427 (−58.7%)**. The two differ only in how $\hat\omega_{ij}$ is constructed: Chiu (2005) aligns it to $\omega_{ij}$ through the rank-based plot position $p_i = (i-0.5)/15$; this repo uses the **percentile method** of §3.2/§3.3, where $\hat y_i$ comes from inverting $\Delta$'s LogNormal quantile at $\Phi(z_i)$ with $z_i$ the individual standardized residual. Note: this repo's numbers come from a **non-converged** sampler (§5.4) and are provisional.

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
| `jax_nuts_zoo/` | **Model zoo (added 2026-10-29):** JAX/NumPyro NUTS fits of 9 error × fatigue-limit models (`rfl_zoo.py`, `runner.py`), y-ASSE scoring (`yasse.py`, `make_table.py`), ADR tool and HL loop (`adr_rfl.py`, `hl_loop.py`, `delta_share.py`, `h1_w1_check.py`, `node_convergence.py`, `sim_study.py`, `analyze_sim.py`, `da_sim_study.py`, `analyze_da_sim.py`), and tests (`test_zoo.py`, `test_quadrature.py`) — see §5.5–5.6 | ⭐⭐ |


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

### Model Zoo (JAX/NumPyro, added 2026-10-29; tens of minutes on 6 workers)

```bash
pip install jax numpyro arviz pandas
cd jax_nuts_zoo
python test_zoo.py          # likelihood equivalence + gradient checks (expects "FAILS: 0")
python test_quadrature.py   # quadrature-accuracy checks quoted in section 5.5
python runner.py full75 6   # fit 9 models into results/ (existing results are skipped)
python make_table.py full75 # y-ASSE table -> results/yasse_full75.csv
python hl_loop.py           # HL loop over structure x heuristic -> results/hl_loop.json
python delta_share.py       # Delta variance share vs y-ASSE -> results/delta_share.csv
python h1_w1_check.py       # H1 ASSE vs the Wasserstein distance it estimates
python sim_study.py select sev lognormal 100   # simulation: which structure does each score pick? (about 10 min per truth)
python sim_study.py noise sev lognormal 60     # simulation: correct structure, sigma x 0.5/1/2
python analyze_sim.py       # summary of the simulation runs -> results/sim_summary.json
python node_convergence.py  # does H0/H1 keep falling as quadrature nodes grow? (uses results/roy_da_thetas.json)
python da_sim_study.py 0 6 ln && python analyze_da_sim.py   # same simulation with the repo's DA+NCP fits (slow: about 3 min per fit)
python adr_rfl.py fit sev lognormal   # what an ADR agent sees (diagnostics + WAIC, no ASSE)
```

`test_zoo.py` imports `rfl_pymc_ncp.py` from the repo root (the numpy reference likelihood).

---

## 10. References

### Key References

1. **Pascual, F. G., & Meeker, W. Q. (1999).** Estimating fatigue curves with the random fatigue-limit model. *Technometrics*, 41(4), 277–289.
   *Foundational RFL model paper. The n=75 dataset used here does not match either of this paper's own two datasets — see §4's provenance caveat. The dataset's working attribution is instead Castillo & Hadi (1995) below, as re-analyzed in this paper's published Response to discussants (§2.2).*

2. **Castillo, E., & Hadi, A. S. (1995).** Modeling lifetime data with application to fatigue models. *Journal of the American Statistical Association*, 90(431), 1041–1054.
   *Working source of the n=75 concrete-fatigue dataset used in this repo (structural + summary-statistic match confirmed via Pascual & Meeker's Response, §2.2; original table not independently obtained — paywalled, see §4).*

3. **Chiu, C.-H. (2005).** *二元分佈族及其統計推論 (A Family of Bivariate Distributions With Some Applications to Statistical Inferences).* Master's Thesis, Graduate Institute of Management Science, Tamkang University, June 2005. Advisor: Wen-Tao Huang. Local copy: `C:/Users/roy/MD thesis.pdf`.
   *Chapter 2 applies the Nor-Nor variant of the RFL family to the Pascual & Meeker (1999) fatigue dataset; Table 3 (thesis p. 23) is the published ASSE comparison against six other S-N models and is reproduced verbatim in §6.2.*

4. **Hoffman, M. D., & Gelman, A. (2014).** The No-U-Turn Sampler: Adaptively setting path lengths in Hamiltonian Monte Carlo. *Journal of Machine Learning Research*, 15(1), 1593–1623.
   *The NUTS algorithm; the core MCMC engine used in this study.*

5. **Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2013).** *Bayesian Data Analysis* (3rd ed.). CRC Press.
   *Methodological basis for Non-Centered Parameterization (the Neal's Funnel fix).*

6. **Salvatier, J., Wiecki, T. V., & Fonnesbeck, C. (2016).** Probabilistic programming in Python using PyMC3. *PeerJ Computer Science*, 2, e55.
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
