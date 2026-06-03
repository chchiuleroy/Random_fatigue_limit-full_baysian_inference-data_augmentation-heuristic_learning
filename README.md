# Full Bayesian RFL Model: Model Z-Score Prediction with Heuristic Learning (State → Strategy → Feedback) Pipeline Selection

> **Roy (2026-06-02)**
> Full Bayesian Inference to improve individual lifetime prediction for the Random Fatigue Limit (RFL) model.
> Proposes replacing sample statistics with model-theoretic moments (Model Z-Score),
> and validates the optimal prediction pipeline via a three-stage Heuristic Learning framework (**State** posterior info → **Strategy** H0–H4 → **Feedback** −ASSE).

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
| Heuristic Learning→(HL) | Systematic search for the optimal prediction pipeline |
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

$$E(\ln N \mid S_j) = \hat{\beta}_0 - \hat{\sigma}\gamma_E + \hat{\beta}_1 \int_0^{S_j} \ln(S_j - \Delta) \cdot g(\Delta \mid \hat{\mu}_\Delta, \hat{\sigma}_\Delta) \thinspace d\Delta$$

where $\gamma_E = 0.5772$ (Euler-Mascheroni constant).

**Theoretical variance** (correctly separating two sources):

$$V(\ln N \mid S_j) = \hat{\beta}_1^2 \cdot \underbrace{\text{Var}[\ln(S_j - \Delta)]}_{\Delta\text{ contribution}} + \underbrace{\frac{\pi^2 \hat{\sigma}^2}{6}}_{\text{SEV term}} \quad \left(\text{Normal: } \hat{\sigma}^2\right)$$

#### Step 2: Complete Prediction Pipeline

```
1. Model z-score:   z_ij = (y_ij - E(lnN|S_j)) / sqrt(V(lnN|S_j))
2. Percentile:      p_ij = Phi(z_ij)    [Normal CDF]
3. Invert for Delta:Delta_ij = exp(mu_Delta + sigma_Delta * z_ij)
4. Predict:         y_ij = beta0 + beta1 * ln(S_j - Delta_ij) - sigma*gamma_E  (SEV)
5. ASSE:            sum|y_ij - y_hat_ij|
```

---

### 3.3 Heuristic Learning (HL) Validation

HL (Weng, 2026) decomposes the problem into three explicit stages to systematically search for the optimal prediction rule:

**Stage 1 — State:** All information available after MCMC inference

$$\text{State} = \Bigl\lbrace \hat{\theta}, \lbrace E(\ln N \mid S_j), \sqrt{V(\ln N \mid S_j)}\rbrace_{j=1}^5, \lbrace(y_{ij}, S_j)\rbrace \Bigr\rbrace$$

- $\hat{\theta}$: posterior mean (MCMC output; 5 parameters, see Section 2.2)
- Theoretical mean and SD per stress level: computed from $\hat{\theta}$ via GL integration
- Observed data $\lbrace(y_{ij}, S_j)\rbrace$: used for final ASSE computation

**Stage 2 — Strategy:** Given the State, choose a heuristic rule mapping $y_{ij}$ to $\hat{y}_{ij}$

$$\text{Strategy}_k : \text{State} \times y_{ij} \longrightarrow \hat{y}_{ij} \qquad k \in \lbrace H0, H1, H2, H3, H4 \rbrace$$

**Stage 3 — Feedback:** Evaluate strategy quality to drive selection

$$\text{Reward}(k) = -\text{ASSE}(k) = -\sum_{i,j} |y_{ij} - \hat{y}_{ij}^{(k)}|$$

HL enumerates the five candidate strategies H0–H4 and selects the one with maximum Reward.

| Heuristic | Step 2 (percentile) | Steps 3–4 (Delta estimation) |
|-----------|---------------------|------------------------------|
| **H0 (model z-score)** | $\Phi(z)$ | LogNormal inverse CDF |
| H1 | Exact marginal CDF (numerical integration) | Same |
| H2 | Cornish-Fisher $\Phi(z + \gamma_1(z^2-1)/6)$ | Same |
| H3 | Per-group z calibration (subtract group mean) | Same |
| H4 | — | DA posterior $E[\Delta_i \mid \mathbf{y}, \theta^{(s)}]$ |

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

The true $y_i$ is unknown for censored observations. Two approaches:

1. **Exclude censored observations:** Compute ASSE using only $\delta_i = 1$ failures (most common)
2. **Conditional expectation imputation:** Substitute $E[\ln N \mid \ln N > c_i, \Delta_i, \theta]$ for $y_i$, incorporating censored observations into ASSE

---

## 4. Data

**Source:** Pascual & Meeker (1999) *Technometrics* — aluminum alloy (R.R. Moore rotating bending fatigue test)

| Data Characteristic | Value |
|---------------------|-------|
| Total observations | 75 |
| Observed failures | 75 |
| Stress levels | 5 (S = 0.675, 0.750, 0.825, 0.900, 0.950 ksi) |
| Specimens per stress level | 15 |
| Response variable | $\ln N$ (log number of cycles to failure) |

---

## 5. Results

### 5.1 MCMC Convergence (SEV + LogNormal, Best Model)

| Metric | Value |
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

**This study: max $\hat{R}$ = 1.032** — slightly above the strict threshold but within acceptable range. The worst parameter is likely $\sigma_\Delta$ (true value ≈ 0.038, near-degenerate, harder to sample).

---

**ESS (Effective Sample Size)**

MCMC chains exhibit autocorrelation; the nominal 4000 samples carry information equivalent to ESS independent samples:

$$\text{ESS} = \frac{4000}{1 + 2\sum_{k=1}^{\infty}\rho_k}$$

where $\rho_k$ is the lag-$k$ autocorrelation. Higher autocorrelation yields smaller ESS.

- ESS > 400: reliable estimation of means and variances
- ESS > 100: minimum acceptable threshold
- ESS < 100: increase sampling count

**This study: min ESS = 117** — barely above threshold, indicating moderate autocorrelation for some parameter (likely $\sigma_\Delta$). For more precise posterior quantiles, increase warm-up or sample count.

---

**Divergences**

During HMC/NUTS numerical integration, if energy conservation is severely violated (typically in regions of sharp posterior curvature, such as Neal's Funnel), the step "diverges." Posterior regions near divergent points are not correctly explored, biasing posterior estimates.

- Divergences = 0: HMC operating normally; flat posterior geometry
- Divergences > 0: reduce step size (increase `target_accept`) or switch to NCP

**This study: Divergences = 0** — NCP successfully flattened Neal's Funnel; all 4 chains have zero divergences.

---

**LOO-ELPD (Leave-One-Out Expected Log Pointwise Predictive Density)**

Leave one observation out each time, train on the remaining 74, compute the log predictive density for the held-out point, and sum:

$$\text{LOO-ELPD} = \sum_{i=1}^{75} \log p(y_i \mid \mathbf{y}_{-i})$$

- Larger values (closer to 0) indicate better predictive performance
- A difference $\Delta\text{ELPD} > 2$ is generally considered meaningful

**This study:** SEV+LN = −76.39, Normal+LN = −79.38, $\Delta = 2.99$. SEV's heavier-tailed behavior better captures fatigue life data.

### 5.2 Posterior Parameter Estimates (SEV + LogNormal)

| Parameter | Posterior Mean | Posterior SD | 95% CI | MLE Reference |
|-----------|:--------------:|:------------:|--------|:-------------:|
| $\beta_0$ | −9.286 | 0.427 | — | −9.370 |
| $\beta_1$ | −8.746 | 1.355 | — | −8.534 |
| $\sigma$ | 0.195 | 0.072 | — | 0.190 |
| $\mu_\Delta$ | −0.660 | 0.082 | — | −0.644 |
| $\sigma_\Delta$ | 0.038 | 0.006 | — | 0.036 |

> Posterior means nearly reproduce MLE — with n=75, Uniform priors are dominated by the likelihood, giving high agreement between Bayesian and frequentist results.

### 5.3 Posterior ASSE Distribution (SEV + LogNormal, n=75)

| Statistic | Posterior Value |
|-----------|:--------------:|
| Posterior Mean ASSE | 13.83 |
| Posterior Median ASSE | 13.13 |
| 95% Posterior CI | [6.13, 24.58] |
| Posterior SD | 4.87 |

**Plugin ASSE** (evaluated at posterior mean $\hat{\theta}$):

| Prediction Formula | ASSE |
|--------------------|:----:|
| SEV + Euler correction ($\hat{y} = \mu - \hat{\sigma}\gamma_E$) | **5.75** |
| No correction ($\hat{y} = \mu$) | 10.19 |

> Plugin y-ASSE = 5.75 nearly equals the in-sample ASSE of MLE SEV+INLA (5.76), confirming that the posterior mean correctly reproduces the MLE.
> The posterior mean ASSE = 13.83 is higher because posterior uncertainty in $\sigma$ (SD = 0.07) degrades predictions for some draws — this is the "honest ASSE with full parameter uncertainty."

### 5.4 Heteroscedastic Structure Across Stress Levels

| $S_j$ | $E(\ln Y \mid S_j)$ | SD (SEV) | SD (Normal) |
|-------|:-------------------:|:--------:|:-----------:|
| 0.675 | 6.82 | **1.158** | 1.139 |
| 0.750 | 3.38 | 0.808 | 0.782 |
| 0.825 | 0.91 | 0.599 | 0.612 |
| 0.900 | −0.99 | 0.470 | 0.515 |
| 0.950 | −2.10 | 0.525 | 0.418 |

The SD at low stress (S=0.675) is **2.5x** that at high stress (S=0.9), revealing a pronounced heteroscedastic structure.

### 5.5 Model Z-Score ASSE (y-ASSE, ln-lifetime space)

| Method | SEV+LN plugin | Normal+LN plugin | SEV+LN post. mean | Normal+LN post. mean |
|--------|:-------------:|:----------------:|:-----------------:|:--------------------:|
| Sample z-score (Chiu-style) | 10.93 | 9.17 | 14.44 | 14.19 |
| **Model z-score (this work)** | **4.32** | **3.53** | **5.29** | **4.65** |
| Improvement (plugin) | **60.5%** | **61.5%** | — | — |

### 5.6 HL Heuristic Search Results (y-ASSE plugin)

| Heuristic | SEV+LN | Normal+LN | Notes |
|-----------|:------:|:---------:|-------|
| **H0: model z-score + Phi(z)** | **4.32** | **3.53** | **Best** (implicit shrinkage) |
| H1: exact marginal CDF | 10.38 | 12.57 | Counterintuitively worse |
| H2: Cornish-Fisher skewness correction | 6.21 | 3.67 | Slightly worse |
| H3: per-group z calibration | 10.93 | 14.05 | Worst |
| H4: DA posterior E[Delta_i|data] | 5.80 | 9.55 | Moderate |

**Key finding:** H1 (exact marginal CDF) performs worse than H0 (Normal approximation $\Phi(z)$). The implicit shrinkage of $\Phi(z)$ for extreme z-values (Normal tails are more conservative than the true marginal) acts as useful regularization with the finite sample n=75. This is a classic **bias-variance tradeoff**: a "more correct" method need not perform better in finite samples.

---

## 6. Comparison with Prior Work

### 6.1 ASSE Metric Notes

> **Important:** Different studies use different prediction evaluation spaces; values are not directly comparable across spaces.

| Space | Definition | Used by |
|-------|------------|---------|
| **ASSE** (z-score space, **z-ASSE**) | $\sum_i \|\omega_{ij} - \hat{\omega}_{ij}\|$ | Chiu (2005), Roy C-2, SEV+INLA z-score |
| **ASSE** (ln-lifetime space, **y-ASSE**) | $\sum_i \|y_{ij} - \hat{y}_{ij}\|$ | This work (model z-score method) |
| **ASSE** (rank space, **rank-ASSE**) | $E = \sum_i \|y_{ij} - \hat{y}_{(ij)}\|$ (rank-matched) | P&M (1999) |

### 6.2 ASSE Results Summary by Method

#### z-score Space (z-ASSE)

| Method | z-ASSE | Improvement vs. Chiu | Source |
|--------|:------:|:--------------------:|--------|
| Chiu (2005) EIV (thesis baseline) | **10.80** | — | Chiu (2005) thesis |
| Roy Method A (MLE sample z-score) | 12.38 | −14.6% | `rfl_chiu.py` |
| Roy Method B (Nelder-Mead optimization) | 10.38 | +3.9% | `rfl_chiu.py` |
| Roy Method C-1 (OLS LAD z-score) | 11.02 | −2.0% | `rfl_chiu.py` |
| **Roy Method C-2 (LAD regression)** | **9.94** | **+8.0%** | `rfl_chiu.py` |
| SEV+INLA z-score (MLE parameters) | 11.50 | −6.5% | `rfl_asse_zscore.py` |
| Burr+INLA z-score | 11.92 | −10.4% | `rfl_asse_zscore.py` |

#### ln-lifetime Space (y-ASSE, in-sample plugin)

| Method | y-ASSE | Model | Source |
|--------|:------:|-------|--------|
| Burr+EM-GMM (unconstrained) | 0.49 | SEV | `rfl_burr_em.py` (overfitting) |
| Burr+EM-GMM (sigma>=0.15, a>=1) Mode A | 4.09 | SEV | `rfl_burr_em.py` |
| Burr+INLA | 5.74 | SEV | `rfl_burr_inla.py` |
| SEV+INLA (MLE in-sample) | 5.76 | SEV | `rfl_profile.py` |
| **Bayes DA Plugin (SEV+Euler)** | **5.75** | SEV | `rfl_bayes_asse.py` |
| SEV sample z-score | 10.93 | SEV | `rfl_model_zscore_asse.py` |
| Normal sample z-score | 9.17 | Normal | `rfl_model_zscore_asse.py` |
| **Model z-score (SEV, H0)** | **4.32** | SEV | `rfl_model_zscore_asse.py` |
| **Model z-score (Normal, H0)** | **3.53** | Normal | `rfl_model_zscore_asse.py` |

#### Rank Space (P&M 1999 criterion, E)

| Method | E | Notes |
|--------|---|-------|
| Normal-Normal MLE (concrete data) | 12.84 | P&M (1999) best result |
| Roy Method B (rank-ASSE, direct optimization) | 12.24 | `rfl_chiu.py` |
| Roy Method C-2 (rank-ASSE) | 12.24 (z), 12.xx (rank) | `rfl_chiu.py` |

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

### 7.2 Relation to the Conditional Posterior

The model z-score pipeline is essentially computing an approximation to the conditional posterior of each individual $\Delta_i$:

$$p(\Delta_i \mid y_i, \theta) \propto f(y_i \mid \Delta_i, \theta) \cdot g(\Delta_i \mid \theta)$$

This method uses the marginal percentile as a proxy for the posterior's "location" — an elegant moment-matching solution to the inverse problem ($y_i \to \Delta_i$) that avoids per-specimen numerical integration and achieves lower ASSE than the DA posterior $E[\Delta_i \mid \mathbf{y}, \theta^{(s)}]$ (H4).

### 7.3 Implicit Regularization Effect of Phi(z)

The exact marginal CDF (H1) performs worse than $\Phi(z)$ (H0) because:

$$\text{true marginal} = \int f_{\text{SEV}}(y; \mu_\Delta(\Delta), \sigma) \cdot g(\Delta) \thinspace d\Delta \neq \text{Normal}$$

The true marginal has heavier tails than Normal, causing extreme z-values to map to higher percentiles, more extreme $\hat{\Delta}$ estimates, and larger prediction errors. $\Phi(z)$ shrinks extreme z-values toward the mean (implicit shrinkage), effectively reducing variance in the finite sample of n=75.

---

## 8. Code Description

### Core Code (This Work)

| File | Function | Priority |
|------|----------|:--------:|
| `rfl_pymc_da.py` | **Main inference:** DA+NCP+NUTS (SEV+LN and Normal+LN models) | ⭐⭐⭐ |
| `rfl_model_zscore_asse.py` | **Model z-score ASSE:** complete implementation and validation of Roy's new method | ⭐⭐⭐ |
| `rfl_hl_asse.py` | **HL search:** systematic comparison of H0–H4 heuristics | ⭐⭐⭐ |
| `rfl_bayes_asse.py` | Posterior y-ASSE direct computation (Euler-corrected) | ⭐⭐ |

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

Output: posterior samples, Rhat, ESS, LOO-ELPD, and posterior mean ASSE.

### Model Z-Score Validation (approx. 7–10 minutes)

```bash
# Compute E(lnY|S), V(lnY|S) per stress level; compare model z-score vs. sample z-score
python rfl_model_zscore_asse.py
```

Output: heteroscedasticity table, y-ASSE comparison (model vs. sample z-score, SEV vs. Normal).

### HL Heuristic Search (approx. 15–20 minutes)

```bash
# Systematic search over five heuristics H0-H4
python rfl_hl_asse.py
```

Output: y-ASSE for H0–H4 (SEV and Normal); confirmation of the best heuristic.

### Posterior ASSE Computation (approx. 3–5 minutes)

```bash
# Compute ASSE for each posterior draw; output posterior distribution statistics
python rfl_bayes_asse.py
```

Output: posterior mean ASSE = 13.83, median = 13.13, 95% CI = [6.13, 24.58].

---

## 10. References

### Key References

1. **Pascual, F. G., & Meeker, W. Q. (1999).** Estimating fatigue curves with the random fatigue-limit model. *Technometrics*, 41(4), 277–289.
   *Foundational RFL model paper; this study uses its n=75 aluminum alloy dataset.*

2. **Chiu, C. (2005).** *Statistical Analysis of Fatigue Data with the Random Fatigue-Limit Model.* PhD Thesis.
   *Proposes the EIV z-score method; z-ASSE = 10.80 is the primary benchmark for this study.*

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
   *The original Heuristic Learning paper; this study applies the HL framework to validate the optimal prediction pipeline.*

9. **Murphy, S. A., & van der Vaart, A. W. (2000).** On profile likelihood. *Journal of the American Statistical Association*, 95(450), 449–465.
   *Theoretical basis for Profile Likelihood; relevant to Roy's semiparametric RFL work.*

10. **Meeker, W. Q., & Escobar, L. A. (1998).** *Statistical Methods for Reliability Data*. John Wiley & Sons.
    *Standard reference for SEV distribution (log-Weibull) and statistical methods for fatigue analysis.*

### Bayesian Computation References

11. **Vehtari, A., Gelman, A., & Gabry, J. (2017).** Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC. *Statistics and Computing*, 27(5), 1413–1432.
    *Methodological basis for PSIS-LOO; used in this study for model comparison (LOO-ELPD).*

12. **Betancourt, M. (2017).** A conceptual introduction to Hamiltonian Monte Carlo. *arXiv preprint arXiv:1701.02434*.
    *In-depth introduction to HMC and NUTS, including the geometric interpretation of NCP.*
