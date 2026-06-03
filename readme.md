# Full Bayesian Inference for Random Fatigue Limit Model with Model Z-Score Prediction

> **Roy (2026-06-02)**  
> 以全貝式推論（Full Bayesian Inference）改進 Random Fatigue Limit（RFL）模型的個別壽命預測，  
> 提出以模型理論矩（Model Z-Score）取代樣本統計量，並以 Heuristic Learning 驗證最佳管線。

---

## 目錄

1. [研究背景](#1-研究背景)
2. [問題定義與動機](#2-問題定義與動機)
3. [研究方法](#3-研究方法)
4. [資料](#4-資料)
5. [實作結果](#5-實作結果)
6. [與先前研究的比較](#6-與先前研究的比較)
7. [理論貢獻](#7-理論貢獻)
8. [程式碼說明](#8-程式碼說明)
9. [執行方式](#9-執行方式)
10. [References](#10-references)

---

## 縮寫對照表

### 核心模型與方法

| 全名 | 說明 |
|------|------|
| Random Fatigue Limit→(RFL) | 隨機疲勞極限模型，本研究主體 |
| Smallest Extreme Value→(SEV) | 最小極值分佈，Weibull 的 log 版本，失效壽命分佈族 |
| Data Augmentation→(DA) | 資料擴增——將潛變數 $\Delta_i$ 納入 MCMC 取樣，避免數值積分 |
| Non-Centered Parameterization→(NCP) | 非中心化參數化——解耦 prior 與超參數，消除 Neal's Funnel |
| No-U-Turn Sampler→(NUTS) | 無折返取樣器——自適應 HMC 演算法，Hoffman & Gelman 2014 |
| Hamiltonian Monte Carlo→(HMC) | 哈密頓蒙地卡羅——NUTS 的底層演算法 |
| Markov Chain Monte Carlo→(MCMC) | 馬可夫鏈蒙地卡羅，通用後驗取樣框架 |
| Heuristic Learning→(HL) | 啟發式學習——系統搜尋最佳預測管線 |
| LogNormal→(LN) | 對數常態分佈，$\Delta_i$ 的先驗分佈族 |

### 預測誤差指標

| 全名 | 說明 |
|------|------|
| Absolute Sum of Standardized-score Errors→(ASSE) | $\sum_i\|y_i - \hat{y}_i\|$；分 z-ASSE（z-score 空間）、y-ASSE（ln 壽命空間）、rank-ASSE（排序空間）三種 |

### 統計推論方法

| 全名 | 說明 |
|------|------|
| Maximum Likelihood Estimation→(MLE) | 最大概似估計，頻率論基準 |
| Integrated Nested Laplace Approximation→(INLA) | 整合巢狀拉普拉斯近似 |
| Errors-In-Variables→(EIV) | 變數含誤差模型，Chiu (2005) z-score 方法的框架 |
| Ordinary Least Squares→(OLS) | 普通最小平方法，Roy Method C-1 基礎 |
| Least Absolute Deviations→(LAD) | 最小絕對偏差法，Roy Method C-2 基礎 |
| Leave-One-Out Expected Log Pointwise Predictive Density→(LOO-ELPD) | 留一交叉驗證期望對數點預測密度，模型比較指標 |
| Pareto-Smoothed Importance Sampling Leave-One-Out→(PSIS-LOO) | Pareto 平滑重要性採樣留一法 |
| Effective Sample Size→(ESS) | 有效樣本數，衡量 MCMC 混合品質 |
| Credible Interval→(CI) | 可信區間，貝式脈絡 |
| Standard Deviation→(SD) | 標準差 |

### 數值計算方法

| 全名 | 說明 |
|------|------|
| Gauss-Legendre→(GL) | 高斯-勒讓德數值積分，用於計算理論矩 $E(\ln N \mid S_j)$ |
| Gauss-Hermite→(GH) | 高斯-厄米特數值積分 |
| Cumulative Distribution Function→(CDF) | 累積分佈函數 |
| Expectation-Maximization→(EM) | 期望最大化演算法，用於混合模型估計 |
| Gaussian Mixture Model→(GMM) | 高斯混合模型 |

---

## 1. 研究背景

**Random Fatigue Limit（RFL）模型**由 Pascual & Meeker（1999）提出，用於分析疲勞試驗資料。核心假設：每個試件有一個隨機的疲勞極限 $\Delta_i$，僅在應力超過 $\Delta_i$ 時才會發生疲勞失效。

### 原始模型（P&M 1999）

$$\ln N_i \mid \Delta_i \sim \text{SEV}(\beta_0 + \beta_1 \ln(S_i - \Delta_i),\; \sigma)$$

$$\Delta_i \sim \text{LogNormal}(\mu_\Delta, \sigma_\Delta)$$

其中 Smallest Extreme Value→(SEV) 是 Weibull 的 log 版本，$N_i$ 是失效循環數，$S_i$ 是施加應力。

**先前研究局限**：
- P&M (1999) 使用 Laplace 近似推論，預測精度有限
- Chiu (2005) 用組內樣本 z-score 估計個別 $\Delta_i$，誤差較大
- 標準 MLE/INLA 方法無法量化參數不確定性

---

## 2. 問題定義與動機

### 2.1 Chiu (2005) 樣本 z-score 的三個缺陷

Chiu 論文用組內樣本統計量估計個別試件的疲勞極限：

$$\omega_{ij} = \frac{y_{ij} - \bar{y}_j}{s_j}$$

**問題一：樣本量不足**
每個應力水準只有 $n_j = 15$ 個觀測，$\bar{y}_j$ 和 $s_j$ 的估計不穩定（標準誤較大）。

**問題二：方差混合不可分**

$$s_j^2 \approx \underbrace{\beta_1^2 \cdot \text{Var}[\ln(S_j - \Delta)]}_{\Delta \text{ 個體差異}} + \underbrace{\frac{\pi^2 \sigma^2}{6}}_{\text{SEV 殘差}}$$

樣本 $s_j^2$ 混合了兩個來源，無法將個別 $\Delta_i$ 的不確定性從殘差 $\sigma$ 中分離。

**問題三：忽略跨應力水準的異方差**

$s_j$ 不隨應力水準 $S_j$ 變化，但理論上 $V(\ln N \mid S_j)$ 因 $\text{Var}[\ln(S_j - \Delta)]$ 而顯著隨 $S_j$ 變化（低應力時 $S_j - \Delta$ 較小且不穩定，方差更大）。這是 Chiu 方法無法捕捉的**異方差結構**。

### 2.2 本研究的核心想法

> **用全貝式後驗推導的理論矩取代樣本統計量。**

全貝式推論給出後驗 $\hat{\theta} = (\hat{\beta}_0, \hat{\beta}_1, \hat{\sigma}, \hat{\mu}_\Delta, \hat{\sigma}_\Delta)$，可計算每個應力水準的精確理論均值和方差，正確分離 $\Delta$ 變異和殘差 $\sigma$ 的貢獻。

---

## 3. 研究方法

### 3.1 全貝式推論：DA + NCP + NUTS

#### 聯合後驗

$$p(\theta, \Delta_{1:n} \mid \mathbf{y}) \propto \prod_{i=1}^{n} f(y_i \mid \Delta_i, \theta) \cdot \prod_{i=1}^{n} g(\Delta_i \mid \theta) \cdot p(\theta)$$

**Data Augmentation→(DA)**：將 $\Delta_i$ 作為明確取樣的潛變數，避免數值積分。  
每個 MCMC 步驟都有 $\Delta_i$ 的當前值，likelihood 直接求值，無需 GL/GH quadrature。

#### Non-Centered Parameterization→(NCP)

直接取樣 $\log \Delta_i \sim \mathcal{N}(\mu_\Delta, \sigma_\Delta)$ 在 $\sigma_\Delta \approx 0.033$（接近退化）時形成 Neal's Funnel，導致大量 divergences。

**解法**：

$$z_{\Delta_i} \sim \mathcal{N}(0, 1), \qquad \log \Delta_i = \mu_\Delta + \sigma_\Delta \cdot z_{\Delta_i}$$

Prior 與超參數 $(\mu_\Delta, \sigma_\Delta)$ 脫耦，後驗幾何更平坦，NUTS 可有效探索。

#### Prior 設計（Uniform + Jeffreys）

| 參數 | Prior | 理由 |
|------|-------|------|
| $\beta_0$ | $\text{Uniform}(-50, 50)$ | flat，無先驗知識 |
| $\beta_1$ | $\text{Uniform}(-30, 0)$ | 硬約束：超額應力大→壽命短 |
| $\log\sigma$ | $\text{Uniform}(\log 0.01, \log 5)$ | Jeffreys scale prior |
| $\mu_\Delta$ | $\text{Uniform}(-5, \log S_{\min})$ | 確保 $\Delta$ 中位數 $< S_{\min}$ |
| $\log\sigma_\Delta$ | $\text{Uniform}(\log 0.001, \log 2)$ | Jeffreys scale prior |

#### 演算法

採用 **PyMC** 的 **No-U-Turn Sampler→(NUTS)**（Hoffman & Gelman 2014），由 pytensor 自動微分計算梯度。  
4 chains 並行，各 1000 warm-up + 1000 sample，共 4000 後驗樣本。

---

### 3.2 模型 z-score 新方法（Roy 提出）

#### Step 1：計算每個應力水準的理論矩

**理論均值**（用 32-point Gauss-Legendre 積分）：

$$E(\ln N \mid S_j) = \hat{\beta}_0 - \hat{\sigma}\gamma_E + \hat{\beta}_1 \int_0^{S_j} \ln(S_j - \Delta)\, g(\Delta \mid \hat{\mu}_\Delta, \hat{\sigma}_\Delta)\, d\Delta$$

其中 $\gamma_E = 0.5772$（Euler-Mascheroni 常數）。

**理論方差**（正確分離兩個來源）：

$$V(\ln N \mid S_j) = \hat{\beta}_1^2 \cdot \underbrace{\text{Var}[\ln(S_j - \Delta)]}_{\Delta \text{ 的貢獻}} + \underbrace{\frac{\pi^2 \hat{\sigma}^2}{6}}_{\text{SEV 本身}} \quad \left(\text{Normal: } \hat{\sigma}^2\right)$$

#### Step 2：完整預測管線

```
1. 模型 z-score:    z_ij = (y_ij - E(lnN|S_j)) / sqrt(V(lnN|S_j))
2. Percentile:      p_ij = Φ(z_ij)      [Normal CDF]
3. 逆推 Δ̂_ij:      Δ̂_ij = exp(μ̂_Δ + σ̂_Δ × z_ij)
4. 預測:            ŷ_ij = β̂₀ + β̂₁ ln(S_j - Δ̂_ij) - σ̂γ_E  (SEV)
5. Absolute Sum of Standardized-score Errors→(ASSE):  Σ|y_ij - ŷ_ij|
```

---

### 3.3 Heuristic Learning→(HL) 驗證

HL（Weng, 2026）將問題拆解為三個明確階段，系統搜尋最佳預測規則：

**Stage 1 — State（狀態）**：MCMC 推論完成後可用的全部資訊

$$\text{State} = \bigl\{\,\hat{\theta},\; \{E(\ln N \mid S_j),\, \sqrt{V(\ln N \mid S_j)}\}_{j=1}^5,\; \{(y_{ij}, S_j)\} \bigr\}$$

- $\hat{\theta} = (\hat{\beta}_0, \hat{\beta}_1, \hat{\sigma}, \hat{\mu}_\Delta, \hat{\sigma}_\Delta)$：後驗均值（MCMC 輸出）
- 各應力水準的理論均值與標準差：由 GL 積分從 $\hat{\theta}$ 計算
- 原始觀測資料 $\{(y_{ij}, S_j)\}$：用於最終計算 ASSE

**Stage 2 — Strategy（策略）**：給定 State，選擇一個 Heuristic 規則將 $y_{ij}$ 映射至 $\hat{y}_{ij}$

$$\text{Strategy}_k : \text{State} \times y_{ij} \;\longrightarrow\; \hat{y}_{ij} \qquad k \in \{H0, H1, H2, H3, H4\}$$

**Stage 3 — Feedback（回饋）**：評估 Strategy 的好壞，驅動選擇

$$\text{Reward}(k) = -\text{ASSE}(k) = -\sum_{i,j} |y_{ij} - \hat{y}_{ij}^{(k)}|$$

HL 枚舉 H0–H4 五個候選 Strategy，取 Reward 最大者為最終管線。

| Heuristic | 步驟 2 (percentile) | 步驟 3–4 (Δ 推算) |
|-----------|--------------------|--------------------|
| **H0（模型 z-score）** | $\Phi(z)$ | LogNormal 逆 CDF |
| H1 | 精確邊際 CDF（數值積分） | 同上 |
| H2 | Cornish-Fisher $\Phi(z + \gamma_1(z^2-1)/6)$ | 同上 |
| H3 | Per-group z 校準（減去組均值） | 同上 |
| H4 | — | DA 後驗 $E[\Delta_i \mid \mathbf{y}, \theta^{(s)}]$ |

---

### 3.4 含 Censoring 資料的推廣

本研究 P&M (1999) 資料全為失效觀測（$n=75$），無設限。若資料存在設限觀測，需修改 Likelihood 貢獻。

#### 設定

設 $\delta_i \in \{0, 1\}$ 為失效指示：$\delta_i = 1$ 表示觀測到失效（$y_i = \ln N_i$ 已知），$\delta_i = 0$ 表示在 $c_i$ 處設限（$\ln N_i > c_i$，真實值未知）。

#### 修改後的聯合 Likelihood

$$p(\mathbf{y},\boldsymbol{\delta} \mid \boldsymbol{\Delta}, \theta) = \prod_{i:\,\delta_i=1} f(y_i \mid \Delta_i, \theta) \;\cdot\; \prod_{i:\,\delta_i=0} S(c_i \mid \Delta_i, \theta)$$

存活函數 $S(\cdot)$ 依模型分佈：

$$S_{\text{SEV}}(c_i \mid \Delta_i, \theta) = \exp\!\left(-\exp\!\left(\frac{c_i - \mu_i}{\sigma}\right)\right), \qquad \mu_i = \beta_0 + \beta_1\ln(S_j - \Delta_i)$$

$$S_{\text{Normal}}(c_i \mid \Delta_i, \theta) = 1 - \Phi\!\left(\frac{c_i - \mu_i}{\sigma}\right)$$

#### 四種設限類型

| 設限類型 | 描述 | 每筆設限觀測的 Likelihood 貢獻 |
|---------|------|-------------------------------|
| **Type I（時間設限）** | 在固定時間 $c$ 停止，未失效者設限 | $S(c \mid \Delta_i, \theta)$ |
| **Type II（失效數設限）** | 等前 $r$ 筆失效後停止，其餘設限 | $S(y_{(r)} \mid \Delta_i, \theta)$（設限值同為第 $r$ 個失效時間） |
| **隨機設限** | 每個試件各有獨立設限時間 $c_i$ | $S(c_i \mid \Delta_i, \theta)$ |
| **區間設限** | 只知失效在 $[L_i, U_i]$ 區間內 | $F(U_i \mid \Delta_i, \theta) - F(L_i \mid \Delta_i, \theta)$ |

#### DA 框架的優勢

設限觀測在 DA 框架中自然處理：$\Delta_i$ 仍被明確取樣，只需把 Likelihood 貢獻從 $f(y_i \mid \cdot)$ 換成 $S(c_i \mid \cdot)$，**不需對每個設限觀測額外做數值積分**，NCP 及 NUTS 結構完全不變。

#### PyMC 實作

```python
# delta: shape (n,)，1=失效，0=設限；c: 設限時間向量
import pytensor.tensor as pt

loglik = pt.switch(
    delta,
    dist.logp(y),                               # 失效：log pdf
    pt.log1p(-pt.exp(dist.logcdf(c)))           # 設限：log survival = log(1 - CDF)
)
pm.Potential('likelihood', loglik.sum())
```

#### ASSE 的處理

設限觀測的真實 $y_i$ 未知，有兩種做法：

1. **排除設限觀測**：僅對 $\delta_i = 1$ 的失效觀測計算 ASSE（最常見）
2. **條件期望替代**：以 $E[\ln N \mid \ln N > c_i,\, \Delta_i, \theta]$ 代入 $y_i$，將設限觀測也納入 ASSE

---

## 4. 資料

**來源**：Pascual & Meeker (1999) *Technometrics* — 鋁合金（R.R. Moore 旋轉彎曲疲勞試驗）

| 資料特性 | 值 |
|---------|---|
| 總觀測數 | 75 |
| 失效觀測 | 75 |
| 應力水準 | 5（S = 0.675, 0.750, 0.825, 0.900, 0.950 ksi） |
| 每應力水準樣本數 | 15 |
| 響應變數 | $\ln N$（log 失效循環數） |

---

## 5. 實作結果

### 5.1 MCMC 收斂（SEV + LogNormal，最佳模型）

| 指標 | 值 |
|------|---|
| $\hat{R}$ 最大值 | 1.032 |
| ESS 最小值 | 117 |
| Divergences | 0 |
| LOO-ELPD | −76.39（優於 Normal+LN = −79.38） |

#### 指標解讀

**$\hat{R}$（Potential Scale Reduction Factor，潛在尺度縮減因子）**

Gelman-Rubin 收斂診斷，比較「鏈內方差」與「鏈間方差」：

$$\hat{R} = \sqrt{\frac{\text{鏈間方差} + \text{鏈內方差（加權）}}{\text{鏈內方差}}}$$

- $\hat{R} = 1$：所有 chain 完全收斂到同一分佈
- $\hat{R} < 1.01$：嚴格標準，收斂良好
- $\hat{R} < 1.05$：寬鬆標準，一般可接受
- $\hat{R} \geq 1.1$：收斂不足，需延長取樣或重新設計 prior

**本研究 $\hat{R}$ 最大值 = 1.032**：稍高於 1.01 嚴格標準，但仍在可接受範圍；最差的參數可能是 $\sigma_\Delta$（真值 ≈ 0.038，接近退化，較難取樣）。

---

**ESS→(Effective Sample Size，有效樣本數)**

MCMC 鏈有自相關，名義上 4000 個樣本實際攜帶的資訊量等同於 ESS 個**獨立**樣本：

$$\text{ESS} = \frac{4000}{1 + 2\sum_{k=1}^{\infty}\rho_k}$$

其中 $\rho_k$ 為 lag-$k$ 自相關係數。自相關越高，ESS 越小。

- ESS > 400：可靠估計均值與方差
- ESS > 100：最低可接受門檻
- ESS < 100：需增加取樣數

**本研究 ESS 最小值 = 117**：剛過門檻，表示某參數（可能是 $\sigma_\Delta$）的鏈有中等自相關；如需更精確的後驗分位數，建議增加 warm-up 或取樣數。

---

**Divergences（發散次數）**

HMC/NUTS 在數值積分時若能量守恆被嚴重破壞（通常發生在後驗幾何急劇彎曲處，如 Neal's Funnel），積分步驟「發散」。發散點附近的後驗**未被正確探索**，會導致後驗估計偏差。

- Divergences = 0：HMC 完全正常運作，後驗幾何平坦
- Divergences > 0：需縮小步長（`target_accept` 提高）或改用 NCP

**本研究 Divergences = 0**：NCP 成功將 Neal's Funnel 展平，所有 4 條 chain 均無發散。

---

**LOO-ELPD→(Leave-One-Out Expected Log Pointwise Predictive Density，留一交叉驗證期望對數點預測密度)**

每次留出一筆觀測，用其餘 74 筆訓練，對留出點計算對數預測密度，再加總：

$$\text{LOO-ELPD} = \sum_{i=1}^{75} \log p(y_i \mid \mathbf{y}_{-i})$$

- 值越大（越靠近 0）= 預測能力越好
- 差值 $\Delta\text{ELPD} > 2$：通常視為有意義的差距

**本研究**：SEV+LN = −76.39，Normal+LN = −79.38，$\Delta = 2.99$；SEV 的尾部行為更符合疲勞壽命資料，預測能力優於 Normal。

### 5.2 後驗參數估計（SEV + LogNormal）

| 參數 | 後驗均值 | 後驗 SD | 95% CI | MLE 參考值 |
|------|---------|---------|--------|-----------|
| $\beta_0$ | −9.286 | 0.427 | — | −9.370 |
| $\beta_1$ | −8.746 | 1.355 | — | −8.534 |
| $\sigma$ | 0.195 | 0.072 | — | 0.190 |
| $\mu_\Delta$ | −0.660 | 0.082 | — | −0.644 |
| $\sigma_\Delta$ | 0.038 | 0.006 | — | 0.036 |

> 後驗均值幾乎完全重現 MLE——Uniform prior 在 n=75 下被 likelihood 主導，貝式與頻率論結果高度一致。

### 5.3 後驗 ASSE 分佈（SEV + LogNormal，n=75 觀測）

| 統計量 | 後驗值 |
|--------|--------|
| 後驗均值 ASSE | 13.83 |
| 後驗中位數 ASSE | 13.13 |
| 95% 後驗 CI | [6.13, 24.58] |
| 後驗標準差 | 4.87 |

**Plugin ASSE**（後驗均值 $\hat{\theta}$ 代入）：

| 預測公式 | ASSE |
|---------|------|
| SEV + Euler 修正（$\hat{y} = \mu - \hat{\sigma}\gamma_E$） | **5.75** |
| 無修正（$\hat{y} = \mu$） | 10.19 |

> Plugin y-ASSE = 5.75 幾乎等於 MLE SEV+INLA 的 in-sample ASSE（5.76），確認後驗均值正確重現 MLE。  
> 後驗均值 ASSE = 13.83 較高，因為 $\sigma$ 的後驗不確定性（SD = 0.07）使部分樣本的預測較差——這是「帶完整參數不確定性的誠實 ASSE」。

### 5.4 各應力水準的異方差結構

| $S_j$ | $E(\ln Y \mid S_j)$ | SD（SEV） | SD（Normal） |
|-------|---------------------|----------|-------------|
| 0.675 | 6.82 | **1.158** | 1.139 |
| 0.750 | 3.38 | 0.808 | 0.782 |
| 0.825 | 0.91 | 0.599 | 0.612 |
| 0.900 | −0.99 | 0.470 | 0.515 |
| 0.950 | −2.10 | 0.525 | 0.418 |

低應力（S=0.675）的 SD 是高應力（S=0.9）的 **2.5 倍**，異方差結構非常顯著。

### 5.5 模型 z-score ASSE（y-ASSE，ln 壽命空間）

| 方法 | SEV+LN plugin | Normal+LN plugin | SEV+LN 後驗均值 | Normal+LN 後驗均值 |
|------|:-------------:|:----------------:|:--------------:|:-----------------:|
| 樣本 z-score（Chiu 風格） | 10.93 | 9.17 | 14.44 | 14.19 |
| **模型 z-score（本方法）** | **4.32** | **3.53** | **5.29** | **4.65** |
| 改善幅度（plugin） | **60.5%** | **61.5%** | — | — |

### 5.6 HL Heuristic 搜尋結果（y-ASSE plugin）

| Heuristic | SEV+LN | Normal+LN | 說明 |
|-----------|:------:|:---------:|------|
| **H0：模型 z-score + Φ(z)** | **4.32** | **3.53** | **最佳**（隱性 shrinkage） |
| H1：精確邊際 CDF | 10.38 | 12.57 | 反而更差 |
| H2：Cornish-Fisher 偏度修正 | 6.21 | 3.67 | 略差 |
| H3：Per-group z 校準 | 10.93 | 14.05 | 最差 |
| H4：DA 後驗 E[Δᵢ\|data] | 5.80 | 9.55 | 中等 |

**關鍵發現**：H1（精確邊際 CDF）比 H0（Normal 近似 $\Phi(z)$）更差。$\Phi(z)$ 對極端 z 值的隱性 shrinkage（Normal 尾部比真實邊際分佈更保守）在 n=75 的有限樣本下是有用的正規化。這是 **bias-variance tradeoff** 的典型案例：「更正確」的方法不一定在有限樣本下更好。

---

## 6. 與先前研究的比較

### 6.1 ASSE 指標說明

> **重要**：不同研究使用不同的預測評估空間，數值不可直接跨空間比較。

| 空間 | 定義 | 使用方法 |
|------|------|---------|
| **Absolute Sum of Standardized-score Errors→(ASSE)**（z-score 空間，**z-ASSE**） | $\sum_i \|\omega_{ij} - \hat{\omega}_{ij}\|$ | Chiu (2005), Roy C-2, SEV+INLA z-score |
| **Absolute Sum of Standardized-score Errors→(ASSE)**（ln 壽命空間，**y-ASSE**） | $\sum_i \|y_{ij} - \hat{y}_{ij}\|$ | 本研究（模型 z-score 方法） |
| **Absolute Sum of Standardized-score Errors→(ASSE)**（排序空間，**rank-ASSE**） | $E = \sum_i \|y_{ij} - \hat{y}_{(ij)}\|$（rank-matched） | P&M (1999) |

### 6.2 各方法 ASSE 結果彙整

#### z-score 空間（z-ASSE）

| 方法 | z-ASSE | 改善 vs Chiu | 來源 |
|------|:------:|:------------:|------|
| Chiu (2005) Errors-In-Variables→(EIV)（論文基準） | **10.80** | — | Chiu (2005) thesis |
| Roy Method A（MLE 樣本 z-score） | 12.38 | −14.6% | `rfl_chiu.py` |
| Roy Method B（Nelder-Mead 最佳化） | 10.38 | +3.9% | `rfl_chiu.py` |
| Roy Method C-1（OLS LAD z-score） | 11.02 | −2.0% | `rfl_chiu.py` |
| **Roy Method C-2（LAD 迴歸）** | **9.94** | **+8.0%** | `rfl_chiu.py` |
| SEV+INLA z-score（MLE 參數） | 11.50 | −6.5% | `rfl_asse_zscore.py` |
| Burr+INLA z-score | 11.92 | −10.4% | `rfl_asse_zscore.py` |

#### ln 壽命空間（y-ASSE，in-sample plugin）

| 方法 | y-ASSE | 模型 | 來源 |
|------|:------:|------|------|
| Burr+EM-GMM（unconstrained） | 0.49 | SEV | `rfl_burr_em.py`（過擬合） |
| Burr+EM-GMM（σ≥0.15, a≥1）Mode A | 4.09 | SEV | `rfl_burr_em.py` |
| Burr+INLA | 5.74 | SEV | `rfl_burr_inla.py` |
| SEV+INLA（MLE in-sample） | 5.76 | SEV | `rfl_profile.py` |
| **Bayes DA Plugin（SEV+Euler）** | **5.75** | SEV | `rfl_bayes_asse.py` |
| SEV 樣本 z-score | 10.93 | SEV | `rfl_model_zscore_asse.py` |
| Normal 樣本 z-score | 9.17 | Normal | `rfl_model_zscore_asse.py` |
| **模型 z-score（SEV，H0）** | **4.32** | SEV | `rfl_model_zscore_asse.py` |
| **模型 z-score（Normal，H0）** | **3.53** | Normal | `rfl_model_zscore_asse.py` |

#### rank 空間（P&M 1999 預測準則，E）

| 方法 | E | 備註 |
|------|---|------|
| Normal-Normal MLE（混凝土資料） | 12.84 | P&M (1999) Response（最優） |
| Roy Method B（rank-ASSE，直接最佳化） | 12.24 | `rfl_chiu.py` |
| Roy Method C-2（rank-ASSE） | 12.24（z），12.xx（rank） | `rfl_chiu.py` |

### 6.3 方法比較總覽

| 方法 | 參數不確定性 | 異方差處理 | 可分離 Δ vs σ | 計算成本 |
|------|:-----------:|:---------:|:-------------:|:-------:|
| Chiu (2005) z-score | ✗ | ✗ | ✗ | 低 |
| MLE + INLA | ✗ | 部分 | ✗ | 中 |
| Bayes DA + sample z-score | ✓ | ✗ | ✗ | 高 |
| **Bayes DA + 模型 z-score（本研究）** | ✓ | ✓ | ✓ | 高 |

---

## 7. 理論貢獻

### 7.1 模型矩替換的正確性

本方法的核心替換：

$$\underbrace{\omega_{ij} = \frac{y_{ij} - \bar{y}_j}{s_j}}_{\text{Chiu：樣本統計，混入噪聲}} \longrightarrow \underbrace{z_{ij} = \frac{y_{ij} - E(\ln N \mid S_j)}{\sqrt{V(\ln N \mid S_j)}}}_{\text{本方法：模型統計，噪聲已分離}}$$

$V(\ln N \mid S_j) = \beta_1^2 \cdot \text{Var}[\ln(S_j - \Delta)] + \pi^2\sigma^2/6$ 的第一項是 $\Delta$ 的貢獻，第二項是 SEV 殘差——兩個來源被全貝式推論清楚地分開。

### 7.2 與條件後驗的關係

模型 z-score 管線本質上是計算個別 $\Delta_i$ 的條件後驗近似：

$$p(\Delta_i \mid y_i, \theta) \propto f(y_i \mid \Delta_i, \theta) \cdot g(\Delta_i \mid \theta)$$

本方法用 marginal percentile 代理後驗的「位置」——這是把反問題（$y_i \to \Delta_i$）以矩匹配求解的優雅做法，避免對每個 $i$ 做數值積分，且比 DA 後驗 $E[\Delta_i \mid \mathbf{y}, \theta^{(s)}]$（H4）有更低的 ASSE。

### 7.3 Φ(z) 的隱性正規化效果

精確邊際 CDF（H1）比 $\Phi(z)$（H0）更差，原因在於：

$$\text{真實邊際分佈} = \int f_{\text{SEV}}(y; \mu_\Delta(\Delta), \sigma) \cdot g(\Delta) \, d\Delta \neq \text{Normal}$$

真實邊際在尾部比 Normal 更厚，導致極端 $z$ 值對應的 percentile 更高，$\hat{\Delta}$ 更極端，預測誤差更大。$\Phi(z)$ 把極端 $z$ 向均值壓縮（隱性 shrinkage），在 $n=75$ 的有限樣本下有效降低 variance。

---

## 8. 程式碼說明

### 核心程式碼（本研究）

| 檔案 | 功能 | 重要度 |
|------|------|:------:|
| `rfl_pymc_da.py` | **主推論**：DA+NCP+NUTS (SEV+LN & Normal+LN 兩種模型) | ⭐⭐⭐ |
| `rfl_model_zscore_asse.py` | **模型 z-score ASSE**：Roy 新方法完整實作與驗證 | ⭐⭐⭐ |
| `rfl_hl_asse.py` | **HL 搜尋**：H0–H4 heuristic 系統比較 | ⭐⭐⭐ |
| `rfl_bayes_asse.py` | 後驗 y-ASSE 直接計算（Euler 修正版） | ⭐⭐ |

### 輔助程式碼（開發過程）

| 檔案 | 功能 | 說明 |
|------|------|------|
| `rfl_bayes_mcmc.py` | 純 NumPy Metropolis MCMC | 無 PyMC 依賴，驗證用 |
| `rfl_bayes_uniform.py` | PyMC + `as_op` + GL 積分 | 邊際 likelihood 版，較慢 |
| `rfl_pymc.py` | PyMC + 數值積分（4 variants） | 早期嘗試 |
| `rfl_pymc_ncp.py` | PyMC + NCP（log space + GH in z-space） | NCP 開發版本 |

---

## 9. 執行方式

### 環境需求

```bash
pip install pymc numpy scipy arviz
```

PyMC 版本建議 ≥ 5.0，使用 pytensor 自動微分。

### 主要推論（約 5–7 分鐘，4 chains）

```bash
# DA+NCP+NUTS 全貝式推論，同時跑 SEV+LN 和 Normal+LN
python rfl_pymc_da.py
```

輸出：後驗樣本、Rhat、ESS、LOO-ELPD，以及後驗均值 ASSE。

### 模型 z-score 方法驗證（約 7–10 分鐘）

```bash
# 計算各應力水準 E(lnY|S), V(lnY|S)，比較模型 z-score vs 樣本 z-score
python rfl_model_zscore_asse.py
```

輸出：異方差結構表、y-ASSE 比較（模型 vs 樣本 z-score，SEV vs Normal）。

### HL 啟發式搜尋（約 15–20 分鐘）

```bash
# H0–H4 五個 heuristic 系統搜尋
python rfl_hl_asse.py
```

輸出：H0–H4 的 y-ASSE（SEV 和 Normal），最佳 Heuristic 確認。

### 後驗 ASSE 計算（約 3–5 分鐘）

```bash
# 每個後驗 draw 各算 ASSE，輸出 ASSE 後驗分佈統計
python rfl_bayes_asse.py
```

輸出：後驗均值 ASSE = 13.83，中位數 = 13.13，95% CI = [6.13, 24.58]。

---

## 10. References

### 主要參考文獻

1. **Pascual, F. G., & Meeker, W. Q. (1999).** Estimating fatigue curves with the random fatigue-limit model. *Technometrics*, 41(4), 277–289.  
   *RFL 模型奠基論文，本研究使用其 n=75 鋁合金資料集。*

2. **Chiu, C. (2005).** *Statistical Analysis of Fatigue Data with the Random Fatigue-Limit Model.* PhD Thesis.  
   *提出 Errors-In-Variables→(EIV) z-score 方法，z-ASSE = 10.80 為本研究的主要比較基準。*

3. **Hoffman, M. D., & Gelman, A. (2014).** The No-U-Turn Sampler: Adaptively setting path lengths in Hamiltonian Monte Carlo. *Journal of Machine Learning Research*, 15(1), 1593–1623.  
   *NUTS 演算法，本研究 MCMC 推論的核心引擎。*

4. **Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2013).** *Bayesian Data Analysis* (3rd ed.). CRC Press.  
   *Non-Centered Parameterization（Neal's Funnel 解法）的方法論依據。*

5. **Salvatier, J., Wiecki, T. V., & Fonnesbeck, C. (2016).** Probabilistic programming in Python using PyMC3. *PeerJ Computer Science*, 2, e55.  
   *PyMC 貝式推論框架。*

### 方法論相關

6. **Tanner, M. A., & Wong, W. H. (1987).** The calculation of posterior distributions by data augmentation. *Journal of the American Statistical Association*, 82(398), 528–540.  
   *Data Augmentation 的理論基礎。*

7. **Neal, R. M. (2003).** Slice sampling. *Annals of Statistics*, 31(3), 705–767.  
   *Neal's Funnel 問題首先由 Neal 描述；NCP 是其標準解法。*

8. **Weng, J. (2026).** Learning beyond gradients: Heuristic learning for sequential decision problems. Preprint.  
   *Heuristic Learning 的原始論文，本研究以 HL 框架驗證最佳預測管線。*

9. **Murphy, S. A., & van der Vaart, A. W. (2000).** On profile likelihood. *Journal of the American Statistical Association*, 95(450), 449–465.  
   *Profile Likelihood 的理論依據，與 Roy 的半參數 RFL 研究相關。*

10. **Meeker, W. Q., & Escobar, L. A. (1998).** *Statistical Methods for Reliability Data*. John Wiley & Sons.  
    *SEV 分佈（Weibull log 版本）及疲勞分析統計方法的標準參考書。*

### 貝式計算相關

11. **Vehtari, A., Gelman, A., & Gabry, J. (2017).** Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC. *Statistics and Computing*, 27(5), 1413–1432.  
    *PSIS-LOO 的方法論依據，本研究用於模型比較（LOO-ELPD）。*

12. **Betancourt, M. (2017).** A conceptual introduction to Hamiltonian Monte Carlo. *arXiv preprint arXiv:1701.02434*.  
    *HMC 與 NUTS 的深度介紹，包含 NCP 的幾何解釋。*
