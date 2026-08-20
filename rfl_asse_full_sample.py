#!/usr/bin/env python3
"""
rfl_asse_full_sample.py — 全樣本模型 z-score ASSE（正式版，取代 H0/model_zscore_asse）
==========================================================================
2026-08-19，Roy 與小c 對話釐清後定案的方法：

  1. 全 75 筆資料一律當精確失效值處理，不做合成設限（P&M(1999) 原始資料
     本身沒有真正的設限，`rfl_bayes_asse.py`/`rfl_hl_asse.py`/
     `rfl_model_zscore_asse.py` 把其中 2 筆併列最大值人工標成設限只是這個
     repo 自己的測試案例，不是原論文設定——見 README.md 既有記錄）
  2. 用 DA+NCP+NUTS 全貝式推論估出 theta_hat=(beta0,beta1,sigma,mu_d,sigma_d)
     的後驗均值（SEV、Normal 各估一次）
  3. 用「模型 z-score」percentile 法算 ASSE（Roy 確認的方法，非 DA/NCP 潛在
     變數 z_delta 後驗反推）：
       E(Y|S_j) = b0 [-sigma*gamma_E if SEV] + b1*E[ln(S_j-Delta)]
       V(Y|S_j) = b1^2*Var[ln(S_j-Delta)] + [pi^2*sigma^2/6 if SEV else sigma^2]
       z_i = (y_i - E(Y|S_j)) / sqrt(V(Y|S_j))
       Delta_hat_i = exp(mu_d + sigma_d * z_i)
       y_hat_i = b0 + b1*ln(S_j - Delta_hat_i) [- sigma*gamma_E if SEV]
       ASSE = sum_i |y_i - y_hat_i|   （全 75 筆）
  4. Gauss-Legendre 積分節點數 N_QUAD=128（非舊版的32）——32-point 版本在
     P(Delta<S_j) 正規化上有到 ~2.8% 的誤差（部分 stress 水準算出來超過1，
     機率上不該發生），64起才收斂到 <1e-5，128/256/512 完全一致，故取128。

取代對象：`rfl_model_zscore_asse.py`（同方法但用了舊的73筆合成設限資料）、
`rfl_hl_asse.py`（H0-H4 heuristic 搜尋，H0 已確認是本方法且是最優解，
H1-H4 的探索已在舊資料上得出「H0最優」的結論，不需要在新資料上重跑整套
搜尋）、`rfl_bayes_asse.py`（DA/NCP 潛在變數 z_delta 後驗反推法——Roy
2026-08-19 指出這個方法用 y_i 自己的資訊反推 Delta_i 再重建 y_hat_i 跟
y_i 比，屬於 in-sample reconstruction，不是這裡採用的方法）。

已知限制（小o code review, 2026-08-19，未在本輪修正，需要更大的重新設計才
能處理，記錄在此避免遺忘）：
  - `Delta < S` 的支撐（support）目前用 `pt.maximum(S_OBS-delta, 1e-8)` 這個
    soft floor 實作，不是真正的硬約束——`Delta >= S` 的樣本仍會拿到有限
    likelihood，邊界處梯度會出現 kink/平坦區。這個寫法是從
    `rfl_bayes_asse.py`/`rfl_pymc_da.py` 沿用下來的既有 repo 慣例，不是本檔
    新引入的問題，但正確修法需要重新參數化（而非再加一層 patch），影響
    範圍是整個 repo 的 DA+NCP 模型族，非本次「重寫 ASSE 計算」的範圍。
  - 硬編碼的 `_raw` 資料跟 `pascual_meeker_1999.csv` 逐筆核對後，`log_life`
    欄位與重新對 `life_cycles` 取 log 之間有微小差異（最大約 1.4e-3，量級
    很小但非零），尚未確認哪個才是 source of truth。
  - P&M(1999) 原始論文的兩個應用資料集規模（125 試件含10右設限；另一組
    246筆縮減至115筆）跟這個 repo 宣稱的「75 筆完整失效」對不上，資料
    provenance 本身尚待查證原始論文核實（見 todo.md 對應條目）。
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import arviz as az
from scipy import stats
import time

EULER_GAMMA = 0.5772156649015329
PI2_OVER_6 = np.pi**2 / 6.0
N_QUAD = 128   # 2026-08-19: 32 有~2.8%正規化誤差(P(Delta<S)>1)，64起收斂，128留安全邊際

# ═══════════════════════════════════════════════════════════════════════
# 1. DATA — 全 75 筆，一律精確失效值，無合成設限
# ═══════════════════════════════════════════════════════════════════════
_raw = {
    0.675: [102.95,280.32,339.83,366.9,485.62,658.96,896.33,
            1241.76,1250.2,1329.78,1399.83,1459.14,3249.82,11748.1,11748.1],
    0.750: [6.71,9.93,12.6,15.58,16.19,17.28,18.62,
            20.3,24.9,26.26,27.94,36.35,48.42,50.09,67.34],
    0.825: [1.246,1.258,1.46,1.492,2.4,2.41,2.59,
            2.903,3.33,3.59,3.847,4.11,4.82,5.56,5.598],
    0.900: [0.201,0.216,0.226,0.252,0.257,0.295,0.311,
            0.342,0.356,0.451,0.457,0.509,0.54,0.68,1.129],
    0.950: [0.037,0.072,0.074,0.076,0.083,0.085,0.105,
            0.109,0.12,0.123,0.143,0.203,0.206,0.217,0.257],
}
S_list, Y_list = [], []
for s, vals in _raw.items():
    for v in vals:
        S_list.append(s); Y_list.append(np.log(v))
S_OBS = np.array(S_list, dtype=float)
Y_OBS = np.array(Y_list, dtype=float)
N_OBS = len(S_OBS)
MIN_S = S_OBS.min()
S_UNIQ = np.array(sorted(set(S_OBS)))

GL_X, GL_W = np.polynomial.legendre.leggauss(N_QUAD)


# ═══════════════════════════════════════════════════════════════════════
# 2. 理論 E(Y|S)/V(Y|S)（邊際化掉 Delta 的 LogNormal 先驗）
# ═══════════════════════════════════════════════════════════════════════

def gl_moments(mu_d, sd_d, S_j, norm_tol=0.01):
    """E[ln(S_j-Delta) | Delta<S_j], Var[ln(S_j-Delta) | Delta<S_j],
    norm=P(Delta<S_j)。

    2026-08-19 修正（小o code review 抓到）：norm_gl（GL 加權和）原本是
    唯一的正規化分母，σ_Δ 很窄時固定節點會漏掉尖峰，算出來的值可能大於1
    （機率不該超過1）。Δ~LogNormal(μ_Δ,σ_Δ) 時 P(Δ<S_j) 其實有解析解
    = Φ((ln S_j - μ_Δ)/σ_Δ)（標準常態 CDF），現在額外算出這個解析值純粹
    當交叉檢查用：E1/Var 仍然除以 norm_gl（GL 積分本身的加總，因為 E1/Var
    的分子沒有封閉解，仍要靠數值積分，128-point 已驗證在目前後驗附近收斂），
    但若 norm_gl 與解析解 norm 差距超過 norm_tol，代表這組 (μ_Δ,σ_Δ)
    落在固定節點會漏峰的區域，直接 raise，不要沉默地用可能錯誤的 norm_gl
    繼續算下去。

    Var 改用加權中心平方差 sum(w*(log_Sd-E1)^2)/norm 而非 E2-E1**2，避免
    E1 較大時 E2-E1**2 的相減消去誤差（小o 建議）。
    """
    d_pts = S_j/2*(GL_X+1); jac = S_j/2
    log_d = np.log(np.maximum(d_pts, 1e-300))
    log_Sd = np.log(np.maximum(S_j-d_pts, 1e-300))
    lg = (-0.5*np.log(2*np.pi)-np.log(sd_d)-(log_d-mu_d)**2/(2*sd_d**2)-log_d)
    w_g = GL_W*np.exp(lg)*jac
    norm_gl = w_g.sum()

    norm = float(stats.norm.cdf((np.log(S_j)-mu_d)/sd_d))  # 解析解，恆在[0,1]
    if norm < 1e-10:
        raise ValueError(
            f"P(Delta<S_j)={norm:.3e} 太小（S_j={S_j}, mu_d={mu_d}, sd_d={sd_d}）"
            "——這組 theta 底下幾乎不可能有觀測 Delta<S_j，公式不適用，需要人工介入。")
    if abs(norm_gl-norm) > norm_tol*max(norm, 1e-12):
        raise ValueError(
            f"GL 積分算出的 P(Delta<S_j)={norm_gl:.4f} 跟解析解 {norm:.4f} 差距"
            f"超過 {norm_tol:.0%}（S_j={S_j}, mu_d={mu_d}, sd_d={sd_d}）——"
            "固定節點可能漏掉 Delta 分佈的尖峰，N_QUAD 需要加大或改用 adaptive quadrature。")

    E1 = np.dot(w_g, log_Sd)/norm_gl
    Var = np.dot(w_g, (log_Sd-E1)**2)/norm_gl
    return E1, max(Var, 0.0), norm


def E_V_sev(b0, b1, sig, mu_d, sd_d, S_j):
    E1, Var, norm = gl_moments(mu_d, sd_d, S_j)
    return b0-sig*EULER_GAMMA+b1*E1, max(b1**2*Var+PI2_OVER_6*sig**2, 1e-8), norm


def E_V_normal(b0, b1, sig, mu_d, sd_d, S_j):
    E1, Var, norm = gl_moments(mu_d, sd_d, S_j)
    return b0+b1*E1, max(b1**2*Var+sig**2, 1e-8), norm


# ═══════════════════════════════════════════════════════════════════════
# 3. 模型 z-score ASSE（全 75 筆，percentile 法）
# ═══════════════════════════════════════════════════════════════════════

def model_zscore_asse_full(b0, b1, sig, mu_d, sd_d, error='sev', verbose=False):
    """對每個 stress 水準 S_j：
      z_i = (y_i - E(Y|S_j)) / sqrt(V(Y|S_j))
      Delta_hat_i = exp(mu_d + sd_d * z_i)
      y_hat_i = b0 + b1*ln(S_j - Delta_hat_i) [- sig*gamma_E if SEV]
    回傳全 75 筆 ASSE = sum|y_i - y_hat_i|。

    2026-08-19 修正（小o code review）：拿掉舊版沿用下來、不在 Roy 拍板
    公式內的 `np.clip(z_j,-6,6)` 與 `D_clip=clip(D_hat,1e-6,S_j*0.999)`——
    這兩個 clip 會把尾端不同的觀測靜默壓成同一個 Delta_hat/y_hat，等於在
    使用者看不到的地方悄悄改變方法。改成：若 Delta_hat_i >= S_j（代表
    ln(S_j-Delta_hat_i) 無定義），直接 raise，讓呼叫者知道這組 theta_hat
    在這個 stress 水準下不適用，而不是靜默 winsorize 出一個看似正常的數字。
    """
    if error not in ('sev', 'normal'):
        raise ValueError(f"error must be 'sev' or 'normal', got {error!r}")
    EV_fn = E_V_sev if error == 'sev' else E_V_normal
    euler = EULER_GAMMA if error == 'sev' else 0.0
    asse = 0.0
    info = []
    for s_j in S_UNIQ:
        idx = np.where(S_OBS == s_j)[0]
        E_j, V_j, norm = EV_fn(b0, b1, sig, mu_d, sd_d, s_j)
        z_j = (Y_OBS[idx]-E_j)/np.sqrt(V_j)
        D_hat = np.exp(mu_d+sd_d*z_j)
        if np.any(D_hat >= s_j):
            raise ValueError(
                f"S_j={s_j}: {int((D_hat>=s_j).sum())} 筆 Delta_hat >= S_j"
                f"（超出 ln(S_j-Delta) 的定義域），theta_hat 在這個 stress 水準"
                "下不適用，需要人工檢查，不做靜默 clip。")
        y_hat = b0+b1*np.log(s_j-D_hat)-sig*euler
        resid = np.abs(Y_OBS[idx]-y_hat)
        asse += resid.sum()
        if verbose:
            info.append((s_j, len(idx), E_j, np.sqrt(V_j), norm, D_hat.mean(), resid.sum()))
    return (asse, info) if verbose else asse


# ═══════════════════════════════════════════════════════════════════════
# 4. 若未來要處理真正的設限資料：方法說明（不計算，此 repo 目前資料無真設限）
# ═══════════════════════════════════════════════════════════════════════
r"""
若資料集裡有真正的右設限觀測（已知試件活過某個時間點仍未失效，真實失效
時間未知，只知道 y_i > threshold_i），本節說明該怎麼延伸，本檔不實作也
不計算（Roy 2026-08-19 明確要求只寫做法，不用跑）。

## (a) 概似函數端（配適 theta_hat 時）——這個 repo 已有現成寫法

失效觀測用精確密度（本檔 build_model() 用的寫法）；設限觀測改用存活機率
P(T > threshold)，SEV 情形：

    z_c = (threshold_i - mu_cond_i) / sigma
    log P(T > threshold_i) = -exp(z_c)        # SEV 生存函數

Normal 情形：

    log P(T > threshold_i) = log(0.5 * erfc(z_c / sqrt(2)))

`rfl_bayes_asse.py::build_sev_lognormal_da()` 與 `rfl_pymc_da.py` 都已經
這樣寫（`obs_cens` potential），theta_hat 的貝式配適端不需要新設計，
直接沿用即可。

另外，本模型有 Delta>=S_j 的機率質量（對應「這個試件在觀測範圍內不會
失效」）——右設限觀測的存活概似要包含這個分支，不能只考慮 Delta<S_j 底下
的有限壽命密度，見 (b2) 說明。

## (b) ASSE 計算端——這裡才是真正需要延伸設計的地方（2026-08-19 依小o
code review 意見修正，原本的「單邊懲罰法」選項已移除，理由見下方說明）

percentile 法的 z_i = (y_i-E(Y|S_j))/sqrt(V(Y|S_j)) 需要精確觀測到的 y_i
才能算；設限觀測只知道 y_i > threshold_i，無法直接代入。兩種可能做法，
各有取捨：

  (b1) 排除法（最簡單，`rfl_bayes_asse.py` 稱 only_fail=True 的既有做法）：
       設限觀測完全不進 ASSE 加總，只用失效觀測。優點簡單、跟 Chiu(2005)
       原始論文的處理方式最接近；缺點是丟棄了「至少活了這麼久」這個資訊，
       樣本數變少，且報告時必須明講「這是 failures-only ASSE，評估母體
       跟全樣本不同」，不能跟全樣本 ASSE 直接並列比較。

  (b2) 條件期望插補法（README 提過的「conditional expectation imputation」，
       目前這個 repo 沒有任何腳本正確實作過）：
       用截斷分布的條件期望 E[Y_i | Y_i > threshold_i, S_j; theta_hat] 當作
       y_i 的代理真值，代入跟失效觀測完全相同的 ASSE 公式。正確做法要注意：
         - 這個條件期望要同時「條件在 Delta<S_j（試件終究會失效）」且
           「邊際化 Delta 的 LogNormal 先驗」——不能只對單一 SEV/Normal 尾部
           積分，還要對 Delta 積分（跟本檔 gl_moments() 邊際化的方式一致）
         - `|E[Y|censored]-y_hat|` 不等於 `E[|Y-y_hat| | censored]`——用條件
           期望值代入 ASSE 公式，本質上是單次 mean-imputation 評分，會丟掉
           插補本身的不確定性，不是「真正」的期望絕對誤差，這點在報告時
           要講清楚，不能跟失效觀測的精確殘差混為一談

  （原本列在這裡的「單邊懲罰法」已移除：小o code review 指出這個 ASSE
  框架的 y_hat_i 本身需要精確 y_i 才能算出 z_i，光把最後一步的絕對誤差
  換成 hinge loss max(threshold-y_hat,0) 沒辦法解決「設限觀測算不出
  y_hat_i」這個根本問題；而且這其實是「與設限集合相容的下界違反分數」，
  跟正式的 censored proper scoring rule（如 Tobit likelihood）不是同一
  回事，也查無文獻支持把它跟原始 ASSE 直接比較，所以不列為選項，避免
  誤導未來的實作者。）

  兩者裡 (b1) 最簡單也最接近既有慣例，若要延伸到真設限資料建議從 (b1)
  開始，(b2) 留待有真正需要且有時間做完整實作時再評估。
"""


# ═══════════════════════════════════════════════════════════════════════
# 5. DA+NCP+NUTS 模型（SEV/Normal，全 75 筆精確概似，無合成設限）
# ═══════════════════════════════════════════════════════════════════════

def build_model(error='sev'):
    """2026-08-19 修正（小o code review）：SEV 分支原本用手寫 potential +
    `pt.clip(z_f,-500,20)`——一旦 z_f 超過 clip 上界，exp(z_f) 項不再變化
    但線性項繼續增加，logp 與梯度方向會算錯，不只是數值保護，是真的公式
    bug。改用 PyMC 內建 `pm.Gumbel`（右偏/max 型）配合正負號反轉的標準
    技巧來實作 SEV（左偏/min 型）：若 Y~SEV(mu,beta)，則 -Y~Gumbel_max(-mu,beta)，
    所以觀測值與位置參數都取負號餵給 `pm.Gumbel`，內建 logp 實作不需要
    手動 clip。"""
    upper_mu_d = float(np.log(MIN_S)-0.05)
    with pm.Model() as m:
        b0 = pm.Uniform("beta0", lower=-50., upper=50.)
        b1 = pm.Uniform("beta1", lower=-30., upper=0.)
        log_sig = pm.Uniform("log_sigma", lower=np.log(0.01), upper=np.log(5.))
        sigma = pm.Deterministic("sigma", pt.exp(log_sig))
        mu_d = pm.Uniform("mu_d", lower=-5., upper=upper_mu_d)
        log_sdd = pm.Uniform("log_sigma_d", lower=np.log(0.001), upper=np.log(2.))
        sigma_d = pm.Deterministic("sigma_d", pt.exp(log_sdd))
        z_delta = pm.Normal("z_delta", mu=0., sigma=1., shape=N_OBS)
        log_delta = mu_d+sigma_d*z_delta
        delta = pm.Deterministic("delta", pt.exp(log_delta))
        mu_cond = b0+b1*pt.log(pt.maximum(S_OBS-delta, 1e-8))
        if error == 'normal':
            pm.Normal("obs_all", mu=mu_cond, sigma=sigma, observed=Y_OBS)
        elif error == 'sev':  # 全 75 筆一律精確密度，無 obs_cens
            pm.Gumbel("obs_all", mu=-mu_cond, beta=sigma, observed=-Y_OBS)
        else:
            raise ValueError(f"error must be 'sev' or 'normal', got {error!r}")
    return m


# ═══════════════════════════════════════════════════════════════════════
# 5b. HL Loop（Roy 2026-08-19 指定：套用 wiki [[concept_heuristic_learning]]
#     的模式到「參數估計」這一步，對應既有前例
#     concept_bayesian_meta_analysis_extended 的 hl_loop.py:run_until_stable
# ═══════════════════════════════════════════════════════════════════════
"""
HL 三層對應（2026-08-19 第二版：改成候選集合+挑最佳，不是逐步升級鏈）：
  第一版用「跑一輪→依規則調整→再跑一輪」的升級鏈，結果 divergence/Rhat
  隨輪數單調變差（3165→3108→4973→7116），不是慢慢收斂再撞頂——代表升級
  規則的假設（調高 target_accept、拉長 tune/draws 會讓下一輪變好）對這個
  模型不成立。Roy 糾正：不該預期單一條鏈剛好收斂，該平行跑幾組候選設定，
  直接挑診斷最好的那組。

  state    = diagnose_fit()：讀 idata 算出 divergence 數、Rhat_max、ESS_min
  policy   = CANDIDATE_TARGET_ACCEPTS：具名候選集合（不是升級規則），
             draws/tune 固定不隨候選變動，只讓 target_accept 這個變因掃過
             {0.85,0.90,0.95,0.99}（沿用本專案其他腳本已驗證過的常用掃法）
  feedback = 每個候選都印出完整診斷，全部跑完後列表比較
  loop     = search_best_config()：跑完全部候選，挑 Rhat_max 最低（若打平
             再看 divergence 較少）的那組當最終 theta_hat，不是「最後一個」
             也不是「第一個達標的」——沒有任何一組達標時，明確報告「以下
             是候選集合中最好的一組，仍未達收斂標準」，不偽稱已收斂。
"""

DIVERGENCE_OK = 50      # 每 8000 draws 允許的 divergence 上限（寬鬆但非零，
                         # 這個模型族已知很難完全歸零，見 README 既有記錄）
RHAT_OK = 1.05
ESS_OK = 200
CANDIDATE_TARGET_ACCEPTS = [0.85, 0.90, 0.95, 0.99]
FIXED_DRAWS = 2000
FIXED_TUNE = 2000


def diagnose_fit(idata):
    """state：從 idata 讀出這輪的收斂診斷。"""
    n_div = int(idata.sample_stats.diverging.sum())
    summ = az.summary(idata, var_names=["beta0", "beta1", "sigma", "mu_d", "sigma_d"])
    rhat_max = float(summ["r_hat"].max())
    ess_min = float(summ["ess_bulk"].min())
    ok = (n_div <= DIVERGENCE_OK) and (rhat_max <= RHAT_OK) and (ess_min >= ESS_OK)
    return dict(n_div=n_div, rhat_max=rhat_max, ess_min=ess_min, ok=ok)


def search_best_config(error, init_dict, label):
    """loop：平行跑 CANDIDATE_TARGET_ACCEPTS 裡每一組 target_accept（draws/
    tune 固定），全部跑完後挑 Rhat_max 最低的一組（打平則看 divergence 較
    少）當最終結果。誠實回報有沒有任何一組真正達標。"""
    candidates = []
    for target_accept in CANDIDATE_TARGET_ACCEPTS:
        print(f"\n  -- HL candidate [{label}]: draws={FIXED_DRAWS} tune={FIXED_TUNE} "
              f"target_accept={target_accept:.2f} --")
        model = build_model(error=error)
        with model:
            idata = pm.sample(
                draws=FIXED_DRAWS, tune=FIXED_TUNE, chains=4, cores=4,
                initvals=init_dict, target_accept=target_accept,
                progressbar=True, random_seed=[0, 1, 2, 3], nuts_sampler='pymc',
            )
        state = diagnose_fit(idata)
        print(f"     state: n_div={state['n_div']} rhat_max={state['rhat_max']:.4f} "
              f"ess_min={state['ess_min']:.1f} ok={state['ok']}")
        pm_ = {v: float(idata.posterior[v].mean()) for v in ["beta0", "beta1", "sigma", "mu_d", "sigma_d"]}
        candidates.append(dict(target_accept=target_accept, state=state, theta=pm_))

    print(f"\n  候選比較表 [{label}]:")
    print(f"  {'target_accept':>13} {'n_div':>6} {'rhat_max':>9} {'ess_min':>8} {'ok':>6}")
    for c in candidates:
        s = c["state"]
        print(f"  {c['target_accept']:>13.2f} {s['n_div']:>6d} {s['rhat_max']:>9.4f} {s['ess_min']:>8.1f} {str(s['ok']):>6}")

    best = min(candidates, key=lambda c: (c["state"]["rhat_max"], c["state"]["n_div"]))
    if best["state"]["ok"]:
        print(f"  -> 選 target_accept={best['target_accept']:.2f}（達標）")
    else:
        print(f"  -> 沒有任何候選達標，選候選集合中 Rhat_max 最低的一組"
              f"（target_accept={best['target_accept']:.2f}，仍不可信，如實回報）")
    return best["theta"], candidates, best


# ═══════════════════════════════════════════════════════════════════════
# 6. MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("="*70)
    print(f"  全樣本模型 z-score ASSE（n={N_OBS}，無合成設限，N_QUAD={N_QUAD}）")
    print("="*70)

    results = {}
    for error, init_sig, label in [('sev', 0.19, 'SEV'), ('normal', 0.60, 'Normal')]:
        print(f"\n[{label}] DA+NCP+NUTS via HL candidate search...")
        init_dict = dict(beta0=-9.4 if error=='sev' else -9.2,
                          beta1=-8.5 if error=='sev' else -8.1,
                          log_sigma=np.log(init_sig),
                          mu_d=-0.65, log_sigma_d=np.log(0.04),
                          z_delta=np.zeros(N_OBS))
        t0 = time.time()
        pm_, candidates, best = search_best_config(error, init_dict, label)
        print(f"  elapsed {time.time()-t0:.1f}s total across {len(candidates)} candidate(s)")

        asse, info = model_zscore_asse_full(
            pm_["beta0"], pm_["beta1"], pm_["sigma"], pm_["mu_d"], pm_["sigma_d"],
            error=error, verbose=True)
        print(f"\n  [{label}] 全75筆模型z-score ASSE = {asse:.4f}（用 target_accept={best['target_accept']:.2f} 這組 theta_hat）")
        print(f"  {'S_j':>6} {'n':>3} {'E(Y|S)':>9} {'SD(Y|S)':>9} {'P(D<S)':>8} {'Dhat_mean':>10} {'group_SAE':>10}")
        for s_j, n, E_j, sd_j, norm, D_m, r in info:
            print(f"  {s_j:>6.3f} {n:>3d} {E_j:>9.4f} {sd_j:>9.4f} {norm:>8.5f} {D_m:>10.4f} {r:>10.4f}")

        results[label] = dict(theta=pm_, asse=asse, best=best, candidates=candidates)

    print(f"\n{'='*70}\n總結\n{'='*70}")
    for label, res in results.items():
        s = res['best']['state']
        print(f"  {label}: ASSE={res['asse']:.4f}  target_accept={res['best']['target_accept']:.2f}  converged={s['ok']}  "
              f"(divergences={s['n_div']}, Rhat_max={s['rhat_max']:.4f}, ESS_min={s['ess_min']:.1f})")
        print(f"    theta_hat={res['theta']}")
