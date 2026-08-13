#!/usr/bin/env python3
"""
rfl_hl_asse.py — Heuristic Learning for RFL Model z-score ASSE
================================================================
HL 框架：把 ASSE 計算管線視為「Policy」，迭代改進。

HL 精神：
  Policy     = (theta, data) -> ASSE 的完整映射程式
  Reward     = -ASSE
  Coding loop = 提出候選 Heuristic → 評估 → 保留最佳

在這裡我們手動實作數個 Heuristic 變體，模擬 HL 一次迭代的過程：
  H0 (baseline) : 模型 z-score + Φ(z) [Normal 近似]
  H1            : 使用實際邊際 CDF（GL 積分）取代 Φ(z)
  H2            : Cornish-Fisher 偏度修正（處理 SEV 右偏）
  H3            : Per-group 校準（用組內殘差校正 z-scale）
  H4            : 直接使用 DA 後驗的 E[Δᵢ|data]（終極對照）
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import arviz as az
from scipy import stats, special
import time

EULER_GAMMA = 0.5772156649015329
PI2_OVER_6  = np.pi**2 / 6.0

# ═══════════════════════════════════════════════════════════════════════
# 1. DATA
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
S_list, Y_list, ev_list = [], [], []
for s, vals in _raw.items():
    mx = max(vals); nm = vals.count(mx)
    for v in vals:
        S_list.append(s); Y_list.append(np.log(v))
        ev_list.append(0 if (v == mx and nm > 1) else 1)

S_OBS  = np.array(S_list,  dtype=float)
Y_OBS  = np.array(Y_list,  dtype=float)
EVENT  = np.array(ev_list, dtype=int)
N_OBS  = len(S_OBS)
MIN_S  = S_OBS.min()
S_UNIQ = np.array(sorted(set(S_OBS)))
fail_idx = np.where(EVENT == 1)[0]
cens_idx = np.where(EVENT == 0)[0]
Y_fail   = Y_OBS[fail_idx]
Y_cens   = Y_OBS[cens_idx]

N_QUAD = 32
GL_X, GL_W = np.polynomial.legendre.leggauss(N_QUAD)


# ═══════════════════════════════════════════════════════════════════════
# 2. 共用：理論 E, V, 邊際 CDF
# ═══════════════════════════════════════════════════════════════════════

def gl_moments(mu_d, sd_d, S_j):
    """E[ln(S-Δ)], Var[ln(S-Δ)] given Δ~LN(mu_d,sd_d), Delta<S_j"""
    d_pts = S_j/2*(GL_X+1); jac = S_j/2
    log_d = np.log(np.maximum(d_pts,1e-300))
    log_Sd= np.log(np.maximum(S_j-d_pts,1e-300))
    lg    = (-0.5*np.log(2*np.pi)-np.log(sd_d)-(log_d-mu_d)**2/(2*sd_d**2)-log_d)
    w_g   = GL_W*np.exp(lg)*jac
    norm  = w_g.sum()
    if norm < 1e-10:
        return 0.0, 0.0, norm
    E1  = np.dot(w_g, log_Sd)/norm
    E2  = np.dot(w_g, log_Sd**2)/norm
    return E1, max(E2-E1**2, 0.0), norm


def E_V_sev(b0, b1, sig, mu_d, sd_d, S_j):
    """SEV: E(lnY|S), V(lnY|S)"""
    E1, Var, norm = gl_moments(mu_d, sd_d, S_j)
    if norm < 1e-10:
        return b0-sig*EULER_GAMMA, PI2_OVER_6*sig**2
    return b0-sig*EULER_GAMMA+b1*E1, max(b1**2*Var+PI2_OVER_6*sig**2, 1e-8)


def E_V_normal(b0, b1, sig, mu_d, sd_d, S_j):
    """Normal: E(lnY|S), V(lnY|S)"""
    E1, Var, norm = gl_moments(mu_d, sd_d, S_j)
    if norm < 1e-10:
        return b0, sig**2
    return b0+b1*E1, max(b1**2*Var+sig**2, 1e-8)


def marginal_cdf(y, b0, b1, sig, mu_d, sd_d, S_j, error='sev'):
    """P(lnY ≤ y | S_j) via GL quadrature — 精確邊際 CDF"""
    d_pts = S_j/2*(GL_X+1); jac = S_j/2
    log_d = np.log(np.maximum(d_pts,1e-300))
    log_Sd= np.log(np.maximum(S_j-d_pts,1e-300))
    lg    = (-0.5*np.log(2*np.pi)-np.log(sd_d)-(log_d-mu_d)**2/(2*sd_d**2)-log_d)
    g     = np.exp(lg)
    mu_c  = b0+b1*log_Sd
    if error=='sev':
        z   = (y-mu_c)/sig
        cdf = 1-np.exp(-np.exp(z))   # SEV CDF = 1 - exp(-exp(z))
    else:
        cdf = stats.norm.cdf(y, mu_c, sig)
    integrand = cdf*g*jac
    norm = (g*jac*GL_W).sum()
    if norm < 1e-10:
        return 0.5
    return float(np.dot(GL_W, integrand)/norm)


def marginal_skewness(b0, b1, sig, mu_d, sd_d, S_j, E_j, V_j):
    """3rd central moment → skewness for Cornish-Fisher"""
    d_pts = S_j/2*(GL_X+1); jac = S_j/2
    log_d = np.log(np.maximum(d_pts,1e-300))
    log_Sd= np.log(np.maximum(S_j-d_pts,1e-300))
    lg    = (-0.5*np.log(2*np.pi)-np.log(sd_d)-(log_d-mu_d)**2/(2*sd_d**2)-log_d)
    w_g   = GL_W*np.exp(lg)*jac
    norm  = w_g.sum()
    if norm < 1e-10: return 0.0
    mu_c  = b0+b1*log_Sd-sig*EULER_GAMMA   # SEV mean
    m3    = np.dot(w_g, (mu_c-E_j)**3)/norm
    return m3 / max(V_j**1.5, 1e-10)


# ═══════════════════════════════════════════════════════════════════════
# 3. HEURISTICS（Policy variants）
# ═══════════════════════════════════════════════════════════════════════

def _pred(b0, b1, sig, D_hat, S_j, error):
    euler = EULER_GAMMA if error=='sev' else 0.0
    D_clip = np.clip(D_hat, 1e-6, S_j*0.999)
    return b0+b1*np.log(S_j-D_clip)-sig*euler


def H0_baseline(b0, b1, sig, mu_d, sd_d, error='sev'):
    """H0: 模型 z-score + Φ(z) (Normal 近似) ← 上一個腳本的方法"""
    EV_fn = E_V_sev if error=='sev' else E_V_normal
    asse = 0.0
    for s_j in S_UNIQ:
        idx = np.where((S_OBS==s_j)&(EVENT==1))[0]
        if len(idx)==0: continue
        E_j, V_j = EV_fn(b0,b1,sig,mu_d,sd_d,s_j)
        z_j = (Y_OBS[idx]-E_j)/np.sqrt(V_j)
        D_hat = np.exp(mu_d+sd_d*np.clip(z_j,-6,6))
        asse += np.abs(Y_OBS[idx]-_pred(b0,b1,sig,D_hat,s_j,error)).sum()
    return asse


def H1_marginal_cdf(b0, b1, sig, mu_d, sd_d, error='sev'):
    """H1: 用精確邊際 CDF 取代 Φ(z) → 更準確的 percentile"""
    asse = 0.0
    for s_j in S_UNIQ:
        idx = np.where((S_OBS==s_j)&(EVENT==1))[0]
        if len(idx)==0: continue
        p_j = np.array([marginal_cdf(y,b0,b1,sig,mu_d,sd_d,s_j,error)
                        for y in Y_OBS[idx]])
        p_j = np.clip(p_j, 1e-6, 1-1e-6)
        z_eq = stats.norm.ppf(p_j)           # p → standard normal quantile
        D_hat = np.exp(mu_d+sd_d*z_eq)
        asse += np.abs(Y_OBS[idx]-_pred(b0,b1,sig,D_hat,s_j,error)).sum()
    return asse


def H2_cornish_fisher(b0, b1, sig, mu_d, sd_d, error='sev'):
    """H2: Cornish-Fisher 偏度修正 — 處理邊際分佈的非對稱性

    已知簡化（2026-08-13 記錄，未修正計算本身，會改變下方已發布 ASSE 數字，
    非本次精確度修正範圍）：marginal_skewness() 只算了條件均值 mu_c(Δ) 隨 Δ
    變化貢獻的偏度（between-Δ），漏掉 SEV 殘差本身固定的偏度（within-Δ，
    SEV/Gumbel-type 分佈的偏度是與參數無關的常數）——依全期望公式
    (law of total cumulants)，完整邊際三階動差應該是兩者之和，目前只算了
    第一項。
    """
    EV_fn = E_V_sev if error=='sev' else E_V_normal
    asse = 0.0
    for s_j in S_UNIQ:
        idx = np.where((S_OBS==s_j)&(EVENT==1))[0]
        if len(idx)==0: continue
        E_j, V_j = EV_fn(b0,b1,sig,mu_d,sd_d,s_j)
        gamma1 = marginal_skewness(b0,b1,sig,mu_d,sd_d,s_j,E_j,V_j) if error=='sev' else 0.0
        z_j = (Y_OBS[idx]-E_j)/np.sqrt(V_j)
        # Cornish-Fisher: z_cf = z + gamma1/6*(z^2-1)
        z_cf = z_j + gamma1/6*(z_j**2-1)
        D_hat = np.exp(mu_d+sd_d*np.clip(z_cf,-6,6))
        asse += np.abs(Y_OBS[idx]-_pred(b0,b1,sig,D_hat,s_j,error)).sum()
    return asse


def H3_calibrated(b0, b1, sig, mu_d, sd_d, error='sev'):
    """H3: Per-group 校準 — 用組內 z_j 的均值/SD 修正 bias"""
    EV_fn = E_V_sev if error=='sev' else E_V_normal
    asse = 0.0
    for s_j in S_UNIQ:
        idx = np.where((S_OBS==s_j)&(EVENT==1))[0]
        if len(idx)==0: continue
        E_j, V_j = EV_fn(b0,b1,sig,mu_d,sd_d,s_j)
        z_j  = (Y_OBS[idx]-E_j)/np.sqrt(V_j)
        # 組內 z_j 的均值和標準差應該是 (0,1) — 若有偏移就校準
        z_mean = z_j.mean()
        z_std  = z_j.std(ddof=1) if len(z_j)>1 else 1.0
        z_cal  = (z_j-z_mean)/max(z_std, 0.5)   # 校準到零均值、單位方差
        D_hat  = np.exp(mu_d+sd_d*np.clip(z_cal,-6,6))
        asse  += np.abs(Y_OBS[idx]-_pred(b0,b1,sig,D_hat,s_j,error)).sum()
    return asse


def H4_da_posterior(idata, error='sev'):
    """H4: 直接用 DA 後驗的 E[Δᵢ|data] — 終極對照（不用 z-score）

    已知簡化（2026-08-13 記錄，2026-08-13 小o review 後修正措辭，未修正
    計算本身，會改變下方已發布 ASSE 數字，非本次精確度修正範圍）：這裡先把
    mu_d/sigma_d 壓成後驗均值（mudp/sddp），再用這組固定值搭配每個 draw
    自己的 z_delta 變換、最後對 draws 取平均——等於算
    exp(mean(mu_d) + mean(sigma_d)*z^(s)) 的平均，不是嚴格定義下「每個
    draw 各自變換再平均」的 E[Δᵢ|data] = mean_s[exp(mu_d^(s) +
    sigma_d^(s)*z_delta_i^(s))]。兩者一般不相等，但不是單純的 Jensen
    不等式問題（exp() 凸函數只保證 E[exp(X)]≥exp(E[X])，而這裡的 plug-in
    版本還把 E[σ_Δ·z] 換成 E[σ_Δ]·E[z]，等於同時忽略了 σ_Δ 與 z_delta
    在聯合後驗中的相關性）——正確說法是 exp() 的非線性加上
    (mu_d,sigma_d,z_delta) 聯合後驗的相關性，讓「先平均再變換」與
    「逐 draw 變換再平均」通常不可交換，兩者也沒有一般固定的大小關係；
    名稱與實作有落差。
    """
    post = idata.posterior
    b0p  = float(post["beta0"].mean()); b1p = float(post["beta1"].mean())
    sigp = float(post["sigma"].mean())
    mudp = float(post["mu_d"].mean());  sddp = float(post["sigma_d"].mean())
    z_d  = post["z_delta"].values       # (n_c, n_d, N_OBS)
    delta_post = np.exp(mudp + sddp * z_d)
    D_mean = delta_post.mean(axis=(0,1))  # posterior mean of each Delta_i
    euler  = EULER_GAMMA if error=='sev' else 0.0
    D_clip = np.clip(D_mean[fail_idx], 1e-6, S_OBS[fail_idx]*0.999)
    y_hat  = b0p + b1p*np.log(S_OBS[fail_idx]-D_clip) - sigp*euler
    return float(np.abs(Y_OBS[fail_idx]-y_hat).sum()), b0p, b1p, sigp, mudp, sddp


# ═══════════════════════════════════════════════════════════════════════
# 4. PyMC DA+NCP 模型（SEV+LN and Normal+LN）
# ═══════════════════════════════════════════════════════════════════════

def build_model(error='sev'):
    upper_mu_d = float(np.log(MIN_S)-0.05)
    with pm.Model() as m:
        b0=pm.Uniform("beta0",lower=-50.,upper=50.)
        b1=pm.Uniform("beta1",lower=-30.,upper=0.)
        ls=pm.Uniform("log_sigma",lower=np.log(0.01),upper=np.log(5.))
        sigma=pm.Deterministic("sigma",pt.exp(ls))
        md=pm.Uniform("mu_d",lower=-5.,upper=upper_mu_d)
        ls2=pm.Uniform("log_sigma_d",lower=np.log(0.001),upper=np.log(2.))
        sd=pm.Deterministic("sigma_d",pt.exp(ls2))
        z_d=pm.Normal("z_delta",mu=0.,sigma=1.,shape=N_OBS)
        ld=md+sd*z_d; delta=pm.Deterministic("delta",pt.exp(ld))
        mu_c=b0+b1*pt.log(pt.maximum(S_OBS-delta,1e-8))
        if error=='normal':
            pm.Normal("of", mu=mu_c[fail_idx], sigma=sigma, observed=Y_fail)
            z_c=(Y_cens-mu_c[cens_idx])/sigma
            pm.Potential("oc", pt.sum(pt.log(0.5*pt.erfc(z_c/pt.sqrt(2.)))))
        else:  # SEV
            z_f=(Y_fail-mu_c[fail_idx])/sigma
            pm.Potential("of",pt.sum(-pt.log(sigma)+z_f-pt.exp(pt.clip(z_f,-500,20))))
            z_c=(Y_cens-mu_c[cens_idx])/sigma
            pm.Potential("oc",pt.sum(-pt.exp(pt.clip(z_c,-500,20))))
    return m


# ═══════════════════════════════════════════════════════════════════════
# 5. MAIN — HL 迭代展示
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("="*70)
    print("  HL for RFL: Heuristic Search on y-ASSE Mapping Pipeline")
    print("="*70)

    # ── 全貝式採樣（SEV+LN 與 Normal+LN 各自獨立建模，共用同一 DA 結構） ──
    print("\n[1] Bayesian Inference: DA+NCP+NUTS (SEV+LogNormal)...")
    model_sev = build_model(error='sev')
    with model_sev:
        idata_sev = pm.sample(
            draws=2000, tune=2000, chains=4, cores=4,
            initvals=dict(beta0=-9.4,beta1=-8.5,log_sigma=np.log(0.19),
                          mu_d=-0.65,log_sigma_d=np.log(0.04),
                          z_delta=np.zeros(N_OBS)),
            target_accept=0.95, progressbar=True, random_seed=[0,1,2,3])  # raised from 0.85 (2026-08-12), see README §5.5

    pm_sev = {v: float(idata_sev.posterior[v].mean())
              for v in ["beta0","beta1","sigma","mu_d","sigma_d"]}
    b0s,b1s,ss,mds,sdds = (pm_sev["beta0"],pm_sev["beta1"],pm_sev["sigma"],
                            pm_sev["mu_d"],pm_sev["sigma_d"])
    print(f"  SEV posterior mean: b0={b0s:.3f} b1={b1s:.3f} sig={ss:.4f} "
          f"mu_d={mds:.4f} sd_d={sdds:.4f}")

    print("\n[2] Bayesian Inference: DA+NCP+NUTS (Normal+LogNormal)...")
    model_nor = build_model(error='normal')
    with model_nor:
        idata_nor = pm.sample(
            draws=2000, tune=2000, chains=4, cores=4,
            initvals=dict(beta0=-9.2,beta1=-8.1,log_sigma=np.log(0.60),
                          mu_d=-0.65,log_sigma_d=np.log(0.04),
                          z_delta=np.zeros(N_OBS)),
            target_accept=0.95, progressbar=True, random_seed=[0,1,2,3])  # raised from 0.85 (2026-08-12), see README §5.5

    pm_nor = {v: float(idata_nor.posterior[v].mean())
              for v in ["beta0","beta1","sigma","mu_d","sigma_d"]}
    b0n,b1n,sn,mdn,sddn = (pm_nor["beta0"],pm_nor["beta1"],pm_nor["sigma"],
                             pm_nor["mu_d"],pm_nor["sigma_d"])
    print(f"  NOR posterior mean: b0={b0n:.3f} b1={b1n:.3f} sig={sn:.4f} "
          f"mu_d={mdn:.4f} sd_d={sddn:.4f}")

    # ── HL 迭代：評估所有 Heuristics ─────────────────────────────────
    print("\n[3] HL Heuristic Evaluation...")
    print(f"  {'Heuristic':<45} {'SEV+LN':>9} {'NOR+LN':>9}")
    print(f"  {'-'*65}")

    rows = []

    h0_s = H0_baseline(b0s,b1s,ss,mds,sdds, 'sev')
    h0_n = H0_baseline(b0n,b1n,sn,mdn,sddn, 'normal')
    print(f"  {'H0: Model z-score + Phi(z)':<45} {h0_s:>9.4f} {h0_n:>9.4f}")
    rows.append(('H0 baseline (model z-score)', h0_s, h0_n))

    print(f"  {'H1: Exact marginal CDF (no Normal approx)':<45}", end='', flush=True)
    h1_s = H1_marginal_cdf(b0s,b1s,ss,mds,sdds,'sev')
    h1_n = H1_marginal_cdf(b0n,b1n,sn,mdn,sddn,'normal')
    print(f" {h1_s:>9.4f} {h1_n:>9.4f}")
    rows.append(('H1: Exact marginal CDF', h1_s, h1_n))

    h2_s = H2_cornish_fisher(b0s,b1s,ss,mds,sdds,'sev')
    h2_n = H2_cornish_fisher(b0n,b1n,sn,mdn,sddn,'normal')
    print(f"  {'H2: Cornish-Fisher skewness correction':<45} {h2_s:>9.4f} {h2_n:>9.4f}")
    rows.append(('H2: Cornish-Fisher', h2_s, h2_n))

    h3_s = H3_calibrated(b0s,b1s,ss,mds,sdds,'sev')
    h3_n = H3_calibrated(b0n,b1n,sn,mdn,sddn,'normal')
    print(f"  {'H3: Per-group z calibration':<45} {h3_s:>9.4f} {h3_n:>9.4f}")
    rows.append(('H3: Per-group calibration', h3_s, h3_n))

    h4_s, *_ = H4_da_posterior(idata_sev, 'sev')
    h4_n, *_ = H4_da_posterior(idata_nor, 'normal')
    print(f"  {'H4: DA posterior E[Delta_i|data]':<45} {h4_s:>9.4f} {h4_n:>9.4f}")
    rows.append(('H4: DA posterior mean of Delta_i', h4_s, h4_n))

    # ── HL 最佳 Heuristic 選擇 ───────────────────────────────────────
    best_sev = min(rows, key=lambda x: x[1])
    best_nor = min(rows, key=lambda x: x[2])

    print(f"\n[4] HL Result — Best Heuristic per model:")
    print(f"  SEV+LN best: {best_sev[0]:<40}  ASSE={best_sev[1]:.4f}")
    print(f"  NOR+LN best: {best_nor[0]:<40}  ASSE={best_nor[2]:.4f}")

    # ── 最終比較 ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  HL SUMMARY — y-ASSE improvement trajectory")
    print(f"{'='*70}")
    print(f"  {'Stage':<40} {'SEV+LN':>9} {'NOR+LN':>9}")
    print(f"  {'-'*60}")
    print(f"  {'Sample z-score (Chiu baseline)':<40} {'~10.9':>9} {'~9.2':>9}")
    for name, vs, vn in rows:
        print(f"  {name:<40} {vs:>9.4f} {vn:>9.4f}")
    print(f"\n  Context (different scale):")
    print(f"    Chiu z-ASSE   = 10.80")
    print(f"    C-2 LAD ASSE  =  9.94")
    print(f"\n  HL insight: H1 (exact CDF) shows whether Normal approx for")
    print(f"  p=Phi(z) is the bottleneck. H3/H4 shows the ceiling from DA.")
    print("\nDone.")
