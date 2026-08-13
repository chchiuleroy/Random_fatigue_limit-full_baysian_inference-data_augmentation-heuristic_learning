#!/usr/bin/env python3
"""
rfl_bayes_asse.py — 全貝式後驗 ASSE 計算
===========================================
用 DA+NCP+NUTS 後驗樣本計算 ASSE 的後驗分佈

兩種 ASSE 定義：
  z-ASSE  (Chiu 2005) = Σ|ω_t − ω̂_t|  → 組內 z-score 空間
  y-ASSE  (本檔，直接)= Σ|y_i − ŷ_i|  → log(壽命) 空間

預測 ŷᵢ（每個後驗樣本 s）：
  Normal: ŷᵢˢ = β₀ˢ + β₁ˢ × ln(Sᵢ − Δᵢˢ)
  SEV   : ŷᵢˢ = β₀ˢ + β₁ˢ × ln(Sᵢ − Δᵢˢ) − σˢ × γ_E
          其中 γ_E = 0.5772... (Euler-Mascheroni 常數)

Δᵢˢ 直接從 DA 後驗取得，不需額外積分或 z-score 轉換。
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import arviz as az
from scipy import special
import time

EULER_GAMMA = 0.5772156649015329   # Euler-Mascheroni 常數

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
S_OBS  = np.array(S_list,  dtype=np.float64)
Y_OBS  = np.array(Y_list,  dtype=np.float64)
EVENT  = np.array(ev_list, dtype=int)
N_OBS  = len(S_OBS)
MIN_S  = S_OBS.min()
LOG_S  = np.log(S_OBS)

fail_idx = np.where(EVENT == 1)[0]   # 73 failures
cens_idx = np.where(EVENT == 0)[0]   # 2 censored
Y_fail   = Y_OBS[fail_idx]
Y_cens   = Y_OBS[cens_idx]


# ═══════════════════════════════════════════════════════════════════════
# 2. DA+NCP MODEL（SEV + LogNormal，最佳 variant）
# ═══════════════════════════════════════════════════════════════════════

def build_sev_lognormal_da():
    upper_mu_d = float(np.log(MIN_S) - 0.05)
    with pm.Model() as model:
        b0      = pm.Uniform("beta0",      lower=-50.,         upper=50.)
        b1      = pm.Uniform("beta1",      lower=-30.,         upper=0.)
        log_sig = pm.Uniform("log_sigma",  lower=np.log(0.01), upper=np.log(5.))
        sigma   = pm.Deterministic("sigma", pt.exp(log_sig))
        mu_d    = pm.Uniform("mu_d",       lower=-5.,          upper=upper_mu_d)
        log_sdd = pm.Uniform("log_sigma_d",lower=np.log(0.001),upper=np.log(2.))
        sigma_d = pm.Deterministic("sigma_d", pt.exp(log_sdd))

        # NCP: z ~ N(0,1), log(Δ) = μ_Δ + σ_Δ · z
        z_delta   = pm.Normal("z_delta", mu=0., sigma=1., shape=N_OBS)
        log_delta = mu_d + sigma_d * z_delta
        delta     = pm.Deterministic("delta", pt.exp(log_delta))

        mu_cond = b0 + b1 * pt.log(pt.maximum(S_OBS - delta, 1e-8))

        # SEV likelihood
        z_f = (Y_fail - mu_cond[fail_idx]) / sigma
        pm.Potential("obs_fail",
            pt.sum(-pt.log(sigma) + z_f - pt.exp(pt.clip(z_f, -500, 20))))
        z_c = (Y_cens - mu_cond[cens_idx]) / sigma
        pm.Potential("obs_cens",
            pt.sum(-pt.exp(pt.clip(z_c, -500, 20))))
    return model


# ═══════════════════════════════════════════════════════════════════════
# 3. ASSE 向量化計算（後驗樣本全部同時算）
# ═══════════════════════════════════════════════════════════════════════

def compute_asse_posterior(idata, error="sev", only_fail=True):
    """
    對每個 posterior draw 計算 ASSE。

    idata  : pm.sample 輸出的 InferenceData
    error  : 'sev' | 'normal'
    only_fail : True → 只算 73 個失效觀測（README「排除設限觀測」做法，
                本檔預設值，headline 13.83/13.13 用的就是這組）；
                False → 全 75 個，見下方已知簡化

    已知簡化（only_fail=False 時，2026-08-13 記錄，未修正計算，非本次
    精確度修正範圍）：設限觀測的殘差直接用 |Y_OBS[cens_idx] - y_pred|
    計算，但 Y_OBS 對設限列存的是設限門檻（同組併列最大值），不是真實
    失效時間，也沒有做條件期望插補——等於把設限門檻當成真實 outcome
    算誤差，README 原本描述的兩種正規做法（排除 / conditional
    expectation imputation）都不是。

    返回 asse_samples (n_chains × n_draws,)
    """
    if error not in ("sev", "normal"):
        # 2026-08-13 fix: 原本 `if error == "sev": ... else: ...` 對任何
        # 打錯字的值（如 "SEV"）都會靜默落入 else 分支（無 Euler
        # correction），沒有任何錯誤訊息，容易算出錯誤結果卻不自知。
        raise ValueError(f"error must be 'sev' or 'normal', got {error!r}")
    post = idata.posterior

    # 取後驗樣本 → (n_c, n_d)
    b0_    = post["beta0"].values      # (n_c, n_d)
    b1_    = post["beta1"].values
    sig_   = post["sigma"].values
    mu_d_  = post["mu_d"].values
    sdd_   = post["sigma_d"].values
    z_d_   = post["z_delta"].values    # (n_c, n_d, N_OBS)

    # Δᵢˢ = exp(μ_Δˢ + σ_Δˢ · zᵢˢ)，shape (n_c, n_d, N_OBS)
    log_delta = mu_d_[:, :, None] + sdd_[:, :, None] * z_d_
    delta     = np.exp(log_delta)

    # μ_cond shape (n_c, n_d, N_OBS)
    S_e    = S_OBS[None, None, :]
    mu_c   = b0_[:,:,None] + b1_[:,:,None] * np.log(np.maximum(S_e - delta, 1e-8))

    # SEV 的條件均值 = μ_cond - σ · γ_E
    if error == "sev":
        ypred = mu_c - sig_[:,:,None] * EULER_GAMMA
    else:
        ypred = mu_c                   # Normal: mean = mode

    # ASSE = Σ|y_i − ŷᵢ|
    Y_ = Y_OBS[None, None, :]
    if only_fail:
        residuals = np.abs(Y_[:,  :, fail_idx] - ypred[:, :, fail_idx])
    else:
        residuals = np.abs(Y_ - ypred)

    asse = residuals.sum(axis=-1)      # (n_c, n_d)

    # flatten → 1D array of all posterior ASSE values
    return asse.flatten()


# ═══════════════════════════════════════════════════════════════════════
# 4. MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"n={N_OBS}  failures={len(fail_idx)}  censored={len(cens_idx)}")
    print("Model: SEV + LogNormal  |  DA + NCP + NUTS\n")

    # ── Sampling ─────────────────────────────────────────────────────
    model = build_sev_lognormal_da()
    t0    = time.time()

    with model:
        idata = pm.sample(
            draws         = 2000,
            tune          = 2000,
            chains        = 4,
            cores         = 4,
            initvals      = dict(beta0=-9.4, beta1=-8.5,
                                 log_sigma=np.log(0.19),
                                 mu_d=-0.65, log_sigma_d=np.log(0.04),
                                 z_delta=np.zeros(N_OBS)),
            target_accept = 0.85,
            progressbar   = True,
            random_seed   = [0,1,2,3],
        )

    elapsed = time.time() - t0
    print(f"\nSampling done: {elapsed:.1f}s")

    # ── Convergence check ────────────────────────────────────────────
    vnames = ["beta0","beta1","sigma","mu_d","sigma_d"]
    summ   = az.summary(idata, var_names=vnames, round_to=4)
    print("\nPosterior summary (global params):")
    print(summ[["mean","sd","eti89_lb","eti89_ub","r_hat","ess_bulk"]].to_string())

    # 2026-08-13 fix: the block above only checked the 5 named global
    # params, never the 75 z_delta latent variables that are also part of
    # the sampled state (80-dim total) -- a global param can look converged
    # while individual latents haven't. Summarize the worst-case across all
    # 75 dims instead of printing all of them (too verbose for routine runs).
    z_summ = az.summary(idata, var_names=["z_delta"], round_to=4)
    print(f"\nz_delta (75 latent Δᵢ, worst-case across all dims): "
          f"max r_hat={z_summ['r_hat'].max():.4f}  "
          f"min ess_bulk={z_summ['ess_bulk'].min():.0f}")

    # ── ASSE 後驗計算 ────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  ASSE POSTERIOR DISTRIBUTION")
    print("="*60)

    # A. y-ASSE（log-壽命空間，僅失效觀測）
    asse_fail = compute_asse_posterior(idata, error="sev", only_fail=True)
    # B. y-ASSE（全部觀測，含設限，設限貢獻其預測值）
    asse_all  = compute_asse_posterior(idata, error="sev", only_fail=False)

    print(f"\n  [A] y-ASSE（73 failures only）:")
    print(f"      Posterior mean   = {asse_fail.mean():.4f}")
    print(f"      Posterior median = {np.median(asse_fail):.4f}")
    print(f"      95% CI           = [{np.percentile(asse_fail,2.5):.4f}, "
          f"{np.percentile(asse_fail,97.5):.4f}]")
    print(f"      SD               = {asse_fail.std():.4f}")

    print(f"\n  [B] y-ASSE（75 obs, incl. censored）:")
    print(f"      Posterior mean   = {asse_all.mean():.4f}")
    print(f"      Posterior median = {np.median(asse_all):.4f}")
    print(f"      95% CI           = [{np.percentile(asse_all,2.5):.4f}, "
          f"{np.percentile(asse_all,97.5):.4f}]")

    # ── Plugin ASSE（後驗均值代入）────────────────────────────────────
    # 2026-08-13 note (wording corrected after 小o review): delta_pm here
    # is exp(E[mu_d]+E[sigma_d]*E[z_delta]) -- a plug-in FITTED POINT
    # ESTIMATE using posterior MEANS of the underlying Normal parameters
    # (and each specimen's own posterior mean z_delta), evaluated once.
    # This is NOT the same quantity as `delta_mean` in the "Per-specimen Δ
    # posterior" section below, which is E[exp(mu_d+sigma_d*z_delta)] --
    # the true posterior mean of Δ, averaged per-draw. This is NOT simply
    # Jensen's inequality (exp is convex, but the plug-in version also
    # replaces E[sigma_d*z] with E[sigma_d]*E[z], discarding the posterior
    # correlation between sigma_d and z_delta) -- the accurate statement is
    # that exp()'s nonlinearity combined with the joint posterior
    # correlation of (mu_d, sigma_d, z_delta) makes "average-then-
    # transform" and "transform-then-average" generally non-commuting,
    # with no fixed ordering between them. They intentionally answer
    # different questions (plug-in fitted estimate vs. posterior
    # expectation) and are not meant to match numerically -- flagged here
    # since both sections reuse the name "delta" and were previously easy
    # to conflate.
    pm_  = {v: float(idata.posterior[v].mean()) for v in
            ["beta0","beta1","sigma","mu_d","sigma_d"]}
    z_pm = idata.posterior["z_delta"].values.mean(axis=(0,1))   # (N_OBS,)
    delta_pm = np.exp(pm_["mu_d"] + pm_["sigma_d"] * z_pm)
    mu_c_pm  = pm_["beta0"] + pm_["beta1"] * np.log(
                   np.maximum(S_OBS - delta_pm, 1e-8))
    ypred_pm_sev  = mu_c_pm - pm_["sigma"] * EULER_GAMMA
    ypred_pm_norm = mu_c_pm

    plugin_asse_fail_sev  = np.abs(Y_OBS[fail_idx] - ypred_pm_sev[fail_idx]).sum()
    plugin_asse_fail_norm = np.abs(Y_OBS[fail_idx] - ypred_pm_norm[fail_idx]).sum()
    plugin_asse_all_sev   = np.abs(Y_OBS - ypred_pm_sev).sum()

    print(f"\n  [C] Plugin ASSE（後驗均值參數代入，73 failures）:")
    print(f"      SEV  correction: {plugin_asse_fail_sev:.4f}")
    print(f"      No correction  : {plugin_asse_fail_norm:.4f}")
    print(f"  [D] Plugin ASSE（75 obs）: {plugin_asse_all_sev:.4f}")

    # ── 對照表 ────────────────────────────────────────────────────────
    print(f"\n  {'─'*58}")
    print(f"  COMPARISON (y-ASSE on failures, log(kilocycle) scale)")
    print(f"  {'─'*58}")
    print(f"  {'Method':<35} {'ASSE':>8}")
    print(f"  {'─'*44}")
    print(f"  {'Chiu (2005) EIV thesis':<35} {'10.80':>8}  ← z-ASSE")
    print(f"  {'Method C-2 LAD (Roy)':<35} {'9.94':>8}  ← z-ASSE")
    print(f"  {'SEV+INLA z-score':<35} {'11.50':>8}  ← z-ASSE")
    print(f"  {'─'*44}")
    print(f"  {'Bayes DA+NCP (posterior mean)':<35} {asse_fail.mean():>8.4f}  ← y-ASSE")
    print(f"  {'Bayes DA+NCP (plugin, SEV)':<35} {plugin_asse_fail_sev:>8.4f}  ← y-ASSE")
    print(f"  {'Bayes DA+NCP 95% CI':<35} "
          f"[{np.percentile(asse_fail,2.5):.2f}, {np.percentile(asse_fail,97.5):.2f}]")
    print(f"\n  注意：z-ASSE 與 y-ASSE 定義不同，數值不可直接比較。")
    print(f"  z-ASSE 在 z-score 空間計算；y-ASSE 在 ln(壽命) 空間計算。")

    # ── Per-specimen Δ 後驗均值 ───────────────────────────────────────
    # delta_mean below is E[exp(mu_d+sigma_d*z_delta)] (properly averaged
    # per posterior draw) -- NOT the same quantity as delta_pm above, see
    # the note there.
    print(f"\n  {'─'*58}")
    print(f"  PER-SPECIMEN Δ POSTERIOR (mean ± sd)")
    print(f"  {'─'*58}")
    delta_post = np.exp(
        idata.posterior["mu_d"].values[:,:,None] +
        idata.posterior["sigma_d"].values[:,:,None] *
        idata.posterior["z_delta"].values
    )   # (n_c, n_d, N_OBS)
    delta_mean = delta_post.mean(axis=(0,1))
    delta_std  = delta_post.std(axis=(0,1))

    print(f"  {'S':>7}  {'mean Δ':>9}  {'sd Δ':>8}  {'Δ/S':>6}  {'resid (y-ŷ)':>12}")
    print(f"  {'-'*52}")
    for s_uniq in sorted(set(S_OBS)):
        idx_s = np.where(S_OBS == s_uniq)[0]
        for i in idx_s[:3]:            # 每組只印前3個
            ev_str = "F" if EVENT[i] == 1 else "C"
            resid  = Y_OBS[i] - (pm_["beta0"] + pm_["beta1"] *
                      np.log(max(s_uniq - delta_mean[i], 1e-8))
                      - pm_["sigma"] * EULER_GAMMA)
            print(f"  {s_uniq:>7.3f}  {delta_mean[i]:>9.4f}  {delta_std[i]:>8.4f}  "
                  f"{delta_mean[i]/s_uniq:>6.3f}  {resid:>12.4f}  [{ev_str}]")
        if len(idx_s) > 3:
            print(f"           ... ({len(idx_s)-3} more in this group)")

    print("\nDone.")
