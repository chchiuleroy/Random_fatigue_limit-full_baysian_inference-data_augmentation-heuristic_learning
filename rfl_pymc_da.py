#!/usr/bin/env python3
"""
rfl_pymc_da.py — RFL Data Augmentation + NUTS
==============================================
改變：不再數值積分掉 Δ，改為明確取樣每個 Δᵢ

  z_Δᵢ ~ N(0, 1)；log(Δᵢ) = μ_Δ + σ_Δ · z_Δᵢ   (NCP，未截斷 LogNormal)
  ↓ Δᵢ < Sᵢ 只靠 likelihood 裡的 pt.maximum(Sᵢ-Δᵢ, 1e-8) soft floor 近似生效，
    不是真正的 hard-support 截斷（見下方「已知簡化」）

優點：
  - 不需 as_op → 可以 pickle → 4 chains 真正平行
  - PyMC autodiff 計算梯度 → NUTS 取代 Metropolis
  - NCP 打破 (μ_Δ,σ_Δ)↔z_Δᵢ 的 Neal's Funnel
  - 總參數空間：5 + 75(Δᵢ) = 80 維，NUTS 擅長

模型：
  Normal error:  ln N_i | Δᵢ ~ N(β₀ + β₁ ln(Sᵢ-Δᵢ), σ²)
  SEV   error:   ln N_i | Δᵢ ~ SEV(β₀ + β₁ ln(Sᵢ-Δᵢ), σ)
  LogNormal g(Δ): Δᵢ ~ LN(μ_Δ, σ_Δ)，**未截斷**（支撐集為全部正實數）

已知簡化（2026-08-12 review 發現，Roy 決定不動——真正加 hard truncation
是額外建模工作，見 README.md §5.1/§5.5、[[concept_model_zscore_bayes_hl]]）：
  Δᵢ < Sᵢ 這個約束沒有反映在 prior 的支撐集或正規化常數上，只靠
  pt.maximum(S_OBS-delta, 1e-8) 在 likelihood 裡防止 log(負數)，這在
  Δᵢ=Sⱼ 處會產生不平滑轉折點（kink），且少了真正截斷該有的正規化常數
  對 μ_Δ/σ_Δ 後驗的影響。docstring 過去誤寫成 TruncatedNormal，已於
  2026-08-13 修正用詞以符合實際實作（不影響任何已發布數字，純屬描述精確度）。
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import arviz as az
import time

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

fail_idx = np.where(EVENT == 1)[0]   # 73 failures
cens_idx = np.where(EVENT == 0)[0]   # 2 censored
Y_fail   = Y_OBS[fail_idx]
Y_cens   = Y_OBS[cens_idx]
S_fail   = S_OBS[fail_idx]
S_cens   = S_OBS[cens_idx]
LOG_S    = np.log(S_OBS)             # upper bounds for log(Δ)


# ═══════════════════════════════════════════════════════════════════════
# 2. MODEL FACTORY
# ═══════════════════════════════════════════════════════════════════════

def build_da_model(error='normal', g_type='lognormal'):
    """
    Data Augmentation RFL model.
    error  : 'normal' | 'sev'
    g_type : 'lognormal' | 'gamma'

    LogNormal g(Δ) — TRUE NCP，未截斷：
      z_Δᵢ ~ Normal(0,1)  →  log(Δᵢ) = μ_Δ + σ_Δ·z_Δᵢ
    Gamma g(Δ) — 同樣未截斷（2026-08-13 修正此處過時 docstring，
      舊版誤寫成「透過 pm.Truncated(Gamma) 直接取樣」，實際程式碼
      下方 121 行本身就寫「不用 Truncated（初始化不穩定）」，兩者矛盾）：
      delta ~ pm.Gamma(alpha_d, beta_r)，Δᵢ<Sᵢ 約束跟 LogNormal 分支
      一樣只靠 likelihood 的 pt.maximum(S-Δ, 1e-8) soft floor 近似生效
    """
    if error not in ('normal', 'sev'):
        # 2026-08-13 fix (小o review 發現同一類問題已在 rfl_bayes_asse.py
        # 修過)：原本 `if error=='normal': ... else: # SEV` 對任何打錯字
        # 的值都會靜默落入 SEV 分支，不報錯。
        raise ValueError(f"error must be 'normal' or 'sev', got {error!r}")
    if g_type not in ('lognormal', 'gamma'):
        raise ValueError(f"g_type must be 'lognormal' or 'gamma', got {g_type!r}")
    upper_mu_d = float(np.log(MIN_S) - 0.05)

    with pm.Model() as model:

        # ── Global parameters ──────────────────────────────────────────
        b0      = pm.Uniform("beta0",     lower=-50., upper=50.)
        b1      = pm.Uniform("beta1",     lower=-30., upper=0.)
        log_sig = pm.Uniform("log_sigma", lower=np.log(0.01), upper=np.log(5.))
        sigma   = pm.Deterministic("sigma", pt.exp(log_sig))

        if g_type == 'lognormal':
            mu_d    = pm.Uniform("mu_d",       lower=-5., upper=upper_mu_d)
            log_sdd = pm.Uniform("log_sigma_d",lower=np.log(0.001), upper=np.log(2.))
            sigma_d = pm.Deterministic("sigma_d", pt.exp(log_sdd))
        else:  # gamma
            # 小 CV 的 Gamma 需要大 α 和 β_r（與 σ_Δ≈0.033 對應）
            # α ≈ 1/CV² ≈ 900, β_r ≈ α/mean ≈ 900/0.52 ≈ 1730
            log_alpha = pm.Uniform("log_alpha_d", lower=np.log(0.5),   upper=np.log(5000.))
            log_betar = pm.Uniform("log_beta_r",  lower=np.log(1.0),   upper=np.log(10000.))
            alpha_d   = pm.Deterministic("alpha_d", pt.exp(log_alpha))
            beta_r    = pm.Deterministic("beta_r",  pt.exp(log_betar))

        # ── Latent Δᵢ ─────────────────────────────────────────────────
        if g_type == 'lognormal':
            # TRUE NCP: z_Δᵢ ~ N(0,1),  log(Δᵢ) = μ_Δ + σ_Δ·z_Δᵢ
            # δᵢ<Sᵢ 約束透過 likelihood 執行（log(S-δ)→-∞ 時 loglik→-∞）
            z_delta   = pm.Normal("z_delta", mu=0., sigma=1., shape=N_OBS)
            log_delta = mu_d + sigma_d * z_delta
            delta     = pm.Deterministic("delta", pt.exp(log_delta))
        else:
            # Gamma: 不用 Truncated（初始化不穩定）
            # 改用 pm.Gamma + soft constraint（2026-08-13 修正措辭，
            # 小o review 指出這不是真的無限大懲罰）：
            # 當 delta≥S_i 時，maximum(S_i-delta,1e-8) 被地板到 1e-8，
            # log(1e-8)≈-18.4 是很大但有限的負值，b1<0 → mu_cond 變成
            # 很大的正值 → likelihood 受到強烈但有限的懲罰，NUTS 通常會
            # 遠離，但這不是嚴格意義上把 likelihood 推到 -∞ 的 hard
            # constraint
            delta = pm.Gamma("delta", alpha=alpha_d, beta=beta_r, shape=N_OBS)

        # ── Conditional mean ────────────────────────────────────────────
        # μ(Sᵢ, Δᵢ) = β₀ + β₁ ln(Sᵢ − Δᵢ)
        # clip(S-δ, 1e-8) → 施加強烈但有限的 likelihood 懲罰（非真正 -∞
        # 的 hard constraint，2026-08-13 修正措辭，見上方 Gamma 分支同一
        # 個機制的說明）
        mu_cond = b0 + b1 * pt.log(pt.maximum(S_OBS - delta, 1e-8))

        # ── Likelihood ──────────────────────────────────────────────────
        if error == 'normal':
            # Failures: ln N_i ~ Normal(μᵢ, σ)
            pm.Normal("obs_fail",
                      mu=mu_cond[fail_idx], sigma=sigma,
                      observed=Y_fail)

            # Censored: log P(ln N > y_c | μᵢ, σ)
            # = log(1 − Φ(z))  where  z = (y_c − μᵢ)/σ
            # Numerically stable: log(0.5 * erfc(z/√2))
            z_c     = (Y_cens - mu_cond[cens_idx]) / sigma
            log_s   = pt.log(0.5 * pt.erfc(z_c / pt.sqrt(2.)))
            pm.Potential("obs_cens", pt.sum(log_s))

        else:  # SEV: log f = -log σ + z − exp(z),  log S = −exp(z)
            # Failures
            z_f   = (Y_fail - mu_cond[fail_idx]) / sigma
            logp_f = pt.sum(-pt.log(sigma) + z_f - pt.exp(pt.clip(z_f, -500, 20)))
            pm.Potential("obs_fail", logp_f)

            # Censored
            z_c   = (Y_cens - mu_cond[cens_idx]) / sigma
            logp_c = pt.sum(-pt.exp(pt.clip(z_c, -500, 20)))
            pm.Potential("obs_cens", logp_c)

    return model


# ═══════════════════════════════════════════════════════════════════════
# 3. INIT VALUES
# ═══════════════════════════════════════════════════════════════════════

def make_init(error, g_type='lognormal'):
    """Starting values near known MLE."""
    if error == 'normal':
        b0_0, b1_0, lsig_0 = -9.2, -8.1, np.log(0.60)
    else:
        b0_0, b1_0, lsig_0 = -9.4, -8.5, np.log(0.19)

    if g_type == 'lognormal':
        return dict(
            beta0=b0_0, beta1=b1_0, log_sigma=lsig_0,
            mu_d=-0.65, log_sigma_d=np.log(0.04),
            z_delta=np.zeros(N_OBS),
        )
    else:
        # Gamma: CV≈0.033 → α≈1/0.033²≈920, mean≈0.52 → β_r≈920/0.52≈1770
        # 初始值設小一點讓 delta 在 S 範圍內
        # mean Δ = 400/800 = 0.5，std = √400/800 = 0.025
        delta_init = np.full(N_OBS, 0.40)   # safe below min S=0.675
        return dict(
            beta0=b0_0, beta1=b1_0, log_sigma=lsig_0,
            log_alpha_d=np.log(400.0), log_beta_r=np.log(800.0),
            delta=delta_init,
        )


# ═══════════════════════════════════════════════════════════════════════
# 4. MAIN
# ═══════════════════════════════════════════════════════════════════════

N_CHAINS      = 4
N_TUNE        = 2000
N_DRAWS       = 2000
N_CORES       = 4   # Data augmentation = pure pytensor → picklable → 真正平行
TARGET_ACCEPT = 0.95   # raised from 0.85 (2026-08-12): SEV+LogNormal shows
                        # persistent NUTS divergences (NCP/CP intermediate-
                        # regime issue, not a boundary-cliff or script bug --
                        # see README.md Section 5.5). 0.95 cuts divergences
                        # from 2214 to 932 (still nonzero; provisional).

if __name__ == "__main__":
    print(f"n={N_OBS}  failures={len(fail_idx)}  censored={len(cens_idx)}")
    print(f"Sampler: NUTS  |  chains={N_CHAINS}  tune={N_TUNE}  draws={N_DRAWS}\n")

    all_res = {}

    RUNS = [
        ("normal", "lognormal"),
        ("sev",    "lognormal"),
        ("sev",    "gamma"),
    ]

    for error, g_type in RUNS:
        name  = f"DA_{error}_{g_type}"
        init  = make_init(error, g_type)
        print(f"\n{'='*65}")
        print(f"  [{name}]")
        print(f"{'='*65}")

        model = build_da_model(error=error, g_type=g_type)
        t0    = time.time()

        with model:
            idata = pm.sample(
                draws         = N_DRAWS,
                tune          = N_TUNE,
                chains        = N_CHAINS,
                cores         = N_CORES,
                initvals      = init,
                target_accept = TARGET_ACCEPT,
                progressbar   = True,
                random_seed   = list(range(N_CHAINS)),
            )

        elapsed = time.time() - t0
        print(f"\n  Elapsed: {elapsed:.1f}s")

        # ── Global parameter summary ──────────────────────────────────
        if g_type == 'lognormal':
            vnames = ["beta0","beta1","sigma","mu_d","sigma_d"]
        else:
            vnames = ["beta0","beta1","sigma","alpha_d","beta_r"]
        summ = az.summary(idata, var_names=vnames, round_to=4)
        print("\n  Global parameters:")
        print(summ[["mean","sd","eti89_lb","eti89_ub","r_hat","ess_bulk"]].to_string())

        # ── Per-specimen Δ summary ─────────────────────────────────────
        delta_post = idata.posterior["delta"].values  # (n_c, n_d, N_OBS)
        delta_mean = delta_post.mean(axis=(0,1))      # (N_OBS,)
        print(f"\n  Δ posterior mean per stress level:")
        for s_uniq in sorted(set(S_OBS)):
            idx_s = np.where(S_OBS == s_uniq)[0]
            dm    = delta_mean[idx_s].mean()
            print(f"    S={s_uniq:.3f}: mean Δ = {dm:.4f}  (Δ/S = {dm/s_uniq:.3f})")

        # ── Rhat & ESS for global params ─────────────────────────────
        rhat_max = summ["r_hat"].max()
        ess_min  = summ["ess_bulk"].min()
        print(f"\n  Rhat max = {rhat_max:.4f}  |  ESS min = {ess_min:.0f}")

        all_res[name] = dict(summ=summ, rhat_max=rhat_max,
                             ess_min=ess_min, elapsed=elapsed,
                             g_type=g_type)

    # ── Comparison ──────────────────────────────────────────────────
    print(f"\n\n{'='*65}")
    print("  DATA AUGMENTATION FINAL COMPARISON")
    print(f"{'='*65}")
    print(f"  {'Model':<25} {'b0':>8} {'b1':>8} {'sigma':>7} {'p1':>10} {'p2':>10} {'Rhat':>7} {'ESS':>6}")
    print(f"  {'-'*85}")
    for name, res in all_res.items():
        pm_ = res["summ"]["mean"].values
        rh  = res["rhat_max"]; es = res["ess_min"]
        print(f"  {name:<25} {pm_[0]:>8.3f} {pm_[1]:>8.3f} {pm_[2]:>7.4f} "
              f"{pm_[3]:>10.4f} {pm_[4]:>10.4f} {rh:>7.4f} {es:>6.0f}")

    print("\nDone.")
