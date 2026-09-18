# -*- coding: utf-8 -*-
# run_sr_analysis.py — Stage 1: 固定 α 下的详细 SR 分析 + 路径特征分布分析
#
# 数据来源：
#   - 前一轮固定 α=0.242 结果：target_sim_alpha_correction_v1/02_fixed_alpha/results/FIXED_ALPHA_RESULTS.csv
#   - 前一轮 α 敏感性结果：target_sim_alpha_correction_v1/03_alpha_sensitivity/results/ALPHA_SENSITIVITY_RESULTS_LONG.csv
#   - 本批缓存：00_data/cache/（df_ep / df_re / re_mean / TARGETS，与原始代码聚合逻辑逐位一致）
#
# 本脚本只做统计计算与绘图，不重新跑仿真、不训练模型。

import json
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from itertools import combinations
from scipy import stats

warnings.filterwarnings("ignore")

PIPELINE_ROOT = os.environ.get('PIPELINE_ROOT', '')
if not PIPELINE_ROOT:
    raise EnvironmentError(
        "PIPELINE_ROOT environment variable must be set to the data workspace root. "
        "See README.md (Data Availability section) for data download instructions."
    )

BASE = os.path.join(PIPELINE_ROOT, "target_sim_sr_focused_v1")
PREV = os.path.join(PIPELINE_ROOT, "target_sim_alpha_correction_v1")
CACHE = f"{BASE}/00_data/cache"
RES = f"{BASE}/01_sr_analysis/results"
FIG = f"{BASE}/01_sr_analysis/figures"
os.makedirs(RES, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

RNG = np.random.default_rng(20260827)

FEAT_COLS = ["nav_eff", "rout_eff", "search_info", "communicability"]
CLASSES = ["personalized", "conventional", "control"]
CLASS_COLORS = {"personalized": "#2a78d6", "conventional": "#eb6834", "control": "#1baf7a"}

# ---------- 图表主题（dataviz skill: light surface / 次级墨色） ----------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE_LINE = "#c3c2b7"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": BASE_LINE, "grid.color": GRID,
    "font.family": "DejaVu Sans", "font.size": 10,
})

# =============================================================================
# 0. 缓存数据 + 前一轮结果
# =============================================================================
df_ep = pd.read_pickle(f"{CACHE}/df_ep.pkl")
df_re = pd.read_pickle(f"{CACHE}/df_re.pkl")
re_mean = np.load(f"{CACHE}/re_mean.npy")
with open(f"{CACHE}/TARGETS.json") as f:
    TARGETS = json.load(f)

regions = [r for k in CLASSES for r in TARGETS[k]]
assert len(regions) == 11, f"expected 11 targets, got {len(regions)}"
class_of = {r: k for k in CLASSES for r in TARGETS[k]}
assert len(class_of) == 11, "target classes must be disjoint"

prev_fixed = pd.read_csv(f"{PREV}/02_fixed_alpha/results/FIXED_ALPHA_RESULTS.csv")
prev_long = pd.read_csv(f"{PREV}/03_alpha_sensitivity/results/ALPHA_SENSITIVITY_RESULTS_LONG.csv")
FIXED_ALPHA = 0.242


def cos_sim(a, b):
    """与原始代码逐行一致。"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)) if (np.linalg.norm(a) * np.linalg.norm(b)) != 0 else 0.95


# 逐位校验：用本批缓存重算固定 α 下的 SR，与前一轮 sr_full 对比
recomputed = {}
for r in regions:
    m_r = df_ep.loc[df_ep["stim_region"] == r, FEAT_COLS].mean().values
    sim_mean = (1 - FIXED_ALPHA) * m_r + FIXED_ALPHA * re_mean
    recomputed[r] = float(cos_sim(sim_mean, re_mean))

prev_map = {row.region: row.sr_full for row in prev_fixed.itertuples()}
sr_diff = {r: abs(recomputed[r] - prev_map[r]) for r in regions}
max_diff = max(sr_diff.values())
print(f"[validation] recomputed SR vs prev sr_full: max abs diff = {max_diff:.3e}")
assert max_diff < 1e-8, "cache recomputation does not reproduce previous round — abort"

sr_full = {r: prev_map[r] for r in regions}
sr_rounded = {r: round(prev_map[r], 3) for r in regions}

# =============================================================================
# 1. 每靶点 SR 表 + 三类汇总
# =============================================================================
rows = []
for r in regions:
    rows.append({
        "target_type": class_of[r], "region": r,
        "sr_full": sr_full[r], "SR": sr_rounded[r],
    })
df_sr = pd.DataFrame(rows)
df_sr.to_csv(f"{RES}/SR_DETAILED_BY_TARGET.csv", index=False)

summary_rows = []
for k in CLASSES:
    vals = np.array([sr_full[r] for r in TARGETS[k]])
    n = len(vals)
    m = float(vals.mean())
    sd = float(vals.std(ddof=1)) if n > 1 else float("nan")
    se = sd / np.sqrt(n) if n > 1 else float("nan")
    tcrit = stats.t.ppf(0.975, n - 1) if n > 1 else float("nan")
    ci_low = m - tcrit * se if n > 1 else float("nan")
    ci_high = m + tcrit * se if n > 1 else float("nan")
    # bootstrap 均值 CI（小样本下更稳健）
    boot = np.array([
        vals[RNG.integers(0, n, size=n)].mean() for _ in range(10000)
    ])
    summary_rows.append({
        "target_type": k, "n": n, "mean_sr": m, "sd": sd, "se": se,
        "ci95_t_low": ci_low, "ci95_t_high": ci_high,
        "ci95_boot_low": float(np.percentile(boot, 2.5)),
        "ci95_boot_high": float(np.percentile(boot, 97.5)),
    })
df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(f"{RES}/SR_DETAILED_BY_TARGET.csv", mode="a", index=False,
                  header=list(df_summary.columns))

# =============================================================================
# 2. 精确置换框架（11 个靶点 → 2/2/7 的全部 1980 种标签分配）
# =============================================================================
def all_labelings(regions_, n_pers=2, n_conv=2):
    """枚举把 11 个靶点分入 2/2/7 三类的全部标签分配（共 1980 种）。"""
    for pers_idx in combinations(range(len(regions_)), n_pers):
        rest = [i for i in range(len(regions_)) if i not in pers_idx]
        for conv_idx in combinations(rest, n_conv):
            ctrl_idx = [i for i in rest if i not in conv_idx]
            yield set(pers_idx), set(conv_idx), set(ctrl_idx)


N_PERM = sum(1 for _ in all_labelings(regions))  # 1980

def perm_stat_region(stat_fn, n_iter=None):
    """对区域级统计量做精确置换 p 值（全部 1980 种标签分配）。

    stat_fn(pers_idx, conv_idx, ctrl_idx) -> float
    返回 (p_two_sided, n_perm, n_extreme, stat_obs_perm_dist 摘要)
    """
    obs = stat_fn(set(range(2)), set(range(2, 4)), set(range(4, 11)))  # 观测标签即 TARGETS 顺序
    extreme = 0
    total = 0
    for p_idx, c_idx, t_idx in all_labelings(regions):
        total += 1
        if n_iter is not None and total > n_iter:
            break
        s = stat_fn(p_idx, c_idx, t_idx)
        if s >= obs:  # F 型统计量越大越极端
            extreme += 1
    p = extreme / total
    return {"stat_obs": float(obs), "p_perm_exact": p, "n_perm": total, "n_extreme": extreme}

def perm_stat_contrast(stat_fn, total_perm=1980):
    """对比统计量（均值差绝对值）的精确置换 p 值。"""
    obs = stat_fn(set(range(2)), set(range(2, 4)), set(range(4, 11)))
    extreme = 0
    total = 0
    for p_idx, c_idx, t_idx in all_labelings(regions):
        total += 1
        if total > total_perm:
            break
        s = stat_fn(p_idx, c_idx, t_idx)
        if abs(s) >= abs(obs):
            extreme += 1
    return {"stat_obs": float(obs), "p_perm_exact": extreme / total, "n_perm": total, "n_extreme": extreme}

# =============================================================================
# 3. SR 统计检验（ANOVA / Welch / MWU / 效应量 / CI / 精确置换）
# =============================================================================
sr_stats = {}

# 3a. ANOVA（区域级，n=11；参数 + 精确置换）
vals = np.array([sr_full[r] for r in regions])
groups = [np.array([sr_full[r] for r in TARGETS[k]]) for k in CLASSES]
f_val, p_val = stats.f_oneway(*groups)
grand = vals.mean()
ssb = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
ssw = sum(((g - g.mean()) ** 2).sum() for g in groups)

def anova_F(p_idx, c_idx, t_idx):
    g1 = vals[list(p_idx)]; g2 = vals[list(c_idx)]; g3 = vals[list(t_idx)]
    gs = [g1, g2, g3]
    m = vals.mean()
    ssb_ = sum(len(g) * (g.mean() - m) ** 2 for g in gs)
    ssw_ = sum(((g - g.mean()) ** 2).sum() for g in gs)
    return (ssb_ / 2) / (ssw_ / 8)

sr_stats["ANOVA"] = {
    "n_total": 11, "n_per_group": [2, 2, 7],
    "F": float(f_val), "p_parametric": float(p_val),
    "SS_between": float(ssb), "SS_within": float(ssw),
    "eta_sq": float(ssb / (ssb + ssw)),
    **perm_stat_region(anova_F),
}

# 3b. 事后检验（Welch t + MWU + Cohen's d + rank-biserial + bootstrap CI + 精确置换）
def contrast_block(ka, kb):
    x = np.array([sr_full[r] for r in TARGETS[ka]])
    y = np.array([sr_full[r] for r in TARGETS[kb]])
    t, p_welch = stats.ttest_ind(x, y, equal_var=False)
    u, p_mwu = stats.mannwhitneyu(x, y, alternative="two-sided", method="exact")
    n1, n2 = len(x), len(y)
    sp = np.sqrt(((n1 - 1) * x.var(ddof=1) + (n2 - 1) * y.var(ddof=1)) / (n1 + n2 - 2))
    d = (x.mean() - y.mean()) / sp if sp > 0 else float("nan")
    # rank-biserial: U1 以 group1 的秩和表示
    ranks = stats.rankdata(np.concatenate([x, y]))
    u1 = n1 * n2 + n1 * (n1 + 1) / 2 - ranks[:n1].sum()
    rb = 2 * u1 / (n1 * n2) - 1
    # bootstrap CI of mean diff（区域级，10000 次）
    boots = np.array([
        x[RNG.integers(0, n1, size=n1)].mean() - y[RNG.integers(0, n2, size=n2)].mean()
        for _ in range(10000)
    ])
    # 精确置换（1980 标签分配下 |mean(pers)-mean(conv/ctrl)|）
    def diff_fn(p_idx, c_idx, t_idx):
        a = vals[list(p_idx)]
        b = vals[list(c_idx)] if kb == "conventional" else vals[list(t_idx)]
        return a.mean() - b.mean()
    perm = perm_stat_contrast(diff_fn)
    return {
        "n1": n1, "n2": n2,
        "mean1": float(x.mean()), "mean2": float(y.mean()),
        "mean_diff": float(x.mean() - y.mean()),
        "welch_t": float(t), "welch_p": float(p_welch),
        "mwu_U": float(u), "mwu_p_exact": float(p_mwu),
        "rank_biserial_r": float(rb),
        "cohens_d": float(d),
        "ci95_boot_low": float(np.percentile(boots, 2.5)),
        "ci95_boot_high": float(np.percentile(boots, 97.5)),
        **perm,
    }

sr_stats["posthoc_pers_vs_conventional"] = contrast_block("personalized", "conventional")
sr_stats["posthoc_pers_vs_control"] = contrast_block("personalized", "control")
sr_stats["posthoc_conventional_vs_control"] = contrast_block("conventional", "control")
sr_stats["note"] = (
    "SR 为固定 α=0.242 下仿真后区域均值特征向量与 re_mean 的余弦相似度；"
    "统计单元为 11 个靶点区域（2/2/7），n 小、功效低；"
    "精确置换基于全部 1980 种 2/2/7 标签分配（非随机抽样）。"
    "CR 未检验：固定 α 下 CR≡α 恒等（前一轮已证，设计局限）。SI 全为 0，无方差。"
)
with open(f"{RES}/SR_STATISTICAL_TESTS.json", "w") as f:
    json.dump(sr_stats, f, indent=2, ensure_ascii=False, default=float)

# 3c. α 敏感性复用（SR 排序 pers > ctrl > conv 是否在所有 α 水平稳定）
alpha_reuse = []
ranking_ok = {}
for a, grp in prev_long.groupby("alpha"):
    row = {"alpha": float(a)}
    for k in CLASSES:
        row[f"mean_sr_{k}"] = float(grp.loc[grp["target_type"] == k, "sr_full"].mean())
    order = np.argsort([row[f"mean_sr_{k}"] for k in CLASSES])[::-1]
    ranking = [CLASSES[i] for i in order]
    row["ranking"] = " > ".join(ranking)
    row["ranking_matches_pers_ctrl_conv"] = ranking == ["personalized", "control", "conventional"]
    alpha_reuse.append(row)
    ranking_ok[str(a)] = row["ranking_matches_pers_ctrl_conv"]
df_alpha = pd.DataFrame(alpha_reuse)
df_alpha.to_csv(f"{RES}/SR_ALPHA_SENSITIVITY_REUSE.csv", index=False)
sr_stats["alpha_sensitivity_ranking"] = {
    "ranking_stable_all_levels": bool(all(ranking_ok.values())),
    "by_alpha": ranking_ok,
}

# =============================================================================
# 4. 原始路径特征分布分析（不依赖 α 的纯位置信号）
# =============================================================================
feat_stats_rows = []
region_feat_mean = {}   # region -> 4-vector (path-level mean)
for r in regions:
    dfr = df_ep[df_ep["stim_region"] == r]
    v = dfr[FEAT_COLS].values
    region_feat_mean[r] = v.mean(axis=0)
    row = {"target_type": class_of[r], "region": r, "n_paths": int(len(dfr))}
    for j, c in enumerate(FEAT_COLS):
        col = v[:, j]
        row[f"{c}_mean"] = float(col.mean())
        row[f"{c}_median"] = float(np.median(col))
        row[f"{c}_sd"] = float(col.std(ddof=1))
        row[f"{c}_se"] = float(col.std(ddof=1) / np.sqrt(len(col)))
    feat_stats_rows.append(row)
df_feat = pd.DataFrame(feat_stats_rows)
df_feat.to_csv(f"{RES}/PATH_FEATURE_DISTRIBUTION_BY_TARGET.csv", index=False)

# 4a. 区域级 MANOVA（4 特征 × 11 区域；Wilks Λ + Pillai + Hotelling；精确置换）
X = np.array([region_feat_mean[r] for r in regions])  # 11×4
p_dim = X.shape[1]
g_count = 3
n_total = X.shape[0]

def manova_stats(p_idx, c_idx, t_idx):
    groups_ = [X[list(p_idx)], X[list(c_idx)], X[list(t_idx)]]
    grand_ = X.mean(axis=0)
    B = np.zeros((p_dim, p_dim)); W = np.zeros((p_dim, p_dim))
    for g in groups_:
        dg = g - g.mean(axis=0)
        B += len(g) * np.outer(g.mean(axis=0) - grand_, g.mean(axis=0) - grand_)
        W += dg.T @ dg
    T = B + W
    lam = np.linalg.det(W) / np.linalg.det(T) if np.linalg.det(T) != 0 else np.nan
    pillai = np.trace(np.linalg.pinv(T) @ B)
    hl = np.trace(np.linalg.pinv(W) @ B)
    return lam, pillai, hl

lam_obs, pillai_obs, hl_obs = manova_stats(set(range(2)), set(range(2, 4)), set(range(4, 11)))
# Rao F 近似（p=4, g=3, n=11 → s=2, df1=8, df2=11）
s_ = np.sqrt((p_dim**2 * (g_count - 1)**2 - 4) / (p_dim**2 + (g_count - 1)**2 - 5))
lam_s = lam_obs ** (1 / s_)
df1 = p_dim * (g_count - 1)
df2 = s_ * (n_total - (p_dim + g_count + 1) / 2) - (p_dim * (g_count - 1) - 2) / 2
f_rao = ((1 - lam_s) / lam_s) * (df2 / df1)
p_rao = float(stats.f.sf(f_rao, df1, df2))

perm_extreme = 0
perm_total = 0
lam_perm_list = []
for p_idx, c_idx, t_idx in all_labelings(regions):
    perm_total += 1
    lam_p, _, _ = manova_stats(p_idx, c_idx, t_idx)
    lam_perm_list.append(lam_p)
    if lam_p <= lam_obs:  # Wilks Λ 越小越极端
        perm_extreme += 1
lam_perm_list = np.array(lam_perm_list)

manova_block = {
    "unit": "region-level mean feature vector (n=11 targets: 2/2/7)",
    "Wilks_lambda": float(lam_obs),
    "Pillai_trace": float(pillai_obs),
    "Hotelling_Lawley_trace": float(hl_obs),
    "rao_F_approx": {
        "F": float(f_rao), "df1": float(df1), "df2": float(df2), "p_parametric": p_rao,
        "note": "Rao 近似；n=11 时近似仅供参考",
    },
    "p_perm_exact": perm_extreme / perm_total,
    "n_perm": perm_total, "n_extreme": perm_extreme,
    "perm_lambda_min_max": [float(lam_perm_list.min()), float(lam_perm_list.max())],
}

# 4b. 单变量 ANOVA（区域级，每特征；参数 + 精确置换 + Bonferroni）
feat_anova = {}
for j, c in enumerate(FEAT_COLS):
    vals_f = X[:, j]
    groups_f = [vals_f[[0, 1]], vals_f[[2, 3]], vals_f[4:]]
    F_f, p_f = stats.f_oneway(*groups_f)
    grand_f = vals_f.mean()
    ssb_f = sum(len(g) * (g.mean() - grand_f) ** 2 for g in groups_f)
    ssw_f = sum(((g - g.mean()) ** 2).sum() for g in groups_f)

    def anova_f_fn(p_idx, c_idx, t_idx):
        gs = [vals_f[list(p_idx)], vals_f[list(c_idx)], vals_f[list(t_idx)]]
        m = vals_f.mean()
        ssb_ = sum(len(g) * (g.mean() - m) ** 2 for g in gs)
        ssw_ = sum(((g - g.mean()) ** 2).sum() for g in gs)
        return (ssb_ / 2) / (ssw_ / 8)

    perm_f = perm_stat_region(anova_f_fn)
    # 路径级 ANOVA（描述性；路径嵌套于区域，p 仅供参考）
    path_groups = [df_ep.loc[df_ep["stim_region"].isin(TARGETS[k]), c].values for k in CLASSES]
    F_path, p_path = stats.f_oneway(*path_groups)
    grand_p = df_ep.loc[df_ep["stim_region"].isin(regions), c].mean()
    ssb_p = sum(len(g) * (g.mean() - grand_p) ** 2 for g in path_groups)
    ssw_p = sum(((g - g.mean()) ** 2).sum() for g in path_groups)

    feat_anova[c] = {
        "region_level": {
            "F": float(F_f), "p_parametric": float(p_f),
            "SS_between": float(ssb_f), "SS_within": float(ssw_f),
            "partial_eta_sq": float(ssb_f / (ssb_f + ssw_f)),
            **perm_f,
        },
        "path_level_descriptive": {
            "F": float(F_path), "p": float(p_path),
            "n_paths_per_group": [int(len(g)) for g in path_groups],
            "partial_eta_sq": float(ssb_p / (ssb_p + ssw_p)),
            "caveat": "路径嵌套于区域（11 个区域），路径级 p 为描述性，非推断单元",
        },
    }

# 4c. 事后检验（区域级，每特征：pers vs conv、pers vs ctrl；Bonferroni 8 次）
posthoc_feat = {}
for c in FEAT_COLS:
    x_c = X[[0, 1], FEAT_COLS.index(c)]
    posthoc_feat[c] = {}
    for kb, idxs in [("conventional", [2, 3]), ("control", list(range(4, 11)))]:
        y_c = X[idxs, FEAT_COLS.index(c)]
        t_c, p_c = stats.ttest_ind(x_c, y_c, equal_var=False)
        u_c, p_u_c = stats.mannwhitneyu(x_c, y_c, alternative="two-sided", method="exact")
        n1, n2 = len(x_c), len(y_c)
        sp_c = np.sqrt(((n1 - 1) * x_c.var(ddof=1) + (n2 - 1) * y_c.var(ddof=1)) / (n1 + n2 - 2))
        d_c = (x_c.mean() - y_c.mean()) / sp_c if sp_c > 0 else float("nan")
        boots_c = np.array([
            x_c[RNG.integers(0, n1, size=n1)].mean() - y_c[RNG.integers(0, n2, size=n2)].mean()
            for _ in range(10000)
        ])
        def diff_c_fn(p_idx, c_idx, t_idx, j=FEAT_COLS.index(c)):
            a = X[list(p_idx), j]
            b = X[list(c_idx), j] if kb == "conventional" else X[list(t_idx), j]
            return a.mean() - b.mean()
        perm_c = perm_stat_contrast(diff_c_fn)
        posthoc_feat[c][f"pers_vs_{kb}"] = {
            "mean_diff": float(x_c.mean() - y_c.mean()),
            "welch_t": float(t_c), "welch_p": float(p_c),
            "mwu_U": float(u_c), "mwu_p_exact": float(p_u_c),
            "cohens_d": float(d_c),
            "ci95_boot_low": float(np.percentile(boots_c, 2.5)),
            "ci95_boot_high": float(np.percentile(boots_c, 97.5)),
            **perm_c,
            "p_bonferroni_8": float(min(1.0, p_c * 8)),
            "p_mwu_bonferroni_8": float(min(1.0, p_u_c * 8)),
        }

# 4d. PERMANOVA（区域级，Euclidean 距离的精确置换）
def permanova_F(p_idx, c_idx, t_idx):
    gs = [X[list(p_idx)], X[list(c_idx)], X[list(t_idx)]]
    grand_ = X.mean(axis=0)
    ssb_ = sum(len(g) * ((g.mean(axis=0) - grand_) ** 2).sum() for g in gs)
    ssw_ = sum(((g - g.mean(axis=0)) ** 2).sum() for g in gs)
    return (ssb_ / (g_count - 1)) / (ssw_ / (n_total - g_count))

permanova_obs = permanova_F(set(range(2)), set(range(2, 4)), set(range(4, 11)))
perm_ext = 0
perm_tot = 0
for p_idx, c_idx, t_idx in all_labelings(regions):
    perm_tot += 1
    if permanova_F(p_idx, c_idx, t_idx) >= permanova_obs:
        perm_ext += 1

feat_stats = {
    "MANOVA": manova_block,
    "ANOVA_by_feature": feat_anova,
    "posthoc_by_feature": posthoc_feat,
    "PERMANOVA_euclidean_region_level": {
        "pseudo_F": float(permanova_obs),
        "p_perm_exact": perm_ext / perm_tot,
        "n_perm": perm_tot, "n_extreme": perm_ext,
        "note": "区域级 4 特征均值向量的欧氏距离 PERMANOVA（非参数替代）；p 为全部 1980 种标签分配的精确置换 p",
    },
    "note": (
        "统计单元 = 11 个靶点区域（2/2/7），样本量小、功效低；"
        "MANOVA/ANOVA/PERMANOVA 均为区域级（路径嵌套于区域）。"
        "路径级 ANOVA 作为描述性补充。Bonferroni 校正按 4 特征 × 2 对比 = 8 次。"
    ),
}
with open(f"{RES}/PATH_FEATURE_STATISTICAL_TESTS.json", "w") as f:
    json.dump(feat_stats, f, indent=2, ensure_ascii=False, default=float)

# =============================================================================
# 5. 特征方向与 SR 的关联（SR 本质 = 区域均值与 re_mean 夹角决定的余弦）
# =============================================================================
link_rows = []
for r in regions:
    m_r = region_feat_mean[r]
    cos_m_re = float(cos_sim(m_r, re_mean))
    sim_mean = (1 - FIXED_ALPHA) * m_r + FIXED_ALPHA * re_mean
    sr_check = float(cos_sim(sim_mean, re_mean))
    link_rows.append({
        "target_type": class_of[r], "region": r,
        "cos(region_mean, re_mean)": cos_m_re,
        "SR_fixed_alpha": sr_full[r],
        "SR_recomputed_identity": sr_check,
        "SR_identity_diff": sr_check - sr_full[r],
    })
df_link = pd.DataFrame(link_rows)
df_link.to_csv(f"{RES}/FEATURE_SR_COSINE_LINK.csv", index=False)

cos_vec = df_link["cos(region_mean, re_mean)"].values
sr_vec = df_link["SR_fixed_alpha"].values
rho, p_rho = stats.spearmanr(cos_vec, sr_vec)
r_pear, p_pear = stats.pearsonr(cos_vec, sr_vec)
tau, p_tau = stats.kendalltau(cos_vec, sr_vec)
rank_same = np.allclose(stats.rankdata(cos_vec), stats.rankdata(sr_vec))
# 特征水平相关性（z 分数后）
feat_corrs = {}
for c in FEAT_COLS:
    zc = (df_feat[f"{c}_mean"].values - df_feat[f"{c}_mean"].mean()) / df_feat[f"{c}_mean"].std(ddof=1)
    rr, pr = stats.spearmanr(zc, sr_vec)
    feat_corrs[c] = {"spearman_rho": float(rr), "p": float(pr)}

link_json = {
    "cosine_vs_SR": {
        "spearman_rho": float(rho), "spearman_p": float(p_rho),
        "kendall_tau": float(tau), "kendall_p": float(p_tau),
        "pearson_r": float(r_pear), "pearson_p": float(p_pear),
        "rank_order_identical": bool(rank_same),
    },
    "SR_identity_check": {
        "formula": "SR = cos((1-α)·region_mean + α·re_mean, re_mean)",
        "max_abs_diff_vs_previous_round": float(max(abs(df_link["SR_identity_diff"]))),
    },
    "feature_zscore_corr_with_SR": feat_corrs,
    "interpretation": (
        "SR 差异的来源 = 各区域原始路径特征均值向量与 re_mean 的方向夹角差异；"
        "cos(region_mean, re_mean) 与 SR 的排序关系见 spearman_rho；rho=1 表示排序完全一致。"
    ),
}
with open(f"{RES}/FEATURE_SR_CORRELATION.json", "w") as f:
    json.dump(link_json, f, indent=2, ensure_ascii=False, default=float)

# =============================================================================
# 6. 图 1：每靶点 SR 柱状图（按类别着色）
# =============================================================================
order_sr = [r for k in CLASSES for r in sorted(TARGETS[k], key=lambda rr: -sr_full[rr])]
fig, ax = plt.subplots(figsize=(8.2, 4.4))
xs = np.arange(11)
colors = [CLASS_COLORS[class_of[r]] for r in order_sr]
bars = ax.bar(xs, [sr_full[r] for r in order_sr], width=0.62, color=colors,
              edgecolor=INK, linewidth=0.6)
ax.set_ylim(0.955, 1.004)
for x, r in zip(xs, order_sr):
    ax.text(x, sr_full[r] + 0.0012, f"{sr_full[r]:.3f}", ha="center", va="bottom",
            fontsize=8, color=INK2)
ax.set_xticks(xs)
ax.set_xticklabels([r.replace("_ROI", "") for r in order_sr], rotation=45, ha="right", fontsize=8.5)
ax.set_ylabel("SR (cosine similarity to reference state)", fontsize=10)
ax.set_title("Fixed-α SR by target region (α = 0.242)", fontsize=11, color=INK, loc="left")
ax.grid(axis="y", alpha=0.35, linewidth=0.6)
ax.set_axisbelow(True)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
legend_handles = [Line2D([0], [0], color=CLASS_COLORS[k], lw=8, label=k) for k in CLASSES]
ax.legend(handles=legend_handles, frameon=False, loc="lower right", fontsize=9)
fig.tight_layout()
fig.savefig(f"{FIG}/sr_by_target_barplot.png", dpi=300)
plt.close(fig)

# =============================================================================
# 7. 图 2：4 特征 × 11 靶点热图（特征内 z 分数）
# =============================================================================
Z = np.zeros((4, 11))
for j, c in enumerate(FEAT_COLS):
    col_vals = df_feat[f"{c}_mean"].values
    Z[j] = (col_vals - col_vals.mean()) / col_vals.std(ddof=1)
# 按 TARGETS 原始顺序（pers, conv, ctrl）排列列
region_order_heat = [r for k in CLASSES for r in TARGETS[k]]
Z = Z[:, [regions.index(r) for r in region_order_heat]]

cmap = LinearSegmentedColormap.from_list("bluegrayred", ["#2a78d6", "#f0efec", "#e34948"], N=256)
fig, ax = plt.subplots(figsize=(9.5, 3.2))
im = ax.imshow(Z, cmap=cmap, aspect="auto", vmin=-2.5, vmax=2.5)
ax.set_xticks(np.arange(11))
ax.set_xticklabels([r.replace("_ROI", "") for r in region_order_heat], rotation=45, ha="right", fontsize=8.5)
ax.set_yticks(np.arange(4))
ax.set_yticklabels(FEAT_COLS, fontsize=9)
for i in range(4):
    for jj in range(11):
        ax.text(jj, i, f"{Z[i, jj]:.1f}", ha="center", va="center", fontsize=7.5,
                color=INK)
# 类别分隔线
ax.axvline(1.5, color=INK, linewidth=0.8, linestyle=(0, (2, 2)), alpha=0.7)
ax.axvline(3.5, color=INK, linewidth=0.8, linestyle=(0, (2, 2)), alpha=0.7)
ax.text(0.5, 3.85, "personalized", fontsize=8, color=CLASS_COLORS["personalized"], ha="center")
ax.text(2.5, 3.85, "conventional", fontsize=8, color=CLASS_COLORS["conventional"], ha="center")
ax.text(7.0, 3.85, "control", fontsize=8, color=CLASS_COLORS["control"], ha="center")
ax.set_ylim(3.5, -0.6)
cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
cb.set_label("z-score within feature", fontsize=8)
cb.ax.tick_params(labelsize=7)
ax.set_title("Path feature means by target (z-scored per feature, pathology state)",
             fontsize=10, color=INK, loc="left")
fig.tight_layout()
fig.savefig(f"{FIG}/path_feature_heatmap.png", dpi=300)
plt.close(fig)

# =============================================================================
# 8. 图 3：4 特征 × 3 类箱线图（路径级分布）
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.2))
for j, (c, ax) in enumerate(zip(FEAT_COLS, axes.ravel())):
    data_by_class = [
        df_ep.loc[df_ep["stim_region"].isin(TARGETS[k]), c].values for k in CLASSES
    ]
    bp = ax.boxplot(data_by_class, widths=0.55, patch_artist=True, showfliers=False,
                    medianprops=dict(color=INK, linewidth=1.4))
    for patch, k in zip(bp["boxes"], CLASSES):
        patch.set_facecolor(CLASS_COLORS[k])
        patch.set_alpha(0.75)
        patch.set_edgecolor(INK)
        patch.set_linewidth(0.6)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(CLASSES, fontsize=8.5)
    ax.set_title(c, fontsize=10, color=INK, loc="left")
    ax.grid(axis="y", alpha=0.35, linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
fig.suptitle("Path-level feature distributions by target class (pathology state, "
             "whiskers = 1.5×IQR, outliers omitted)",
             fontsize=10.5, color=INK, x=0.02, ha="left")
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(f"{FIG}/feature_distribution_boxplot.png", dpi=300)
plt.close(fig)

# =============================================================================
# 9. 汇总说明
# =============================================================================
print("\n================ SR 详细统计 ================")
print(df_sr[["target_type", "region", "SR"]].to_string(index=False))
print("\n三类汇总：")
print(df_summary.round(6).to_string(index=False))
print("\nSR ANOVA: F={:.4f}, p={:.4f}, perm p={}".format(
    sr_stats["ANOVA"]["F"], sr_stats["ANOVA"]["p_parametric"],
    sr_stats["ANOVA"]["p_perm_exact"]))
print("pers vs conv: Welch p={:.4f}, perm p={}".format(
    sr_stats["posthoc_pers_vs_conventional"]["welch_p"],
    sr_stats["posthoc_pers_vs_conventional"]["p_perm_exact"]))
print("pers vs ctrl: Welch p={:.4f}, perm p={}".format(
    sr_stats["posthoc_pers_vs_control"]["welch_p"],
    sr_stats["posthoc_pers_vs_control"]["p_perm_exact"]))
print("α 敏感性排序稳定: {}".format(sr_stats["alpha_sensitivity_ranking"]["ranking_stable_all_levels"]))

print("\n================ 路径特征分布 ================")
print("MANOVA Wilks Λ={:.4f}, Pillai={:.4f}, perm p={}".format(
    manova_block["Wilks_lambda"], manova_block["Pillai_trace"], manova_block["p_perm_exact"]))
for c in FEAT_COLS:
    blk = feat_anova[c]["region_level"]
    print(f"{c}: region ANOVA F={blk['F']:.3f}, p={blk['p_parametric']:.4f}, "
          f"perm p={blk['p_perm_exact']}")
print("PERMANOVA pseudo-F={:.3f}, perm p={}".format(
    feat_stats["PERMANOVA_euclidean_region_level"]["pseudo_F"],
    feat_stats["PERMANOVA_euclidean_region_level"]["p_perm_exact"]))

print("\n================ 特征-SR 关联 ================")
print("cos(region_mean, re_mean) vs SR: Spearman rho={:.4f} (p={:.4g}), 排序一致={}".format(
    rho, p_rho, rank_same))

print("\n[run_sr_analysis] 全部输出已落盘")
print("  ", RES)
print("  ", FIG)
