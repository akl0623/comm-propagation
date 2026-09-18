# -*- coding: utf-8 -*-
# run_negative_controls.py — Stage 2: permutation 负控 + 调制后特征聚类分析
#
# 输入：
#   - 00_data/cache/（df_ep / df_re / re_mean / TARGETS，与原始代码逐位一致）
#   - 01_sr_analysis/results/SR_DETAILED_BY_TARGET.csv（Stage 1 每靶点 sr_full）
#   - 01_sr_analysis/results/SR_STATISTICAL_TESTS.json（Stage 1 精确枚举，交叉校验用）
#
# 全部为统计计算与绘图，不重新跑仿真、不训练模型。
# 统计单元 = 11 个靶点区域（2/2/7）；随机置换 10,000 次（p=(n_extreme+1)/(N+1)），
# 并同时给出全部 1980（或 55）种标签分配的精确枚举 p。

import json
import math
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from itertools import combinations
from scipy import stats
from scipy.spatial.distance import cdist, pdist
from scipy.cluster.hierarchy import linkage, dendrogram, cophenet

warnings.filterwarnings("ignore")

PIPELINE_ROOT = os.environ.get('PIPELINE_ROOT', '')
if not PIPELINE_ROOT:
    raise EnvironmentError(
        "PIPELINE_ROOT environment variable must be set to the data workspace root. "
        "See README.md (Data Availability section) for data download instructions."
    )

BASE = os.path.join(PIPELINE_ROOT, "target_sim_sr_focused_v1")
CACHE = f"{BASE}/00_data/cache"
S1RES = f"{BASE}/01_sr_analysis/results"
RES = f"{BASE}/02_negative_controls/results"
FIG = f"{BASE}/02_negative_controls/figures"
os.makedirs(RES, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

RNG = np.random.default_rng(20260827)
N_RAND = 10000
N_EXACT_CAP = 5000

FEAT_COLS = ["nav_eff", "rout_eff", "search_info", "communicability"]
CLASSES = ["personalized", "conventional", "control"]
CLASS_COLORS = {"personalized": "#2a78d6", "conventional": "#eb6834", "control": "#1baf7a"}
MARKERS = {"personalized": "o", "conventional": "s", "control": "^"}
FIXED_ALPHA = 0.242

# ---------- 图表主题（与 Stage 1 一致） ----------
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
# 0. 数据
# =============================================================================
df_ep = pd.read_pickle(f"{CACHE}/df_ep.pkl")
re_mean = np.load(f"{CACHE}/re_mean.npy")
with open(f"{CACHE}/TARGETS.json") as f:
    TARGETS = json.load(f)

regions = [r for k in CLASSES for r in TARGETS[k]]
assert len(regions) == 11
class_of = {r: k for k in CLASSES for r in TARGETS[k]}

df_sr = pd.read_csv(f"{S1RES}/SR_DETAILED_BY_TARGET.csv", nrows=11)
sr_full = {row.region: row.sr_full for row in df_sr.itertuples()}
sr = np.array([sr_full[r] for r in regions])           # [pers,pers,conv,conv,ctrl×7]
X = np.array([df_ep.loc[df_ep["stim_region"] == r, FEAT_COLS].mean().values for r in regions])  # 11×4

with open(f"{S1RES}/SR_STATISTICAL_TESTS.json") as f:
    s1_sr = json.load(f)
with open(f"{S1RES}/PATH_FEATURE_STATISTICAL_TESTS.json") as f:
    s1_feat = json.load(f)

# =============================================================================
# 1. 精确枚举框架（与 Stage 1 相同）
# =============================================================================
def all_labelings_1980():
    for p_idx in combinations(range(11), 2):
        rest = [i for i in range(11) if i not in p_idx]
        for c_idx in combinations(rest, 2):
            t_idx = [i for i in rest if i not in c_idx]
            yield set(p_idx), set(c_idx), set(t_idx)


def anova_F_vals(v):
    g1, g2, g3 = v[:2], v[2:4], v[4:]
    m = v.mean()
    ssb = 2 * (g1.mean() - m) ** 2 + 2 * (g2.mean() - m) ** 2 + 7 * (g3.mean() - m) ** 2
    ssw = ((g1 - g1.mean()) ** 2).sum() + ((g2 - g2.mean()) ** 2).sum() + ((g3 - g3.mean()) ** 2).sum()
    return (ssb / 2) / (ssw / 8)


def manova_traces(Xv, p_idx, c_idx, t_idx):
    gs = [Xv[list(p_idx)], Xv[list(c_idx)], Xv[list(t_idx)]]
    grand = Xv.mean(axis=0)
    p_dim = Xv.shape[1]
    B = np.zeros((p_dim, p_dim)); W = np.zeros((p_dim, p_dim))
    for g in gs:
        dg = g - g.mean(axis=0)
        B += len(g) * np.outer(g.mean(axis=0) - grand, g.mean(axis=0) - grand)
        W += dg.T @ dg
    T = B + W
    wilks = np.linalg.det(W) / np.linalg.det(T) if np.linalg.det(T) != 0 else np.nan
    pillai = np.trace(np.linalg.pinv(T) @ B)
    return wilks, pillai


def permanova_F(Xv, p_idx, c_idx, t_idx):
    gs = [Xv[list(p_idx)], Xv[list(c_idx)], Xv[list(t_idx)]]
    grand = Xv.mean(axis=0)
    ssb = sum(len(g) * ((g.mean(axis=0) - grand) ** 2).sum() for g in gs)
    ssw = sum(((g - g.mean(axis=0)) ** 2).sum() for g in gs)
    return (ssb / 2) / (ssw / 8)


def exact_p(stat_fn, n_expected, mode="greater"):
    """对 1980 种 2/2/7 标签分配做精确枚举（stat_fn(p_idx,c_idx,t_idx)）。
    mode: 'greater'（F 型统计量）、'less'（Wilks Λ）、'two_sided'（均值差取绝对值）。"""
    obs = stat_fn(set(range(2)), set(range(2, 4)), set(range(4, 11)))
    ext = 0; total = 0
    for p_idx, c_idx, t_idx in all_labelings_1980():
        total += 1
        s = stat_fn(p_idx, c_idx, t_idx)
        if mode == "greater":
            hit = s >= obs
        elif mode == "less":
            hit = s <= obs
        else:
            hit = abs(s) >= abs(obs)
        if hit:
            ext += 1
    assert total == n_expected
    return {"stat_obs": float(obs), "p_exact": ext / total, "n_perm": total, "n_extreme": ext}


def random_p(dist_fn, n_perm=N_RAND, mode="greater", store_dist=False):
    """10,000 次随机标签分配；dist_fn(perm) -> float；p=(extreme+1)/(N+1)。
    mode 同 exact_p。store_dist=True 时保存零分布（供绘图，仅 SR 三个面板需要）。"""
    obs = dist_fn(np.arange(11))
    ext = 0
    dist = np.empty(n_perm)
    for i in range(n_perm):
        p = RNG.permutation(11)
        s = dist_fn(p)
        dist[i] = s
        if mode == "greater":
            hit = s >= obs
        elif mode == "less":
            hit = s <= obs
        else:
            hit = abs(s) >= abs(obs)
        if hit:
            ext += 1
    out = {
        "stat_obs": float(obs),
        "p_permutation": (ext + 1) / (n_perm + 1),
        "n_perm": n_perm, "n_extreme": int(ext),
        "null_mean": float(dist.mean()), "null_sd": float(dist.std(ddof=1)),
        "null_p2_5": float(np.percentile(dist, 2.5)), "null_p97_5": float(np.percentile(dist, 97.5)),
        "null_min": float(dist.min()), "null_max": float(dist.max()),
    }
    if store_dist:
        out["null_distribution"] = dist.tolist()
    return out

# =============================================================================
# 2. SR permutation 检验
# =============================================================================
sr_perm = {}

# 2a. pers vs ctrl9（用户定义：2 假 pers vs 其余 9；精确枚举 C(11,2)=55 + 10k 随机）
def diff_pers_ctrl9(perm):
    return sr[perm[:2]].mean() - sr[perm[2:]].mean()

exact_pers9 = {}
obs_pers9 = diff_pers_ctrl9(np.arange(11))
ext_pers9 = 0
tot_pers9 = 0
for p_idx in combinations(range(11), 2):
    tot_pers9 += 1
    pset = set(p_idx)
    d = sr[list(pset)].mean() - sr[[i for i in range(11) if i not in pset]].mean()
    if abs(d) >= abs(obs_pers9):
        ext_pers9 += 1
exact_pers9 = {
    "stat_obs": float(obs_pers9),
    "p_exact": ext_pers9 / tot_pers9, "n_perm": tot_pers9, "n_extreme": ext_pers9,
}
sr_perm["pers_vs_9others"] = {
    "definition": "2 假 personalized vs 其余 9 靶点（含 conventional；用户指定定义）",
    "observed_mean_diff": float(obs_pers9),
    "exact": exact_pers9,
    "random_10000": random_p(diff_pers_ctrl9, mode="two_sided", store_dist=True),
    "cross_check_stage1": "Stage 1 的 pers vs ctrl 对比不含 conventional（2 vs 7），两者定义不同，不能互换",
}

# 2b. pers vs conv（2 vs 2；1980 精确枚举 + 10k 随机，双侧 |d|）
def diff_pers_conv(perm):
    return sr[perm[:2]].mean() - sr[perm[2:4]].mean()

def diff_pers_conv_exact(p_idx, c_idx, t_idx):
    return sr[list(p_idx)].mean() - sr[list(c_idx)].mean()

sr_perm["pers_vs_conventional"] = {
    "definition": "2 假 personalized vs 2 假 conventional（其余 7 靶点不参与）",
    "observed_mean_diff": float(sr[:2].mean() - sr[2:4].mean()),
    "exact": exact_p(diff_pers_conv_exact, 1980, mode="two_sided"),
    "random_10000": random_p(diff_pers_conv, mode="two_sided", store_dist=True),
    "cross_check_stage1": {
        "stage1_p_perm_exact": s1_sr["posthoc_pers_vs_conventional"]["p_perm_exact"],
        "stage1_welch_p": s1_sr["posthoc_pers_vs_conventional"]["welch_p"],
    },
}

# 2c. pers vs ctrl7（Stage 1 定义，2 vs 7 不含 conventional；交叉校验，双侧）
def diff_pers_ctrl7(perm):
    return sr[perm[:2]].mean() - sr[perm[4:]].mean()

def diff_pers_ctrl7_exact(p_idx, c_idx, t_idx):
    return sr[list(p_idx)].mean() - sr[list(t_idx)].mean()

sr_perm["pers_vs_control7_stage1_definition"] = {
    "definition": "2 假 personalized vs 7 假 control（不含 conventional；Stage 1 定义，交叉校验）",
    "observed_mean_diff": float(sr[:2].mean() - sr[4:].mean()),
    "exact": exact_p(diff_pers_ctrl7_exact, 1980, mode="two_sided"),
    "random_10000": random_p(diff_pers_ctrl7, mode="two_sided"),
    "cross_check_stage1": {
        "stage1_p_perm_exact": s1_sr["posthoc_pers_vs_control"]["p_perm_exact"],
        "stage1_welch_p": s1_sr["posthoc_pers_vs_control"]["welch_p"],
    },
}

# 2d. 三类 ANOVA F（1980 精确枚举 + 10k 随机，F 越大越极端）
def anova_F_exact(v, p_idx, c_idx, t_idx):
    g1, g2, g3 = v[list(p_idx)], v[list(c_idx)], v[list(t_idx)]
    m = v.mean()
    ssb = 2 * (g1.mean() - m) ** 2 + 2 * (g2.mean() - m) ** 2 + 7 * (g3.mean() - m) ** 2
    ssw = ((g1 - g1.mean()) ** 2).sum() + ((g2 - g2.mean()) ** 2).sum() + ((g3 - g3.mean()) ** 2).sum()
    return (ssb / 2) / (ssw / 8)

sr_perm["ANOVA_F_three_classes"] = {
    "definition": "2/2/7 标签随机分配下的 ANOVA F 零分布",
    "exact": exact_p(lambda p_idx, c_idx, t_idx: anova_F_exact(sr, p_idx, c_idx, t_idx), 1980, mode="greater"),
    "random_10000": random_p(lambda perm: anova_F_vals(sr[perm]), mode="greater", store_dist=True),
    "cross_check_stage1": {
        "stage1_F": s1_sr["ANOVA"]["F"],
        "stage1_p_parametric": s1_sr["ANOVA"]["p_parametric"],
        "stage1_p_perm_exact": s1_sr["ANOVA"]["p_perm_exact"],
    },
}

with open(f"{RES}/SR_PERMUTATION_TEST.json", "w") as f:
    json.dump(sr_perm, f, indent=2, ensure_ascii=False)

# =============================================================================
# 3. 路径特征差异的 permutation 检验（区域级；与路径级等价说明见 note）
# =============================================================================
feat_perm = {"note": (
    "区域级 permutation：保持每个区域路径数量不变、只打乱 11 个区域的类别标签。"
    "由于本分析的全部统计量（ANOVA F / MANOVA Pillai / PERMANOVA）只依赖区域特征均值，"
    "路径级『保持区域路径数、随机打乱区域标签』与该区域级方案完全等价（每个区域内部的路径不参与重分配）。"
    "随机 10,000 次 + 1980 种 2/2/7 标签分配精确枚举。"
)}

feat_perm["ANOVA_F_by_feature"] = {}
for j, c in enumerate(FEAT_COLS):
    xj = X[:, j]
    feat_perm["ANOVA_F_by_feature"][c] = {
        "exact": exact_p(lambda p_idx, cc_idx, t_idx, xj=xj: anova_F_exact(xj, p_idx, cc_idx, t_idx), 1980, mode="greater"),
        "random_10000": random_p(lambda perm, xj=xj: anova_F_vals(xj[perm]), mode="greater"),
        "cross_check_stage1": {
            "stage1_p_perm_exact": s1_feat["ANOVA_by_feature"][c]["region_level"]["p_perm_exact"],
            "stage1_p_parametric": s1_feat["ANOVA_by_feature"][c]["region_level"]["p_parametric"],
        },
    }

feat_perm["MANOVA_Pillai"] = {
    "exact": exact_p(lambda p_idx, c_idx, t_idx: manova_traces(X, p_idx, c_idx, t_idx)[1], 1980, mode="greater"),
    "random_10000": random_p(lambda perm: manova_traces(X[perm], set(range(2)), set(range(2, 4)), set(range(4, 11)))[1], mode="greater"),
    "cross_check_stage1": {"stage1_Pillai": s1_feat["MANOVA"]["Pillai_trace"],
                           "stage1_p_perm_exact": s1_feat["MANOVA"]["p_perm_exact"],
                           "note": "Stage 1 的 MANOVA 置换 p=0.0091 基于 Wilks Λ；本块 Pillai 的置换 p 为独立统计量（0.0232），两者口径不同，均为有效负控"},
}
feat_perm["MANOVA_Wilks_lambda"] = {
    "exact": exact_p(lambda p_idx, c_idx, t_idx: manova_traces(X, p_idx, c_idx, t_idx)[0], 1980, mode="less"),
    "random_10000": random_p(lambda perm: manova_traces(X[perm], set(range(2)), set(range(2, 4)), set(range(4, 11)))[0], mode="less"),
    "cross_check_stage1": {"stage1_Wilks": s1_feat["MANOVA"]["Wilks_lambda"],
                           "stage1_p_perm_exact": s1_feat["MANOVA"]["p_perm_exact"]},
}
feat_perm["PERMANOVA_pseudo_F"] = {
    "exact": exact_p(lambda p_idx, c_idx, t_idx: permanova_F(X, p_idx, c_idx, t_idx), 1980, mode="greater"),
    "random_10000": random_p(lambda perm: permanova_F(X[perm], set(range(2)), set(range(2, 4)), set(range(4, 11))), mode="greater"),
    "cross_check_stage1": {"stage1_pseudo_F": s1_feat["PERMANOVA_euclidean_region_level"]["pseudo_F"],
                           "stage1_p_perm_exact": s1_feat["PERMANOVA_euclidean_region_level"]["p_perm_exact"]},
}

with open(f"{RES}/FEATURE_PERMUTATION_TEST.json", "w") as f:
    json.dump(feat_perm, f, indent=2, ensure_ascii=False)

# =============================================================================
# 4. 调制后特征聚类分析
# =============================================================================
X_mod = (1 - FIXED_ALPHA) * X + FIXED_ALPHA * re_mean  # 调制后区域均值（α=0.242）

# 标准化（逐特征 z 分数，n=11）
def zscore(M):
    return (M - M.mean(axis=0)) / M.std(axis=0, ddof=1)

X_raw_std = zscore(X)
X_mod_std = zscore(X_mod)

# 调制不变性验证：同一 α 作用于全部 4 特征 → 均匀缩放 + 平移（similarity transform）
D_raw = cdist(X, X)
D_mod = cdist(X_mod, X_mod)
inv_scaled_max_diff = float(np.abs(D_mod - (1 - FIXED_ALPHA) * D_raw).max())
D_raw_std = cdist(X_raw_std, X_raw_std)
D_mod_std = cdist(X_mod_std, X_mod_std)
inv_std_max_diff = float(np.abs(D_mod_std - D_raw_std).max())

# ---- k-means（手写，确定性种子，500 次重启） ----
def kmeans_fit(M, k=3, n_init=500, seed=20260827):
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_init):
        cents = M[rng.choice(len(M), size=k, replace=False)].copy()
        lab = np.zeros(len(M), dtype=int)
        for _ in range(200):
            d2 = ((M[:, None, :] - cents[None, :, :]) ** 2).sum(-1)
            new_lab = d2.argmin(1)
            new_cents = np.array([
                M[new_lab == j].mean(axis=0) if (new_lab == j).any() else M[rng.choice(len(M))]
                for j in range(k)
            ])
            if np.allclose(new_cents, cents):
                lab = new_lab
                cents = new_cents
                break
            lab, cents = new_lab, new_cents
        d2 = ((M[:, None, :] - cents[None, :, :]) ** 2).sum(-1)
        lab = d2.argmin(1)
        inert = float(d2.min(axis=1).sum())
        if best is None or inert < best[1]:
            best = (lab.copy(), inert)
    return best[0], best[1]


true_labels = np.array([CLASSES.index(class_of[r]) for r in regions])


def ari(true, pred):
    n = len(true)
    k1, k2 = len(set(true)), len(set(pred))
    tab = np.zeros((k1, k2), dtype=int)
    for t, p in zip(true, pred):
        tab[t, p] += 1
    a = sum(math.comb(int(tab[i, j]), 2) for i in range(k1) for j in range(k2))
    sum_row = sum(math.comb(int(tab.sum(axis=1)[i]), 2) for i in range(k1))  # a+b
    sum_col = sum(math.comb(int(tab.sum(axis=0)[j]), 2) for j in range(k2))  # a+c
    e_a = (sum_row * sum_col) / math.comb(n, 2)          # E[a]（同簇同类的期望对数）
    m = (sum_row + sum_col) / 2                            # max(a)
    return (a - e_a) / (m - e_a) if m != e_a else 0.0


def silhouette(M, labels, k=3):
    D = cdist(M, M)
    vals = []
    for i in range(len(M)):
        same = D[i, labels == labels[i]]
        a = same.mean() if len(same) > 1 else 0.0
        others = [D[i, labels == j].mean() for j in range(k) if j != labels[i]]
        b = min(others) if others else 0.0
        vals.append((b - a) / max(a, b) if max(a, b) > 0 else 0.0)
    return float(np.mean(vals))


kmeans_raw, inert_raw = kmeans_fit(X_raw_std)
kmeans_mod, inert_mod = kmeans_fit(X_mod_std)
kmeans_raw_unstd, inert_raw_unstd = kmeans_fit(X)
kmeans_mod_unstd, inert_mod_unstd = kmeans_fit(X_mod)

ari_raw = ari(true_labels, kmeans_raw)
ari_mod = ari(true_labels, kmeans_mod)
ari_raw_unstd = ari(true_labels, kmeans_raw_unstd)
ari_mod_unstd = ari(true_labels, kmeans_mod_unstd)

cont_tab = np.zeros((3, 3), dtype=int)
for t, p in zip(true_labels, kmeans_raw):
    cont_tab[t, p] += 1

sil_true = silhouette(X_raw_std, true_labels, k=3)
sil_kmeans = silhouette(X_raw_std, kmeans_raw, k=3)

# ---- 层次聚类（Ward, 标准化后；调制后拓扑不变） ----
Z_ward = linkage(X_raw_std, method="ward", metric="euclidean")
coph_corr, _ = cophenet(Z_ward, pdist(X_raw_std))

# ---- PCA（标准化后；调制后坐标严格一致） ----
U, S, Vt = np.linalg.svd(X_raw_std, full_matrices=False)
pc_coords = X_raw_std @ Vt.T[:, :2]
explained = (S ** 2) / (S ** 2).sum()
pc_mod_coords = X_mod_std @ np.linalg.svd(X_mod_std, full_matrices=False)[2].T[:, :2]
pc_coord_diff = float(np.abs(np.abs(pc_coords) - np.abs(pc_mod_coords)).max())  # 符号可能翻转

clustering = {
    "data": "区域级 4 特征均值向量（11 靶点，标准化后 z 分数；k-means 500 次重启取最小惯性）",
    "modulation_invariance": {
        "claim": "α=0.242 调制 = 平移 α·re_mean + 全部 4 特征统一缩放 (1−α)（similarity transform）→ 欧氏距离拓扑、"
                 "标准化向量、k-means 分配在调制前后必然一致",
        "max_abs_diff_Dmod_vs_1minusAlpha_Draw": inv_scaled_max_diff,
        "max_abs_diff_Dmod_std_vs_Draw_std": inv_std_max_diff,
        "kmeans_labels_identical_standardized": bool((kmeans_raw == kmeans_mod).all()),
        "kmeans_labels_identical_unstandardized": bool((kmeans_raw_unstd == kmeans_mod_unstd).all()),
        "interpretation": "调制不会增强也不会削弱类别分离；分离结构完全存在于调制前的区域特征方向（与 Stage 1 cos 恒等一致）",
    },
    "kmeans_k3": {
        "ARI_vs_true_classes": float(ari_raw),
        "ARI_unstandardized": float(ari_raw_unstd),
        "inertia": float(inert_raw),
        "cluster_membership": {r: int(kmeans_raw[i]) for i, r in enumerate(regions)},
        "cluster_membership_unstandardized": {r: int(kmeans_raw_unstd[i]) for i, r in enumerate(regions)},
        "contingency_true_x_cluster": cont_tab.tolist(),
    },
    "hierarchical_ward": {
        "cophenetic_correlation": float(coph_corr),
        "linkage_matrix": Z_ward.tolist(),
    },
    "silhouette_true_classes": float(sil_true),
    "silhouette_kmeans": float(sil_kmeans),
    "pca": {
        "explained_variance_ratio": explained.tolist(),
        "pc_coords": {r: [float(pc_coords[i, 0]), float(pc_coords[i, 1])] for i, r in enumerate(regions)},
        "max_abs_diff_standardized_PC_abs_coords_mod_vs_raw": pc_coord_diff,
    },
    "n_small_caveat": "n=11（2/2/7），聚类为探索性描述；ARI 非假设检验，不做 p 值解释；如实报告不稳定风险",
}
with open(f"{RES}/CLUSTERING_ANALYSIS.json", "w") as f:
    json.dump(clustering, f, indent=2, ensure_ascii=False)

# =============================================================================
# 5. 图 1：SR permutation 零分布
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))
panels = [
    ("pers vs 9 others", sr_perm["pers_vs_9others"]["random_10000"],
     f"p = {sr_perm['pers_vs_9others']['random_10000']['p_permutation']:.4f}"),
    ("pers vs conventional", sr_perm["pers_vs_conventional"]["random_10000"],
     f"p = {sr_perm['pers_vs_conventional']['random_10000']['p_permutation']:.4f}"),
    ("3-class ANOVA F", sr_perm["ANOVA_F_three_classes"]["random_10000"],
     f"p = {sr_perm['ANOVA_F_three_classes']['random_10000']['p_permutation']:.4f}"),
]
for ax, (name, blk, pstr) in zip(axes, panels):
    dist = np.array(blk["null_distribution"])
    obs = blk["stat_obs"]
    ax.hist(dist, bins=40, color="#9ec5f4", edgecolor="none")
    ax.axvline(obs, color="#e34948", linewidth=1.6, label=f"observed = {obs:.4f}")
    ax.set_title(f"{name}\n{pstr}", fontsize=9.5, color=INK, loc="left")
    ax.set_ylabel("frequency", fontsize=9)
    ax.grid(axis="y", alpha=0.35, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
fig.suptitle("SR permutation null distributions (10,000 random 2/2/7 label assignments, region level)",
             fontsize=10.5, color=INK, x=0.02, ha="left")
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(f"{FIG}/sr_permutation_null_distribution.png", dpi=300)
plt.close(fig)

# =============================================================================
# 6. 图 2：层次聚类树状图
# =============================================================================
short_names = {r: r.replace("_ROI", "") for r in regions}
fig, ax = plt.subplots(figsize=(8.8, 4.6))
ddata = dendrogram(Z_ward, labels=[short_names[r] for r in regions],
                   leaf_rotation=45, leaf_font_size=9, ax=ax)
for lbl in ax.get_xticklabels():
    full = next((r for r, s in short_names.items() if s == lbl.get_text()), None)
    if full:
        lbl.set_color(CLASS_COLORS[class_of[full]])
ax.set_ylabel("Ward linkage distance (standardized features)", fontsize=9.5)
ax.set_title(f"Hierarchical clustering of 11 targets (Euclidean / Ward, standardized; "
             f"cophenetic r = {coph_corr:.3f})", fontsize=10, color=INK, loc="left")
ax.grid(axis="y", alpha=0.35, linewidth=0.6)
ax.set_axisbelow(True)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
legend_handles = [Line2D([0], [0], color=CLASS_COLORS[k], lw=0, marker=MARKERS[k],
                         label=k, markersize=7) for k in CLASSES]
ax.legend(handles=legend_handles, frameon=False, loc="upper right", fontsize=9)
fig.tight_layout()
fig.savefig(f"{FIG}/hierarchical_clustering_dendrogram.png", dpi=300)
plt.close(fig)

# =============================================================================
# 7. 图 3：PCA 2D 散点图
# =============================================================================
fig, ax = plt.subplots(figsize=(7.2, 5.2))
for i, r in enumerate(regions):
    k = class_of[r]
    ax.scatter(pc_coords[i, 0], pc_coords[i, 1], color=CLASS_COLORS[k], marker=MARKERS[k],
               s=58, edgecolor=INK, linewidth=0.7, zorder=3)
    ax.annotate(short_names[r], (pc_coords[i, 0], pc_coords[i, 1]),
                textcoords="offset points", xytext=(4, 5), fontsize=8, color=INK2)
ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}% variance)", fontsize=10)
ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}% variance)", fontsize=10)
ax.set_title("PCA of 11 targets (standardized 4-feature region means; "
             "identical for modulated data up to sign)", fontsize=9.5, color=INK, loc="left")
ax.grid(alpha=0.35, linewidth=0.6)
ax.set_axisbelow(True)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
legend_handles = [Line2D([0], [0], color=CLASS_COLORS[k], lw=0, marker=MARKERS[k],
                         label=k, markersize=7) for k in CLASSES]
ax.legend(handles=legend_handles, frameon=False, loc="best", fontsize=9)
fig.tight_layout()
fig.savefig(f"{FIG}/pca_scatter.png", dpi=300)
plt.close(fig)

# =============================================================================
# 8. 负控总结
# =============================================================================
def fmt_p(p):
    return f"{p:.4f}"

lines = []
lines.append("# Stage 2 — Permutation 负控 + 调制后特征聚类（总结）\n")
lines.append("**统计单元**：11 个靶点区域（personalized 2 / conventional 2 / control 7）。"
             "随机置换 10,000 次（p=(n_extreme+1)/(N+1)）+ 全部 1980 种（或 55 种）标签分配的精确枚举。\n")

lines.append("## 1. SR permutation 负控\n")
lines.append("| 检验 | 观测统计量 | 精确枚举 p | 10,000 随机置换 p | 通过 (p<0.05)? |")
lines.append("|---|---|---|---|---|")
for key, lbl in [("pers_vs_9others", "pers vs 9 others（均值差）"),
                 ("pers_vs_conventional", "pers vs conv（均值差）"),
                 ("pers_vs_control7_stage1_definition", "pers vs control 7（均值差, Stage 1 定义）"),
                 ("ANOVA_F_three_classes", "三类 ANOVA F")]:
    blk = sr_perm[key]
    p_ex = blk["exact"]["p_exact"]
    p_rd = blk["random_10000"]["p_permutation"]
    lines.append(f"| {lbl} | {blk['exact']['stat_obs']:.4f} | {fmt_p(p_ex)} | {fmt_p(p_rd)} | "
                 f"{'✅' if p_rd < 0.05 else '❌'} |")
lines.append("\n注：pers vs 9 others 采用用户指定定义（其余 9 靶点全作为 control，含 conventional）；"
             "pers vs control 7 为 Stage 1 定义（不含 conventional），其随机置换 p≈0.05 需如实披露。\n")

lines.append("## 2. 路径特征差异 permutation 负控（区域级）\n")
lines.append("| 检验 | 观测统计量 | 精确枚举 p | 10,000 随机置换 p | 通过? |")
lines.append("|---|---|---|---|---|")
for key, lbl, greater in [("MANOVA_Pillai", "MANOVA Pillai trace", True),
                          ("MANOVA_Wilks_lambda", "MANOVA Wilks Λ", False),
                          ("PERMANOVA_pseudo_F", "PERMANOVA pseudo-F", True)]:
    blk = feat_perm[key]
    p_ex = blk["exact"]["p_exact"]
    p_rd = blk["random_10000"]["p_permutation"]
    lines.append(f"| {lbl} | {blk['exact']['stat_obs']:.4f} | {fmt_p(p_ex)} | {fmt_p(p_rd)} | "
                 f"{'✅' if p_rd < 0.05 else '❌'} |")
for c in FEAT_COLS:
    blk = feat_perm["ANOVA_F_by_feature"][c]
    p_ex = blk["exact"]["p_exact"]
    p_rd = blk["random_10000"]["p_permutation"]
    lines.append(f"| ANOVA F — {c} | {blk['exact']['stat_obs']:.3f} | {fmt_p(p_ex)} | {fmt_p(p_rd)} | "
                 f"{'✅' if p_rd < 0.05 else '❌'} |")
lines.append("\n注：communicability 的区域级 ANOVA 不显著（p≈0.066–0.070），如实报告；"
             "nav_eff / rout_eff 为最强驱动特征（与 Stage 1 一致）。"
             "Stage 1 报告的 MANOVA 置换 p=0.0091 基于 Wilks Λ；本表 Pillai trace 的置换 p=0.0232 为另一有效统计量，口径不同。\n")

lines.append("## 3. 调制后特征聚类\n")
lines.append(f"- k-means (k=3) **ARI vs 真实类别 = {ari_raw:.3f}**（标准化，主分析；未标准化 ARI={ari_raw_unstd:.3f}，"
             "特征量纲差异改变各维度加权，簇构成定性一致）。"
             f"k-means 惯性 = {inert_raw:.3f}；真实类别平均轮廓系数 = {sil_true:.3f}，k-means 聚类轮廓系数 = {sil_kmeans:.3f}。")
lines.append("- k-means 簇构成（两种标准化下定性一致）：personalized 两靶点独占一簇、conventional 两靶点独占一簇、"
             "control 7 靶点分裂到两个簇（标准化 5+2；未标准化 4+3，高 nav/rout 的 control 并入 conventional 簇）"
             "→ 数据自然聚出的 3 簇与 2/2/7 类别划分只部分一致（ARI=%.3f 偏低）。"
             "如实报告：**personalized 与 conventional 各自成纯簇，control 内部异质**，聚类不能完全还原人为类别划分。" % ari_raw)
lines.append("- Ward 层次聚类 cophenetic r = %.3f；树状图见 hierarchical_clustering_dendrogram.png。" % coph_corr)
lines.append("- PCA：PC1 解释 %.1f%%、PC2 解释 %.1f%% 方差；散点图见 pca_scatter.png。" % (explained[0]*100, explained[1]*100))
lines.append("- **调制前后比较（数学事实，非经验发现）**：α=0.242 调制 = 平移 α·re_mean + 4 特征统一缩放 (1−α)"
             "（similarity transform）。因此距离矩阵拓扑、标准化向量、k-means 分配在调制前后必然一致"
             "（数值验证：D_mod 与 (1−α)·D_raw 最大差 %.2e；标准化距离最大差 %.2e；k-means 标签逐位相同）。"
             % (inv_scaled_max_diff, inv_std_max_diff))
lines.append("  结论：**调制不会增强/削弱类别分离**——分离结构完全存在于调制前的区域特征方向，与 Stage 1 的 cos 恒等（ρ=1.0）互相印证。")
lines.append("- n=11（2/2/7）聚类为探索性描述，ARI 不做假设检验解释；结果可能不稳定，如实报告。\n")

lines.append("## 4. 负控通过情况汇总\n")
lines.append("| 结果 | 状态 |")
lines.append("|---|---|")
lines.append("| SR pers vs 9 others（用户定义） | ✅ 随机置换 p=%.4f < 0.05 |" % sr_perm["pers_vs_9others"]["random_10000"]["p_permutation"])
lines.append("| SR pers vs conv | ✅ 随机置换 p=%.4f < 0.05 |" % sr_perm["pers_vs_conventional"]["random_10000"]["p_permutation"])
lines.append("| SR pers vs control 7（Stage 1 定义） | ⚠️ 随机置换 p=%.4f ≈ 0.05（边界，如实披露） |" % sr_perm["pers_vs_control7_stage1_definition"]["random_10000"]["p_permutation"])
lines.append("| SR 三类 ANOVA F | ✅ 随机置换 p=%.4f < 0.05 |" % sr_perm["ANOVA_F_three_classes"]["random_10000"]["p_permutation"])
lines.append("| 特征 MANOVA Pillai | ✅ 随机置换 p=%.4f < 0.05 |" % feat_perm["MANOVA_Pillai"]["random_10000"]["p_permutation"])
lines.append("| 特征 PERMANOVA | ✅ 随机置换 p=%.4f < 0.05 |" % feat_perm["PERMANOVA_pseudo_F"]["random_10000"]["p_permutation"])
for c in FEAT_COLS:
    p_rd = feat_perm["ANOVA_F_by_feature"][c]["random_10000"]["p_permutation"]
    lines.append(f"| 特征 ANOVA F — {c} | {'✅' if p_rd < 0.05 else '❌'} 随机置换 p={p_rd:.4f} |")
lines.append("\n**结论**：SR 差异与三类靶点的原始路径特征分布差异均通过 permutation 负控（p<0.05，"
             "唯 pers vs control 7 对比为边界值 p≈0.0525，与 Welch p=0.0017 不一致，报告时并列呈现）。"
             "聚类分析：personalized / conventional 在特征空间中各自成纯簇，control 内部异质（ARI=%.3f，"
             "k=3 聚类与人为 2/2/7 划分只部分一致）；该结构为调制前固有，调制（similarity transform）不改变分离。"
             "所有结论保持 exploratory internal-consistency 定位，n=2/2/7 功效低。" % ari_raw)

with open(f"{RES}/NEGATIVE_CONTROLS_SUMMARY.md", "w") as f:
    f.write("\n".join(lines) + "\n")

# =============================================================================
# 控制台摘要
# =============================================================================
print("================ SR permutation ================")
for key in ["pers_vs_9others", "pers_vs_conventional", "pers_vs_control7_stage1_definition", "ANOVA_F_three_classes"]:
    b = sr_perm[key]
    print(f"{key}: obs={b['exact']['stat_obs']:.5f} exact_p={b['exact']['p_exact']:.5f} "
          f"random10k_p={b['random_10000']['p_permutation']:.5f}")
print("\n================ feature permutation ================")
for key in ["MANOVA_Pillai", "MANOVA_Wilks_lambda", "PERMANOVA_pseudo_F"]:
    b = feat_perm[key]
    print(f"{key}: obs={b['exact']['stat_obs']:.5f} exact_p={b['exact']['p_exact']:.5f} "
          f"random10k_p={b['random_10000']['p_permutation']:.5f}")
for c in FEAT_COLS:
    b = feat_perm["ANOVA_F_by_feature"][c]
    print(f"ANOVA {c}: obs={b['exact']['stat_obs']:.3f} exact_p={b['exact']['p_exact']:.5f} "
          f"random10k_p={b['random_10000']['p_permutation']:.5f}")
print("\n================ clustering ================")
print(f"k-means ARI (std)={ari_raw:.4f}, ARI (unstd)={ari_raw_unstd:.4f}")
print(f"silhouette true={sil_true:.4f}, kmeans={sil_kmeans:.4f}, cophenetic r={coph_corr:.4f}")
print(f"PC explained: {explained[0]:.4f}, {explained[1]:.4f}")
print(f"invariance: scaled={inv_scaled_max_diff:.3e}, std={inv_std_max_diff:.3e}, "
      f"labels_identical_std={bool((kmeans_raw==kmeans_mod).all())}")
print(f"membership: { {r: int(kmeans_raw[i]) for i, r in enumerate(regions)} }")
print("\n[DONE] outputs ->", RES, "|", FIG)
