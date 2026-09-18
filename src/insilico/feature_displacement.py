# -*- coding: utf-8 -*-
# fixed_alpha_simulation.py — Stage 2 固定 α 主分析
# 基于原始代码 stimulations_for_modu.py
#
# ★ 唯一计算修改：simulate_dynamic 中的 α 分配由「按 target_type 差异化分配」
#   改为「所有靶点统一使用 FIXED_ALPHA = 0.242」。
#   （0.242 = Stage 1 确定的三类靶点 α 总体中位数 0.241991 的四舍五入，
#    见 01_audit/FIXED_ALPHA_DECISION.md）
# ★ 非计算修改：输入 CSV 绝对路径、结果落盘（原代码仅 print + 存 PNG）。
# 其余一切（数据加载、聚合、靶点分类、calc_api_single、cos_sim、CR/SR/SI 指标、
# ANOVA、绘图）与原始代码逐行一致。

import os
import pandas as pd
import numpy as np
import warnings
import json
warnings.filterwarnings('ignore')

PIPELINE_ROOT = os.environ.get('PIPELINE_ROOT', '')
if not PIPELINE_ROOT:
    raise EnvironmentError(
        "PIPELINE_ROOT environment variable must be set to the data workspace root. "
        "See README.md (Data Availability section) for data download instructions."
    )

# =============================================================================
# 1. 基础配置 + 加载数据 + 完整路径聚合（与原始代码一致）
# =============================================================================
feat_cols = ["nav_eff", "rout_eff", "search_info", "communicability"]

# 加载你的数据
df_ep_sub = pd.read_csv(os.path.join(PIPELINE_ROOT, "raw_inputs", "subpath_samples_1129_1.csv"))
df_re_sub = pd.read_csv(os.path.join(PIPELINE_ROOT, "raw_inputs", "subpath_samples_1130_1.csv"))

# 子路径 → 完整路径（核心不变）
def aggregate_to_full_path(df_sub):
    return df_sub.groupby(["path_id", "stim_region"])[feat_cols].mean().reset_index()

df_ep = aggregate_to_full_path(df_ep_sub)
df_re = aggregate_to_full_path(df_re_sub)
re_mean = df_re[feat_cols].mean().values  # 康复基准特征

# =============================================================================
# 2. 自动批量筛选有效靶点（与原始代码一致）
# =============================================================================
def get_valid_regions(df, min_paths=50):
    count = df["stim_region"].value_counts()
    return count[count >= min_paths].index.tolist()

valid_regions = get_valid_regions(df_ep)

df_mapping = pd.read_csv(os.path.join(PIPELINE_ROOT, "raw_inputs", "mmp_to_yeo7_mapping.csv"))

limbic_regions = df_mapping[
    df_mapping["Yeo7_Network_Label"].str.contains("Limbic|LIM", case=False, na=False)
]["MMP_Label"].tolist()

personalized = [r for r in valid_regions if any(lim in r for lim in limbic_regions)][:10]
conventional = [r for r in valid_regions if any(x in r for x in ["H", "PH", "EC"])][:10]
control = [r for r in valid_regions if any(x in r for x in ["V1", "V2", "1_ROI"])][:10]

TARGETS = {"personalized": personalized, "conventional": conventional, "control": control}

print("🧠 选定的 Limbic 网络个性化靶点 (前10):", personalized)

# =============================================================================
# 3. 指标函数（与原始代码一致）
# =============================================================================
def calc_api_single(feat_vector, re_feat_vector):
    """计算【单条路径】的异常指数（修复维度问题）"""
    diff = np.abs(feat_vector - re_feat_vector).sum()
    return np.clip(diff / 4, 0, 1)

def cos_sim(a, b):
    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)) if (np.linalg.norm(a)*np.linalg.norm(b))!=0 else 0.95

# =============================================================================
# ★ 修改点（Stage 2 唯一计算修改）：固定 α
# 原始代码：
#   if target_type == "personalized":
#       alpha = np.clip(0.2 + api_before * 0.75, 0.3, 0.75)
#   else:
#       alpha = np.clip(0.2 + api_before * 0.5, 0.2, 0.7)
# 修改后：所有靶点使用相同的 FIXED_ALPHA，分离"α 分配效应"与"靶点位置效应"。
# 依据：Stage 1 → 三类靶点 11 个 α 的总体中位数 = 0.241991 → FIXED_ALPHA = 0.242
# =============================================================================
FIXED_ALPHA = 0.242

# 动态仿真（除 α 分配外与原始代码一致）
def simulate_dynamic(df_region, re_feat):
    df_sim = df_region.copy()
    mean_feat = df_region[feat_cols].mean().values
    api_before = calc_api_single(mean_feat, re_feat)  # 保留计算：用于 CR 与记录
    alpha = FIXED_ALPHA  # ← 固定 α（原代码在此处按 target_type 差异化分配）

    for col in feat_cols:
        df_sim[col] = df_sim[col] * (1-alpha) + re_mean[feat_cols.index(col)] * alpha
    return df_sim, api_before

# =============================================================================
# 4. 批量计算3个指标（CR+SR+SI，与原始代码一致）
# =============================================================================
results = []
for target_type, regions in TARGETS.items():
    for region in regions:
        df_region = df_ep[df_ep["stim_region"] == region].copy()
        if len(df_region) < 10:
            continue

        # 仿真
        df_sim, api_before = simulate_dynamic(df_region, re_mean)

        # 指标1：矫正率 CR
        api_after = calc_api_single(df_sim[feat_cols].mean().values, re_mean)
        cr = np.clip((api_before - api_after)/api_before, 0, 1) if api_before>0 else 0
        cr_full = (api_before - api_after)/api_before if api_before>0 else 0.0  # 未四舍五入，供统计检验

        # 指标2：正常相似度 SR
        sr = cos_sim(df_sim[feat_cols].mean().values, re_mean)

        # 指标3：调控特异性 SI（与原始代码一致：遍历仿真前的 df_region）
        normal_paths = 0
        for idx, row in df_region.iterrows():
            api_path = calc_api_single(row[feat_cols].values, re_mean)
            if api_path < 0.2:
                normal_paths +=1
        si = 1 - (normal_paths / len(df_region))

        # 保存结果
        results.append({
            "target_type": target_type, "region": region,
            "api_before": round(float(api_before), 6),
            "alpha": FIXED_ALPHA,
            "CR": round(cr,3), "SR": round(sr,3), "SI": round(si,3),
            "cr_full": round(float(cr_full), 10), "sr_full": round(float(sr), 10),
        })

# =============================================================================
# 5. 输出结果 + 统计检验 + 绘图（与原始代码一致）
# =============================================================================
df_result = pd.DataFrame(results)
print("="*85)
print("📊 固定 α 仿真结果（FIXED_ALPHA=0.242 | 3指标 | 完整路径 | 批量靶点）")
print("="*85)
print(df_result[["target_type","region","CR","SR","SI"]])

summary = df_result.groupby("target_type")[["CR","SR","SI"]].mean().round(3)
print("\n🎯 三类靶点最终汇总（固定 α）")
print(summary)

from scipy.stats import f_oneway
print("\n📈 统计检验结果（固定 α）：")
anova_results = []
for metric in ["CR","SR","SI"]:
    p_data = df_result[df_result["target_type"]=="personalized"][metric]
    c_data = df_result[df_result["target_type"]=="conventional"][metric]
    ctrl_data = df_result[df_result["target_type"]=="control"][metric]
    if len(p_data)>0 and len(c_data)>0 and len(ctrl_data)>0:
        f_val, p_val = f_oneway(p_data, c_data, ctrl_data)
        print(f"{metric}：F={f_val:.2f}，p={p_val:.4f} {'✅显著' if p_val<0.05 else '⚪不显著'}")
        anova_results.append({"metric": metric, "F": round(float(f_val),4) if not np.isnan(f_val) else None,
                              "p": round(float(p_val),4) if not np.isnan(p_val) else None})

# 论文级绘图（保持原始样式）
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
summary.plot(kind="bar", figsize=(10,5), colormap="viridis")
plt.title(f"Experiment 3: Target Optimization (FIXED alpha={FIXED_ALPHA})", fontweight="bold")
plt.ylabel("Score (0~1, Higher=Better)", fontweight="bold")
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(os.path.join(PIPELINE_ROOT, "target_sim_alpha_correction_v1", "02_fixed_alpha", "figures", "experiment3_fixed_alpha.png"), dpi=300)
print("\n✅ 固定 α 高清论文图已保存！")

# =============================================================================
# 结果落盘（原代码无此部分，仅新增保存）
# =============================================================================
RES = os.path.join(PIPELINE_ROOT, "target_sim_alpha_correction_v1", "02_fixed_alpha", "results")
df_result.to_csv(f"{RES}/FIXED_ALPHA_RESULTS.csv", index=False)
summary.to_csv(f"{RES}/FIXED_ALPHA_SUMMARY.csv")
with open(f"{RES}/FIXED_ALPHA_ANOVA.json", "w") as f:
    json.dump(anova_results, f, indent=2, ensure_ascii=False)
print(f"\n[fixed_alpha] 结果已落盘到 {RES}/")
