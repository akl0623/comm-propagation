import pandas as pd
import numpy as np
import json
import os
import time
import warnings
warnings.filterwarnings('ignore')

BASE = os.environ.get('PIPELINE_ROOT', '/home/yuanyufei/revision_v2') + '/target_sim_sr_focused_v1'
CACHE = f"{BASE}/00_data/cache"
os.makedirs(CACHE, exist_ok=True)

feat_cols = ["nav_eff", "rout_eff", "search_info", "communicability"]
USECOLS = ["stim_region", "path_id"] + feat_cols

RAW_SUBPATH_DIR = os.environ.get('RAW_SUBPATH_DIR', os.path.dirname(os.environ.get('PIPELINE_ROOT', '/home/yuanyufei/revision_v2')))

# 注意：与原始代码一致使用默认 float64（float32 会改变 groupby 均值的小数位，
# 破坏与前一轮固定 α 结果的逐位复现）。仅通过 usecols 省内存/IO。
t0 = time.time()
print("[build_cache] loading ep subpaths ...", flush=True)
df_ep_sub = pd.read_csv(
    os.path.join(RAW_SUBPATH_DIR, 'subpath_samples_1129_1.csv'), usecols=USECOLS,
)
print(f"[build_cache] ep loaded: {len(df_ep_sub):,} rows in {time.time()-t0:.1f}s", flush=True)

t0 = time.time()
print("[build_cache] loading re subpaths ...", flush=True)
df_re_sub = pd.read_csv(
    os.path.join(RAW_SUBPATH_DIR, 'subpath_samples_1130_1.csv'), usecols=USECOLS,
)
print(f"[build_cache] re loaded: {len(df_re_sub):,} rows in {time.time()-t0:.1f}s", flush=True)


# 子路径 → 完整路径（与原始代码一致）
def aggregate_to_full_path(df_sub):
    return df_sub.groupby(["path_id", "stim_region"])[feat_cols].mean().reset_index()


t0 = time.time()
print("[build_cache] aggregating ep ...", flush=True)
df_ep = aggregate_to_full_path(df_ep_sub)
del df_ep_sub
print(f"[build_cache] df_ep: {len(df_ep):,} full paths in {time.time()-t0:.1f}s", flush=True)

t0 = time.time()
print("[build_cache] aggregating re ...", flush=True)
df_re = aggregate_to_full_path(df_re_sub)
del df_re_sub
print(f"[build_cache] df_re: {len(df_re):,} full paths in {time.time()-t0:.1f}s", flush=True)

re_mean = df_re[feat_cols].mean().values  # 康复基准特征（与原始代码一致）

# 靶点分类（与原始代码一致）
def get_valid_regions(df, min_paths=50):
    count = df["stim_region"].value_counts()
    return count[count >= min_paths].index.tolist()

valid_regions = get_valid_regions(df_ep)
MAPPING_P = os.environ.get('MMP_YEO7_MAPPING', os.path.join(RAW_SUBPATH_DIR, 'mmp_to_yeo7_mapping.csv'))
df_mapping = pd.read_csv(MAPPING_P)
limbic_regions = df_mapping[
    df_mapping["Yeo7_Network_Label"].str.contains("Limbic|LIM", case=False, na=False)
]["MMP_Label"].tolist()

personalized = [r for r in valid_regions if any(lim in r for lim in limbic_regions)][:10]
conventional = [r for r in valid_regions if any(x in r for x in ["H", "PH", "EC"])][:10]
control = [r for r in valid_regions if any(x in r for x in ["V1", "V2", "1_ROI"])][:10]
TARGETS = {"personalized": personalized, "conventional": conventional, "control": control}

print("[build_cache] TARGETS:", json.dumps(TARGETS, indent=2, ensure_ascii=False), flush=True)

# 互斥性检查
sets = [set(TARGETS[k]) for k in TARGETS]
overlap = (sets[0] & sets[1]) | (sets[0] & sets[2]) | (sets[1] & sets[2])
print(f"[build_cache] class overlap: {sorted(overlap)}", flush=True)

# 落盘
t0 = time.time()
df_ep.to_pickle(f"{CACHE}/df_ep.pkl")
df_re.to_pickle(f"{CACHE}/df_re.pkl")
np.save(f"{CACHE}/re_mean.npy", re_mean)
with open(f"{CACHE}/valid_regions.json", "w") as f:
    json.dump(valid_regions, f, indent=2)
with open(f"{CACHE}/TARGETS.json", "w") as f:
    json.dump(TARGETS, f, indent=2, ensure_ascii=False)
print(f"[build_cache] saved to {CACHE}/ in {time.time()-t0:.1f}s", flush=True)
print(f"[build_cache] re_mean = {re_mean}", flush=True)
print("[build_cache] DONE", flush=True)
