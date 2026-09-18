import numpy as np
import pandas as pd
import nibabel as nib
# --------------------------
# 输入输出文件路径
# --------------------------
input_tsv = 'sub-01_task-ccepcoreg_space-MNI152NLin2009aSym_electrodes.tsv'  # 输入的 RAS 坐标系电极坐标
output_tsv = 'sub-01_task-ccepcoreg_space-LIA_electrodes.tsv'             # 输出的 LIA 坐标系电极坐标
t1w_nii_path = 'sub-01_T1w.nii'                                              # 受试者 T1w 结构像
# --------------------------
# 步骤 1: 读取 T1w 的仿射矩阵
# --------------------------
t1_img = nib.load(t1w_nii_path)
t1_affine = t1_img.affine  # 体素坐标 → LIA 世界坐标的变换矩阵
# --------------------------
# 步骤 2: 读取电极坐标（假设为 RAS 世界坐标系）
# --------------------------
df = pd.read_csv(input_tsv, sep='\t')
ras_coords = df[['x', 'y', 'z']].values  # (n_electrodes, 3)
# --------------------------
# 步骤 3: 将 RAS 世界坐标 → T1w 体素坐标 → LIA 世界坐标
# --------------------------
# RAS 世界坐标 → T1w 体素坐标
voxel_coords = nib.affines.apply_affine(np.linalg.inv(t1_affine), ras_coords)
# T1w 体素坐标 → LIA 世界坐标（验证对齐）
lia_coords = nib.affines.apply_affine(t1_affine, voxel_coords)
# --------------------------
# 步骤 4: 保存 LIA 世界坐标
# --------------------------
df['x'] = lia_coords[:, 0]
df['y'] = lia_coords[:, 1]
df['z'] = lia_coords[:, 2]
df.to_csv(output_tsv, sep='\t', index=False)


# # --------------------------
# # 步骤 4: 保存 T1w 体素坐标
# # --------------------------
# df['x'] = voxel_coords[:, 0]
# df['y'] = voxel_coords[:, 1]
# df['z'] = voxel_coords[:, 2]
# df.to_csv(output_tsv, sep='\t', index=False)
