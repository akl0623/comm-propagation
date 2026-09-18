import numpy as np
import pandas as pd
import os
from glob import glob
from collections import defaultdict
import re

# 定义360个脑区的顺序（请在此处粘贴完整的regions列表）
regions = ['L_V1_ROI', 'L_MST_ROI', 'L_V6_ROI', 'L_V2_ROI', 'L_V3_ROI', 'L_V4_ROI', 'L_V8_ROI', 'L_4_ROI', 'L_3b_ROI', 'L_FEF_ROI', 'L_PEF_ROI', 'L_55b_ROI', 'L_V3A_ROI', 'L_RSC_ROI', 'L_POS2_ROI', 'L_V7_ROI', 'L_IPS1_ROI', 'L_FFC_ROI', 'L_V3B_ROI', 'L_LO1_ROI', 'L_LO2_ROI', 'L_PIT_ROI', 'L_MT_ROI', 'L_A1_ROI', 'L_PSL_ROI', 'L_SFL_ROI', 'L_PCV_ROI', 'L_STV_ROI', 'L_7Pm_ROI', 'L_7m_ROI', 'L_POS1_ROI', 'L_23d_ROI', 'L_v23ab_ROI', 'L_d23ab_ROI', 'L_31pv_ROI', 'L_5m_ROI', 'L_5mv_ROI', 'L_23c_ROI', 'L_5L_ROI', 'L_24dd_ROI', 'L_24dv_ROI', 'L_7AL_ROI', 'L_SCEF_ROI', 'L_6ma_ROI', 'L_7Am_ROI', 'L_7PL_ROI', 'L_7PC_ROI', 'L_LIPv_ROI', 'L_VIP_ROI', 'L_MIP_ROI', 'L_1_ROI', 'L_2_ROI', 'L_3a_ROI', 'L_6d_ROI', 'L_6mp_ROI', 'L_6v_ROI', 'L_p24pr_ROI', 'L_33pr_ROI', 'L_a24pr_ROI', 'L_p32pr_ROI', 'L_a24_ROI', 'L_d32_ROI', 'L_8BM_ROI', 'L_p32_ROI', 'L_10r_ROI', 'L_47m_ROI', 'L_8Av_ROI', 'L_8Ad_ROI', 'L_9m_ROI', 'L_8BL_ROI', 'L_9p_ROI', 'L_10d_ROI', 'L_8C_ROI', 'L_44_ROI', 'L_45_ROI', 'L_47l_ROI', 'L_a47r_ROI', 'L_6r_ROI', 'L_IFJa_ROI', 'L_IFJp_ROI', 'L_IFSp_ROI', 'L_IFSa_ROI', 'L_p9-46v_ROI', 'L_46_ROI', 'L_a9-46v_ROI', 'L_9-46d_ROI', 'L_9a_ROI', 'L_10v_ROI', 'L_a10p_ROI', 'L_10pp_ROI', 'L_11l_ROI', 'L_13l_ROI', 'L_OFC_ROI', 'L_47s_ROI', 'L_LIPd_ROI', 'L_6a_ROI', 'L_i6-8_ROI', 'L_s6-8_ROI', 'L_43_ROI', 'L_OP4_ROI', 'L_OP1_ROI', 'L_OP2-3_ROI', 'L_52_ROI', 'L_RI_ROI', 'L_PFcm_ROI', 'L_PoI2_ROI', 'L_TA2_ROI', 'L_FOP4_ROI', 'L_MI_ROI', 'L_Pir_ROI', 'L_AVI_ROI', 'L_AAIC_ROI', 'L_FOP1_ROI', 'L_FOP3_ROI', 'L_FOP2_ROI', 'L_PFt_ROI', 'L_AIP_ROI', 'L_EC_ROI', 'L_PreS_ROI', 'L_H_ROI', 'L_ProS_ROI', 'L_PeEc_ROI', 'L_STGa_ROI', 'L_PBelt_ROI', 'L_A5_ROI', 'L_PHA1_ROI', 'L_PHA3_ROI', 'L_STSda_ROI', 'L_STSdp_ROI', 'L_STSvp_ROI', 'L_TGd_ROI', 'L_TE1a_ROI', 'L_TE1p_ROI', 'L_TE2a_ROI', 'L_TF_ROI', 'L_TE2p_ROI', 'L_PHT_ROI', 'L_PH_ROI', 'L_TPOJ1_ROI', 'L_TPOJ2_ROI', 'L_TPOJ3_ROI', 'L_DVT_ROI', 'L_PGp_ROI', 'L_IP2_ROI', 'L_IP1_ROI', 'L_IP0_ROI', 'L_PFop_ROI', 'L_PF_ROI', 'L_PFm_ROI', 'L_PGi_ROI', 'L_PGs_ROI', 'L_V6A_ROI', 'L_VMV1_ROI', 'L_VMV3_ROI', 'L_PHA2_ROI', 'L_V4t_ROI', 'L_FST_ROI', 'L_V3CD_ROI', 'L_LO3_ROI', 'L_VMV2_ROI', 'L_31pd_ROI', 'L_31a_ROI', 'L_VVC_ROI', 'L_25_ROI', 'L_s32_ROI', 'L_pOFC_ROI', 'L_PoI1_ROI', 'L_Ig_ROI', 'L_FOP5_ROI', 'L_p10p_ROI', 'L_p47r_ROI', 'L_TGv_ROI', 'L_MBelt_ROI', 'L_LBelt_ROI', 'L_A4_ROI', 'L_STSva_ROI', 'L_TE1m_ROI', 'L_PI_ROI', 'L_a32pr_ROI', 'L_p24_ROI', 'R_V1_ROI', 'R_MST_ROI', 'R_V6_ROI', 'R_V2_ROI', 'R_V3_ROI', 'R_V4_ROI', 'R_V8_ROI', 'R_4_ROI', 'R_3b_ROI', 'R_FEF_ROI', 'R_PEF_ROI', 'R_55b_ROI', 'R_V3A_ROI', 'R_RSC_ROI', 'R_POS2_ROI', 'R_V7_ROI', 'R_IPS1_ROI', 'R_FFC_ROI', 'R_V3B_ROI', 'R_LO1_ROI', 'R_LO2_ROI', 'R_PIT_ROI', 'R_MT_ROI', 'R_A1_ROI', 'R_PSL_ROI', 'R_SFL_ROI', 'R_PCV_ROI', 'R_STV_ROI', 'R_7Pm_ROI', 'R_7m_ROI', 'R_POS1_ROI', 'R_23d_ROI', 'R_v23ab_ROI', 'R_d23ab_ROI', 'R_31pv_ROI', 'R_5m_ROI', 'R_5mv_ROI', 'R_23c_ROI', 'R_5L_ROI', 'R_24dd_ROI', 'R_24dv_ROI', 'R_7AL_ROI', 'R_SCEF_ROI', 'R_6ma_ROI', 'R_7Am_ROI', 'R_7PL_ROI', 'R_7PC_ROI', 'R_LIPv_ROI', 'R_VIP_ROI', 'R_MIP_ROI', 'R_1_ROI', 'R_2_ROI', 'R_3a_ROI', 'R_6d_ROI', 'R_6mp_ROI', 'R_6v_ROI', 'R_p24pr_ROI', 'R_33pr_ROI', 'R_a24pr_ROI', 'R_p32pr_ROI', 'R_a24_ROI', 'R_d32_ROI', 'R_8BM_ROI', 'R_p32_ROI', 'R_10r_ROI', 'R_47m_ROI', 'R_8Av_ROI', 'R_8Ad_ROI', 'R_9m_ROI', 'R_8BL_ROI', 'R_9p_ROI', 'R_10d_ROI', 'R_8C_ROI', 'R_44_ROI', 'R_45_ROI', 'R_47l_ROI', 'R_a47r_ROI', 'R_6r_ROI', 'R_IFJa_ROI', 'R_IFJp_ROI', 'R_IFSp_ROI', 'R_IFSa_ROI', 'R_p9-46v_ROI', 'R_46_ROI', 'R_a9-46v_ROI', 'R_9-46d_ROI', 'R_9a_ROI', 'R_10v_ROI', 'R_a10p_ROI', 'R_10pp_ROI', 'R_11l_ROI', 'R_13l_ROI', 'R_OFC_ROI', 'R_47s_ROI', 'R_LIPd_ROI', 'R_6a_ROI', 'R_i6-8_ROI', 'R_s6-8_ROI', 'R_43_ROI', 'R_OP4_ROI', 'R_OP1_ROI', 'R_OP2-3_ROI', 'R_52_ROI', 'R_RI_ROI', 'R_PFcm_ROI', 'R_PoI2_ROI', 'R_TA2_ROI', 'R_FOP4_ROI', 'R_MI_ROI', 'R_Pir_ROI', 'R_AVI_ROI', 'R_AAIC_ROI', 'R_FOP1_ROI', 'R_FOP3_ROI', 'R_FOP2_ROI', 'R_PFt_ROI', 'R_AIP_ROI', 'R_EC_ROI', 'R_PreS_ROI', 'R_H_ROI', 'R_ProS_ROI', 'R_PeEc_ROI', 'R_STGa_ROI', 'R_PBelt_ROI', 'R_A5_ROI', 'R_PHA1_ROI', 'R_PHA3_ROI', 'R_STSda_ROI', 'R_STSdp_ROI', 'R_STSvp_ROI', 'R_TGd_ROI', 'R_TE1a_ROI', 'R_TE1p_ROI', 'R_TE2a_ROI', 'R_TF_ROI', 'R_TE2p_ROI', 'R_PHT_ROI', 'R_PH_ROI', 'R_TPOJ1_ROI', 'R_TPOJ2_ROI', 'R_TPOJ3_ROI', 'R_DVT_ROI', 'R_PGp_ROI', 'R_IP2_ROI', 'R_IP1_ROI', 'R_IP0_ROI', 'R_PFop_ROI', 'R_PF_ROI', 'R_PFm_ROI', 'R_PGi_ROI', 'R_PGs_ROI', 'R_V6A_ROI', 'R_VMV1_ROI', 'R_VMV3_ROI', 'R_PHA2_ROI', 'R_V4t_ROI', 'R_FST_ROI', 'R_V3CD_ROI', 'R_LO3_ROI', 'R_VMV2_ROI', 'R_31pd_ROI', 'R_31a_ROI', 'R_VVC_ROI', 'R_25_ROI', 'R_s32_ROI', 'R_pOFC_ROI', 'R_PoI1_ROI', 'R_Ig_ROI', 'R_FOP5_ROI', 'R_p10p_ROI', 'R_p47r_ROI', 'R_TGv_ROI', 'R_MBelt_ROI', 'R_LBelt_ROI', 'R_A4_ROI', 'R_STSva_ROI', 'R_TE1m_ROI', 'R_PI_ROI', 'R_a32pr_ROI', 'R_p24_ROI']  # 完整列表

# 创建脑区名称到索引的映射
region_to_idx = {region: idx for idx, region in enumerate(regions)}

# 定义时间轴和响应期
time = np.linspace(-300, 700, 1001)
baseline_mask = (time >= -300) & (time <= -50)
response_mask = (time >= 0) & (time <= 100)  

def process_ccep_per_trial(data):
    """对每个trial和每个通道单独处理，不进行trial平均"""
    n_trials, n_channels, n_timepoints = data.shape
    
    z_scores = np.zeros_like(data)
    for trial in range(n_trials):
        for channel in range(n_channels):
            baseline_data = data[trial, channel, baseline_mask]
            baseline_mean = np.mean(baseline_data)
            baseline_std = np.std(baseline_data)
            
            if baseline_std > 0:
                z_scores[trial, channel, :] = (data[trial, channel, :] - baseline_mean) / baseline_std
            else:
                z_scores[trial, channel, :] = 0
    
    return z_scores

def identify_ccep_per_trial(z_scores, threshold=5):
    """在每个trial中识别CCEP特征"""
    n_trials, n_channels, n_timepoints = z_scores.shape
    all_ccep_features = []
    
    for trial in range(n_trials):
        trial_features = []
        for channel in range(n_channels):
            signal = z_scores[trial, channel, :]
            abs_signal = np.abs(signal)
            
            response_indices = np.where(response_mask)[0]
            response_signal = abs_signal[response_mask]
            
            above_threshold = response_signal > threshold
            
            if np.any(above_threshold):
                first_above_idx = np.argmax(above_threshold)
                global_idx = response_indices[first_above_idx]
                
                onset_delay = time[global_idx]
                
                peak_idx_in_response = np.argmax(response_signal[first_above_idx:]) + first_above_idx
                peak_delay = time[response_indices[peak_idx_in_response]]
                
                features = {
                    'trial': trial,
                    'channel': channel,
                    'onset_delay_ms': onset_delay,
                    'peak_delay_ms': peak_delay,
                    'peak_amplitude': signal[response_indices[peak_idx_in_response]],
                    'is_significant': True
                }
                
                trial_features.append(features)
            else:
                trial_features.append({
                    'trial': trial,
                    'channel': channel,
                    'is_significant': False
                })
        
        all_ccep_features.append(trial_features)
    
    return all_ccep_features

def get_stimulated_region(subject_id, run_number, electrodes_df):
    """获取刺激脑区"""
    subject_pattern = f"{subject_id}_task"
    electrode_info = electrodes_df[electrodes_df['subject_id'].str.startswith(subject_pattern)]
    
    if len(electrode_info) == 0:
        print(f"Warning: No electrode info found for subject {subject_id}")
        return None
    
    try:
        run_num = int(run_number)
        run_info = electrode_info[electrode_info['run_number'] == run_num]
        
        if len(run_info) == 0:
            run_info = electrode_info.iloc[[0]]
            print(f"Warning: No exact run match for {subject_id} run {run_number}, using first run")
    except ValueError:
        run_info = electrode_info.iloc[[0]]
        print(f"Warning: Invalid run number {run_number} for {subject_id}, using first run")
    
    if len(run_info) == 0:
        print(f"Warning: No electrode info found for {subject_id} run {run_number}")
        return None
    
    electrode_name = run_info.iloc[0]['electrode']
    print(f"Found electrode {electrode_name} for subject {subject_id} run {run_number}")
    return electrode_name

def get_region_mapping(mapping_file):
    """从映射文件中获取电极到脑区的映射关系"""
    mapping_df = pd.read_csv(mapping_file)
    electrode_to_region = {}
    
    for _, row in mapping_df.iterrows():
        electrode = row['electrode']
        parcel = row['parcel']
        electrode_to_region[electrode] = parcel
    
    return electrode_to_region

def process_all_experiments():
    """处理所有实验数据并构建360×360矩阵"""
    
    # 初始化结果矩阵
    response_probability = np.full((360, 360), np.nan)
    median_onset_time = np.full((360, 360), np.nan)
    
    # 用于累积统计的字典
    # 使用三层嵌套的defaultdict来存储：stimulated_idx -> observed_idx -> {'total_trials': int, 'onset_times': list}
    stimulation_stats = defaultdict(lambda: defaultdict(lambda: {'total_trials': 0, 'onset_times': []}))
    
    # 读取电极信息
    electrodes_df = pd.read_csv('./subject_electrodes.csv')
    
    # 找到所有数据文件
    data_files = glob('./output/sub-*_task-ccepcoreg_run-*_parcel_seeg_data.npy')
    
    print(f"找到 {len(data_files)} 个数据文件")
    
    processed_count = 0
    
    for data_file in data_files:
        filename = os.path.basename(data_file)
        
        match = re.match(r'sub-(\d+)_task-ccepcoreg_run-(\d+)_parcel_seeg_data\.npy', filename)
        if not match:
            print(f"Warning: 文件名格式不匹配: {filename}")
            continue
            
        subject_id = match.group(1)
        run_number = match.group(2)
        
        print(f"处理: subject {subject_id}, run {run_number}")
        
        # 获取刺激脑区
        stimulated_electrode = get_stimulated_region(subject_id, run_number, electrodes_df)
        if stimulated_electrode is None:
            continue
        
        # 获取映射文件
        mapping_file = f"./output/sub-{subject_id}_task-ccepcoreg_run-{run_number}_electrode_to_parcel_mapping.csv"
        if not os.path.exists(mapping_file):
            alt_mapping_file = f"./output/sub-{subject_id}_task-ccepcoreg_run-01_electrode_to_parcel_mapping.csv"
            if os.path.exists(alt_mapping_file):
                mapping_file = alt_mapping_file
                print(f"Using alternative mapping file: {alt_mapping_file}")
            else:
                print(f"Warning: No mapping file found for subject {subject_id}")
                continue
        
        # 获取电极到脑区的映射
        electrode_to_region = get_region_mapping(mapping_file)
        
        # 获取刺激脑区名称
        stimulated_region = electrode_to_region.get(stimulated_electrode)
        if stimulated_region is None:
            print(f"Warning: No region mapping found for electrode {stimulated_electrode}")
            continue
        
        # 获取刺激脑区索引
        if stimulated_region not in region_to_idx:
            print(f"Warning: Stimulated region {stimulated_region} not in regions list")
            continue
        
        stimulated_idx = region_to_idx[stimulated_region]
        
        # 加载数据
        try:
            data = np.load(data_file)
        except Exception as e:
            print(f"Error loading data file {data_file}: {e}")
            continue
        
        # 获取活跃脑区列表
        active_parcels_file = f"./output/sub-{subject_id}_task-ccepcoreg_run-{run_number}_active_parcels.csv"
        if not os.path.exists(active_parcels_file):
            print(f"Warning: No active parcels file found for {filename}")
            continue
        
        active_parcels_df = pd.read_csv(active_parcels_file)
        active_regions = active_parcels_df['parcel_name'].tolist()
        
        # 处理数据
        z_scores = process_ccep_per_trial(data)
        all_ccep_features = identify_ccep_per_trial(z_scores)
        
        # 统计每个观测脑区的响应
        n_trials = data.shape[0]
        
        for channel_idx, region_name in enumerate(active_regions):
            if region_name not in region_to_idx:
                continue
                
            observed_idx = region_to_idx[region_name]
            
            # 更新总试验次数
            stimulation_stats[stimulated_idx][observed_idx]['total_trials'] += n_trials
            
            # 统计该脑区在所有trial中的响应
            for trial in range(n_trials):
                if channel_idx < len(all_ccep_features[trial]):
                    feature = all_ccep_features[trial][channel_idx]
                    if feature['is_significant']:
                        stimulation_stats[stimulated_idx][observed_idx]['onset_times'].append(
                            feature['onset_delay_ms']
                        )
        
        processed_count += 1
        print(f"成功处理 {processed_count} 个文件")
    
    # 计算最终矩阵
    print("计算最终矩阵...")
    
    for stim_idx in range(360):
        for obs_idx in range(360):
            stats = stimulation_stats[stim_idx][obs_idx]
            total_trials = stats['total_trials']
            onset_times = stats['onset_times']
            
            if total_trials > 0:
                # 计算响应概率
                response_prob = len(onset_times) / total_trials
                response_probability[stim_idx, obs_idx] = response_prob
                
                # 计算起始时间中位数
                if onset_times:
                    median_onset_time[stim_idx, obs_idx] = np.median(onset_times)
                else:
                    median_onset_time[stim_idx, obs_idx] = np.nan
            else:
                # 没有试验数据
                response_probability[stim_idx, obs_idx] = np.nan
                median_onset_time[stim_idx, obs_idx] = np.nan
    
    return response_probability, median_onset_time, processed_count

# 执行处理
print("开始处理所有SEEG数据...")
response_prob_matrix, onset_time_matrix, processed_count = process_all_experiments()

print(f"成功处理了 {processed_count} 个文件")

# 保存矩阵为txt文件
print("保存结果矩阵...")

# 保存响应概率矩阵
with open('./response_probability_matrix_251114.txt', 'w') as f:
    for i in range(360):
        row = []
        for j in range(360):
            if np.isnan(response_prob_matrix[i, j]):
                row.append('nan')
            else:
                row.append(f'{response_prob_matrix[i, j]:.6f}')
        f.write(' '.join(row) + '\n')

# 保存起始时间中位数矩阵
with open('./median_onset_time_matrix_251114.txt', 'w') as f:
    for i in range(360):
        row = []
        for j in range(360):
            if np.isnan(onset_time_matrix[i, j]):
                row.append('nan')
            else:
                row.append(f'{onset_time_matrix[i, j]:.6f}')
        f.write(' '.join(row) + '\n')

print("处理完成！")
print(f"响应概率矩阵已保存到: ./response_probability_matrix_251114.txt")
print(f"起始时间中位数矩阵已保存到: ./median_onset_time_matrix_251114.txt")

# 输出一些统计信息
valid_responses = np.sum(~np.isnan(response_prob_matrix))
print(f"\n统计信息:")
print(f"矩阵中有有效值的元素数量: {valid_responses}/129600")
if valid_responses > 0:
    print(f"响应概率范围: [{np.nanmin(response_prob_matrix):.3f}, {np.nanmax(response_prob_matrix):.3f}]")
    valid_onset_times = onset_time_matrix[~np.isnan(onset_time_matrix)]
    if len(valid_onset_times) > 0:
        print(f"起始时间中位数范围: [{np.nanmin(onset_time_matrix):.1f}, {np.nanmax(onset_time_matrix):.1f}] ms")
    else:
        print("没有有效的起始时间数据")
else:
    print("没有有效的响应数据")
