import numpy as np
from scipy import stats

# 假设数据形状: trials x channels x timepoints
data = np.load('./output/sub-01_task-ccepcoreg_run-02_parcel_seeg_data.npy')  # shape: (40, 43, 1001)

# 定义时间轴 (假设采样率1kHz，-300ms到700ms)
time = np.linspace(-300, 700, 1001)
baseline_mask = (time >= -300) & (time <= -50)  # 基线期[-300, -50]ms
response_mask = (time >= 0) & (time <= 50)     # 响应期[0, 500]ms

def process_ccep_per_trial(data):
    """
    对每个trial和每个通道单独处理，不进行trial平均
    """
    n_trials, n_channels, n_timepoints = data.shape
    
    # 对每个trial和通道进行Z-score标准化
    z_scores = np.zeros_like(data)
    for trial in range(n_trials):
        for channel in range(n_channels):
            baseline_data = data[trial, channel, baseline_mask]
            baseline_mean = np.mean(baseline_data)
            baseline_std = np.std(baseline_data)
            
            # 避免除零
            if baseline_std > 0:
                z_scores[trial, channel, :] = (data[trial, channel, :] - baseline_mean) / baseline_std
            else:
                z_scores[trial, channel, :] = 0
    
    return z_scores

def identify_ccep_per_trial(z_scores, threshold=5):
    """
    在每个trial中识别CCEP特征
    """
    n_trials, n_channels, n_timepoints = z_scores.shape
    all_ccep_features = []
    
    for trial in range(n_trials):
        trial_features = []
        for channel in range(n_channels):
            signal = z_scores[trial, channel, :]
            abs_signal = np.abs(signal)
            
            # 在响应期内寻找显著响应
            response_indices = np.where(response_mask)[0]
            response_signal = abs_signal[response_mask]
            
            # 寻找超过阈值的位置
            above_threshold = response_signal > threshold
            
            if np.any(above_threshold):
                # 找到第一个超过阈值的位置
                first_above_idx = np.argmax(above_threshold)
                global_idx = response_indices[first_above_idx]
                
                # 计算延迟时间
                onset_delay = time[global_idx]  # 起始延迟
                
                # 找到峰值位置
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

def summarize_ccep_by_channel(all_ccep_features, n_channels):
    """
    按脑区汇总CCEP统计信息
    """
    channel_stats = []
    
    for channel in range(n_channels):
        # 收集该脑区在所有trial中的显著CCEPs
        channel_cceps = []
        for trial_features in all_ccep_features:
            feature = trial_features[channel]
            if feature['is_significant']:
                channel_cceps.append(feature)
        
        # 统计信息
        n_significant = len(channel_cceps)
        
        if n_significant > 0:
            onset_times = [ccep['onset_delay_ms'] for ccep in channel_cceps]
            peak_times = [ccep['peak_delay_ms'] for ccep in channel_cceps]
            amplitudes = [ccep['peak_amplitude'] for ccep in channel_cceps]
            
            stats_info = {
                'channel': channel,
                'n_significant_trials': n_significant,
                'onset_times': onset_times,
                'peak_times': peak_times,
                'amplitudes': amplitudes,
                'mean_onset_ms': np.mean(onset_times),
                'std_onset_ms': np.std(onset_times),
                'mean_peak_ms': np.mean(peak_times),
                'std_peak_ms': np.std(peak_times),
                'mean_amplitude': np.mean(amplitudes)
            }
        else:
            stats_info = {
                'channel': channel,
                'n_significant_trials': 0,
                'onset_times': [],
                'peak_times': [],
                'amplitudes': [],
                'mean_onset_ms': None,
                'std_onset_ms': None,
                'mean_peak_ms': None,
                'std_peak_ms': None,
                'mean_amplitude': None
            }
        
        channel_stats.append(stats_info)
    
    return channel_stats

# 执行处理
z_scores = process_ccep_per_trial(data)
all_ccep_features = identify_ccep_per_trial(z_scores)
channel_stats = summarize_ccep_by_channel(all_ccep_features, data.shape[1])

# 输出结果
print(f"总trial数: {data.shape[0]}")
print(f"总脑区数: {data.shape[1]}")
print("\n各脑区CCEP统计:")

for stats in channel_stats:
    if stats['n_significant_trials'] > 0:
        print(f"脑区 {stats['channel']:2d}: {stats['n_significant_trials']:2d}个trial有CCEP, "
              f"起始延迟: {stats['mean_onset_ms']:6.1f} ± {stats['std_onset_ms']:5.1f} ms, "
              f"峰值延迟: {stats['mean_peak_ms']:6.1f} ± {stats['std_peak_ms']:5.1f} ms")

# 总体统计
total_significant = sum(stats['n_significant_trials'] for stats in channel_stats)
max_channel = max(channel_stats, key=lambda x: x['n_significant_trials'])
print(f"\n总体统计:")
print(f"所有脑区总共检测到 {total_significant} 个CCEP事件")
print(f"最活跃脑区: {max_channel['channel']} (在{max_channel['n_significant_trials']}个trial中检测到CCEP)")

# 可选: 保存详细结果
import pandas as pd
# 创建详细记录表格
detailed_records = []
for trial_features in all_ccep_features:
    for feature in trial_features:
        if feature['is_significant']:
            detailed_records.append(feature)

if detailed_records:
    df_detailed = pd.DataFrame(detailed_records)
    df_detailed.to_csv('./output/ccep_detailed_records.csv', index=False)
    print(f"\n详细记录已保存到: ./output/ccep_detailed_records.csv")

# 创建脑区统计表格
df_channel_stats = pd.DataFrame(channel_stats)
df_channel_stats.to_csv('./output/ccep_channel_statistics.csv', index=False)
print(f"脑区统计已保存到: ./output/ccep_channel_statistics.csv")
