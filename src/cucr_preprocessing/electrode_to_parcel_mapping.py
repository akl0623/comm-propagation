import numpy as np
import pandas as pd
from nibabel.freesurfer import read_annot, read_geometry
from scipy.spatial.distance import cdist
import glob
import os

FREESURFER_SUBJECTS_DIR = os.environ.get('FREESURFER_SUBJECTS_DIR', '')
if not FREESURFER_SUBJECTS_DIR:
    raise EnvironmentError(
        "FREESURFER_SUBJECTS_DIR environment variable must be set to the FreeSurfer "
        "subjects directory (e.g. the directory containing sub-XX_recon). "
        "See src/cucr_preprocessing/README.md."
    )

def compute_face_areas(verts, faces):
    """计算每个三角面片的面积"""
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    vec1 = v1 - v0
    vec2 = v2 - v0
    cross_prod = np.cross(vec1, vec2)
    return 0.5 * np.linalg.norm(cross_prod, axis=1)

def compute_vertex_areas(verts, faces):
    """计算每个顶点的面积权重"""
    n_verts = len(verts)
    face_areas = compute_face_areas(verts, faces)
    vertex_areas = np.zeros(n_verts)
    
    for i, face in enumerate(faces):
        area = face_areas[i] / 3.0  # 均分到三个顶点
        vertex_areas[face[0]] += area
        vertex_areas[face[1]] += area
        vertex_areas[face[2]] += area
        
    return vertex_areas

def compute_parcel_centroid(parcel_data, vertex_areas):
    """计算脑区的面积加权质心"""
    parcel_verts = parcel_data['coords']
    parcel_vert_indices = parcel_data['indices']
    
    # 获取该脑区顶点的面积权重
    parcel_areas = vertex_areas[parcel_vert_indices]
    
    # 面积加权平均计算质心
    if np.sum(parcel_areas) > 0:
        centroid = np.average(parcel_verts, axis=0, weights=parcel_areas)
    else:
        centroid = np.mean(parcel_verts, axis=0)
    
    return centroid

def load_parcel_vertices(annot_path, surf_path):
    """加载脑区顶点坐标并计算面积加权质心"""
    labels, ctab, names = read_annot(annot_path)
    verts, faces = read_geometry(surf_path)
    
    # 计算整个皮层的顶点面积
    vertex_areas = compute_vertex_areas(verts, faces)
    
    # 提取有效脑区名称（跳过第一个非脑区条目）
    parcel_names = [name.decode('utf-8') for name in names[1:]]  # 转换为普通字符串
    
    parcel_data = {}
    for i, name in enumerate(parcel_names, start=1):  # 索引从1开始
        # 获取属于该脑区的顶点索引（整数数组）
        vert_indices = np.where(labels == i)[0].astype(int)
        parcel_verts = verts[vert_indices]
        
        # 计算面积加权质心
        centroid = compute_parcel_centroid({
            'coords': parcel_verts, 
            'indices': vert_indices
        }, vertex_areas)
        
        parcel_data[name] = {
            'indices': vert_indices,
            'coords': parcel_verts,
            'centroid': centroid,  # 添加质心坐标
            'area': np.sum(vertex_areas[vert_indices])  # 脑区总面积
        }
    return parcel_names, parcel_data

def find_closest_parcel(elec_pos, all_parcels):
    """找到距离电极质心最近的脑区"""
    min_distance = float('inf')
    closest_parcel = None
    
    for parcel_name, parcel_data in all_parcels.items():
        # 计算电极到脑区质心的距离
        distance = np.linalg.norm(elec_pos - parcel_data['centroid'])
        
        if distance < min_distance:
            min_distance = distance
            closest_parcel = parcel_name
    
    return closest_parcel, min_distance

def process_single_run(run_number, electrodes, electrode_names, all_parcels, parcel_electrodes, electrode_to_parcel):
    """处理单个run的数据"""
    print(f"\n开始处理 run-{run_number:02d}...")
    
    # 构建文件名
    epochs_file = f'sub-01_task-ccepcoreg_run-{run_number:02d}_epochs.npy'
    channels_file = f'sub-01_task-ccepcoreg_run-{run_number:02d}_channels.tsv'
    
    # 检查文件是否存在
    if not os.path.exists(epochs_file):
        print(f"警告: 文件 {epochs_file} 不存在，跳过该run")
        return None
    if not os.path.exists(channels_file):
        print(f"警告: 文件 {channels_file} 不存在，跳过该run")
        return None
    
    # 读取SEEG数据和通道信息
    print(f"读取SEEG数据和通道信息...")
    seeg_data = np.load(epochs_file)
    channels_data = pd.read_csv(channels_file, sep='\t')
    
    # 处理通道名称：去除单引号
    channels_data['name'] = channels_data['name'].str.replace("'", "")
    
    # 只保留状态为good的通道
    good_channels = channels_data[channels_data['status'] == 'good']['name'].values
    print(f"总共 {len(channels_data)} 个通道，其中 {len(good_channels)} 个状态为good")
    
    # 创建电极名称到数据索引的映射
    channel_to_index = {name: idx for idx, name in enumerate(channels_data['name'])}
    
    # 构建脑区SEEG数据
    print("构建脑区SEEG数据...")
    
    # 找出所有有电极的脑区
    active_parcels = set()
    for elec_name in good_channels:
        if elec_name in electrode_to_parcel:
            active_parcels.add(electrode_to_parcel[elec_name]['parcel'])
    
    active_parcels = sorted(list(active_parcels))
    print(f"有信号的脑区数量: {len(active_parcels)}")
    
    # 创建脑区名称到输出索引的映射
    parcel_to_output_index = {parcel: idx for idx, parcel in enumerate(active_parcels)}
    
    # 初始化脑区SEEG数据
    n_sessions, n_channels, n_timepoints = seeg_data.shape
    parcel_seeg_data = np.zeros((n_sessions, len(active_parcels), n_timepoints))
    
    # 对每个脑区，平均分配到该脑区的所有good电极的信号
    for parcel in active_parcels:
        parcel_electrodes_list = parcel_electrodes.get(parcel, [])
        good_parcel_electrodes = [elec for elec in parcel_electrodes_list if elec in good_channels]
        
        if not good_parcel_electrodes:
            continue
            
        parcel_idx = parcel_to_output_index[parcel]
        
        # 收集该脑区所有good电极的信号
        electrode_signals = []
        for elec_name in good_parcel_electrodes:
            if elec_name in channel_to_index:
                elec_idx = channel_to_index[elec_name]
                electrode_signals.append(seeg_data[:, elec_idx, :])
        
        if electrode_signals:
            # 对电极信号进行平均
            avg_signal = np.mean(electrode_signals, axis=0)
            parcel_seeg_data[:, parcel_idx, :] = avg_signal
    
    print(f"脑区SEEG数据形状: {parcel_seeg_data.shape}")
    
    # 保存脑区SEEG数据
    output_filename = f'sub-01_task-ccepcoreg_run-{run_number:02d}_parcel_seeg_data.npy'
    np.save(output_filename, parcel_seeg_data)
    print(f"脑区SEEG数据已保存到 {output_filename}")
    
    # 保存该run的活跃脑区列表（包含脑区面积信息）
    active_parcels_data = []
    for parcel in active_parcels:
        active_parcels_data.append({
            'parcel_name': parcel,
            'area_mm2': all_parcels[parcel]['area'],
            'centroid_x': all_parcels[parcel]['centroid'][0],
            'centroid_y': all_parcels[parcel]['centroid'][1],
            'centroid_z': all_parcels[parcel]['centroid'][2],
            'electrode_count': len([elec for elec in parcel_electrodes.get(parcel, []) if elec in good_channels])
        })
    
    active_parcels_df = pd.DataFrame(active_parcels_data)
    active_parcels_filename = f'sub-01_task-ccepcoreg_run-{run_number:02d}_active_parcels.csv'
    active_parcels_df.to_csv(active_parcels_filename, index=False)
    print(f"活跃脑区列表已保存到 {active_parcels_filename}")
    
    # 输出统计信息
    print(f"\n=== run-{run_number:02d} 统计信息 ===")
    print(f"有信号的脑区数: {len(active_parcels)}")
    print(f"Good通道数: {len(good_channels)}")
    print(f"输出数据维度: {parcel_seeg_data.shape}")
    
    return parcel_seeg_data.shape

# 主程序开始
def main():
    # 读取电极坐标和名称
    electrode_data = pd.read_csv(
        'sub-01_task-ccepcoreg_space-MNI152NLin2009aSym_electrodes_cleaned.tsv',
        sep='\t'
    )
    
    # 处理电极名称：去除单引号
    electrode_data['name'] = electrode_data['name'].str.replace("'", "")
    electrode_names = electrode_data['name'].values
    electrodes = electrode_data[['x', 'y', 'z']].values * 1000  # 转换为毫米
    
    # 加载左右半球脑区
    print("加载左半球脑区数据...")
    lh_names, lh_parcels = load_parcel_vertices(
        os.path.join(FREESURFER_SUBJECTS_DIR, 'sub-01_recon', 'label', 'lh.HCP-MMP1_on_MNI152_ICBM2009a_nlin.annot'),
        os.path.join(FREESURFER_SUBJECTS_DIR, 'sub-01_recon', 'surf', 'lh.white')
    )
    
    print("加载右半球脑区数据...")
    rh_names, rh_parcels = load_parcel_vertices(
        os.path.join(FREESURFER_SUBJECTS_DIR, 'sub-01_recon', 'label', 'rh.HCP-MMP1_on_MNI152_ICBM2009a_nlin.annot'),
        os.path.join(FREESURFER_SUBJECTS_DIR, 'sub-01_recon', 'surf', 'rh.white')
    )
    
    # 合并脑区信息
    all_parcels = {**lh_parcels, **rh_parcels}
    parcel_names = lh_names + rh_names  # 完整的脑区名称列表
    
    print(f"总共加载了 {len(all_parcels)} 个脑区")
    print(f"总共加载了 {len(electrodes)} 个电极")
    
    # 将每个电极分配到最近的脑区（基于质心距离）
    print("开始将电极分配到脑区（基于质心距离）...")
    electrode_to_parcel = {}
    parcel_electrodes = {}
    
    for i, (elec_name, elec_pos) in enumerate(zip(electrode_names, electrodes)):
        closest_parcel, distance = find_closest_parcel(elec_pos, all_parcels)
        electrode_to_parcel[elec_name] = {
            'parcel': closest_parcel,
            'distance': distance,
            'parcel_centroid': all_parcels[closest_parcel]['centroid'],
            'parcel_area': all_parcels[closest_parcel]['area']
        }
        
        # 记录分配到每个脑区的电极
        if closest_parcel not in parcel_electrodes:
            parcel_electrodes[closest_parcel] = []
        parcel_electrodes[closest_parcel].append(elec_name)
        
        if i % 10 == 0:  # 每10个电极打印一次进度
            print(f"已处理 {i+1}/{len(electrodes)} 个电极")
    
    print("电极分配完成!")
    
    # 保存电极到脑区的映射（包含更多信息）
    mapping_data = []
    for elec_name in electrode_names:
        if elec_name in electrode_to_parcel:
            mapping_info = electrode_to_parcel[elec_name]
            mapping_data.append({
                'electrode': elec_name,
                'parcel': mapping_info['parcel'],
                'distance_to_centroid_mm': mapping_info['distance'],
                'parcel_area_mm2': mapping_info['parcel_area'],
                'parcel_centroid_x': mapping_info['parcel_centroid'][0],
                'parcel_centroid_y': mapping_info['parcel_centroid'][1],
                'parcel_centroid_z': mapping_info['parcel_centroid'][2]
            })
    
    mapping_df = pd.DataFrame(mapping_data)
    mapping_filename = 'sub-01_electrode_to_parcel_mapping.csv'
    mapping_df.to_csv(mapping_filename, index=False)
    print(f"电极到脑区映射已保存到 {mapping_filename}")
    
    # 自动检测并处理所有run
    print("\n开始自动检测并处理所有run...")
    
    # 查找所有epochs文件
    epochs_files = glob.glob('sub-01_task-ccepcoreg_run-*_epochs.npy')
    run_numbers = []
    
    for file in epochs_files:
        # 从文件名中提取run编号
        try:
            run_num = int(file.split('run-')[1].split('_')[0])
            run_numbers.append(run_num)
        except (IndexError, ValueError):
            continue
    
    if not run_numbers:
        print("未找到任何run文件，请检查文件命名格式")
        return
    
    run_numbers.sort()
    print(f"找到 {len(run_numbers)} 个run: {run_numbers}")
    
    # 处理每个run
    all_run_stats = []
    for run_num in run_numbers:
        result = process_single_run(run_num, electrodes, electrode_names, all_parcels, parcel_electrodes, electrode_to_parcel)
        if result is not None:
            all_run_stats.append({
                'run': run_num,
                'data_shape': result
            })
    
    # 输出总体统计信息
    print("\n=== 所有run处理完成 ===")
    print(f"成功处理的run数量: {len(all_run_stats)}")
    for stat in all_run_stats:
        print(f"run-{stat['run']:02d}: 输出数据形状 {stat['data_shape']}")
    
    print("所有处理完成!")

if __name__ == "__main__":
    main()
