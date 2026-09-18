import numpy as np
import pandas as pd
import mne
import bct
from bct import rout_efficiency, diffusion_efficiency, search_information
from scipy.linalg import expm

def navigation_efficiency(L, D, max_hops=None):
    """
    Navigation of connectivity length matrix L guided by nodal distance D
    
    Parameters:
    L : numpy.ndarray
        Weighted/unweighted directed/undirected NxN SC matrix of connection *lengths*
        L(i,j) is the strength-to-length remapping of the connection weight between i and j.
        L(i,j) = 0 denotes the lack of a connection between i and j.
        
    D : numpy.ndarray
        Symmetric NxN nodal distance matrix (e.g., Euclidean distance between node centroids)
        
    max_hops : int, optional
        Limits the maximum number of hops of navigation paths. Default is number of nodes.
    
    Returns:
    sr : float
        Success ratio - proportion of node pairs successfully reached by navigation.
    
    PL_bin : numpy.ndarray
        NxN matrix of binary navigation path length (number of hops). 
        Inf values indicate failed navigation paths.
    
    PL_wei : numpy.ndarray
        NxN matrix of weighted navigation path length (sum of connection weights along path). 
        Inf values indicate failed paths.
    
    PL_dis : numpy.ndarray
        NxN matrix of distance-based navigation path length (sum of D distances along path). 
        Inf values indicate failed paths.
    
    paths : list of lists
        NxN list of lists containing nodes comprising navigation paths.
    
    Reference: Seguin et al. (2018) PNAS.
    """
    N = L.shape[0]
    if max_hops is None:
        max_hops = N
    
    # Initialize output matrices
    PL_bin = np.zeros((N, N))
    PL_wei = np.zeros((N, N))
    PL_dis = np.zeros((N, N))
    paths = [[[] for _ in range(N)] for _ in range(N)]
    
    # Initialize navigation efficiency matrix with zeros
    E = np.zeros((N, N))
    
    # Counter for successful paths
    successful_pairs = 0
    
    for i in range(N):
        for j in range(N):
            if i != j:
                curr_node = i
                last_node = curr_node
                target = j
                path = [curr_node]
                pl_bin = 0
                pl_wei = 0
                pl_dis = 0
                success = True
                
                while curr_node != target:
                    # Find neighbors (non-zero connections)
                    neighbors = np.where(L[curr_node, :] != 0)[0]
                    
                    # If no neighbors, navigation fails
                    if len(neighbors) == 0:
                        success = False
                        break
                    
                    # Find neighbor closest to target
                    neighbor_dists = D[j, neighbors]
                    min_index = np.argmin(neighbor_dists)
                    next_node = neighbors[min_index]
                    
                    # Check for backtracking or max hops exceeded
                    if next_node == last_node or pl_bin >= max_hops:
                        success = False
                        break
                    
                    # Update path and metrics
                    path.append(next_node)
                    pl_bin += 1
                    pl_wei += L[curr_node, next_node]
                    pl_dis += D[curr_node, next_node]
                    
                    # Move to next node
                    last_node = curr_node
                    curr_node = next_node
                
                if success and curr_node == target:
                    PL_bin[i, j] = pl_bin
                    PL_wei[i, j] = pl_wei
                    PL_dis[i, j] = pl_dis
                    paths[i][j] = path.copy()
                    
                    # Calculate navigation efficiency for this pair
                    E[i, j] = 1 / pl_wei if pl_wei > 0 else 0
                    successful_pairs += 1
                else:
                    PL_bin[i, j] = np.inf
                    PL_wei[i, j] = np.inf
                    PL_dis[i, j] = np.inf
                    paths[i][j] = []
                    E[i, j] = 0  # Failed path efficiency is 0
            else:
                # Diagonal elements (self-connections)
                paths[i][j] = []
                PL_bin[i, j] = np.inf
                PL_wei[i, j] = np.inf
                PL_dis[i, j] = np.inf
                E[i, j] = 0  # Self-connection efficiency is 0
    
    # Calculate success ratio
    valid_pairs = N * (N - 1)
    failed = np.sum(PL_bin == np.inf) - N  # Subtract diagonal elements
    sr = 1 - failed / valid_pairs
    
    # Calculate global navigation efficiency
    # Only consider non-diagonal elements
    non_diag_mask = ~np.eye(N, dtype=bool)
    nav_eff_global = np.mean(E[non_diag_mask])
    
    return sr, PL_bin, PL_wei, PL_dis, paths, E, nav_eff_global

def calculate_cmy(W):
    """
    计算脑网络通信模型中的Communicability矩阵(CMY)
    
    参数:
    W : numpy.ndarray, 形状为(N, N)
        结构连接(SC)权重矩阵，W[i][j]表示区域i和j之间的连接强度
        
    返回:
    CMY : numpy.ndarray, 形状为(N, N)
        Communicability矩阵，CMY[i][j]表示区域i到j的通信能力
    """
    N = W.shape[0]
    
    # 1. 计算节点强度向量s (每个节点的总连接强度)
    s = np.sum(W, axis=1)
    
    # 2. 处理强度为0的节点（避免除以0）
    # 将强度为0的节点替换为1（这样归一化因子为1，不影响结果）
    s_safe = np.where(s > 0, s, 1)
    
    # 3. 创建归一化因子对角矩阵
    # D_ii = 1/sqrt(s_i)
    D = np.diag(1 / np.sqrt(s_safe))
    
    # 4. 计算归一化连接矩阵 W_prime
    # W_prime = D * W * D
    W_prime = D @ W @ D
    
    # 5. 计算矩阵指数得到Communicability
    # CMY = e^{W_prime}
    CMY = expm(W_prime)
    
    return CMY

# 步骤1: 读取原始权重矩阵，并转换为array
L = pd.read_csv('averageConnectivity_tractLength_0.25density.csv', header=None).values
L = np.array(L, dtype=float)
print("原始权重矩阵 L 的形状:", L.shape)

# 关键步骤：将所有NaN替换为0（表示无连接）
L[np.isnan(L)] = 0

print("转换后矩阵统计:")
print(f"最小值: {np.min(L)}")
print(f"最大值: {np.max(L)}")
print(f"零值比例: {np.sum(L == 0)/L.size:.2%}")
print(f"正连接比例: {np.sum(L > 0)/L.size:.2%}")
print(f"负连接比例: {np.sum(L < 0)/L.size:.2%}")
print("矩阵中 NaN 的数量:", np.isnan(L).sum())

# 步骤2: 计算欧氏距离
D = pd.read_csv('fsaverage_parcel_distance_matrix.csv', index_col=0).values

# 步骤3: 计算导航效率
sr, PL_bin, PL_wei, PL_dis, paths, E, nav_eff_global = navigation_efficiency(L, D, max_hops=360)
print(E)
# print(PL_bin)
# PL_bin[np.isinf(PL_bin)] = 1
# print(f"最小值: {np.min(PL_bin)}")
# print(f"最大值: {np.max(PL_bin)}")

# 4. 分析结果
print(f"\n导航成功率: {sr:.2%}")
print(f"全局导航效率: {nav_eff_global:.6f}")

# 导航效率矩阵统计
non_diag_mask = ~np.eye(L.shape[0], dtype=bool)
E_non_diag = E[non_diag_mask]
successful_efficiency = E_non_diag[E_non_diag > 0]

print("\n导航效率矩阵统计:")
print(f"最小值: {np.min(successful_efficiency):.6f}" if len(successful_efficiency) > 0 else "无成功路径")
print(f"最大值: {np.max(successful_efficiency):.6f}" if len(successful_efficiency) > 0 else "无成功路径")
print(f"平均值: {np.mean(successful_efficiency):.6f}" if len(successful_efficiency) > 0 else "无成功路径")
print(f"中位数: {np.median(successful_efficiency):.6f}" if len(successful_efficiency) > 0 else "无成功路径")
print(f"成功路径比例: {len(successful_efficiency)/len(E_non_diag):.2%}")

# 5. 保存导航效率矩阵
np.savetxt('navigation_efficiency_matrix_0.25density.csv', E, delimiter=',', fmt='%.6f')
print("\n导航效率矩阵已保存为 navigation_efficiency_matrix_0.25density.csv")


# 计算路由（最短路径）效率（使用 transform=None，因为已手动转换）
GErout, Erout, Eloc = rout_efficiency(L, transform=None)
np.savetxt('rout_efficiency_matrix_0.25density.csv', Erout, delimiter=',', fmt='%.6f')
print("\n最短路径效率矩阵已保存为 rout_efficiency_matrix_0.25density.csv")

S = pd.read_csv('averageConnectivity_tractStrength_0.25density.csv', header=None).values
with np.errstate(divide='ignore', invalid='ignore'):  # 忽略除以零的警告
    result = np.where(S > 0, np.log(S + 1e-12), 0)

# 保存结果到新CSV文件
np.savetxt('transformed_matrix_0.25density.csv', result, delimiter=',')
S_log = pd.read_csv('transformed_matrix_0.25density.csv', header=None).values
S_log = np.array(S_log, dtype=float)
# print("原始权重矩阵 S_log 的形状:", S_log.shape)


gediff, ediff = diffusion_efficiency(S_log)
np.savetxt('diffusion_efficiency_matrix_0.25density.csv', ediff, delimiter=',', fmt='%.6f')
print("\n扩散效率矩阵已保存为 diffusion_efficiency_matrix_0.25density.csv")

SI = search_information(S, transform='log')
np.savetxt('search_information_matrix_0.25density.csv', SI, delimiter=',', fmt='%.6f')
print("\n搜索信息矩阵已保存为 search_information_matrix_0.25density.csv")

CMY = calculate_cmy(S)
np.savetxt('communicability_matrix_0.25density.csv', CMY, delimiter=',', fmt='%.6f')
print("\n通信能力矩阵已保存为 communicability_matrix_0.25density.csv")

print("矩阵中 NaN 的数量:", np.isnan(SI).sum())
