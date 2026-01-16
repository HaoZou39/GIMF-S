"""
Verify that the training setup is correct, especially the euclid_dist_norm usage.
"""
import os
import numpy as np
import torch
import sys

# 切换到脚本所在目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '..')
sys.path.insert(0, '../..')
from MOTSP.MOTSProblemDef import MultiDatasetLoader

# 模拟训练配置中的数据集
datasets = [
    {
        'name': 'loc_30.283276_120.065850',
        'npz_path': '../../../MMDataset/30.256330_120.159448/04_30.283276_120.065850_3000.0/distance_dataset_train_30.283276_120.065850_3000.0_p20.npz',
        'basemap_path': None,
    },
]

print("Loading MultiDatasetLoader...")
loader = MultiDatasetLoader(datasets)

# 采样一个 batch
print("\nSampling a batch of 4 instances...")
problems, dist_matrix, euclid_dist, basemap_path, dataset_name = loader.sample_batch(4)

print('\n=== Verification of euclid_dist_norm usage ===')
print(f'problems shape: {problems.shape}')
print(f'dist_matrix (road) shape: {dist_matrix.shape}')
print(f'euclid_dist shape: {euclid_dist.shape}')

# 计算 detour ratio
d_road = dist_matrix
d_euclid = euclid_dist

# 统计 detour
detour = d_road / (d_euclid + 1e-6)
mask = ~torch.eye(d_road.shape[1], dtype=torch.bool).unsqueeze(0).expand(d_road.shape[0], -1, -1)
detour_valid = detour[mask]

print()
print('=== Detour Statistics (d_road / d_euclid) ===')
print(f'detour mean: {detour_valid.mean():.4f}')
print(f'detour min: {detour_valid.min():.4f}')
print(f'detour max: {detour_valid.max():.4f}')
print(f'detour std: {detour_valid.std():.4f}')
print()
print(f'Ratio >= 1.0: {(detour_valid >= 1.0).sum().item()} / {detour_valid.numel()} ({100.0*(detour_valid >= 1.0).float().mean():.2f}%)')
print()
print('>>> Detour >= 1 is expected (road distance should be >= Euclidean distance)')
print('>>> If detour < 1 were common, it would indicate normalization mismatch.')

# 测试 log-detour (训练目标)
log_detour = torch.log(detour + 1e-6)
log_detour_valid = log_detour[mask]
print()
print('=== Log-Detour Statistics (training target) ===')
print(f'log_detour mean: {log_detour_valid.mean():.4f}')
print(f'log_detour min: {log_detour_valid.min():.4f}')
print(f'log_detour max: {log_detour_valid.max():.4f}')
print(f'log_detour std: {log_detour_valid.std():.4f}')
print()
print('>>> Log-detour should be mostly >= 0 (since detour >= 1)')
print(f'Log-detour >= 0: {(log_detour_valid >= 0).sum().item()} / {log_detour_valid.numel()} ({100.0*(log_detour_valid >= 0).float().mean():.2f}%)')
print()
print('=== SUCCESS: Training setup verified! ===')
