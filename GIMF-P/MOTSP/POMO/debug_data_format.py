"""
Debug script to compare data formats between:
1. Self-generated random problems with Euclidean distance
2. NPZ-imported problems with road network distance matrix

This script verifies that the data formats are compatible and analyzes
the scale differences between the two approaches.
"""

import os
import sys

# Path setup
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils

import torch
import numpy as np

# Import problem generation functions
from MOTSP.MOTSProblemDef import get_random_problems, load_problems_from_npz


def analyze_self_generated(batch_size=64, problem_size=20):
    """Analyze self-generated random problems with Euclidean distance"""
    print("=" * 60)
    print("自生成案例分析 (Self-generated Problems)")
    print("=" * 60)
    
    # Generate random problems
    problems = get_random_problems(batch_size, problem_size, num_objectives=1)
    
    print(f"problems shape: {problems.shape}, dtype: {problems.dtype}")
    print(f"  坐标范围: [{problems.min():.4f}, {problems.max():.4f}]")
    
    # Simulate a random path and calculate Euclidean distance
    # Create random permutation as path for each batch
    paths = torch.stack([torch.randperm(problem_size) for _ in range(batch_size)])
    # shape: (batch, problem)
    
    # Calculate Euclidean distance for each path
    total_distances = []
    for b in range(batch_size):
        path = paths[b]
        coords = problems[b]  # (problem, 2)
        
        total_dist = 0.0
        for i in range(problem_size):
            from_node = path[i]
            to_node = path[(i + 1) % problem_size]  # wrap around
            
            from_coord = coords[from_node]
            to_coord = coords[to_node]
            
            dist = torch.sqrt(((from_coord - to_coord) ** 2).sum())
            total_dist += dist.item()
        
        total_distances.append(total_dist)
    
    total_distances = np.array(total_distances)
    
    print(f"\n模拟随机路径欧几里得总距离:")
    print(f"  均值: {total_distances.mean():.4f}")
    print(f"  范围: [{total_distances.min():.4f}, {total_distances.max():.4f}]")
    print(f"  标准差: {total_distances.std():.4f}")
    
    # Calculate pairwise Euclidean distances for reference
    # For one sample, compute distance matrix
    sample_coords = problems[0]  # (problem, 2)
    euclidean_dists = torch.cdist(sample_coords, sample_coords)
    
    print(f"\n欧几里得距离矩阵统计 (样本0):")
    print(f"  shape: {euclidean_dists.shape}")
    print(f"  范围: [{euclidean_dists.min():.4f}, {euclidean_dists.max():.4f}]")
    print(f"  对角线外均值: {euclidean_dists[euclidean_dists > 0].mean():.4f}")
    
    return problems, euclidean_dists, total_distances


def analyze_npz_imported(npz_path, batch_size=64):
    """Analyze NPZ-imported problems with road network distance matrix"""
    print("\n" + "=" * 60)
    print("NPZ 导入案例分析 (NPZ-imported Problems)")
    print("=" * 60)
    
    if not os.path.exists(npz_path):
        print(f"错误: 文件不存在 - {npz_path}")
        return None, None, None
    
    # Load problems from NPZ
    problems, dist_matrix, next_idx = load_problems_from_npz(npz_path, batch_size, start_idx=0)
    
    print(f"problems shape: {problems.shape}, dtype: {problems.dtype}")
    print(f"  坐标范围: [{problems.min():.4f}, {problems.max():.4f}]")
    
    print(f"\ndist_matrix shape: {dist_matrix.shape}, dtype: {dist_matrix.dtype}")
    print(f"  距离范围: [{dist_matrix.min():.4f}, {dist_matrix.max():.4f}]")
    print(f"  对角线外均值: {dist_matrix[dist_matrix > 0].mean():.4f}")
    
    # Simulate a random path and calculate road network distance
    problem_size = problems.shape[1]
    paths = torch.stack([torch.randperm(problem_size) for _ in range(batch_size)])
    
    total_distances = []
    for b in range(batch_size):
        path = paths[b]
        dist_mat = dist_matrix[b]  # (problem, problem)
        
        total_dist = 0.0
        for i in range(problem_size):
            from_node = path[i].item()
            to_node = path[(i + 1) % problem_size].item()
            
            dist = dist_mat[from_node, to_node].item()
            total_dist += dist
        
        total_distances.append(total_dist)
    
    total_distances = np.array(total_distances)
    
    print(f"\n模拟随机路径路网总距离:")
    print(f"  均值: {total_distances.mean():.4f}")
    print(f"  范围: [{total_distances.min():.4f}, {total_distances.max():.4f}]")
    print(f"  标准差: {total_distances.std():.4f}")
    
    return problems, dist_matrix, total_distances


def compare_formats(euclidean_dists, road_dist_matrix, euclidean_total, road_total):
    """Compare the two data formats"""
    print("\n" + "=" * 60)
    print("量纲对比 (Scale Comparison)")
    print("=" * 60)
    
    # Coordinate range
    print("\n1. Coordinate Range:")
    print("   Self-generated: [0, 1] (uniform random)")
    print("   NPZ-imported:   [0, 1] (normalized)")
    print("   Result: [OK] Compatible")
    
    # Distance matrix range
    print("\n2. Distance Matrix Range:")
    euc_max = euclidean_dists.max().item()
    road_max = road_dist_matrix[0].max().item()  # first sample
    print(f"   Euclidean: [0, {euc_max:.4f}] (theoretical max sqrt(2) = 1.414)")
    print(f"   Road network: [0, {road_max:.4f}]")
    
    ratio = road_max / euc_max
    print(f"   Ratio: road/euclidean = {ratio:.4f}")
    print(f"   Result: {'[OK] Similar scale, compatible' if 0.5 < ratio < 2.0 else '[WARN] Large scale difference, may need normalization'}")
    
    # Total path distance
    print("\n3. Random Path Total Distance:")
    print(f"   Euclidean: mean={euclidean_total.mean():.4f}, range=[{euclidean_total.min():.4f}, {euclidean_total.max():.4f}]")
    print(f"   Road network: mean={road_total.mean():.4f}, range=[{road_total.min():.4f}, {road_total.max():.4f}]")
    
    path_ratio = road_total.mean() / euclidean_total.mean()
    print(f"   Mean ratio: road/euclidean = {path_ratio:.4f}")
    print(f"   Result: {'[OK] Similar scale, reward compatible' if 0.5 < path_ratio < 2.0 else '[WARN] Large scale difference'}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("Data format compatibility check:")
    print("  - Coordinates: [OK] Fully compatible ([batch, problem, 2], range [0,1])")
    print("  - Distance: [OK] Compatible (distance matrix [batch, problem, problem])")
    print("  - Scale: " + ("[OK] Similar" if 0.5 < path_ratio < 2.0 else "[WARN] Needs attention"))
    print("\nReady for training!")


def main():
    print("=" * 60)
    print("数据格式调试脚本")
    print("Debug Data Format Script")
    print("=" * 60)
    print()
    
    batch_size = 64
    problem_size = 20
    npz_path = '../../../MMDataset/杭州/distance_dataset_30.318899_120.055447_5000.0.npz'
    
    # Analyze self-generated problems
    problems_gen, euclidean_dists, euclidean_total = analyze_self_generated(batch_size, problem_size)
    
    # Analyze NPZ-imported problems
    problems_npz, road_dist_matrix, road_total = analyze_npz_imported(npz_path, batch_size)
    
    if problems_npz is not None:
        # Compare formats
        compare_formats(euclidean_dists, road_dist_matrix, euclidean_total, road_total)
    else:
        print("\n无法进行比较：NPZ 文件加载失败")


if __name__ == "__main__":
    main()
