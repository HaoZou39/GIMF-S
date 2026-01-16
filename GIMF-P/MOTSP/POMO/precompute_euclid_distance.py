"""
预计算欧式距离矩阵并添加到 npz 文件中
使用 matched_node_xy_3857 (米制坐标) 计算，然后用 meta_distance_ref_m 归一化
这样欧式距离和道路距离使用相同的归一化方式
"""
import os
import sys
import numpy as np
from glob import glob

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def compute_euclid_distance_matrix(coords_3857):
    """
    计算欧式距离矩阵 (米为单位)
    
    Args:
        coords_3857: (num_instances, num_nodes, 2) - EPSG:3857 坐标 (米)
    
    Returns:
        dist_matrix: (num_instances, num_nodes, num_nodes) - 米
    """
    # coords_3857 shape: (N, n, 2)
    diff = coords_3857[:, :, np.newaxis, :] - coords_3857[:, np.newaxis, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1)).astype(np.float32)
    return dist


def process_npz_file(npz_path, overwrite=False):
    """
    处理单个 npz 文件，添加归一化的欧式距离矩阵
    
    Args:
        npz_path: npz 文件路径
        overwrite: 是否覆盖已存在的字段
    
    Returns:
        success: 是否成功
    """
    print(f"\nProcessing: {npz_path}")
    
    # 加载现有数据
    try:
        data = dict(np.load(npz_path, allow_pickle=True))
    except Exception as e:
        print(f"  [ERROR] Failed to load: {e}")
        return False
    
    # 检查是否已经存在
    if 'euclid_dist_norm' in data and not overwrite:
        print(f"  [SKIP] 'euclid_dist_norm' already exists. Use --overwrite to regenerate.")
        return True
    
    # 检查必需字段
    required_fields = ['matched_node_xy_3857', 'meta_distance_ref_m']
    for field in required_fields:
        if field not in data:
            print(f"  [ERROR] Missing required field: {field}")
            return False
    
    # 获取坐标和归一化参考值
    coords_3857 = data['matched_node_xy_3857']  # (N, n, 2)
    distance_ref_m = float(data['meta_distance_ref_m'])
    
    print(f"  Coordinates shape: {coords_3857.shape}")
    print(f"  Distance reference: {distance_ref_m:.2f} m")
    
    # 计算欧式距离矩阵 (米)
    print("  Computing Euclidean distance matrix...")
    euclid_dist_m = compute_euclid_distance_matrix(coords_3857)
    
    # 归一化 (使用与道路距离相同的归一化因子)
    euclid_dist_norm = (euclid_dist_m / distance_ref_m).astype(np.float32)
    
    print(f"  Euclidean distance (m): mean={euclid_dist_m.mean():.2f}, max={euclid_dist_m.max():.2f}")
    print(f"  Euclidean distance (norm): mean={euclid_dist_norm.mean():.4f}, max={euclid_dist_norm.max():.4f}")
    
    # 验证 detour ratio
    road_dist_norm = data.get('undirected_dist_norm', None)
    if road_dist_norm is not None:
        # 排除对角线
        n = coords_3857.shape[1]
        mask = ~np.eye(n, dtype=bool)
        
        road_flat = road_dist_norm[:, mask].flatten()
        euclid_flat = euclid_dist_norm[:, mask].flatten()
        
        valid_mask = (euclid_flat > 1e-6) & (road_flat > 0)
        if valid_mask.sum() > 0:
            detour = road_flat[valid_mask] / euclid_flat[valid_mask]
            print(f"  Detour ratio: mean={detour.mean():.4f}, median={np.median(detour):.4f}")
            print(f"  Detour < 1.0: {(detour < 1.0).sum() / len(detour) * 100:.1f}%")
    
    # 添加新字段
    data['euclid_dist_m'] = euclid_dist_m
    data['euclid_dist_norm'] = euclid_dist_norm
    
    # 保存更新后的 npz
    print(f"  Saving updated npz...")
    np.savez_compressed(npz_path, **data)
    
    print(f"  [OK] Added 'euclid_dist_m' and 'euclid_dist_norm' to {os.path.basename(npz_path)}")
    return True


def find_all_dataset_files(base_dir):
    """
    查找所有数据集 npz 文件
    """
    patterns = [
        os.path.join(base_dir, '**', 'distance_dataset_*.npz'),
    ]
    
    files = []
    for pattern in patterns:
        files.extend(glob(pattern, recursive=True))
    
    return sorted(set(files))


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Precompute Euclidean distance matrix for datasets')
    parser.add_argument('--base-dir', type=str, 
                        default='../../../MMDataset/30.256330_120.159448',
                        help='Base directory containing dataset folders')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing euclid_dist_norm fields')
    parser.add_argument('--file', type=str, default=None,
                        help='Process a single file instead of scanning directory')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Precompute Euclidean Distance Matrix")
    print("=" * 70)
    
    if args.file:
        # 处理单个文件
        files = [args.file]
    else:
        # 扫描目录
        print(f"\nScanning: {args.base_dir}")
        files = find_all_dataset_files(args.base_dir)
        print(f"Found {len(files)} dataset files")
    
    # 处理每个文件
    success_count = 0
    for f in files:
        if process_npz_file(f, overwrite=args.overwrite):
            success_count += 1
    
    print("\n" + "=" * 70)
    print(f"Completed: {success_count}/{len(files)} files processed successfully")
    print("=" * 70)


if __name__ == "__main__":
    main()
