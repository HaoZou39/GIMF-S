import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

# Path configuration
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils

from MOTSPEnv import TSPEnv
from MOTSP.MOTSProblemDef import MultiDatasetLoader

@torch.no_grad()
def debug_dump_xy_img(env_params, batch_size=4, out_dir="__debug_xyimg__", use_custom_dataset=False, suffix=""):
    """
    Dump xy_img for visualization.
    
    Args:
        env_params: Environment parameters
        batch_size: Number of samples
        out_dir: Output directory
        use_custom_dataset: Whether to use custom dataset
        suffix: Suffix for output filenames (e.g., "_basemap" or "_no_basemap")
    """
    os.makedirs(out_dir, exist_ok=True)

    env = TSPEnv(**env_params)
    use_basemap = env_params.get('use_basemap', False)

    # 1) 采样问题（尽量复用你训练时的逻辑）
    if use_custom_dataset:
        datasets = env_params["datasets"]
        mdl = MultiDatasetLoader(datasets, switch_interval=env_params.get("dataset_switch_interval", 50))
        problems, dist_matrix, basemap_path, dataset_name = mdl.sample_batch(batch_size)
        print(f"[debug{suffix}] dataset:", dataset_name, "basemap:", basemap_path)
        env.load_problems(batch_size, problems=problems, distance_matrix=dist_matrix, basemap_path=basemap_path)
    else:
        env.load_problems(batch_size)

    reset_state, _, _ = env.reset()
    xy_img = reset_state.xy_img  # (B, C, H, W)

    print(f"[debug{suffix}] use_basemap={use_basemap}")
    print(f"[debug{suffix}] offsets shape: {env.offsets.shape} (3x3={env.offsets.shape[0]==9}, single={env.offsets.shape[0]==1})")
    print(f"[debug{suffix}] xy_img shape:", tuple(xy_img.shape), "dtype:", xy_img.dtype)
    for c in range(xy_img.size(1)):
        v = xy_img[:, c]
        print(f"[debug{suffix}] channel {c}: min={v.min().item():.4f} max={v.max().item():.4f} mean={v.mean().item():.4f}")

    # 2) 保存可视化
    B = min(batch_size, xy_img.size(0))
    for b in range(B):
        # 点通道
        p = xy_img[b, 0].detach().cpu().numpy()
        plt.figure()
        plt.imshow(p, cmap="gray")
        mode_str = "black_bg_white_pts" if use_basemap else "white_bg_black_pts"
        plt.title(f"b{b}_points({mode_str})")
        plt.axis("off")
        plt.savefig(os.path.join(out_dir, f"b{b}_points{suffix}.png"), bbox_inches="tight", pad_inches=0)
        plt.close()

        # basemap 通道（若存在）
        if xy_img.size(1) > 1:
            m = xy_img[b, 1].detach().cpu().numpy()
            plt.figure()
            plt.imshow(m, cmap="gray")
            plt.title(f"b{b}_basemap(ch1)")
            plt.axis("off")
            plt.savefig(os.path.join(out_dir, f"b{b}_basemap{suffix}.png"), bbox_inches="tight", pad_inches=0)
            plt.close()

            # 叠加：basemap 为底，点用红色标出来
            plt.figure()
            plt.imshow(m, cmap="gray")
            # numpy 的 nonzero 返回 (行索引数组, 列索引数组)，即 (y, x)
            # 根据 use_basemap 选择正确的阈值
            if use_basemap:
                point_mask = p > 0.5  # 新表示：点是 1（白），背景是 0（黑）
            else:
                point_mask = p < 0.5  # 原表示：点是 0（黑），背景是 1（白）
            yy, xx = np.where(point_mask)
            scatter_size = 3 if use_basemap else 10  # 3x3膨胀点用小标记，单像素用大标记
            plt.scatter(xx, yy, s=scatter_size, c='red', marker='o')
            plt.title(f"b{b}_overlay(points on basemap)")
            plt.axis("off")
            plt.savefig(os.path.join(out_dir, f"b{b}_overlay{suffix}.png"), bbox_inches="tight", pad_inches=0)
            plt.close()

    print(f"[debug{suffix}] saved to: {out_dir}")


##########################################################################################
# Configuration
##########################################################################################

# Multi-dataset configuration (与 train_motsp_n20_single_obj.py 保持一致)
DATASETS = [
    {
        'name': '杭州',
        'npz_path': '../../../MMDataset/杭州/distance_dataset_30.318899_120.055447_5000.0.npz',
        'basemap_path': '../../../MMDataset/杭州/mask_prob_30.318899_120.055447_5000.0_z16.float32.tif',
    },
    {
        'name': '上海',
        'npz_path': '../../../MMDataset/上海/distance_dataset_31.240186_121.496062_5000.0.npz',
        'basemap_path': '../../../MMDataset/上海/mask_prob_31.240186_121.496062_5000.0_z16.float32.tif',
    },
    {
        'name': '柏林',
        'npz_path': '../../../MMDataset/柏林/distance_dataset_52.516298_13.377914_5000.0.npz',
        'basemap_path': '../../../MMDataset/柏林/mask_prob_52.516298_13.377914_5000.0_z16.float32.tif',
    },
    {
        'name': '鹤岗',
        'npz_path': '../../../MMDataset/鹤岗/distance_dataset_47.332394_130.278898_5000.0.npz',
        'basemap_path': '../../../MMDataset/鹤岗/mask_prob_47.332394_130.278898_5000.0_z16.float32.tif',
    },
]

# Base environment parameters
BASE_ENV_PARAMS = {
    'problem_size': 20,
    'pomo_size': 20,
    'num_objectives': 1,
    
    # Multi-dataset configuration
    'use_custom_dataset': True,
    'datasets': DATASETS,
    'use_distance_matrix': True,
    'dataset_switch_interval': 10,
    
    # Default basemap for random generation mode
    'basemap_dir': 'data',
    'basemap_pattern': 'basemap_{id}.tif',
    'default_basemap_id': '0',
    
    # Image parameters
    'img_size': 256,  # ceil(20^0.5 * 56 / 16) * 16 = 256
    'patch_size': 16,
}


@torch.no_grad()
def debug_compare_representations(batch_size=2, out_dir="__debug_xyimg__"):
    """
    Compare two representations using the SAME problem data.
    This ensures we're comparing apples to apples.
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # ========== Step 1: Sample problem data ONCE ==========
    print("\n[Step 1] 采样问题数据...")
    mdl = MultiDatasetLoader(DATASETS, switch_interval=10)
    problems, dist_matrix, basemap_path, dataset_name = mdl.sample_batch(batch_size)
    print(f"  数据集: {dataset_name}")
    print(f"  basemap: {basemap_path}")
    print(f"  problems shape: {problems.shape}")
    
    # ========== Step 2: 配置1 - 有 basemap（黑底白点 + 3x3 膨胀）==========
    print("\n[Step 2] 配置1: use_basemap=True（黑底白点 + 3x3 膨胀）")
    
    env_params_with_basemap = BASE_ENV_PARAMS.copy()
    env_params_with_basemap.update({
        'use_basemap': True,
        'in_channels': 2,
    })
    
    env1 = TSPEnv(**env_params_with_basemap)
    env1.load_problems(batch_size, problems=problems.clone(), distance_matrix=dist_matrix.clone(), basemap_path=basemap_path)
    reset_state1, _, _ = env1.reset()
    xy_img1 = reset_state1.xy_img
    
    print(f"  offsets shape: {env1.offsets.shape} (3x3={env1.offsets.shape[0]==9})")
    print(f"  xy_img channel 0: min={xy_img1[:,0].min():.4f} max={xy_img1[:,0].max():.4f}")
    
    # ========== Step 3: 配置2 - 无 basemap（白底黑点 + 单像素）==========
    print("\n[Step 3] 配置2: use_basemap=False（白底黑点 + 单像素）")
    
    env_params_no_basemap = BASE_ENV_PARAMS.copy()
    env_params_no_basemap.update({
        'use_basemap': False,
        'in_channels': 1,
    })
    
    env2 = TSPEnv(**env_params_no_basemap)
    env2.load_problems(batch_size, problems=problems.clone(), distance_matrix=dist_matrix.clone())
    reset_state2, _, _ = env2.reset()
    xy_img2 = reset_state2.xy_img
    
    print(f"  offsets shape: {env2.offsets.shape} (single={env2.offsets.shape[0]==1})")
    print(f"  xy_img channel 0: min={xy_img2[:,0].min():.4f} max={xy_img2[:,0].max():.4f}")
    
    # ========== Step 4: 可视化对比 ==========
    print("\n[Step 4] 生成对比图...")
    
    # 获取 basemap 用于 overlay
    basemap_tensor = env1.basemap_resized.cpu().numpy() if env1.basemap_resized is not None else None
    
    for b in range(batch_size):
        # 点通道对比图
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 配置1：黑底白点
        p1 = xy_img1[b, 0].cpu().numpy()
        axes[0].imshow(p1, cmap="gray", vmin=0, vmax=1)
        axes[0].set_title(f"use_basemap=True\n(black bg, white pts, 3x3)")
        axes[0].axis("off")
        
        # 配置2：白底黑点
        p2 = xy_img2[b, 0].cpu().numpy()
        axes[1].imshow(p2, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title(f"use_basemap=False\n(white bg, black pts, single)")
        axes[1].axis("off")
        
        plt.suptitle(f"Batch {b} - Points Channel Comparison (SAME problem data)", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"b{b}_compare_points.png"), dpi=150, bbox_inches="tight")
        plt.close()
        
        # Overlay 对比图（如果有 basemap）
        if basemap_tensor is not None:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # 配置1 overlay
            axes[0].imshow(basemap_tensor, cmap="gray")
            mask1 = p1 > 0.5  # 白点
            yy1, xx1 = np.where(mask1)
            axes[0].scatter(xx1, yy1, s=3, c='red', marker='o')
            axes[0].set_title(f"use_basemap=True\n({len(xx1)} pixels)")
            axes[0].axis("off")
            
            # 配置2 overlay
            axes[1].imshow(basemap_tensor, cmap="gray")
            mask2 = p2 < 0.5  # 黑点
            yy2, xx2 = np.where(mask2)
            axes[1].scatter(xx2, yy2, s=10, c='red', marker='o')
            axes[1].set_title(f"use_basemap=False\n({len(xx2)} pixels)")
            axes[1].axis("off")
            
            plt.suptitle(f"Batch {b} - Overlay Comparison (SAME problem data)", fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"b{b}_compare_overlay.png"), dpi=150, bbox_inches="tight")
            plt.close()
    
    print(f"\n[完成] 对比图已保存到: {out_dir}")
    print("  - b*_compare_points.png: 点通道对比")
    print("  - b*_compare_overlay.png: Overlay 对比（红点位置应相同）")


##########################################################################################
# Main
##########################################################################################

if __name__ == "__main__":
    print("=" * 70)
    print("Debug: 对比验证 - 有/无 basemap 时的 xy_img 表示")
    print("  使用相同的问题数据进行对比")
    print("=" * 70)
    
    debug_compare_representations(batch_size=2, out_dir="__debug_xyimg__")
    
    print("\n" + "=" * 70)
    print("验证完成！请检查 __debug_xyimg__ 文件夹中的对比图片：")
    print("  - b*_compare_points.png: 两种表示的点通道对比")
    print("  - b*_compare_overlay.png: 两种表示的 overlay 对比")
    print("=" * 70)
