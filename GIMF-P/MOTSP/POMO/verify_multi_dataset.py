"""
Verify multi-dataset loading:
1. Check that instances from each dataset are loaded correctly
2. Verify that basemap corresponds to the correct dataset
3. Visualize samples from each dataset
"""

import os
import sys

# Path setup
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Import the multi-dataset loader
from MOTSP.MOTSProblemDef import MultiDatasetLoader


# Dataset configuration (same as in train script)
DATASETS = [
    {
        'name': 'hangzhou',
        'npz_path': '../../../MMDataset/杭州/distance_dataset_30.318899_120.055447_5000.0.npz',
        'basemap_path': '../../../MMDataset/杭州/mask_prob_30.318899_120.055447_5000.0_z16.float32.tif',
    },
    {
        'name': 'shanghai',
        'npz_path': '../../../MMDataset/上海/distance_dataset_31.240186_121.496062_5000.0.npz',
        'basemap_path': '../../../MMDataset/上海/mask_prob_31.240186_121.496062_5000.0_z16.float32.tif',
    },
    {
        'name': 'berlin',
        'npz_path': '../../../MMDataset/柏林/distance_dataset_52.516298_13.377914_5000.0.npz',
        'basemap_path': '../../../MMDataset/柏林/mask_prob_52.516298_13.377914_5000.0_z16.float32.tif',
    },
    {
        'name': 'hegang',
        'npz_path': '../../../MMDataset/鹤岗/distance_dataset_47.332394_130.278898_5000.0.npz',
        'basemap_path': '../../../MMDataset/鹤岗/mask_prob_47.332394_130.278898_5000.0_z16.float32.tif',
    },
]


def load_basemap(path, target_size=256):
    """Load and resize basemap"""
    img = Image.open(path)
    arr = np.array(img, dtype=np.float32)
    # Resize
    img_resized = Image.fromarray(arr).resize((target_size, target_size), Image.Resampling.BOX)
    return np.array(img_resized, dtype=np.float32)


def visualize_sample(problems, basemap, dataset_name, sample_idx=0):
    """Visualize a sample: nodes overlaid on basemap"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # Show basemap
    ax.imshow(basemap, cmap='gray', extent=[0, 1, 0, 1], origin='lower')
    
    # Plot nodes
    coords = problems[sample_idx].numpy()
    ax.scatter(coords[:, 0], coords[:, 1], c='red', s=100, marker='o', edgecolors='white', linewidths=2)
    
    # Number the nodes
    for i, (x, y) in enumerate(coords):
        ax.annotate(str(i), (x, y), fontsize=8, ha='center', va='center', color='white')
    
    ax.set_title(f'Dataset: {dataset_name}\nNodes on Basemap')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    return fig


def main():
    print("="*60)
    print("Multi-Dataset Verification Script")
    print("="*60)
    
    # Check if datasets exist
    print("\n1. Checking dataset files...")
    valid_datasets = []
    for ds in DATASETS:
        npz_exists = os.path.exists(ds['npz_path'])
        basemap_exists = os.path.exists(ds['basemap_path'])
        
        status = "[OK]" if (npz_exists and basemap_exists) else "[MISSING]"
        print(f"  {ds['name']}: {status}")
        if not npz_exists:
            print(f"    - NPZ not found: {ds['npz_path']}")
        if not basemap_exists:
            print(f"    - Basemap not found: {ds['basemap_path']}")
        
        if npz_exists and basemap_exists:
            valid_datasets.append(ds)
    
    if not valid_datasets:
        print("\nNo valid datasets found! Exiting.")
        return
    
    print(f"\n{len(valid_datasets)}/{len(DATASETS)} datasets are valid.")
    
    # Initialize loader with valid datasets
    print("\n2. Initializing MultiDatasetLoader...")
    loader = MultiDatasetLoader(valid_datasets)
    
    # Sample from each dataset and verify
    print("\n3. Sampling and verifying data correspondence...")
    batch_size = 4
    
    fig, axes = plt.subplots(2, len(valid_datasets), figsize=(5*len(valid_datasets), 10))
    if len(valid_datasets) == 1:
        axes = axes.reshape(-1, 1)
    
    for ds_idx, ds in enumerate(valid_datasets):
        print(f"\n  Dataset: {ds['name']}")
        
        # Manually sample from this specific dataset
        data_cache = loader.data_cache[ds_idx]
        problems = torch.tensor(data_cache['problems'][:batch_size], dtype=torch.float32)
        dist_matrix = torch.tensor(data_cache['dist_matrix'][:batch_size], dtype=torch.float32)
        basemap_path = data_cache['basemap_path']
        
        print(f"    Problems shape: {problems.shape}")
        print(f"    Dist matrix shape: {dist_matrix.shape}")
        print(f"    Basemap path: {basemap_path}")
        
        # Load basemap
        basemap = load_basemap(basemap_path, target_size=256)
        print(f"    Basemap shape: {basemap.shape}")
        
        # Verify coordinate range
        coord_min, coord_max = problems.min().item(), problems.max().item()
        print(f"    Coord range: [{coord_min:.4f}, {coord_max:.4f}]")
        
        # Verify distance matrix
        dist_min, dist_max = dist_matrix.min().item(), dist_matrix.max().item()
        print(f"    Distance range: [{dist_min:.4f}, {dist_max:.4f}]")
        
        # Visualize
        # Row 0: Basemap
        ax0 = axes[0, ds_idx]
        ax0.imshow(basemap, cmap='gray', origin='lower')  # 统一使用 origin='lower'
        ax0.set_title(f'{ds["name"]}\nBasemap')
        ax0.axis('off')
        
        # Row 1: Nodes on basemap
        ax1 = axes[1, ds_idx]
        ax1.imshow(basemap, cmap='gray', extent=[0, 1, 0, 1], origin='lower')
        coords = problems[0].numpy()
        ax1.scatter(coords[:, 0], coords[:, 1], c='red', s=50, marker='o', edgecolors='white', linewidths=1)
        ax1.set_title(f'Sample nodes\n(n={len(coords)})')
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
    
    plt.tight_layout()
    save_path = 'multi_dataset_verification.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n4. Visualization saved to: {save_path}")
    
    # Test sampling
    print("\n5. Testing random sampling...")
    for i in range(5):
        problems, dist_matrix, basemap_path, dataset_name = loader.sample_batch(batch_size)
        print(f"  Sample {i+1}: Dataset={dataset_name}, Basemap={os.path.basename(basemap_path)}")
    
    print("\n" + "="*60)
    print("Verification Complete!")
    print("="*60)
    print("\nKey points:")
    print("  - Each sample batch comes from a single dataset")
    print("  - Basemap path is returned along with the samples")
    print("  - The Env will load the corresponding basemap automatically")
    
    plt.show()


if __name__ == "__main__":
    main()
