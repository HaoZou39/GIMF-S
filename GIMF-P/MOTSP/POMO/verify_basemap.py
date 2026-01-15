"""
Verify and compare basemap images:
1. Original basemap (data/basemap_0.tif)
2. New dataset basemap (MMDataset/杭州/mask_prob_*.tif)

This script visualizes both images and computes similarity metrics.
"""

import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Path setup
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def load_tif_image(path):
    """Load a TIF image and return as numpy array"""
    if not os.path.exists(path):
        print(f"Error: File not found - {path}")
        return None
    
    img = Image.open(path)
    arr = np.array(img, dtype=np.float32)
    return arr


def analyze_image(name, arr):
    """Analyze and print image statistics"""
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")
    print(f"  Shape: {arr.shape}")
    print(f"  Dtype: {arr.dtype}")
    print(f"  Value range: [{arr.min():.4f}, {arr.max():.4f}]")
    print(f"  Mean: {arr.mean():.4f}")
    print(f"  Std: {arr.std():.4f}")
    
    # Value distribution
    print(f"  Percentiles:")
    for p in [0, 25, 50, 75, 100]:
        print(f"    {p}%: {np.percentile(arr, p):.4f}")
    
    return arr


def compare_images(img1, img2, name1, name2):
    """Compare two images visually and statistically"""
    print(f"\n{'='*60}")
    print(f"Comparison: {name1} vs {name2}")
    print(f"{'='*60}")
    
    # Resize to same shape if needed
    if img1.shape != img2.shape:
        print(f"  Different shapes: {img1.shape} vs {img2.shape}")
        # Resize img2 to match img1 for comparison
        from PIL import Image as PILImage
        img2_pil = PILImage.fromarray(img2)
        img2_resized = img2_pil.resize((img1.shape[1], img1.shape[0]), PILImage.Resampling.BILINEAR)
        img2 = np.array(img2_resized, dtype=np.float32)
        print(f"  Resized img2 to: {img2.shape}")
    
    # Normalize both to [0, 1] for fair comparison
    img1_norm = (img1 - img1.min()) / (img1.max() - img1.min() + 1e-8)
    img2_norm = (img2 - img2.min()) / (img2.max() - img2.min() + 1e-8)
    
    # Compute similarity metrics
    # 1. Mean Absolute Error
    mae = np.abs(img1_norm - img2_norm).mean()
    print(f"  Mean Absolute Error (normalized): {mae:.4f}")
    
    # 2. Correlation coefficient
    corr = np.corrcoef(img1_norm.flatten(), img2_norm.flatten())[0, 1]
    print(f"  Correlation coefficient: {corr:.4f}")
    
    # 3. Structural similarity (simplified)
    # Using mean and std comparison
    mean_diff = abs(img1_norm.mean() - img2_norm.mean())
    std_diff = abs(img1_norm.std() - img2_norm.std())
    print(f"  Mean difference: {mean_diff:.4f}")
    print(f"  Std difference: {std_diff:.4f}")
    
    return img1_norm, img2_norm


def visualize_comparison(img1, img2, name1, name2, save_path=None):
    """Create visualization comparing two images"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Original images
    ax1 = axes[0, 0]
    im1 = ax1.imshow(img1, cmap='gray')
    ax1.set_title(f'{name1}\nShape: {img1.shape}')
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046)
    
    ax2 = axes[0, 1]
    im2 = ax2.imshow(img2, cmap='gray')
    ax2.set_title(f'{name2}\nShape: {img2.shape}')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046)
    
    # Resize img2 if needed for difference map
    if img1.shape != img2.shape:
        from PIL import Image as PILImage
        img2_pil = PILImage.fromarray(img2)
        img2_resized = img2_pil.resize((img1.shape[1], img1.shape[0]), PILImage.Resampling.BILINEAR)
        img2_compare = np.array(img2_resized, dtype=np.float32)
    else:
        img2_compare = img2
    
    # Normalize for comparison
    img1_norm = (img1 - img1.min()) / (img1.max() - img1.min() + 1e-8)
    img2_norm = (img2_compare - img2_compare.min()) / (img2_compare.max() - img2_compare.min() + 1e-8)
    
    # Difference map
    ax3 = axes[0, 2]
    diff = np.abs(img1_norm - img2_norm)
    im3 = ax3.imshow(diff, cmap='hot')
    ax3.set_title(f'Absolute Difference\nMean: {diff.mean():.4f}')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046)
    
    # Row 2: Histograms
    ax4 = axes[1, 0]
    ax4.hist(img1.flatten(), bins=50, alpha=0.7, color='blue', label=name1)
    ax4.set_title(f'{name1} Histogram')
    ax4.set_xlabel('Pixel Value')
    ax4.set_ylabel('Count')
    ax4.legend()
    
    ax5 = axes[1, 1]
    ax5.hist(img2.flatten(), bins=50, alpha=0.7, color='orange', label=name2)
    ax5.set_title(f'{name2} Histogram')
    ax5.set_xlabel('Pixel Value')
    ax5.set_ylabel('Count')
    ax5.legend()
    
    # Overlay histogram (normalized)
    ax6 = axes[1, 2]
    ax6.hist(img1_norm.flatten(), bins=50, alpha=0.5, color='blue', label=name1, density=True)
    ax6.hist(img2_norm.flatten(), bins=50, alpha=0.5, color='orange', label=name2, density=True)
    ax6.set_title('Normalized Histogram Overlay')
    ax6.set_xlabel('Normalized Pixel Value')
    ax6.set_ylabel('Density')
    ax6.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to: {save_path}")
    
    plt.show()


def main():
    print("="*60)
    print("Basemap Verification Script")
    print("="*60)
    
    # Define paths
    original_basemap_path = 'data/basemap_0.tif'
    new_basemap_path = '../../../MMDataset/杭州/mask_prob_30.318899_120.055447_5000.0_z16.float32.tif'
    
    # Alternative: try satellite image too
    satellite_path = '../../../MMDataset/杭州/crop_30.318899_120.055447_5000.0_z16.tif'
    
    print(f"\nPaths:")
    print(f"  Original basemap: {original_basemap_path}")
    print(f"  New basemap (mask_prob): {new_basemap_path}")
    print(f"  Satellite image: {satellite_path}")
    
    # Load images
    print("\n" + "="*60)
    print("Loading images...")
    print("="*60)
    
    original = load_tif_image(original_basemap_path)
    new_mask = load_tif_image(new_basemap_path)
    satellite = load_tif_image(satellite_path)
    
    # Analyze each image
    if original is not None:
        analyze_image("Original Basemap (data/basemap_0.tif)", original)
    
    if new_mask is not None:
        analyze_image("New Mask Prob (MMDataset/杭州/mask_prob_*.tif)", new_mask)
    
    if satellite is not None:
        analyze_image("Satellite Image (MMDataset/杭州/crop_*.tif)", satellite)
    
    # Compare images
    if original is not None and new_mask is not None:
        compare_images(original, new_mask, "Original", "New Mask")
        visualize_comparison(original, new_mask, "Original Basemap", "New Mask Prob",
                           save_path='basemap_comparison.png')
    
    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    
    if original is not None and new_mask is not None:
        # Check if they are the same type of data
        orig_range = original.max() - original.min()
        new_range = new_mask.max() - new_mask.min()
        
        print(f"\nData type analysis:")
        print(f"  Original range: {orig_range:.4f}")
        print(f"  New mask range: {new_range:.4f}")
        
        if orig_range <= 1.0 and new_range <= 1.0:
            print("  Both appear to be normalized probability maps [0, 1]")
        elif orig_range > 1.0 and new_range > 1.0:
            print("  Both appear to be raw pixel values (not normalized)")
        else:
            print("  WARNING: Different value ranges - may need normalization!")
        
        print("\nRecommendation:")
        print("  If correlation > 0.5 and similar histograms: Images are compatible")
        print("  If correlation < 0.3 or very different histograms: May need to check data")
    else:
        print("\nCould not compare - one or more images failed to load")


if __name__ == "__main__":
    main()
