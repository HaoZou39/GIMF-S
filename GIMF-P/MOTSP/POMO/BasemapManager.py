"""
BasemapManager: Manage multiple basemaps with automatic caching and downscaling
"""
import os
import numpy as np
import torch
from PIL import Image


class BasemapManager:
    """
    Manages basemap loading, caching, and downscaling.

    Features:
    - Automatically cache downscaled versions
    - Lazy loading and smart caching

    Cache structure:
        data/basemap_cache/
            basemap_0_256.npy
            basemap_0_448.npy
            ...
    """
    
    def __init__(self, cache_dir='data/basemap_cache'):
        """
        Initialize BasemapManager
        
        Args:
            cache_dir: Directory to store cached downscaled basemaps
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # In-memory cache: {(map_id, resolution): torch.Tensor}
        self.memory_cache = {}
        
        # Original basemap registry: {map_id: original_array}
        self.original_basemaps = {}

    def _get_map_id(self, basemap_path):
        """
        Extract map_id from basemap path (simplified version)
        """
        filename = os.path.basename(basemap_path)
        name_without_ext = os.path.splitext(filename)[0]

        if name_without_ext.startswith('basemap_'):
            return name_without_ext.replace('basemap_', '')
        else:
            return name_without_ext

    def _get_cache_path(self, map_id, resolution):
        """
        Get cache file path for a specific map_id and resolution
        
        Args:
            map_id: Map identifier
            resolution: Target resolution (e.g., 256)
            
        Returns:
            cache_path: Path to cached .npy file
        """
        cache_filename = f'basemap_{map_id}_{resolution}.npy'
        return os.path.join(self.cache_dir, cache_filename)
    
    def _load_original_basemap(self, basemap_path):
        """
        Load original basemap from file
        
        Args:
            basemap_path: Path to original basemap file
            
        Returns:
            basemap_array: numpy array (H, W), float32, [0, 1]
        """
        if not os.path.exists(basemap_path):
            raise FileNotFoundError(f"Basemap not found: {basemap_path}")
        
        print(f"Loading original basemap from: {basemap_path}")
        img = Image.open(basemap_path)
        basemap_array = np.array(img, dtype=np.float32)
        
        print(f"  Original size: {basemap_array.shape}")
        print(f"  Value range: [{basemap_array.min():.4f}, {basemap_array.max():.4f}]")
        
        return basemap_array
    
    def _downscale_basemap(self, basemap_array, target_resolution, method='AREA'):
        """
        Downscale basemap to target resolution
        
        Args:
            basemap_array: Original basemap array (H, W)
            target_resolution: Target size (e.g., 256 for 256x256)
            method: Downscaling method ('AREA' recommended for confidence maps)
            
        Returns:
            downscaled_array: numpy array (target_resolution, target_resolution), float32
        """
        original_size = basemap_array.shape[0]
        
        if original_size == target_resolution:
            # No need to resize
            return basemap_array
        
        # Convert to PIL Image
        basemap_pil = Image.fromarray(basemap_array, mode='F')
        
        # Downscale using specified method
        if method == 'AREA':
            resample_method = Image.Resampling.BOX
        elif method == 'LANCZOS':
            resample_method = Image.Resampling.LANCZOS
        elif method == 'BICUBIC':
            resample_method = Image.Resampling.BICUBIC
        else:
            raise ValueError(f"Unknown downscaling method: {method}")
        
        downscaled_pil = basemap_pil.resize((target_resolution, target_resolution), resample_method)
        downscaled_array = np.array(downscaled_pil, dtype=np.float32)
        
        # Ensure [0, 1] range
        downscaled_array = np.clip(downscaled_array, 0, 1)
        
        print(f"  Downscaled: {original_size}x{original_size} -> {target_resolution}x{target_resolution} ({method})")
        print(f"  Result range: [{downscaled_array.min():.4f}, {downscaled_array.max():.4f}]")
        
        return downscaled_array
    
    def _load_from_cache(self, cache_path):
        """
        Load cached basemap from disk
        
        Args:
            cache_path: Path to cached .npy file
            
        Returns:
            basemap_array: numpy array or None if cache doesn't exist
        """
        if os.path.exists(cache_path):
            basemap_array = np.load(cache_path)
            print(f"  Loaded from cache: {cache_path}")
            return basemap_array
        return None
    
    def _save_to_cache(self, basemap_array, cache_path):
        """
        Save downscaled basemap to disk cache
        
        Args:
            basemap_array: Downscaled basemap array
            cache_path: Path to save .npy file
        """
        np.save(cache_path, basemap_array)
        print(f"  Saved to cache: {cache_path}")
    
    def get_basemap(self, basemap_path, resolution, map_id=None, downscale_method='AREA'):
        """
        Get basemap at specified resolution (with smart caching)

        Workflow:
        1. Check in-memory cache
        2. Check disk cache
        3. Load original and downscale
        4. Save to both caches

        Args:
            basemap_path: Path to original basemap file
            resolution: Target resolution (e.g., 256)
            map_id: Optional map identifier (auto-extracted if None)
            downscale_method: 'AREA' (default), 'LANCZOS', 'BICUBIC'

        Returns:
            basemap_tensor: torch.Tensor (resolution, resolution), float32
        """
        # Extract map_id if not provided
        if map_id is None:
            map_id = self._get_map_id(basemap_path)

        cache_key = (map_id, resolution)
        
        # 1. Check in-memory cache
        if cache_key in self.memory_cache:
            print(f"Basemap [{map_id}, {resolution}x{resolution}] loaded from memory")
            return self.memory_cache[cache_key]
        
        # 2. Check disk cache
        cache_path = self._get_cache_path(map_id, resolution)
        cached_array = self._load_from_cache(cache_path)
        
        if cached_array is not None:
            # Load from disk cache
            basemap_tensor = torch.from_numpy(cached_array).float()
            self.memory_cache[cache_key] = basemap_tensor
            return basemap_tensor
        
        # 3. Load original and downscale
        print(f"Basemap [{map_id}, {resolution}x{resolution}] not in cache, generating...")
        
        # Load original basemap (cache in registry)
        if map_id not in self.original_basemaps:
            original_array = self._load_original_basemap(basemap_path)
            self.original_basemaps[map_id] = original_array
        else:
            original_array = self.original_basemaps[map_id]
            print(f"  Using cached original basemap: {original_array.shape}")
        
        # Downscale
        downscaled_array = self._downscale_basemap(original_array, resolution, method=downscale_method)
        
        # 4. Save to both caches
        self._save_to_cache(downscaled_array, cache_path)
        
        basemap_tensor = torch.from_numpy(downscaled_array).float()
        self.memory_cache[cache_key] = basemap_tensor
        
        return basemap_tensor
    
# Singleton instance for global access
_global_basemap_manager = None

def get_basemap_manager(cache_dir='data/basemap_cache'):
    """
    Get global BasemapManager instance (singleton pattern)
    
    Args:
        cache_dir: Cache directory (only used on first call)
        
    Returns:
        BasemapManager instance
    """
    global _global_basemap_manager
    if _global_basemap_manager is None:
        _global_basemap_manager = BasemapManager(cache_dir=cache_dir)
    return _global_basemap_manager
