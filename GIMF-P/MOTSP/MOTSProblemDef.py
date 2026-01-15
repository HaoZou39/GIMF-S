import torch
import numpy as np
import random


class MultiDatasetLoader:
    """
    Manager for loading problems from multiple datasets.
    Ensures that basemap corresponds correctly to the sampled instances.
    
    Optimization: Uses switch_interval to reduce basemap switching overhead.
    Instead of randomly selecting a dataset for each batch, we use the same
    dataset for `switch_interval` consecutive batches before switching.
    """
    
    def __init__(self, datasets, switch_interval=50):
        """
        Initialize with a list of dataset configurations.
        
        Args:
            datasets: List of dicts, each containing:
                - 'name': Dataset name (e.g., '杭州')
                - 'npz_path': Path to the NPZ file
                - 'basemap_path': Path to the corresponding basemap
            switch_interval: Number of batches to use the same dataset before switching.
                             Set to 1 for random selection each batch (original behavior).
                             Higher values reduce basemap switching overhead.
        """
        self.datasets = datasets
        self.num_datasets = len(datasets)
        self.switch_interval = switch_interval
        
        # Track current dataset and batch count for interval-based switching
        self.current_dataset_idx = 0
        self.batches_since_switch = 0
        
        # Load all NPZ data into memory for faster access
        self.data_cache = {}
        for i, ds in enumerate(datasets):
            print(f"Loading dataset {i+1}/{self.num_datasets}: {ds['name']}...")
            data = np.load(ds['npz_path'], allow_pickle=True)
            self.data_cache[i] = {
                'name': ds['name'],
                'problems': data['matched_node_norm'],
                'dist_matrix': data['undirected_dist_norm'],
                'basemap_path': ds['basemap_path'],
                'total_instances': data['matched_node_norm'].shape[0],
                'current_idx': 0,
            }
            print(f"  Loaded {self.data_cache[i]['total_instances']} instances")
        
        print(f"All {self.num_datasets} datasets loaded successfully!")
        print(f"Dataset switch interval: {switch_interval} batches")
    
    def sample_batch(self, batch_size, strategy='random'):
        """
        Sample a batch of problems from the datasets.
        
        Args:
            batch_size: Number of instances to sample
            strategy: Sampling strategy
                - 'random': Randomly select a dataset for the entire batch
                - 'mixed': Mix instances from different datasets in one batch
        
        Returns:
            problems: torch.Tensor (batch_size, problem_size, 2)
            dist_matrix: torch.Tensor (batch_size, problem_size, problem_size)
            basemap_path: str (path to the corresponding basemap)
            dataset_name: str (name of the selected dataset)
        """
        if strategy == 'random':
            return self._sample_single_dataset(batch_size)
        elif strategy == 'mixed':
            return self._sample_mixed(batch_size)
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")
    
    def _sample_single_dataset(self, batch_size):
        """
        Sample all instances from a single dataset.
        Uses interval-based switching to reduce basemap overhead.
        
        Sampling strategy:
        - Dataset selection: Random selection every switch_interval batches
        - Instance sampling: Random sampling within dataset (with replacement)
        """
        # Check if we need to switch to a new dataset
        if self.batches_since_switch >= self.switch_interval:
            # Randomly select a new dataset
            self.current_dataset_idx = random.randint(0, self.num_datasets - 1)
            self.batches_since_switch = 0
        
        self.batches_since_switch += 1
        ds_idx = self.current_dataset_idx
        ds = self.data_cache[ds_idx]
        
        # Random sampling: randomly select batch_size instances from the dataset
        total = ds['total_instances']
        random_indices = np.random.randint(0, total, size=batch_size)
        
        problems = ds['problems'][random_indices]
        dist_matrix = ds['dist_matrix'][random_indices]
        
        return (
            torch.tensor(problems, dtype=torch.float32),
            torch.tensor(dist_matrix, dtype=torch.float32),
            ds['basemap_path'],
            ds['name']
        )
    
    def _sample_mixed(self, batch_size):
        """
        Sample instances from multiple datasets (mixed batch).
        Note: This returns the basemap of the first sampled dataset.
        For mixed batches, the basemap may not perfectly match all instances.
        Use with caution - 'random' strategy is recommended for consistency.
        """
        # For simplicity, use the same dataset for now
        # Mixed sampling requires more complex handling of basemaps
        return self._sample_single_dataset(batch_size)
    
    def get_dataset_info(self):
        """Return information about loaded datasets"""
        info = []
        for i, ds in self.data_cache.items():
            info.append({
                'index': i,
                'name': ds['name'],
                'total_instances': ds['total_instances'],
                'basemap_path': ds['basemap_path'],
            })
        return info


def get_random_problems(batch_size, problem_size, num_objectives=2):
    """
    Generate random problem instances.
    
    Args:
        batch_size: Number of problem instances
        problem_size: Number of nodes
        num_objectives: Number of objectives (1 for single-objective, 2+ for multi-objective)
    
    Returns:
        problems: torch.Tensor
            - If num_objectives == 1: shape (batch_size, problem_size, 2)
              Single pair of coordinates [x, y] per node
            - If num_objectives >= 2: shape (batch_size, problem_size, 2*num_objectives)
              Multiple pairs [x1, y1, x2, y2, ...] per node
    """
    if num_objectives == 1:
        # Single objective: only generate one pair of coordinates [x, y]
        coord_dim = 2
    else:
        # Multi-objective: generate coordinates for each objective
        coord_dim = 2 * num_objectives
    
    problems = torch.rand(size=(batch_size, problem_size, coord_dim))
    return problems


def load_problems_from_npz(npz_path, batch_size, start_idx=0):
    """
    从 npz 加载预生成的问题实例
    
    Args:
        npz_path: npz 文件路径
        batch_size: 批次大小
        start_idx: 起始索引（用于遍历整个数据集）
    
    Returns:
        problems: torch.Tensor (batch_size, problem_size, 2) 节点坐标
        dist_matrix: torch.Tensor (batch_size, problem_size, problem_size) 距离矩阵
        next_idx: 下一个起始索引
    """
    data = np.load(npz_path, allow_pickle=True)
    total_instances = data['matched_node_norm'].shape[0]  # e.g., 10000
    
    end_idx = min(start_idx + batch_size, total_instances)
    actual_batch = end_idx - start_idx
    
    # 如果到达末尾，从头开始（循环）
    if actual_batch < batch_size:
        # 取剩余部分 + 从头开始的部分
        problems_part1 = data['matched_node_norm'][start_idx:end_idx]
        dist_part1 = data['undirected_dist_norm'][start_idx:end_idx]
        
        remaining = batch_size - actual_batch
        problems_part2 = data['matched_node_norm'][0:remaining]
        dist_part2 = data['undirected_dist_norm'][0:remaining]
        
        problems = np.concatenate([problems_part1, problems_part2], axis=0)
        dist_matrix = np.concatenate([dist_part1, dist_part2], axis=0)
        next_idx = remaining
    else:
        problems = data['matched_node_norm'][start_idx:end_idx]
        dist_matrix = data['undirected_dist_norm'][start_idx:end_idx]
        next_idx = end_idx % total_instances
    
    return torch.tensor(problems, dtype=torch.float32), \
           torch.tensor(dist_matrix, dtype=torch.float32), \
           next_idx


def augment_xy_data_by_8_fold(xy_data, num_objectives=1):
    """
    Augment data by 8-fold using symmetry transformations.
    Applies to each objective space independently.
    
    Args:
        xy_data: (batch, nodes, 2*num_objectives)
        num_objectives: number of objectives
    
    Returns:
        Augmented data: (batch*8, nodes, 2*num_objectives)
    """
    # Extract coordinates for each objective
    coords = []
    for obj_idx in range(num_objectives):
        x = xy_data[:, :, [2*obj_idx]]      # x coordinate
        y = xy_data[:, :, [2*obj_idx + 1]]  # y coordinate
        coords.append((x, y))
    
    # Generate 8 augmentations for each objective
    aug_transforms = []
    for obj_idx in range(num_objectives):
        x, y = coords[obj_idx]
        dat = {}
        dat[0] = torch.cat((x, y), dim=2)
        dat[1] = torch.cat((1-x, y), dim=2)
        dat[2] = torch.cat((x, 1-y), dim=2)
        dat[3] = torch.cat((1-x, 1-y), dim=2)
        dat[4] = torch.cat((y, x), dim=2)
        dat[5] = torch.cat((1-y, x), dim=2)
        dat[6] = torch.cat((y, 1-x), dim=2)
        dat[7] = torch.cat((1-y, 1-x), dim=2)
        aug_transforms.append(dat)
    
    # Combine augmentations
    dat_aug = []
    for i in range(8):
        # Concatenate all objectives with the same transformation index
        obj_list = [aug_transforms[obj_idx][i] for obj_idx in range(num_objectives)]
        dat = torch.cat(obj_list, dim=2)
        dat_aug.append(dat)
    
    aug_problems = torch.cat(dat_aug, dim=0)
    return aug_problems


def augment_xy_data_by_64_fold_2obj(xy_data):
    """
    Legacy function for 2-objective 64-fold augmentation.
    Kept for backward compatibility.
    """
    x1 = xy_data[:, :, [0]]
    y1 = xy_data[:, :, [1]]
    x2 = xy_data[:, :, [2]]
    y2 = xy_data[:, :, [3]]

    dat1 = {}
    dat2 = {}

    dat_aug = []

    dat1[0] = torch.cat((x1, y1), dim=2)
    dat1[1]= torch.cat((1-x1, y1), dim=2)
    dat1[2] = torch.cat((x1, 1-y1), dim=2)
    dat1[3] = torch.cat((1-x1, 1-y1), dim=2)
    dat1[4]= torch.cat((y1, x1), dim=2)
    dat1[5] = torch.cat((1-y1, x1), dim=2)
    dat1[6] = torch.cat((y1, 1-x1), dim=2)
    dat1[7] = torch.cat((1-y1, 1-x1), dim=2)

    dat2[0] = torch.cat((x2, y2), dim=2)
    dat2[1]= torch.cat((1-x2, y2), dim=2)
    dat2[2] = torch.cat((x2, 1-y2), dim=2)
    dat2[3] = torch.cat((1-x2, 1-y2), dim=2)
    dat2[4]= torch.cat((y2, x2), dim=2)
    dat2[5] = torch.cat((1-y2, x2), dim=2)
    dat2[6] = torch.cat((y2, 1-x2), dim=2)
    dat2[7] = torch.cat((1-y2, 1-x2), dim=2)

    for i in range(8):
        for j in range(8):
            dat = torch.cat((dat1[i], dat2[j]), dim=2)
            dat_aug.append(dat)

    aug_problems = torch.cat(dat_aug, dim=0)

    return aug_problems


