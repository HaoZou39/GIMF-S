from dataclasses import dataclass
import torch
import os

from MOTSP.MOTSProblemDef import get_random_problems, augment_xy_data_by_64_fold_2obj, augment_xy_data_by_8_fold
from BasemapManager import get_basemap_manager

@dataclass
class Reset_State:
    problems: torch.Tensor
    # shape: (batch, problem, 2)
    xy_img: torch.Tensor = None

@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor
    POMO_IDX: torch.Tensor
    # shape: (batch, pomo)
    current_node: torch.Tensor = None
    # shape: (batch, pomo)
    ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, node)


class TSPEnv:
    def __init__(self, **env_params):

        # Const @INIT
        ####################################
        self.env_params = env_params
        self.problem_size = env_params['problem_size']
        self.pomo_size = env_params['pomo_size']
        self.num_objectives = env_params['num_objectives']

        # Const @Load_Problem
        ####################################
        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        # IDX.shape: (batch, pomo)
        self.problems = None
        # shape: (batch, node, 2*num_objectives)

        # Dynamic
        ####################################
        self.selected_count = None
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = None
        # shape: (batch, pomo, 0~problem)

        self.channels = env_params['in_channels']
        self.img_size = env_params['img_size']
        self.patch_size = env_params['patch_size']
        self.patches = self.img_size // self.patch_size
        
        # Basemap configuration (must be set before offsets)
        ####################################
        self.use_basemap = env_params.get('use_basemap', False)
        
        # Point representation configuration
        # These parameters are independent of use_basemap for flexibility:
        # - point_style: 'white_on_black' (白点黑底) or 'black_on_white' (黑点白底)
        # - point_dilation: '3x3' (3x3放大) or '1x1' (单像素)
        # Default behavior when not specified:
        # - With basemap: white_on_black + 3x3
        # - Without basemap: black_on_white + 1x1
        self.point_style = env_params.get('point_style', 
                                          'white_on_black' if self.use_basemap else 'black_on_white')
        self.point_dilation = env_params.get('point_dilation',
                                             '3x3' if self.use_basemap else '1x1')
        
        # Set offsets based on point_dilation
        if self.point_dilation == '3x3':
            # 3x3 dilation for better point visibility
            self.offsets = torch.tensor([
                [-1, -1], [-1, 0], [-1, 1],
                [0, -1],  [0, 0],  [0, 1],
                [1, -1],  [1, 0],  [1, 1]
            ])
        else:
            # Single pixel (1x1)
            self.offsets = torch.tensor([[0, 0]])
        self._offsets_device = None  # Track device for offsets
        self.basemap_dir = env_params.get('basemap_dir', 'data')
        self.basemap_pattern = env_params.get('basemap_pattern', 'basemap_{id}.tif')
        self.default_basemap_id = env_params.get('default_basemap_id', '0')
        self.basemap_resized = None  # Resized basemap tensor
        self.basemap_manager = get_basemap_manager() if self.use_basemap else None

        # Distance matrix configuration
        ####################################
        self.use_distance_matrix = env_params.get('use_distance_matrix', False)
        self.distance_matrix = None  # (batch, problem, problem)
        
        # Basemap cache for multi-dataset support
        ####################################
        self.basemap_cache = {}  # Cache for different basemaps: {path: tensor}
        self.current_basemap_path = None  # Currently active basemap path

    def load_problems(self, batch_size, aug_factor=1, problems=None, distance_matrix=None, basemap_path=None):
        """
        Load problems with basemap support

        Args:
            batch_size: Number of problem instances
            aug_factor: Data augmentation factor
            problems: Pre-generated problems (optional)
            distance_matrix: Pre-computed distance matrix (optional), shape (batch, problem, problem)
            basemap_path: Path to the basemap file (optional, for multi-dataset support)
        """
        self.batch_size = batch_size

        # Determine the device based on default tensor type
        device = torch.tensor([0.0]).device  # Get current default device

        if problems is not None:
            # Move to correct device if needed
            self.problems = problems.to(device) if problems.device != device else problems
        else:
            self.problems = get_random_problems(batch_size, self.problem_size, self.num_objectives)

        # Store distance matrix if provided
        if distance_matrix is not None:
            # Move to correct device if needed
            self.distance_matrix = distance_matrix.to(device) if distance_matrix.device != device else distance_matrix
            self.use_distance_matrix = True

        # problems.shape: (batch, problem, 2*num_objectives)
        if aug_factor > 1:
            if aug_factor == 8:
                # Use 8-fold augmentation for any number of objectives
                self.batch_size = self.batch_size * 8
                self.problems = augment_xy_data_by_8_fold(self.problems, self.num_objectives)
            elif aug_factor == 64 and self.num_objectives == 2:
                # 64-fold only for 2 objectives (legacy)
                self.batch_size = self.batch_size * 64
                self.problems = augment_xy_data_by_64_fold_2obj(self.problems)
            else:
                raise NotImplementedError(f"aug_factor={aug_factor} not supported for {self.num_objectives} objectives")

        # Load basemap
        if self.use_basemap and self.basemap_manager is not None:
            # Determine basemap path: use provided path or default
            if basemap_path is not None:
                actual_basemap_path = basemap_path
            else:
                actual_basemap_path = os.path.join(
                    self.basemap_dir,
                    self.basemap_pattern.format(id=self.default_basemap_id)
                )
            
            # Check if we need to load a new basemap (path changed or not loaded)
            if (self.current_basemap_path != actual_basemap_path or 
                self.basemap_resized is None or 
                self.basemap_resized.shape[0] != self.img_size):
                
                # Try to get from cache first
                cache_key = (actual_basemap_path, self.img_size)
                if cache_key in self.basemap_cache:
                    self.basemap_resized = self.basemap_cache[cache_key]
                else:
                    # Load and cache the basemap
                    self.basemap_resized = self.basemap_manager.get_basemap(
                        basemap_path=actual_basemap_path,
                        resolution=self.img_size,
                        downscale_method='AREA'
                    )
                    self.basemap_cache[cache_key] = self.basemap_resized
                
                self.current_basemap_path = actual_basemap_path

        # Ensure all tensors are on the same device as problems
        device = self.problems.device
        
        # Move offsets to correct device if needed
        if self._offsets_device != device:
            self.offsets = self.offsets.to(device)
            self._offsets_device = device
        
        # Move basemap to correct device if needed
        if self.basemap_resized is not None and self.basemap_resized.device != device:
            self.basemap_resized = self.basemap_resized.to(device)
            # Update cache with device-correct tensor
            if self.current_basemap_path is not None:
                cache_key = (self.current_basemap_path, self.img_size)
                self.basemap_cache[cache_key] = self.basemap_resized

        # Image construction
        # Point representation is controlled by point_style parameter:
        # - 'white_on_black': black bg (0) + white points (1) - positive signals consistent with basemap
        # - 'black_on_white': white bg (1) + black points (0) - original representation
        if self.point_style == 'white_on_black':
            self.xy_img = torch.zeros((self.batch_size, self.channels, self.img_size, self.img_size), device=device)
            point_value = 1  # White points on black background
        else:  # 'black_on_white'
            self.xy_img = torch.ones((self.batch_size, self.channels, self.img_size, self.img_size), device=device)
            point_value = 0  # Black points on white background
        
        # Single objective with multiple channels: Channel 0 = point image, Channel 1+ = basemap
        if self.num_objectives == 1 and self.channels > 1:
            # Channel 0: Point image
            # Use (img_size - 1) to ensure coordinates in [0, 1] map to valid indices [0, img_size-1]
            xy_img = self.problems[:, :, 0:2] * (self.img_size - 1)
            xy_img = xy_img.int()
            # Clamp to ensure no index overflow
            xy_img = torch.clamp(xy_img, 0, self.img_size - 1)
            block_indices = xy_img // self.patch_size
            self.block_indices = block_indices[:, :, 0] * self.patches + block_indices[:, :, 1]
            # Apply offsets (3x3 dilation with basemap, single pixel without)
            xy_img = xy_img[:, :, None, :] + self.offsets[None, None, :, :].expand(self.batch_size, self.problem_size,
                                                                                              self.offsets.shape[0],
                                                                                              self.offsets.shape[1])
            # Clamp dilated coordinates to valid range (handle edge cases)
            xy_img = torch.clamp(xy_img, 0, self.img_size - 1)
            xy_img_idx = xy_img.reshape(-1, 2)
            BATCH_IDX = torch.arange(self.batch_size, device=device)[:, None, None].expand(self.batch_size, self.problem_size,
                                                                            self.offsets.shape[0]).reshape(-1)
            self.xy_img[BATCH_IDX, 0, xy_img_idx[:, 0], xy_img_idx[:, 1]] = point_value
            
            # Channel 1+: Basemap (roadnet)
            if self.use_basemap and self.basemap_resized is not None:
                for i in range(1, self.channels):
                    self.xy_img[:, i, :, :] = self.basemap_resized.unsqueeze(0).expand(self.batch_size, -1, -1)
            else:
                raise ValueError(f"Single objective with in_channels={self.channels} requires basemap. "
                                 f"Set use_basemap=True or reduce in_channels to 1.")
        else:
            # General case: Each channel uses its corresponding objective's coordinates
            # For num_objectives==1 with channels==1, this also works correctly
            for i in range(self.channels):
                # Use (img_size - 1) to ensure coordinates in [0, 1] map to valid indices [0, img_size-1]
                xy_img = self.problems[:, :, 2 * i: 2 * i + 2] * (self.img_size - 1)
                xy_img = xy_img.int()
                # Clamp to ensure no index overflow
                xy_img = torch.clamp(xy_img, 0, self.img_size - 1)
                block_indices = xy_img // self.patch_size
                self.block_indices = block_indices[:, :, 0] * self.patches + block_indices[:, :, 1]
                xy_img = xy_img[:, :, None, :] + self.offsets[None, None, :, :].expand(self.batch_size, self.problem_size,
                                                                                                  self.offsets.shape[0],
                                                                                                  self.offsets.shape[1])
                # Clamp dilated coordinates to valid range (handle edge cases)
                xy_img = torch.clamp(xy_img, 0, self.img_size - 1)
                xy_img_idx = xy_img.reshape(-1, 2)
                BATCH_IDX = torch.arange(self.batch_size, device=device)[:, None, None].expand(self.batch_size, self.problem_size,
                                                                                self.offsets.shape[0]).reshape(-1)
                self.xy_img[BATCH_IDX, i, xy_img_idx[:, 0], xy_img_idx[:, 1]] = point_value

        self.BATCH_IDX = torch.arange(self.batch_size, device=device)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size, device=device)[None, :].expand(self.batch_size, self.pomo_size)

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, pomo)
        device = self.problems.device
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long, device=device)
        # shape: (batch, pomo, 0~problem)

        # CREATE STEP STATE
        self.step_state = Step_State(BATCH_IDX=self.BATCH_IDX, POMO_IDX=self.POMO_IDX)
        self.step_state.ninf_mask = torch.zeros((self.batch_size, self.pomo_size, self.problem_size), device=device)
        # shape: (batch, pomo, problem)

        reward = None
        done = False
        return Reset_State(self.problems, self.xy_img), reward, done

    def pre_step(self):
        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected):
        # selected.shape: (batch, pomo)

        self.selected_count += 1
        self.current_node = selected
        # shape: (batch, pomo)
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)
        # shape: (batch, pomo, 0~problem)

        # UPDATE STEP STATE
        self.step_state.current_node = self.current_node
        # shape: (batch, pomo)
        self.step_state.ninf_mask[self.BATCH_IDX, self.POMO_IDX, self.current_node] = float('-inf')
        # shape: (batch, pomo, node)

        # returning values
        done = (self.selected_count == self.problem_size)
        if done:
            reward = -self._get_travel_distance()  # note the minus sign!
        else:
            reward = None

        return self.step_state, reward, done

    def _get_travel_distance(self):
        """
        Calculate travel distance for each objective.
        
        Returns:
            travel_distances_vec: (batch, pomo, num_objectives)
        """
        # Use distance matrix if available
        if self.use_distance_matrix and self.distance_matrix is not None:
            return self._get_travel_distance_from_matrix()
        
        # Otherwise use Euclidean distance
        return self._get_travel_distance_euclidean()

    def _get_travel_distance_from_matrix(self):
        """
        Calculate travel distance using pre-computed distance matrix.
        
        Returns:
            travel_distances_vec: (batch, pomo, 1) - single objective
        """
        batch_size = self.selected_node_list.size(0)
        pomo_size = self.selected_node_list.size(1)
        problem_size = self.selected_node_list.size(2)
        
        seq = self.selected_node_list  # (batch, pomo, problem)
        seq_next = torch.roll(seq, shifts=-1, dims=2)
        
        # Create batch indices for advanced indexing
        batch_idx = torch.arange(batch_size, device=seq.device)[:, None, None].expand(-1, pomo_size, problem_size)
        
        # Look up distances from matrix: distance_matrix[batch, from_node, to_node]
        distances = self.distance_matrix[batch_idx, seq, seq_next]  # (batch, pomo, problem)
        
        # Sum over all segments to get total path length
        travel_distances = distances.sum(dim=2)  # (batch, pomo)
        
        # Return shape: (batch, pomo, 1) for single objective
        return travel_distances.unsqueeze(2)

    def _get_travel_distance_euclidean(self):
        """
        Calculate travel distance using Euclidean distance.
        
        Returns:
            travel_distances_vec: (batch, pomo, num_objectives)
        """
        coord_dim = 2 * self.num_objectives
        gathering_index = self.selected_node_list.unsqueeze(3).expand(self.batch_size, -1, self.problem_size, coord_dim)
        # shape: (batch, pomo, problem, 2*num_objectives)
        seq_expanded = self.problems[:, None, :, :].expand(self.batch_size, self.pomo_size, self.problem_size, coord_dim)

        ordered_seq = seq_expanded.gather(dim=2, index=gathering_index)
        # shape: (batch, pomo, problem, 2*num_objectives)
        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        
        # Calculate distance for each objective separately
        travel_distances_list = []
        for obj_idx in range(self.num_objectives):
            # Extract coordinates for this objective: [x_i, y_i]
            start_idx = 2 * obj_idx
            end_idx = 2 * obj_idx + 2
            
            obj_coords = ordered_seq[:, :, :, start_idx:end_idx]
            obj_coords_next = rolled_seq[:, :, :, start_idx:end_idx]
            
            # Euclidean distance: sqrt((x_i - x_j)^2 + (y_i - y_j)^2)
            segment_lengths = ((obj_coords - obj_coords_next)**2).sum(3).sqrt()
            # shape: (batch, pomo, problem)
            
            # Sum over all segments to get total distance
            travel_distance = segment_lengths.sum(2)
            # shape: (batch, pomo)
            
            travel_distances_list.append(travel_distance)
        
        # Stack all objectives
        travel_distances_vec = torch.stack(travel_distances_list, dim=2)
        # shape: (batch, pomo, num_objectives)
        
        return travel_distances_vec

