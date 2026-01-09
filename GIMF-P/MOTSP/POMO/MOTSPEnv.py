from dataclasses import dataclass
import torch

from MOTSP.MOTSProblemDef import get_random_problems, augment_xy_data_by_64_fold_2obj


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
        self.offsets = torch.tensor([[0, 0]])

    def load_problems(self, batch_size, aug_factor=1, problems=None):
        self.batch_size = batch_size
        if problems is not None:
            self.problems = problems
        else:
            self.problems = get_random_problems(batch_size, self.problem_size, self.num_objectives)
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

        # TODO: In the future, this image construction logic will be replaced with other methods
        # that don't rely on coordinate-based visualization. The current implementation is temporary.
        self.xy_img = torch.ones((self.batch_size, self.channels, self.img_size, self.img_size))
        
        # Special case: Single objective with multiple channels
        # All channels use the same coordinates to build identical images
        if self.num_objectives == 1 and self.channels > 1:
            # problems.shape: (batch, problem_size, 2) - only [x, y]
            # Build the same image for all channels
            for i in range(self.channels):
                xy_img = self.problems[:, :, 0:2] * self.img_size  # Use the same [x, y] for all channels
                if batch_size == 1:  # special out of index for KroAB
                    xy_img = self.problems[:, :, 0:2] * (self.img_size - 1)
                xy_img = xy_img.int()
                block_indices = xy_img // self.patch_size
                self.block_indices = block_indices[:, :, 0] * self.patches + block_indices[:, :, 1]
                xy_img = xy_img[:, :, None, :] + self.offsets[None, None, :, :].expand(self.batch_size, 1,
                                                                                                  self.offsets.shape[0],
                                                                                                  self.offsets.shape[1])
                xy_img_idx = xy_img.reshape(-1, 2)
                BATCH_IDX = torch.arange(self.batch_size)[:, None, None].expand(self.batch_size, self.problem_size,
                                                                                self.offsets.shape[0]).reshape(-1)
                self.xy_img[BATCH_IDX, i, xy_img_idx[:, 0], xy_img_idx[:, 1]] = 0
        else:
            # General case: Each channel uses its corresponding objective's coordinates
            # For num_objectives==1 with channels==1, this also works correctly
            for i in range(self.channels):
                xy_img = self.problems[:, :, 2 * i: 2 * i + 2] * self.img_size
                if batch_size == 1:  # special out of index for KroAB
                    xy_img = self.problems[:, :, 2 * i: 2 * i + 2] * (self.img_size - 1)
                xy_img = xy_img.int()
                block_indices = xy_img // self.patch_size
                self.block_indices = block_indices[:, :, 0] * self.patches + block_indices[:, :, 1]
                xy_img = xy_img[:, :, None, :] + self.offsets[None, None, :, :].expand(self.batch_size, 1,
                                                                                                  self.offsets.shape[0],
                                                                                                  self.offsets.shape[1])
                xy_img_idx = xy_img.reshape(-1, 2)
                BATCH_IDX = torch.arange(self.batch_size)[:, None, None].expand(self.batch_size, self.problem_size,
                                                                                self.offsets.shape[0]).reshape(-1)
                self.xy_img[BATCH_IDX, i, xy_img_idx[:, 0], xy_img_idx[:, 1]] = 0

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long)
        # shape: (batch, pomo, 0~problem)

        # CREATE STEP STATE
        self.step_state = Step_State(BATCH_IDX=self.BATCH_IDX, POMO_IDX=self.POMO_IDX)
        self.step_state.ninf_mask = torch.zeros((self.batch_size, self.pomo_size, self.problem_size))
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

