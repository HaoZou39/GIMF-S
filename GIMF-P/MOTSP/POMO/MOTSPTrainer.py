import torch
from logging import getLogger
from tqdm import tqdm
import time
import os

from MOTSPEnv import TSPEnv as Env
from MOTSPModel import TSPModel as Model

from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler

from utils.utils import *

import numpy as np

# TensorBoard support
from torch.utils.tensorboard import SummaryWriter


class OptimalSolutionValidator:
    """
    Validator for computing optimality gap using pre-computed optimal solutions.
    """
    def __init__(self, datasets_config, script_dir, logger):
        """
        Args:
            datasets_config: List of dataset configurations with 'name' and 'npz_path'
            script_dir: Directory of the training script (for resolving relative paths)
            logger: Logger instance
        """
        self.logger = logger
        self.optimal_data = {}  # {dataset_name: {sample_indices, optimal_tours, optimal_distances_norm, ...}}
        self.source_data = {}   # {dataset_name: {matched_node_norm, distance matrices, ...}}
        
        for ds_config in datasets_config:
            name = ds_config['name']
            npz_path = ds_config['npz_path']
            basemap_path = ds_config.get('basemap_path', None)
            
            # Resolve paths
            source_path = os.path.normpath(os.path.join(script_dir, npz_path))
            
            # Construct optimal solution file path
            base_name = os.path.basename(source_path).replace('.npz', '')
            optimal_path = os.path.join(os.path.dirname(source_path), 
                                        f'{base_name}_optimal_undirected_exact_dp.npz')
            
            if not os.path.exists(optimal_path):
                logger.warning(f'Optimal solution file not found for {name}: {optimal_path}')
                continue
            
            if not os.path.exists(source_path):
                logger.warning(f'Source dataset not found for {name}: {source_path}')
                continue
            
            # Load optimal solutions
            opt_data = np.load(optimal_path, allow_pickle=True)
            self.optimal_data[name] = {
                'sample_indices': opt_data['sample_indices'],
                'optimal_tours': opt_data['optimal_tours'],
                'optimal_distances_m': opt_data['optimal_distances_m'],
                'optimal_distances_norm': opt_data['optimal_distances_norm'],
            }
            
            # Load source data for the sampled instances
            src_data = np.load(source_path, allow_pickle=True)
            sample_indices = opt_data['sample_indices']
            
            # Load Euclidean distance matrix if available
            euclid_dist_norm = None
            if 'euclid_dist_norm' in src_data.files:
                euclid_dist_norm = src_data['euclid_dist_norm'][sample_indices]
                logger.info(f'  [OK] Loaded precomputed euclid_dist_norm for {name}')
            else:
                logger.warning(f'  [WARN] euclid_dist_norm not found for {name}. Run precompute_euclid_distance.py first!')
            
            self.source_data[name] = {
                'matched_node_norm': src_data['matched_node_norm'][sample_indices],  # (num_samples, problem_size, 2) 使用匹配到路网的节点坐标
                'undirected_dist_norm': src_data['undirected_dist_norm'][sample_indices],  # (num_samples, problem_size, problem_size)
                'euclid_dist_norm': euclid_dist_norm,  # (num_samples, problem_size, problem_size) 欧式距离矩阵
                'basemap_path': os.path.normpath(os.path.join(script_dir, basemap_path)) if basemap_path else None,
            }
            
            logger.info(f'Loaded {len(sample_indices)} optimal solutions for {name}')
        
        self.dataset_names = list(self.optimal_data.keys())
        logger.info(f'OptimalSolutionValidator initialized with {len(self.dataset_names)} datasets')
    
    def get_validation_batch(self, dataset_name, batch_indices):
        """
        Get a batch of validation problems.
        
        Args:
            dataset_name: Name of the dataset
            batch_indices: Indices within the optimal solution set (0 to num_samples-1)
        
        Returns:
            problems: (batch_size, problem_size, 2)
            distance_matrix: (batch_size, problem_size, problem_size)
            euclid_distance_matrix: (batch_size, problem_size, problem_size) or None
            optimal_distances: (batch_size,)
            basemap_path: Path to basemap file or None
        """
        src = self.source_data[dataset_name]
        opt = self.optimal_data[dataset_name]
        
        problems = torch.from_numpy(src['matched_node_norm'][batch_indices]).float()  # 使用匹配到路网的节点坐标
        distance_matrix = torch.from_numpy(src['undirected_dist_norm'][batch_indices]).float()
        
        # Load Euclidean distance matrix if available
        euclid_distance_matrix = None
        if src['euclid_dist_norm'] is not None:
            euclid_distance_matrix = torch.from_numpy(src['euclid_dist_norm'][batch_indices]).float()
        
        optimal_distances = opt['optimal_distances_norm'][batch_indices]
        basemap_path = src['basemap_path']
        
        return problems, distance_matrix, euclid_distance_matrix, optimal_distances, basemap_path
    
    def get_num_samples(self, dataset_name):
        """Get number of samples for a dataset."""
        return len(self.optimal_data[dataset_name]['sample_indices'])

class TSPTrainer:
    def __init__(self,
                 env_params,
                 model_params,
                 optimizer_params,
                 trainer_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()
        self.result_log = LogData()

        # cuda
        USE_CUDA = self.trainer_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.trainer_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')

        # Main Components
        self.model = Model(**self.model_params)
        
        self.env = Env(**self.env_params)
        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params['scheduler'])

        # Restore
        self.start_epoch = 1
        model_load = trainer_params['model_load']
        if model_load['enable']:
            checkpoint_fullname = '{path}/checkpoint_motsp-{epoch}.pt'.format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.start_epoch = 1 + model_load['epoch']
            self.result_log.set_raw_data(checkpoint['result_log'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.last_epoch = model_load['epoch']-1
            self.logger.info('Saved Model Loaded !!')

        # utility
        self.time_estimator = TimeEstimator()
        
        # TensorBoard writer
        tensorboard_log_dir = f'{self.result_folder}/tensorboard'
        self.tb_writer = SummaryWriter(log_dir=tensorboard_log_dir)
        self.logger.info(f'TensorBoard logs will be saved to: {tensorboard_log_dir}')
        
        # TensorBoard logging settings
        self.tb_batch_log_interval = 100  # Log batch-level metrics every N batches
        self.global_step = 0  # Global step counter for batch-level logging

        # Custom dataset management
        self.use_custom_dataset = env_params.get('use_custom_dataset', False)
        if self.use_custom_dataset:
            datasets = env_params.get('datasets', [])
            if datasets:
                # Multi-dataset mode
                from MOTSP.MOTSProblemDef import MultiDatasetLoader
                switch_interval = env_params.get('dataset_switch_interval', 50)
                self.multi_dataset_loader = MultiDatasetLoader(datasets, switch_interval=switch_interval)
                self.logger.info(f'Using {len(datasets)} datasets (switch every {switch_interval} batches):')
                for ds in self.multi_dataset_loader.get_dataset_info():
                    self.logger.info(f'  - {ds["name"]}: {ds["total_instances"]} instances')
            else:
                # Legacy single dataset mode (backward compatibility)
                self.multi_dataset_loader = None
                self.dataset_path = env_params.get('dataset_path', '')
                self.dataset_idx = 0
                self.logger.info(f'Using single dataset: {self.dataset_path}')
        
        # Optimal solution validator for computing optimality gap
        self.optimal_validator = None
        self.validation_interval = trainer_params.get('validation_interval', 10)  # Validate every N epochs
        self.validation_batch_size = trainer_params.get('validation_batch_size', 64)
        
        if self.use_custom_dataset and env_params.get('datasets', []):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            try:
                self.optimal_validator = OptimalSolutionValidator(
                    datasets_config=env_params['datasets'],
                    script_dir=script_dir,
                    logger=self.logger
                )
                if not self.optimal_validator.dataset_names:
                    self.logger.warning('No optimal solution files found. Validation disabled.')
                    self.optimal_validator = None
                else:
                    self.logger.info(f'Optimality gap validation enabled (every {self.validation_interval} epochs)')
            except Exception as e:
                self.logger.warning(f'Failed to initialize OptimalSolutionValidator: {e}')
                self.optimal_validator = None

    def run(self):
        self.time_estimator.reset(self.start_epoch)
        
        # Create epoch-level progress bar
        total_epochs = self.trainer_params['epochs']
        epoch_pbar = tqdm(range(self.start_epoch, total_epochs + 1), 
                          desc='Training', unit='epoch',
                          bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        
        for epoch in epoch_pbar:
            self.logger.info('=================================================================')

            # Train
            epoch_start_time = time.time()
            train_score_obj1, train_score_obj2, train_loss, epoch_metrics = self._train_one_epoch(epoch)
            epoch_time = time.time() - epoch_start_time
            
            # Update epoch progress bar with current metrics
            epoch_pbar.set_postfix({
                'Loss': f'{train_loss:.4f}', 
                'Score': f'{train_score_obj1:.4f}',
                'Time': f'{epoch_time:.1f}s'
            })
            
            # LR Decay (after optimizer.step())
            self.scheduler.step()
            self.result_log.append('train_score_obj1', epoch, train_score_obj1)
            self.result_log.append('train_score_obj2', epoch, train_score_obj2)
            self.result_log.append('train_loss', epoch, train_loss)
            
            # TensorBoard logging - Epoch level
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Main metrics
            self.tb_writer.add_scalar('Train/Loss', train_loss, epoch)
            self.tb_writer.add_scalar('Train/Score_Obj1', train_score_obj1, epoch)
            self.tb_writer.add_scalar('Train/Score_Obj2', train_score_obj2, epoch)
            
            # Learning rate and timing
            self.tb_writer.add_scalar('Train/Learning_Rate', current_lr, epoch)
            self.tb_writer.add_scalar('Train/Epoch_Time_Sec', epoch_time, epoch)
            
            # Gradient and advantage statistics (from epoch_metrics)
            if 'grad_norm_avg' in epoch_metrics:
                self.tb_writer.add_scalar('Train/Gradient_Norm_Avg', epoch_metrics['grad_norm_avg'], epoch)
                self.tb_writer.add_scalar('Train/Gradient_Norm_Max', epoch_metrics['grad_norm_max'], epoch)
            if 'advantage_std_avg' in epoch_metrics:
                self.tb_writer.add_scalar('Train/Advantage_Std_Avg', epoch_metrics['advantage_std_avg'], epoch)
                self.tb_writer.add_scalar('Train/Advantage_Mean_Avg', epoch_metrics['advantage_mean_avg'], epoch)
            
            # Edge-aware auxiliary losses (epoch-level)
            if 'edge_sup_loss_avg' in epoch_metrics:
                self.tb_writer.add_scalar('Train/Edge_Supervised_Loss', epoch_metrics['edge_sup_loss_avg'], epoch)
            if 'edge_rank_loss_avg' in epoch_metrics:
                self.tb_writer.add_scalar('Train/Edge_Ranking_Loss', epoch_metrics['edge_rank_loss_avg'], epoch)
            if epoch_metrics.get('in_pretrain', False):
                self.tb_writer.add_scalar('Train/In_Pretrain', 1.0, epoch)
            else:
                self.tb_writer.add_scalar('Train/In_Pretrain', 0.0, epoch)
            
            ############################
            # Logs & Checkpoint
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params['epochs'])
            self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
                epoch, self.trainer_params['epochs'], elapsed_time_str, remain_time_str))

            all_done = (epoch == self.trainer_params['epochs'])
            
            # Optimality gap validation (every N epochs or at the end)
            if self.optimal_validator is not None and (epoch % self.validation_interval == 0 or all_done):
                self.logger.info(f'Running optimality gap validation at epoch {epoch}...')
                val_start_time = time.time()
                gap_results = self._validate_with_optimal()
                val_time = time.time() - val_start_time
                
                # Log to TensorBoard
                for dataset_name, metrics in gap_results.items():
                    safe_name = dataset_name.replace(' ', '_')  # TensorBoard doesn't like spaces
                    self.tb_writer.add_scalar(f'Validation/{safe_name}/Gap_Percent', metrics['gap_percent'], epoch)
                    self.tb_writer.add_scalar(f'Validation/{safe_name}/Model_Distance', metrics['model_distance'], epoch)
                    self.tb_writer.add_scalar(f'Validation/{safe_name}/Optimal_Distance', metrics['optimal_distance'], epoch)
                    
                    self.logger.info(f'  {dataset_name}: Gap={metrics["gap_percent"]:.2f}%, '
                                   f'Model={metrics["model_distance"]:.4f}, Optimal={metrics["optimal_distance"]:.4f}')
                
                # Log average gap across all datasets
                if gap_results:
                    avg_gap = np.mean([m['gap_percent'] for m in gap_results.values()])
                    self.tb_writer.add_scalar('Validation/Average_Gap_Percent', avg_gap, epoch)
                    self.logger.info(f'  Average Gap: {avg_gap:.2f}%')
                    self.tb_writer.add_scalar('Validation/Time_Sec', val_time, epoch)
            model_save_interval = self.trainer_params['logging']['model_save_interval']
       
            if epoch == self.start_epoch or all_done or (epoch % model_save_interval) == 0:
                self.logger.info("Saving trained_model")
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'result_log': self.result_log.get_raw_data()
                }
                torch.save(checkpoint_dict, '{}/checkpoint_motsp-{}.pt'.format(self.result_folder, epoch))

            if all_done:
                self.logger.info(" *** Training Done *** ")
                self.logger.info("Now, printing log array...")
                util_print_log_array(self.logger, self.result_log)
                
                # Close TensorBoard writer
                self.tb_writer.close()
                self.logger.info("TensorBoard writer closed.")

    def _train_one_epoch(self, epoch):

        score_AM_obj1 = AverageMeter()
        score_AM_obj2 = AverageMeter()
    
        loss_AM = AverageMeter()
        
        # Additional metrics for TensorBoard
        grad_norm_AM = AverageMeter()
        grad_norm_max = 0.0
        advantage_std_AM = AverageMeter()
        advantage_mean_AM = AverageMeter()
        
        # Edge-aware auxiliary loss metrics
        edge_sup_loss_AM = AverageMeter()
        edge_rank_loss_AM = AverageMeter()
        in_pretrain_count = 0

        train_num_episode = self.trainer_params['train_episodes']
        episode = 0
        loop_cnt = 0
        batch_cnt = 0
        
        while episode < train_num_episode:

            remaining = train_num_episode - episode
            batch_size = min(self.trainer_params['train_batch_size'], remaining)

            avg_score_obj1, avg_score_obj2, avg_loss, batch_metrics = self._train_one_batch(batch_size, epoch)
            score_AM_obj1.update(avg_score_obj1, batch_size)
            score_AM_obj2.update(avg_score_obj2, batch_size)
            loss_AM.update(avg_loss, batch_size)
            
            # Update gradient and advantage metrics
            if 'grad_norm' in batch_metrics:
                grad_norm_AM.update(batch_metrics['grad_norm'], 1)
                grad_norm_max = max(grad_norm_max, batch_metrics['grad_norm'])
            if 'advantage_std' in batch_metrics:
                advantage_std_AM.update(batch_metrics['advantage_std'], batch_size)
                advantage_mean_AM.update(batch_metrics['advantage_mean'], batch_size)
            
            # Update edge-aware auxiliary loss metrics
            if 'edge_sup_loss' in batch_metrics:
                edge_sup_loss_AM.update(batch_metrics['edge_sup_loss'], batch_size)
            if 'edge_rank_loss' in batch_metrics:
                edge_rank_loss_AM.update(batch_metrics['edge_rank_loss'], batch_size)
            if batch_metrics.get('in_pretrain', 0) > 0:
                in_pretrain_count += 1

            episode += batch_size
            batch_cnt += 1
            self.global_step += 1
            
            # TensorBoard batch-level logging (every N batches)
            if batch_cnt % self.tb_batch_log_interval == 0:
                self.tb_writer.add_scalar('Batch/Loss', avg_loss, self.global_step)
                self.tb_writer.add_scalar('Batch/Score_Obj1', avg_score_obj1, self.global_step)
                if 'grad_norm' in batch_metrics:
                    self.tb_writer.add_scalar('Batch/Gradient_Norm', batch_metrics['grad_norm'], self.global_step)
                # Edge-aware auxiliary losses
                if 'edge_sup_loss' in batch_metrics:
                    self.tb_writer.add_scalar('Batch/Edge_Supervised_Loss', batch_metrics['edge_sup_loss'], self.global_step)
                if 'edge_rank_loss' in batch_metrics:
                    self.tb_writer.add_scalar('Batch/Edge_Ranking_Loss', batch_metrics['edge_rank_loss'], self.global_step)

            # Log First 10 Batch, only at the first epoch
            if epoch == self.start_epoch:
                loop_cnt += 1
                if loop_cnt <= 10:
                    self.logger.info('Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%)  Obj1 Score: {:.4f}, Obj2 Score: {:.4f},  Loss: {:.4f}'
                                     .format(epoch, episode, train_num_episode, 100. * episode / train_num_episode,
                                             score_AM_obj1.avg, score_AM_obj2.avg, loss_AM.avg))

        # Log Once, for each epoch
        self.logger.info('Epoch {:3d}: Train ({:3.0f}%)  Obj1 Score: {:.4f}, Obj2 Score: {:.4f},  Loss: {:.4f}'
                         .format(epoch, 100. * episode / train_num_episode,
                                 score_AM_obj1.avg, score_AM_obj2.avg, loss_AM.avg))

        # Prepare epoch metrics for TensorBoard
        epoch_metrics = {
            'grad_norm_avg': grad_norm_AM.avg if grad_norm_AM.count > 0 else 0.0,
            'grad_norm_max': grad_norm_max,
            'advantage_std_avg': advantage_std_AM.avg if advantage_std_AM.count > 0 else 0.0,
            'advantage_mean_avg': advantage_mean_AM.avg if advantage_mean_AM.count > 0 else 0.0,
            # Edge-aware auxiliary losses
            'edge_sup_loss_avg': edge_sup_loss_AM.avg if edge_sup_loss_AM.count > 0 else 0.0,
            'edge_rank_loss_avg': edge_rank_loss_AM.avg if edge_rank_loss_AM.count > 0 else 0.0,
            'in_pretrain': in_pretrain_count > 0,
        }

        return score_AM_obj1.avg, score_AM_obj2.avg, loss_AM.avg, epoch_metrics

    def _train_one_batch(self, batch_size, epoch: int):

        # Prep
        ###############################################
        self.model.train()
        
        # Load problems: either from custom dataset or randomly generated
        if self.use_custom_dataset:
            if hasattr(self, 'multi_dataset_loader') and self.multi_dataset_loader is not None:
                # Multi-dataset mode: sample from multiple datasets with basemap
                # Returns: problems, dist_matrix, euclid_dist, basemap_path, dataset_name
                problems, dist_matrix, euclid_dist, basemap_path, dataset_name = self.multi_dataset_loader.sample_batch(batch_size)
                self.env.load_problems(batch_size, problems=problems, distance_matrix=dist_matrix, 
                                       euclid_distance_matrix=euclid_dist, basemap_path=basemap_path)
            else:
                # Legacy single dataset mode
                from MOTSP.MOTSProblemDef import load_problems_from_npz
                problems, dist_matrix, self.dataset_idx = load_problems_from_npz(
                    self.dataset_path, batch_size, self.dataset_idx
                )
                self.env.load_problems(batch_size, problems=problems, distance_matrix=dist_matrix)
        else:
            self.env.load_problems(batch_size)

        # Generate preference vector
        num_objectives = self.env.num_objectives
        if num_objectives == 1:
            # For single objective, use [1.0]
            pref = torch.tensor([1.0]).float()
        else:
            # For multi-objective, sample from Dirichlet distribution
            alpha = 1
            alpha_vec = tuple([alpha] * num_objectives)
            pref = np.random.dirichlet(alpha_vec, None)
            pref = torch.tensor(pref).float()


        reset_state, _, _ = self.env.reset()
        
        self.model.decoder.assign(pref)
        self.model.pre_forward(reset_state)

        # ------------------------------------------------------------
        # Edge-aware auxiliary losses (optional)
        # ------------------------------------------------------------
        edge_sup_cfg = self.trainer_params.get('edge_supervised', {})
        edge_rank_cfg = self.trainer_params.get('edge_ranking', {})
        pretrain_cfg = self.trainer_params.get('edge_pretrain', {})

        # Default: disabled unless explicitly enabled
        edge_sup_enable = bool(edge_sup_cfg.get('enable', False))
        edge_rank_enable = bool(edge_rank_cfg.get('enable', False))

        # Compute edge losses only if we have a distance matrix (roadnet supervision)
        loss_edge_sup = torch.tensor(0.0, device=reset_state.problems.device)
        loss_edge_rank = torch.tensor(0.0, device=reset_state.problems.device)

        if (edge_sup_enable or edge_rank_enable) and getattr(self.env, 'distance_matrix', None) is not None:
            dist_mat = self.env.distance_matrix
            euclid_dist_mat = getattr(self.env, 'euclid_distance_matrix', None)  # Precomputed Euclidean distance
            
            if edge_sup_enable and hasattr(self.model, 'compute_edge_supervised_loss'):
                loss_edge_sup = self.model.compute_edge_supervised_loss(
                    problems=reset_state.problems,
                    distance_matrix=dist_mat,
                    euclid_distance_matrix=euclid_dist_mat,  # Pass precomputed Euclidean distance
                    unreachable_threshold=edge_sup_cfg.get('unreachable_threshold', None),
                    eps=float(edge_sup_cfg.get('eps', 1e-6)),
                )
            if edge_rank_enable and hasattr(self.model, 'compute_edge_hard_ranking_loss'):
                loss_edge_rank = self.model.compute_edge_hard_ranking_loss(
                    problems=reset_state.problems,
                    distance_matrix=dist_mat,
                    euclid_distance_matrix=euclid_dist_mat,  # Pass precomputed Euclidean distance
                    euclid_topk=int(edge_rank_cfg.get('euclid_topk', 5)),
                    margin=float(edge_rank_cfg.get('margin', 0.5)),
                    unreachable_threshold=edge_rank_cfg.get('unreachable_threshold', edge_sup_cfg.get('unreachable_threshold', None)),
                    eps=float(edge_rank_cfg.get('eps', 1e-6)),
                )

        # Optional pretrain stage: for first K epochs, train only edge losses (no RL rollout)
        pretrain_enable = bool(pretrain_cfg.get('enable', False))
        pretrain_epochs = int(pretrain_cfg.get('epochs', 0))
        in_pretrain = pretrain_enable and (epoch <= pretrain_epochs)

        if in_pretrain:
            lam_sup = float(edge_sup_cfg.get('weight', 1.0))
            lam_rank = float(edge_rank_cfg.get('weight', 0.1))
            loss_mean = lam_sup * loss_edge_sup + lam_rank * loss_edge_rank

            # Backward & step
            self.model.zero_grad()
            loss_mean.backward()

            total_norm = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            grad_norm = total_norm ** 0.5

            self.optimizer.step()

            batch_metrics = {
                'grad_norm': grad_norm,
                'advantage_std': 0.0,
                'advantage_mean': 0.0,
                'edge_sup_loss': float(loss_edge_sup.item()),
                'edge_rank_loss': float(loss_edge_rank.item()),
                'in_pretrain': 1.0,
            }

            # No RL score in this stage
            return 0.0, 0.0, loss_mean.item(), batch_metrics
        
        prob_list = torch.zeros(size=(batch_size, self.env.pomo_size, 0))
      
        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step()
        
        while not done:
            selected, prob = self.model(state)
            # shape: (batch, pomo)
            state, reward, done = self.env.step(selected)
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)
            
        # Loss
        ###############################################
        # reward was negative, here we set it to positive to calculate TCH
        reward = - reward
        if self.trainer_params['dec_method'] == "WS":
            tch_reward = (pref * reward).sum(dim=2)
        elif self.trainer_params['dec_method'] == "TCH":
            z = torch.ones(reward.shape).cuda() * 0.0
            tch_reward = pref * (reward - z)
            tch_reward, _ = tch_reward.max(dim=2)
        else:
            return NotImplementedError
        
        # set back reward to negative
        reward = -reward
        tch_reward = -tch_reward

        log_prob = prob_list.log().sum(dim=2)

        # shape = (batch, group)
    
        tch_advantage = tch_reward - tch_reward.mean(dim=1, keepdim=True)
    
        tch_loss = -tch_advantage * log_prob # Minus Sign
        # shape = (batch, group)
        loss_mean = tch_loss.mean()

        # Add edge-aware auxiliary losses (multi-task)
        if (edge_sup_enable or edge_rank_enable):
            lam_sup = float(edge_sup_cfg.get('weight', 0.0))
            lam_rank = float(edge_rank_cfg.get('weight', 0.0))
            loss_mean = loss_mean + lam_sup * loss_edge_sup + lam_rank * loss_edge_rank
        
        # Collect advantage statistics for TensorBoard
        advantage_std = tch_advantage.std().item()
        advantage_mean = tch_advantage.mean().item()
        
        # Score
        ###############################################
        _ , max_idx = tch_reward.max(dim=1)
        max_idx = max_idx.reshape(max_idx.shape[0],1)
        
        # Extract best reward for each objective
        num_objectives = self.env.num_objectives
        score_means = []
        for obj_idx in range(num_objectives):
            max_reward_obj = reward[:,:,obj_idx].gather(1, max_idx)
            score_mean_obj = - max_reward_obj.float().mean()
            score_means.append(score_mean_obj)
        
        # For backward compatibility, keep obj1 and obj2 names
        score_mean_obj1 = score_means[0]
        score_mean_obj2 = score_means[1] if num_objectives > 1 else score_means[0]
    
        #Step & Return
        ################################################
        self.model.zero_grad()
        loss_mean.backward()
        
        # Calculate gradient norm before optimizer step
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        grad_norm = total_norm ** 0.5
        
        self.optimizer.step()
        
        # Prepare batch metrics for TensorBoard
        batch_metrics = {
            'grad_norm': grad_norm,
            'advantage_std': advantage_std,
            'advantage_mean': advantage_mean,
            'edge_sup_loss': float(loss_edge_sup.item()),
            'edge_rank_loss': float(loss_edge_rank.item()),
            'in_pretrain': 0.0,
        }
        
        return score_mean_obj1.item(), score_mean_obj2.item(), loss_mean.item(), batch_metrics

    def _validate_with_optimal(self):
        """
        Validate model performance against optimal solutions.
        
        Returns:
            dict: {dataset_name: {'gap_percent': float, 'model_distance': float, 'optimal_distance': float}}
        """
        self.model.eval()
        results = {}
        
        with torch.no_grad():
            for dataset_name in self.optimal_validator.dataset_names:
                num_samples = self.optimal_validator.get_num_samples(dataset_name)
                batch_size = min(self.validation_batch_size, num_samples)
                
                model_distances = []
                optimal_distances = []
                
                # Process all samples in batches
                for start_idx in range(0, num_samples, batch_size):
                    end_idx = min(start_idx + batch_size, num_samples)
                    batch_indices = np.arange(start_idx, end_idx)
                    actual_batch_size = len(batch_indices)
                    
                    # Get validation batch
                    problems, dist_matrix, euclid_dist, opt_dist, basemap_path = \
                        self.optimal_validator.get_validation_batch(dataset_name, batch_indices)
                    
                    # Load problems into environment
                    self.env.load_problems(
                        actual_batch_size, 
                        problems=problems, 
                        distance_matrix=dist_matrix,
                        euclid_distance_matrix=euclid_dist,
                        basemap_path=basemap_path
                    )
                    
                    # Generate preference vector (single objective)
                    pref = torch.tensor([1.0]).float()
                    
                    # Reset environment
                    reset_state, _, _ = self.env.reset()
                    
                    # Model forward
                    self.model.decoder.assign(pref)
                    self.model.pre_forward(reset_state)
                    
                    # POMO Rollout (greedy)
                    state, reward, done = self.env.pre_step()
                    while not done:
                        selected, _ = self.model(state)
                        state, reward, done = self.env.step(selected)
                    
                    # reward shape: (batch, pomo, num_objectives)
                    # For single objective, get the best among POMO
                    # reward is negative distance, so we negate and take min
                    tour_distances = -reward[:, :, 0]  # (batch, pomo)
                    best_distances = tour_distances.min(dim=1).values  # (batch,)
                    
                    model_distances.extend(best_distances.cpu().numpy().tolist())
                    optimal_distances.extend(opt_dist.tolist())
                
                # Calculate gap: (model - optimal) / optimal * 100%
                model_distances = np.array(model_distances)
                optimal_distances = np.array(optimal_distances)
                
                gap_percent = np.mean((model_distances - optimal_distances) / optimal_distances * 100)
                
                results[dataset_name] = {
                    'gap_percent': gap_percent,
                    'model_distance': np.mean(model_distances),
                    'optimal_distance': np.mean(optimal_distances),
                }
        
        self.model.train()
        return results