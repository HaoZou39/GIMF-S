##########################################################################################
# Machine Environment Config
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train MOTSP with single objective')

# Basic settings
parser.add_argument('--gpu', type=int, default=0, help='GPU device number to use (default: 0)')
parser.add_argument('--debug', action='store_true', help='Enable debug mode')

# Training settings
parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs (default: 100)')
parser.add_argument('--batch_size', type=int, default=64, help='Training batch size (default: 64)')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate (default: 1e-4)')
parser.add_argument('--seed', type=int, default=1234, help='Random seed (default: 1234)')
parser.add_argument('--validation_interval', type=int, default=10, help='Validate every N epochs (default: 10, 0 to disable)')

# Environment settings
parser.add_argument('--use_basemap', type=int, default=1, help='Use basemap as additional channel (default: 1)')
parser.add_argument('--point_style', type=str, default='white_on_black', 
                    choices=['white_on_black', 'black_on_white'], help='Point representation style')
parser.add_argument('--point_dilation', type=str, default='3x3', 
                    choices=['3x3', '1x1'], help='Point dilation size')
parser.add_argument('--basemap_normalize', type=str, default='zscore', 
                    choices=['none', 'zscore'], help='Basemap normalization method')
parser.add_argument('--basemap_norm_clip', type=float, default=3.0, help='Clip value after zscore (default: 3.0)')
parser.add_argument('--use_distance_matrix', type=int, default=1, help='Use roadnet distance matrix (default: 1)')

# Model settings
parser.add_argument('--use_edge_head', type=int, default=1, help='Enable edge prediction head (default: 1)')
parser.add_argument('--use_edge_bias', type=int, default=0, help='Enable edge bias in decoder (default: 0, 不影响路径选择)')

# Auxiliary loss settings
parser.add_argument('--edge_pretrain_enable', type=int, default=0, help='Enable pretrain stage (default: 0)')
parser.add_argument('--edge_pretrain_epochs', type=int, default=5, help='Pretrain epochs (default: 5)')
parser.add_argument('--edge_sup_enable', type=int, default=1, help='Enable edge supervised loss (default: 1)')
parser.add_argument('--edge_sup_weight', type=float, default=1.0, help='Edge supervised loss weight (default: 1.0)')
parser.add_argument('--edge_rank_enable', type=int, default=1, help='Enable edge ranking loss (default: 1)')
parser.add_argument('--edge_rank_weight', type=float, default=0.1, help='Edge ranking loss weight (default: 0.1)')

args = parser.parse_args()

DEBUG_MODE = args.debug
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = args.gpu

##########################################################################################
# Path Config
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils

##########################################################################################
# import
import logging
from utils.utils import create_logger, copy_all_src, get_result_folder
import math
import torch
import numpy as np
import random
from MOTSPTrainer import TSPTrainer as Trainer

##########################################################################################
# parameters - SINGLE OBJECTIVE VERSION

# Multi-dataset configuration
# Each dataset entry contains: name, npz_path, basemap_path
# Using new dataset from MMDataset/30.256330_120.159448
# Dataset structure: MMDataset/30.256330_120.159448/xx_lat_lon_radius/distance_dataset_train_*.npz
# Training set: 12 locations (75% of 16 total locations)
# Test set: 4 locations (25% of 16 total locations) - reserved for testing (folders 00-03)
DATASETS = [
    # Training locations (folders 04-15)
    {
        'name': 'loc_30.283276_120.065850',
        'npz_path': '../../../MMDataset/30.256330_120.159448/04_30.283276_120.065850_3000.0/distance_dataset_train_30.283276_120.065850_3000.0_p20.npz',
        'basemap_path': '../../../MMDataset/30.256330_120.159448/04_30.283276_120.065850_3000.0/mask_prob_30.283276_120.065850_3000.0_z16.tif',
    },
    {
        'name': 'loc_30.283276_120.128249',
        'npz_path': '../../../MMDataset/30.256330_120.159448/05_30.283276_120.128249_3000.0/distance_dataset_train_30.283276_120.128249_3000.0_p20.npz',
        'basemap_path': '../../../MMDataset/30.256330_120.159448/05_30.283276_120.128249_3000.0/mask_prob_30.283276_120.128249_3000.0_z16.tif',
    },
    {
        'name': 'loc_30.283276_120.190647',
        'npz_path': '../../../MMDataset/30.256330_120.159448/06_30.283276_120.190647_3000.0/distance_dataset_train_30.283276_120.190647_3000.0_p20.npz',
        'basemap_path': '../../../MMDataset/30.256330_120.159448/06_30.283276_120.190647_3000.0/mask_prob_30.283276_120.190647_3000.0_z16.tif',
    },
    {
        'name': 'loc_30.283276_120.253046',
        'npz_path': '../../../MMDataset/30.256330_120.159448/07_30.283276_120.253046_3000.0/distance_dataset_train_30.283276_120.253046_3000.0_p20.npz',
        'basemap_path': '../../../MMDataset/30.256330_120.159448/07_30.283276_120.253046_3000.0/mask_prob_30.283276_120.253046_3000.0_z16.tif',
    },
    {
        'name': 'loc_30.229377_120.065850',
        'npz_path': '../../../MMDataset/30.256330_120.159448/08_30.229377_120.065850_3000.0/distance_dataset_train_30.229377_120.065850_3000.0_p20.npz',
        'basemap_path': '../../../MMDataset/30.256330_120.159448/08_30.229377_120.065850_3000.0/mask_prob_30.229377_120.065850_3000.0_z16.tif',
    },
    {
        'name': 'loc_30.229377_120.128249',
        'npz_path': '../../../MMDataset/30.256330_120.159448/09_30.229377_120.128249_3000.0/distance_dataset_train_30.229377_120.128249_3000.0_p20.npz',
        'basemap_path': '../../../MMDataset/30.256330_120.159448/09_30.229377_120.128249_3000.0/mask_prob_30.229377_120.128249_3000.0_z16.tif',
    },
    {
        'name': 'loc_30.229377_120.190647',
        'npz_path': '../../../MMDataset/30.256330_120.159448/10_30.229377_120.190647_3000.0/distance_dataset_train_30.229377_120.190647_3000.0_p20.npz',
        'basemap_path': '../../../MMDataset/30.256330_120.159448/10_30.229377_120.190647_3000.0/mask_prob_30.229377_120.190647_3000.0_z16.tif',
    },
    {
        'name': 'loc_30.229377_120.253046',
        'npz_path': '../../../MMDataset/30.256330_120.159448/11_30.229377_120.253046_3000.0/distance_dataset_train_30.229377_120.253046_3000.0_p20.npz',
        'basemap_path': '../../../MMDataset/30.256330_120.159448/11_30.229377_120.253046_3000.0/mask_prob_30.229377_120.253046_3000.0_z16.tif',
    },
    {
        'name': 'loc_30.175448_120.065850',
        'npz_path': '../../../MMDataset/30.256330_120.159448/12_30.175448_120.065850_3000.0/distance_dataset_train_30.175448_120.065850_3000.0_p20.npz',
        'basemap_path': '../../../MMDataset/30.256330_120.159448/12_30.175448_120.065850_3000.0/mask_prob_30.175448_120.065850_3000.0_z16.tif',
    },
    {
        'name': 'loc_30.175448_120.128249',
        'npz_path': '../../../MMDataset/30.256330_120.159448/13_30.175448_120.128249_3000.0/distance_dataset_train_30.175448_120.128249_3000.0_p20.npz',
        'basemap_path': '../../../MMDataset/30.256330_120.159448/13_30.175448_120.128249_3000.0/mask_prob_30.175448_120.128249_3000.0_z16.tif',
    },
    {
        'name': 'loc_30.175448_120.190647',
        'npz_path': '../../../MMDataset/30.256330_120.159448/14_30.175448_120.190647_3000.0/distance_dataset_train_30.175448_120.190647_3000.0_p20.npz',
        'basemap_path': '../../../MMDataset/30.256330_120.159448/14_30.175448_120.190647_3000.0/mask_prob_30.175448_120.190647_3000.0_z16.tif',
    },
    {
        'name': 'loc_30.175448_120.253046',
        'npz_path': '../../../MMDataset/30.256330_120.159448/15_30.175448_120.253046_3000.0/distance_dataset_train_30.175448_120.253046_3000.0_p20.npz',
        'basemap_path': '../../../MMDataset/30.256330_120.159448/15_30.175448_120.253046_3000.0/mask_prob_30.175448_120.253046_3000.0_z16.tif',
    },
]

env_params = {
    'problem_size': 20,
    'pomo_size': 20,
    'num_objectives': 1,
    'use_basemap': bool(args.use_basemap),  # Enable basemap as additional channel
    
    # Point representation configuration
    'point_style': args.point_style,  # 'white_on_black' or 'black_on_white'
    'point_dilation': args.point_dilation,  # '3x3' or '1x1'
    'basemap_normalize': args.basemap_normalize,  # 'none' or 'zscore'
    'basemap_norm_clip': args.basemap_norm_clip if args.basemap_norm_clip > 0 else None,  # clip after zscore
    
    # Multi-dataset configuration
    'use_custom_dataset': True,  # Set to False to use random generated problems (BENCHMARK TEST)
    'datasets': DATASETS,  # List of dataset configurations
    'use_distance_matrix': bool(args.use_distance_matrix),  # Use pre-computed road network distance matrix
    'dataset_switch_interval': 1,  # Batches before switching to next dataset (reduces basemap overhead)
    
    # Default basemap for random generation mode (when use_custom_dataset=False)
    'basemap_dir': 'data',
    'basemap_pattern': 'basemap_{id}.tif',
    'default_basemap_id': '0',
}

model_params = {
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128**(1/2),
    'encoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'eval_type': 'argmax',
    'hyper_hidden_dim': 256,
    'num_objectives': 1,  # Single objective!
    'in_channels': 2 if args.use_basemap else 1,  # 2 channels with basemap, 1 without
    'patch_size': 16,
    'pixel_density': 56,  # 56 for 256x256, 10 for 48x48
    'fusion_layer_num': 3,
    'bn_num': 10,
    'bn_img_num': 10,

    # Edge-aware auxiliary module
    'use_edge_head': bool(args.use_edge_head),
    'use_edge_bias': bool(args.use_edge_bias),  # 禁用edge_bias，避免干扰RL决策
    'edge_head_hidden_dim': 256,
    'edge_head_use_euclid': True,
    'edge_bias_alpha': 1.0,
    'edge_bias_clip': 5.0,
}

model_params['img_size'] = math.ceil(
    env_params['problem_size'] ** (1 / 2) * model_params['pixel_density'] / model_params['patch_size']) * model_params[
                               'patch_size']
env_params['img_size'] = model_params['img_size']
env_params['patch_size'] = model_params['patch_size']
env_params['in_channels'] = model_params['in_channels']

# Consistency checks
assert env_params['num_objectives'] == model_params['num_objectives'], \
    "num_objectives must match in env and model"

num_objectives = model_params['num_objectives']
in_channels = model_params['in_channels']

if num_objectives == 1:
    # Single objective: in_channels=1 (points only) or in_channels>1 (with basemap)
    if in_channels > 1:
        assert env_params.get('use_basemap', False), \
            f"Single objective with in_channels={in_channels} requires basemap (set use_basemap=True)"
        print(f"Using {in_channels}-channel input: Channel 0=Points, Channel 1+=Basemap")
    else:
        print(f"Using {in_channels}-channel input: Points only (no basemap)")
else:
    # Multi-objective: in_channels must equal num_objectives
    assert in_channels == num_objectives, \
        f"Multi-objective (num_objectives={num_objectives}) requires in_channels={num_objectives}"

optimizer_params = {
    'optimizer': {
        'lr': args.lr, 
        'weight_decay': 1e-6
    },
    'scheduler': {
        'milestones': [180,],
        'gamma': 0.1
    }
}

trainer_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'dec_method': 'WS',  # For single objective, WS is just identity
    'epochs': args.epochs,
    'train_episodes': 100 * 1000,
    'train_batch_size': args.batch_size,
    'random_seed': args.seed,  # Fixed random seed for reproducibility
    
    # Optimality gap validation settings
    'validation_interval': args.validation_interval,  # Validate every N epochs (set to 0 to disable)
    'validation_batch_size': 64,  # Batch size for validation
    
    # Edge-aware multi-task training (roadnet supervision)
    'edge_pretrain': {
        'enable': bool(args.edge_pretrain_enable),
        'epochs': args.edge_pretrain_epochs,
    },
    'edge_supervised': {
        'enable': bool(args.edge_sup_enable),
        'weight': args.edge_sup_weight,
        'unreachable_threshold': None,
        'eps': 1e-6,
    },
    'edge_ranking': {
        'enable': bool(args.edge_rank_enable),
        'weight': args.edge_rank_weight,
        'euclid_topk': 5,
        'margin': 0.5,
        'unreachable_threshold': None,
        'eps': 1e-6,
    },

    'logging': {
        'model_save_interval': 5,
        'img_save_interval': 10,
        'log_image_params_1': {
            'json_foldername': 'log_image_style',
            'filename': 'style_tsp_20_single.json'
        },
        'log_image_params_2': {
            'json_foldername': 'log_image_style',
            'filename': 'style_loss_1_single.json'
        },
    },
    'model_load': {
        'enable': False,  # enable loading pre-trained model
        'path': './result/saved_tsp20_single_model',  # directory path of pre-trained model and log files saved.
        'epoch': 1,  # epoch version of pre-trained model to laod.
    }
}

# Generate logger desc based on dataset and model configuration
def _get_dataset_names():
    """Extract dataset names for logging - use short format to avoid long paths"""
    if env_params.get('use_custom_dataset', False):
        datasets = env_params.get('datasets', [])
        if datasets:
            # Use dataset count and short source identifier to avoid long paths
            num_datasets = len(datasets)
            # Extract source identifier from first dataset path
            first_path = datasets[0].get('npz_path', '')
            if '30.256330_120.159448' in first_path:
                source_id = 'newdata'  # Short identifier for new dataset
            else:
                source_id = 'custom'
            return f'{num_datasets}loc_{source_id}'
        return 'custom'
    return 'random'

def _get_auxiliary_training_suffix():
    """Generate suffix based on auxiliary training tasks configuration"""
    aux_parts = []
    
    # Edge-aware auxiliary training tasks
    edge_pretrain = trainer_params.get('edge_pretrain', {})
    edge_supervised = trainer_params.get('edge_supervised', {})
    edge_ranking = trainer_params.get('edge_ranking', {})
    
    if edge_pretrain.get('enable', False):
        aux_parts.append(f"pretrain{edge_pretrain.get('epochs', 0)}ep")
    
    if edge_supervised.get('enable', False):
        weight = edge_supervised.get('weight', 1.0)
        aux_parts.append(f"sup{weight:.1f}")
    
    if edge_ranking.get('enable', False):
        weight = edge_ranking.get('weight', 0.1)
        aux_parts.append(f"rank{weight:.1f}")
    
    # Edge bias status (important for distinguishing experiments)
    use_edge_bias = model_params.get('use_edge_bias', False)
    if use_edge_bias:
        aux_parts.append("bias_on")
    else:
        aux_parts.append("bias_off")
    
    if aux_parts:
        return 'aux_' + '_'.join(aux_parts)
    else:
        return 'no_aux'

def _get_model_config_suffix():
    """Generate suffix based on model configuration for avoiding overwrites"""
    suffix_parts = []
    
    # Basemap configuration & point representation
    use_basemap = env_params.get('use_basemap', False)
    point_style = env_params.get('point_style', 'white_on_black' if use_basemap else 'black_on_white')
    point_dilation = env_params.get('point_dilation', '3x3' if use_basemap else '1x1')
    
    if use_basemap:
        suffix_parts.append(f"ch{model_params['in_channels']}")  # e.g., ch2
        # Point style: white_on_black -> blackW, black_on_white -> whiteB
        if point_style == 'white_on_black':
            style_str = "blackW"  # black background, white points
        else:
            style_str = "whiteB"  # white background, black points
        # Point dilation: 3x3 or 1x1
        dilation_str = point_dilation
        suffix_parts.append(f"{style_str}{dilation_str}")
    else:
        suffix_parts.extend(["no_basemap", "whiteB1px"])  # white background, black points, 1 pixel
    
    # Distance matrix configuration
    if env_params.get('use_distance_matrix', False):
        suffix_parts.append("roadnet")
    else:
        suffix_parts.append("euclidean")
    
    return '_'.join(suffix_parts)

dataset_names = _get_dataset_names()
model_config = _get_model_config_suffix()
auxiliary_config = _get_auxiliary_training_suffix()
logger_desc = f'train__tsp_n20_single_obj_{dataset_names}_{model_config}_{auxiliary_config}'

logger_params = {
    'log_file': {
        'desc': logger_desc,
        'filename': 'run_log'
    }
}

##########################################################################################
# main
def main():
    if DEBUG_MODE:
        _set_debug_mode()

    # Set random seed for reproducibility
    _set_random_seed(trainer_params['random_seed'])

    create_logger(**logger_params)
    _print_config()

    trainer = Trainer(env_params=env_params,
                      model_params=model_params,
                      optimizer_params=optimizer_params,
                      trainer_params=trainer_params)

    copy_all_src(trainer.result_folder)

    trainer.run()

def _set_random_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # For deterministic behavior (may affect performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed} for reproducibility")

def _set_debug_mode():
    global trainer_params
    trainer_params['epochs'] = 2
    trainer_params['train_episodes'] = 10
    trainer_params['train_batch_size'] = 4


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(f"{g_key}{globals()[g_key]}") for g_key in globals() if g_key.endswith('params')]
    
    # Print TensorBoard command for easy access
    # result_folder is set by create_logger (called before this function)
    result_folder = get_result_folder()
    tensorboard_log_dir = f'{result_folder}/tensorboard'
    logger.info('=' * 80)
    logger.info('TensorBoard Command:')
    logger.info(f'  tensorboard --logdir={tensorboard_log_dir}')
    logger.info('=' * 80)

##########################################################################################

if __name__ == "__main__":
    main()

