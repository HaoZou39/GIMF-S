##########################################################################################
# Machine Environment Config
DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0

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
from utils.utils import create_logger, copy_all_src
import math
import torch
import numpy as np
import random
from MOTSPTrainer import TSPTrainer as Trainer

##########################################################################################
# parameters - SINGLE OBJECTIVE VERSION
env_params = {
    'problem_size': 20,
    'pomo_size': 20,
    'num_objectives': 1,
    'use_basemap': True,  # Enable basemap as additional channel(s)
    'basemap_dir': 'data',
    'basemap_pattern': 'basemap_{id}.tif',
    'default_basemap_id': '0',  # Default basemap to use
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
    'in_channels': 2,  # 2 channels: points + basemap
    'patch_size': 16,
    'pixel_density': 56,  # 56 for 256x256, 10 for 48x48
    'fusion_layer_num': 3,
    'bn_num': 10,
    'bn_img_num': 10,
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
    # Multi-objective: in_channels must equal num_objectives
    assert in_channels == num_objectives, \
        f"Multi-objective (num_objectives={num_objectives}) requires in_channels={num_objectives}"

optimizer_params = {
    'optimizer': {
        'lr': 1e-4, 
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
    'epochs': 200,
    'train_episodes': 100 * 1000,
    'train_batch_size': 64,
    'random_seed': 1234,  # Fixed random seed for reproducibility
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

logger_params = {
    'log_file': {
        'desc': 'train__tsp_n20_single_obj',
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
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]

##########################################################################################

if __name__ == "__main__":
    main()

