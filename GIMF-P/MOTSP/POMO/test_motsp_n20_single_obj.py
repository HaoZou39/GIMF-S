##########################################################################################
# Machine Environment Config
DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0

##########################################################################################
# Path Config
import os
import sys
import torch
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils

##########################################################################################
# import
import logging
from utils.utils import create_logger, copy_all_src
import math
from MOTSPTester import TSPTester as Tester
from MOTSProblemDef import get_random_problems

##########################################################################################
import time

##########################################################################################
# parameters - SINGLE OBJECTIVE VERSION
env_params = {
    'problem_size': 20,
    'pomo_size': 20,
    'num_objectives': 1,  # Single objective
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
    'num_objectives': 1,  # Single objective
    'in_channels': 2,  # Can be >= 1 for single objective
    'patch_size': 16,
    'pixel_density': 56,
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
assert model_params['in_channels'] >= 1, "in_channels must be at least 1"

tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    "dec_method": "WS",
    'model_load': {
        'path': './result/train__tsp_n20_single_obj',
        'info': "Single-Objective TSP20",
        'epoch': 50,
    },
    'test_episodes': 200,
    'test_batch_size': 200,
    'augmentation_enable': True,
    'aug_factor': 1,  # or 8 for single objective
    'aug_batch_size': 200,
}
if tester_params['aug_factor'] > 1:
    tester_params['test_batch_size'] = tester_params['aug_batch_size']

logger_params = {
    'log_file': {
        'desc': 'test__tsp_n20_single_obj',
        'filename': 'run_log'
    }
}

##########################################################################################
def _set_debug_mode():
    global tester_params
    tester_params['test_episodes'] = 100

def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]

##########################################################################################
def main():
    if DEBUG_MODE:
        _set_debug_mode()
    
    create_logger(**logger_params)
    _print_config()

    timer_start = time.time()
    tester = Tester(env_params=env_params,
                    model_params=model_params,
                    tester_params=tester_params)
    
    copy_all_src(tester.result_folder)
    
    test_path = f"./data/testdata_tsp_size{env_params['problem_size']}.pt"
    shared_problem = torch.load(test_path).to(device=CUDA_DEVICE_NUM)
    
    # Convert to single-objective format: only use first 2 dimensions [x, y]
    # Original test data is (batch, problem_size, 4) for 2 objectives
    # Single objective needs (batch, problem_size, 2)
    if shared_problem.shape[2] == 4:
        # Take only the first objective's coordinates [x1, y1]
        shared_problem = shared_problem[:, :, :2]
        print(f"Converted test data from 4D (2 objectives) to 2D (single objective)")
    elif shared_problem.shape[2] != 2:
        raise ValueError(f"Unexpected test data dimension: {shared_problem.shape[2]}. Expected 2 or 4.")

    # Single objective: only one preference vector [1.0]
    pref = torch.tensor([1.0]).cuda()

    batch_size = shared_problem.shape[0]
    
    test_timer_start = time.time()
    aug_score = tester.run(shared_problem, pref)
    test_timer_end = time.time()
    test_time = test_timer_end - test_timer_start
    
    timer_end = time.time()
    total_time = timer_end - timer_start

    # Extract scores (single objective, so only first element)
    scores = np.array(aug_score[0].flatten())
    
    # Statistics
    mean_score = scores.mean()
    std_score = scores.std()
    min_score = scores.min()
    max_score = scores.max()
    
    # Concorde baseline for TSP20
    concorde_baseline = 3.83
    gap = ((mean_score - concorde_baseline) / concorde_baseline) * 100
    
    # Print results
    print('\n' + '='*80)
    print('Single-Objective TSP20 Test Results')
    print('='*80)
    print(f'Test Instances: {batch_size}')
    print(f'Mean Tour Length: {mean_score:.4f}')
    print(f'Std Tour Length: {std_score:.4f}')
    print(f'Min Tour Length: {min_score:.4f}')
    print(f'Max Tour Length: {max_score:.4f}')
    print(f'Concorde Baseline: {concorde_baseline:.4f}')
    print(f'Optimality Gap: {gap:.2f}%')
    print(f'Test Time: {test_time:.4f}s')
    print(f'Total Time: {total_time:.4f}s')
    print('='*80 + '\n')
    
    # Save results
    result_file = f"{tester.result_folder}/single_obj_tsp{env_params['problem_size']}_result.txt"
    with open(result_file, 'w') as f:
        f.write(f"Single-Objective TSP{env_params['problem_size']}\n")
        f.write(f"Model Path: {tester_params['model_load']['path']}\n")
        f.write(f"Model Epoch: {tester_params['model_load']['epoch']}\n")
        f.write(f"Hyper Hidden Dim: {model_params['hyper_hidden_dim']}\n")
        f.write(f"In Channels: {model_params['in_channels']}\n")
        f.write(f"Test Instances: {batch_size}\n")
        f.write(f"Aug Factor: {tester_params['aug_factor']}\n")
        f.write(f"Mean Tour Length: {mean_score:.4f}\n")
        f.write(f"Std Tour Length: {std_score:.4f}\n")
        f.write(f"Min Tour Length: {min_score:.4f}\n")
        f.write(f"Max Tour Length: {max_score:.4f}\n")
        f.write(f"Concorde Baseline: {concorde_baseline:.4f}\n")
        f.write(f"Optimality Gap: {gap:.2f}%\n")
        f.write(f"Test Time: {test_time:.4f}s\n")
        f.write(f"Total Time: {total_time:.4f}s\n")
        f.write(f"Info: {tester_params['model_load']['info']}\n")
    
    # Save all scores
    scores_file = f"{tester.result_folder}/all_scores_n{env_params['problem_size']}.txt"
    np.savetxt(scores_file, scores, delimiter='\n', fmt="%.4f")
    
    print(f"Results saved to: {tester.result_folder}")

##########################################################################################
if __name__ == "__main__":
    main()

