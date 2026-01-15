#!/usr/bin/env python3
"""
Test trained MOTSP models on optimal solution benchmark.

This script evaluates trained models (with or without basemap) against 
pre-computed optimal solutions and reports detailed performance metrics.

Usage:
    # Test a single model with basemap
    python test_model.py --checkpoint path/to/checkpoint.pt --use_basemap
    
    # Test a single model without basemap
    python test_model.py --checkpoint path/to/checkpoint.pt --no_basemap
    
    # Compare two models (with and without basemap)
    python test_model.py --compare \
        --checkpoint_basemap path/to/with_basemap.pt \
        --checkpoint_no_basemap path/to/without_basemap.pt
    
    # Specify GPU
    python test_model.py --checkpoint path/to/checkpoint.pt --gpu 0
"""

import argparse
import os
import sys
import math
import time

# Path Config
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils

import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from MOTSPEnv import TSPEnv as Env
from MOTSPModel import TSPModel as Model


##########################################################################################
# Dataset Configuration
##########################################################################################

DATASETS = [
    {
        'name': '杭州',
        'npz_path': '../../../MMDataset/杭州/distance_dataset_30.318899_120.055447_5000.0.npz',
        'basemap_path': '../../../MMDataset/杭州/mask_prob_30.318899_120.055447_5000.0_z16.float32.tif',
    },
    {
        'name': '上海',
        'npz_path': '../../../MMDataset/上海/distance_dataset_31.240186_121.496062_5000.0.npz',
        'basemap_path': '../../../MMDataset/上海/mask_prob_31.240186_121.496062_5000.0_z16.float32.tif',
    },
    {
        'name': '柏林',
        'npz_path': '../../../MMDataset/柏林/distance_dataset_52.516298_13.377914_5000.0.npz',
        'basemap_path': '../../../MMDataset/柏林/mask_prob_52.516298_13.377914_5000.0_z16.float32.tif',
    },
    {
        'name': '鹤岗',
        'npz_path': '../../../MMDataset/鹤岗/distance_dataset_47.332394_130.278898_5000.0.npz',
        'basemap_path': '../../../MMDataset/鹤岗/mask_prob_47.332394_130.278898_5000.0_z16.float32.tif',
    },
    {
        'name': '苏州',
        'npz_path': '../../../MMDataset/苏州/distance_dataset_test_31.298909_120.579205_5000.0.npz',
        'basemap_path': '../../../MMDataset/苏州/mask_prob_31.298909_120.579205_5000.0_z16.float32.tif',
    },
]


##########################################################################################
# Helper Functions
##########################################################################################

def get_env_params(use_basemap=True, point_style=None, point_dilation=None, problem_size=20):
    """
    Get environment parameters based on configuration.
    
    Args:
        use_basemap: Whether to use basemap (in_channels=2 if True, 1 if False)
        point_style: 'white_on_black' or 'black_on_white' (default depends on use_basemap)
        point_dilation: '3x3' or '1x1' (default depends on use_basemap)
        problem_size: Number of nodes in TSP
    """
    in_channels = 2 if use_basemap else 1
    pixel_density = 56
    patch_size = 16
    img_size = math.ceil(problem_size ** (1/2) * pixel_density / patch_size) * patch_size
    
    # Set defaults based on use_basemap if not specified
    if point_style is None:
        point_style = 'white_on_black' if use_basemap else 'black_on_white'
    if point_dilation is None:
        point_dilation = '3x3' if use_basemap else '1x1'
    
    return {
        'problem_size': problem_size,
        'pomo_size': problem_size,
        'num_objectives': 1,
        'use_basemap': use_basemap,
        'point_style': point_style,
        'point_dilation': point_dilation,
        'use_custom_dataset': True,
        'datasets': DATASETS,
        'use_distance_matrix': True,
        'dataset_switch_interval': 10,
        'basemap_dir': 'data',
        'basemap_pattern': 'basemap_{id}.tif',
        'default_basemap_id': '0',
        'img_size': img_size,
        'patch_size': patch_size,
        'in_channels': in_channels,
    }


def get_model_params(use_basemap=True, problem_size=20):
    """Get model parameters based on basemap usage."""
    in_channels = 2 if use_basemap else 1
    pixel_density = 56
    patch_size = 16
    img_size = math.ceil(problem_size ** (1/2) * pixel_density / patch_size) * patch_size
    
    return {
        'embedding_dim': 128,
        'sqrt_embedding_dim': 128**(1/2),
        'encoder_layer_num': 6,
        'qkv_dim': 16,
        'head_num': 8,
        'logit_clipping': 10,
        'ff_hidden_dim': 512,
        'eval_type': 'argmax',
        'hyper_hidden_dim': 256,
        'num_objectives': 1,
        'in_channels': in_channels,
        'patch_size': patch_size,
        'pixel_density': pixel_density,
        'fusion_layer_num': 3,
        'bn_num': 10,
        'bn_img_num': 10,
        'img_size': img_size,
    }


# Pre-defined model configurations
MODEL_CONFIGS = {
    'no_basemap': {
        'name': 'No Basemap (黑点白底 1x1)',
        'use_basemap': False,
        'point_style': 'black_on_white',
        'point_dilation': '1x1',
    },
    'basemap_blackW1x1': {
        'name': 'Basemap + 黑点白底 1x1 (ch2_roadnet)',
        'use_basemap': True,
        'point_style': 'black_on_white',
        'point_dilation': '1x1',
    },
    'basemap_whiteB3x3': {
        'name': 'Basemap + 白点黑底 3x3 (ch2_blackW3x3)',
        'use_basemap': True,
        'point_style': 'white_on_black',
        'point_dilation': '3x3',
    },
}


def load_optimal_solutions(script_dir, dataset_names=None):
    """Load pre-computed optimal solutions for all datasets.
    
    Args:
        script_dir: Directory of the script for resolving relative paths.
        dataset_names: Optional list of dataset names to load. If None, load all.
    """
    optimal_data = {}
    source_data = {}
    
    datasets_to_load = DATASETS
    if dataset_names:
        datasets_to_load = [d for d in DATASETS if d['name'] in dataset_names]
        if not datasets_to_load:
            print(f"Warning: No matching datasets found for: {dataset_names}")
            print(f"Available datasets: {[d['name'] for d in DATASETS]}")
    
    for ds_config in datasets_to_load:
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
            print(f'Warning: Optimal solution file not found for {name}: {optimal_path}')
            continue
        
        if not os.path.exists(source_path):
            print(f'Warning: Source dataset not found for {name}: {source_path}')
            continue
        
        # Load optimal solutions
        opt_data = np.load(optimal_path, allow_pickle=True)
        optimal_data[name] = {
            'sample_indices': opt_data['sample_indices'],
            'optimal_tours': opt_data['optimal_tours'],
            'optimal_distances_m': opt_data['optimal_distances_m'],
            'optimal_distances_norm': opt_data['optimal_distances_norm'],
        }
        
        # Load source data for the sampled instances
        src_data = np.load(source_path, allow_pickle=True)
        sample_indices = opt_data['sample_indices']
        
        source_data[name] = {
            'matched_node_norm': src_data['matched_node_norm'][sample_indices],  # 使用匹配到路网的节点坐标
            'undirected_dist_norm': src_data['undirected_dist_norm'][sample_indices],
            'basemap_path': os.path.normpath(os.path.join(script_dir, basemap_path)) if basemap_path else None,
        }
        
        print(f'Loaded {len(sample_indices)} optimal solutions for {name}')
    
    return optimal_data, source_data


def evaluate_model(model, env, optimal_data, source_data, batch_size=64, use_basemap=True, 
                   point_style=None, point_dilation=None, device='cpu'):
    """
    Evaluate model on optimal solution test set.
    
    Returns:
        dict: {dataset_name: {'gap_percent': float, 'gap_std': float, 'model_distances': array, ...}}
    """
    model.eval()
    results = {}
    
    with torch.no_grad():
        for dataset_name in optimal_data.keys():
            opt = optimal_data[dataset_name]
            src = source_data[dataset_name]
            
            num_samples = len(opt['sample_indices'])
            
            model_distances = []
            all_gaps = []
            inference_times = []
            
            # Process all samples in batches
            pbar = tqdm(range(0, num_samples, batch_size), 
                       desc=f'Testing {dataset_name}', leave=False)
            
            for start_idx in pbar:
                end_idx = min(start_idx + batch_size, num_samples)
                batch_indices = np.arange(start_idx, end_idx)
                actual_batch_size = len(batch_indices)
                
                # Get batch data (使用匹配到路网的节点坐标，与距离矩阵一致)
                problems = torch.from_numpy(src['matched_node_norm'][batch_indices]).float().to(device)
                dist_matrix = torch.from_numpy(src['undirected_dist_norm'][batch_indices]).float().to(device)
                opt_dist = opt['optimal_distances_norm'][batch_indices]
                basemap_path = src['basemap_path'] if use_basemap else None
                
                # Load problems into environment
                env.load_problems(
                    actual_batch_size, 
                    problems=problems, 
                    distance_matrix=dist_matrix,
                    basemap_path=basemap_path
                )
                
                # Generate preference vector (single objective)
                pref = torch.tensor([1.0]).float().to(device)
                
                # Reset environment
                reset_state, _, _ = env.reset()
                
                # Model forward
                model.decoder.assign(pref)
                model.pre_forward(reset_state)
                
                # POMO Rollout (greedy)
                start_time = time.time()
                state, reward, done = env.pre_step()
                while not done:
                    selected, _ = model(state)
                    state, reward, done = env.step(selected)
                inference_time = time.time() - start_time
                
                # Get best tour distance from POMO
                tour_distances = -reward[:, :, 0]  # (batch, pomo)
                best_distances = tour_distances.min(dim=1).values  # (batch,)
                
                # Calculate gaps for this batch
                batch_model_dist = best_distances.cpu().numpy()
                batch_gaps = (batch_model_dist - opt_dist) / opt_dist * 100
                
                model_distances.extend(batch_model_dist.tolist())
                all_gaps.extend(batch_gaps.tolist())
                inference_times.append(inference_time)
                
                # Update progress bar
                current_gap = np.mean(batch_gaps)
                pbar.set_postfix({'Gap': f'{current_gap:.2f}%'})
            
            # Calculate statistics
            model_distances = np.array(model_distances)
            optimal_distances = opt['optimal_distances_norm']
            all_gaps = np.array(all_gaps)
            
            results[dataset_name] = {
                'gap_percent_mean': np.mean(all_gaps),
                'gap_percent_std': np.std(all_gaps),
                'gap_percent_min': np.min(all_gaps),
                'gap_percent_max': np.max(all_gaps),
                'model_distance_mean': np.mean(model_distances),
                'model_distance_std': np.std(model_distances),
                'optimal_distance_mean': np.mean(optimal_distances),
                'optimal_distance_std': np.std(optimal_distances),
                'num_samples': num_samples,
                'inference_time_total': sum(inference_times),
                'inference_time_per_instance': sum(inference_times) / num_samples,
                'model_distances': model_distances,
                'optimal_distances': optimal_distances,
                'gaps': all_gaps,
            }
    
    model.train()
    return results


def print_results(results, model_name="Model"):
    """Print formatted evaluation results."""
    print(f"\n{'='*70}")
    print(f" {model_name} - Evaluation Results")
    print(f"{'='*70}")
    
    all_gaps = []
    
    for dataset_name, metrics in results.items():
        print(f"\n{dataset_name}:")
        print(f"  Samples: {metrics['num_samples']}")
        print(f"  Gap:     {metrics['gap_percent_mean']:.2f}% ± {metrics['gap_percent_std']:.2f}%")
        print(f"           [Min: {metrics['gap_percent_min']:.2f}%, Max: {metrics['gap_percent_max']:.2f}%]")
        print(f"  Model:   {metrics['model_distance_mean']:.4f} ± {metrics['model_distance_std']:.4f}")
        print(f"  Optimal: {metrics['optimal_distance_mean']:.4f} ± {metrics['optimal_distance_std']:.4f}")
        print(f"  Time:    {metrics['inference_time_per_instance']*1000:.2f} ms/instance")
        
        all_gaps.extend(metrics['gaps'].tolist())
    
    all_gaps = np.array(all_gaps)
    print(f"\n{'─'*70}")
    print(f"Overall Average Gap: {np.mean(all_gaps):.2f}% ± {np.std(all_gaps):.2f}%")
    print(f"{'='*70}")
    
    return np.mean(all_gaps)


def print_comparison(results_basemap, results_no_basemap):
    """Print comparison between two models."""
    print(f"\n{'='*80}")
    print(f" Model Comparison: With Basemap vs Without Basemap")
    print(f"{'='*80}")
    
    print(f"\n{'Dataset':<12} {'With Basemap':<20} {'Without Basemap':<20} {'Improvement':<15}")
    print(f"{'─'*80}")
    
    improvements = []
    
    for dataset_name in results_basemap.keys():
        gap_with = results_basemap[dataset_name]['gap_percent_mean']
        gap_without = results_no_basemap[dataset_name]['gap_percent_mean']
        improvement = gap_without - gap_with  # Positive means basemap is better
        
        print(f"{dataset_name:<12} {gap_with:>6.2f}% ± {results_basemap[dataset_name]['gap_percent_std']:<6.2f}  "
              f"{gap_without:>6.2f}% ± {results_no_basemap[dataset_name]['gap_percent_std']:<6.2f}  "
              f"{improvement:>+.2f}%")
        
        improvements.append(improvement)
    
    avg_improvement = np.mean(improvements)
    print(f"{'─'*80}")
    print(f"{'Average':<12} {np.mean([r['gap_percent_mean'] for r in results_basemap.values()]):>6.2f}%          "
          f"     {np.mean([r['gap_percent_mean'] for r in results_no_basemap.values()]):>6.2f}%          "
          f"     {avg_improvement:>+.2f}%")
    print(f"{'='*80}")
    
    if avg_improvement > 0:
        print(f"\n✓ Model WITH basemap performs better by {avg_improvement:.2f}% on average")
    elif avg_improvement < 0:
        print(f"\n✓ Model WITHOUT basemap performs better by {-avg_improvement:.2f}% on average")
    else:
        print(f"\n✓ Both models perform equally")


def test_single_model(checkpoint_path, config_name, gpu, batch_size, dataset_names=None):
    """
    Test a single model with specified configuration.
    
    Args:
        checkpoint_path: Path to model checkpoint
        config_name: Configuration name from MODEL_CONFIGS or dict with config
        gpu: GPU device number (-1 for CPU)
        batch_size: Batch size for evaluation
        dataset_names: Optional list of dataset names to evaluate on
    """
    # Get configuration
    if isinstance(config_name, str):
        if config_name not in MODEL_CONFIGS:
            raise ValueError(f"Unknown config: {config_name}. Available: {list(MODEL_CONFIGS.keys())}")
        config = MODEL_CONFIGS[config_name]
    else:
        config = config_name
    
    use_basemap = config['use_basemap']
    point_style = config.get('point_style')
    point_dilation = config.get('point_dilation')
    model_name = config.get('name', f"Model ({config_name})")
    
    # Setup device
    if gpu >= 0 and torch.cuda.is_available():
        device = torch.device('cuda', gpu)
        torch.cuda.set_device(gpu)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        print(f"Using GPU: {gpu}")
    else:
        device = torch.device('cpu')
        torch.set_default_tensor_type('torch.FloatTensor')
        print("Using CPU")
    
    # Load model
    print(f"\nLoading model from: {checkpoint_path}")
    print(f"Configuration: {model_name}")
    print(f"  use_basemap: {use_basemap}")
    print(f"  point_style: {point_style}")
    print(f"  point_dilation: {point_dilation}")
    
    env_params = get_env_params(
        use_basemap=use_basemap,
        point_style=point_style,
        point_dilation=point_dilation
    )
    model_params = get_model_params(use_basemap=use_basemap)
    
    model = Model(**model_params)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Create environment
    env = Env(**env_params)
    
    # Load optimal solutions
    script_dir = os.path.dirname(os.path.abspath(__file__))
    optimal_data, source_data = load_optimal_solutions(script_dir, dataset_names=dataset_names)
    
    if not optimal_data:
        print("Error: No optimal solution data found. Run solve_optimal_ortools.py first.")
        return None
    
    # Evaluate
    print("\nEvaluating model...")
    results = evaluate_model(
        model, env, optimal_data, source_data,
        batch_size=batch_size,
        use_basemap=use_basemap,
        point_style=point_style,
        point_dilation=point_dilation,
        device=device
    )
    
    # Print results
    avg_gap = print_results(results, model_name)
    
    return results, avg_gap


def compare_models(checkpoint_basemap, checkpoint_no_basemap, gpu, batch_size, dataset_names=None):
    """Compare two models: with and without basemap (legacy function)."""
    # Use the new test_multiple_models function
    checkpoints = {
        'basemap_whiteB3x3': checkpoint_basemap,
        'no_basemap': checkpoint_no_basemap,
    }
    return test_multiple_models(checkpoints, gpu, batch_size, dataset_names=dataset_names)


def test_multiple_models(checkpoints, gpu, batch_size, dataset_names=None):
    """
    Test multiple models with different configurations.
    
    Args:
        checkpoints: Dict of {config_name: checkpoint_path}
        gpu: GPU device number (-1 for CPU)
        batch_size: Batch size for evaluation
        dataset_names: Optional list of dataset names to evaluate on
    
    Returns:
        Dict of {config_name: results}
    """
    # Setup device
    if gpu >= 0 and torch.cuda.is_available():
        device = torch.device('cuda', gpu)
        torch.cuda.set_device(gpu)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        print(f"Using GPU: {gpu}")
    else:
        device = torch.device('cpu')
        torch.set_default_tensor_type('torch.FloatTensor')
        print("Using CPU")
    
    # Load optimal solutions
    script_dir = os.path.dirname(os.path.abspath(__file__))
    optimal_data, source_data = load_optimal_solutions(script_dir, dataset_names=dataset_names)
    
    if not optimal_data:
        print("Error: No optimal solution data found. Run solve_optimal_ortools.py first.")
        return
    
    all_results = {}
    
    for config_name, checkpoint_path in checkpoints.items():
        if config_name not in MODEL_CONFIGS:
            print(f"Warning: Unknown config '{config_name}', skipping")
            continue
        
        config = MODEL_CONFIGS[config_name]
        use_basemap = config['use_basemap']
        point_style = config.get('point_style')
        point_dilation = config.get('point_dilation')
        model_name = config['name']
        
        print(f"\n{'='*70}")
        print(f"Testing: {model_name}")
        print(f"{'='*70}")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Config: use_basemap={use_basemap}, point_style={point_style}, point_dilation={point_dilation}")
        
        env_params = get_env_params(
            use_basemap=use_basemap,
            point_style=point_style,
            point_dilation=point_dilation
        )
        model_params = get_model_params(use_basemap=use_basemap)
        
        model = Model(**model_params)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        
        env = Env(**env_params)
        
        print("\nEvaluating...")
        results = evaluate_model(
            model, env, optimal_data, source_data,
            batch_size=batch_size,
            use_basemap=use_basemap,
            point_style=point_style,
            point_dilation=point_dilation,
            device=device
        )
        
        all_results[config_name] = results
        
        # Clean up
        del model, env
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Print individual results
    for config_name, results in all_results.items():
        model_name = MODEL_CONFIGS[config_name]['name']
        print_results(results, model_name)
    
    # Print comparison table
    print_multi_comparison(all_results)
    
    return all_results


def print_multi_comparison(all_results):
    """Print comparison table for multiple models."""
    if len(all_results) < 2:
        return
    
    print(f"\n{'='*100}")
    print(f" Multi-Model Comparison")
    print(f"{'='*100}")
    
    # Get dataset names
    first_results = list(all_results.values())[0]
    dataset_names = list(first_results.keys())
    
    # Header
    header = f"{'Dataset':<12}"
    for config_name in all_results.keys():
        model_name = MODEL_CONFIGS[config_name]['name']
        # Truncate model name if too long
        short_name = model_name[:20] if len(model_name) > 20 else model_name
        header += f" {short_name:<22}"
    print(header)
    print(f"{'─'*100}")
    
    # Data rows
    for dataset_name in dataset_names:
        row = f"{dataset_name:<12}"
        for config_name, results in all_results.items():
            gap = results[dataset_name]['gap_percent_mean']
            std = results[dataset_name]['gap_percent_std']
            row += f" {gap:>6.2f}% ± {std:<6.2f}      "
        print(row)
    
    # Average row
    print(f"{'─'*100}")
    row = f"{'Average':<12}"
    avg_gaps = []
    for config_name, results in all_results.items():
        avg_gap = np.mean([r['gap_percent_mean'] for r in results.values()])
        avg_gaps.append((config_name, avg_gap))
        row += f" {avg_gap:>6.2f}%                  "
    print(row)
    print(f"{'='*100}")
    
    # Find best model
    best_config, best_gap = min(avg_gaps, key=lambda x: x[1])
    best_name = MODEL_CONFIGS[best_config]['name']
    print(f"\n✓ Best Model: {best_name} with average gap {best_gap:.2f}%")


def main():
    parser = argparse.ArgumentParser(
        description='Test trained MOTSP models on optimal solution benchmark',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test a single model with specific config
  python test_model.py --checkpoint path/to/checkpoint.pt --config basemap_whiteB3x3
  
  # Test a single model without basemap
  python test_model.py --checkpoint path/to/checkpoint.pt --config no_basemap
  
  # Compare two models (legacy)
  python test_model.py --compare \\
      --checkpoint_basemap path/to/with_basemap.pt \\
      --checkpoint_no_basemap path/to/without_basemap.pt
  
  # Test three models with different configurations
  python test_model.py --test_all \\
      --ckpt_no_basemap path/to/no_basemap.pt \\
      --ckpt_basemap_black_1x1 path/to/ch2_roadnet.pt \\
      --ckpt_basemap_white_3x3 path/to/ch2_blackW3x3.pt
        
Available configurations:
  - no_basemap:        No basemap, black points on white bg, 1x1
  - basemap_blackW1x1: Basemap + black points on white bg, 1x1  (ch2_roadnet)
  - basemap_whiteB3x3: Basemap + white points on black bg, 3x3  (ch2_blackW3x3)
        """
    )
    
    # Mode selection
    parser.add_argument('--compare', action='store_true',
                        help='Compare two models (with and without basemap) - legacy mode')
    parser.add_argument('--test_all', action='store_true',
                        help='Test three models with different configurations')
    
    # Single model testing
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint for single model testing')
    parser.add_argument('--config', type=str, default=None,
                        choices=['no_basemap', 'basemap_blackW1x1', 'basemap_whiteB3x3'],
                        help='Model configuration (default: auto-detect from checkpoint name)')
    
    # Legacy comparison mode
    parser.add_argument('--checkpoint_basemap', type=str, default=None,
                        help='Path to model checkpoint WITH basemap (legacy)')
    parser.add_argument('--checkpoint_no_basemap', type=str, default=None,
                        help='Path to model checkpoint WITHOUT basemap (legacy)')
    
    # Three-model testing
    parser.add_argument('--ckpt_no_basemap', type=str, default=None,
                        help='Checkpoint for no_basemap model')
    parser.add_argument('--ckpt_basemap_black_1x1', type=str, default=None,
                        help='Checkpoint for basemap + black points + 1x1 model')
    parser.add_argument('--ckpt_basemap_white_3x3', type=str, default=None,
                        help='Checkpoint for basemap + white points + 3x3 model')
    
    # Common options
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device number (default: 0, use -1 for CPU)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for evaluation (default: 64)')
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                        help='Specific dataset names to evaluate (default: all). E.g., --datasets 苏州')
    
    args = parser.parse_args()
    
    if args.test_all:
        # Test three models
        checkpoints = {}
        if args.ckpt_no_basemap:
            checkpoints['no_basemap'] = args.ckpt_no_basemap
        if args.ckpt_basemap_black_1x1:
            checkpoints['basemap_blackW1x1'] = args.ckpt_basemap_black_1x1
        if args.ckpt_basemap_white_3x3:
            checkpoints['basemap_whiteB3x3'] = args.ckpt_basemap_white_3x3
        
        if len(checkpoints) < 2:
            parser.error("--test_all requires at least 2 checkpoints")
        
        test_multiple_models(checkpoints, args.gpu, args.batch_size, dataset_names=args.datasets)
    
    elif args.compare:
        # Legacy comparison mode
        if not args.checkpoint_basemap or not args.checkpoint_no_basemap:
            parser.error("--compare requires both --checkpoint_basemap and --checkpoint_no_basemap")
        
        compare_models(
            args.checkpoint_basemap,
            args.checkpoint_no_basemap,
            args.gpu,
            args.batch_size,
            dataset_names=args.datasets
        )
    else:
        # Single model testing
        if not args.checkpoint:
            parser.error("Please provide --checkpoint for single model testing")
        
        # Auto-detect config from checkpoint name if not specified
        config = args.config
        if config is None:
            checkpoint_name = args.checkpoint.lower()
            if 'no_basemap' in checkpoint_name:
                config = 'no_basemap'
            elif 'blackw3x3' in checkpoint_name or 'black_w3x3' in checkpoint_name:
                config = 'basemap_whiteB3x3'
            elif 'ch2' in checkpoint_name:
                # Default ch2 without blackW3x3 is black points 1x1
                config = 'basemap_blackW1x1'
            else:
                print("Warning: Could not auto-detect config, defaulting to basemap_whiteB3x3")
                config = 'basemap_whiteB3x3'
            print(f"Auto-detected config: {config}")
        
        test_single_model(
            args.checkpoint,
            config,
            args.gpu,
            args.batch_size,
            dataset_names=args.datasets
        )


if __name__ == "__main__":
    main()
