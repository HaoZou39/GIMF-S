#!/usr/bin/env python3
"""
Baseline methods for TSP evaluation.

This module implements core baseline solvers for TSP:
1. Greedy (Nearest Neighbor): Always choose the nearest unvisited node
2. POMO-style Greedy: Greedy from all starting points, pick best
3. 2-opt: Local search improvement
4. POMO-style 2-opt: 2-opt from all starting points, pick best

Usage:
    python baselines.py --datasets 苏州
    python baselines.py --datasets 苏州 --methods greedy pomo_greedy two_opt pomo_two_opt
"""

import argparse
import os
import sys
import time
import numpy as np
from abc import ABC, abstractmethod
from tqdm import tqdm

# Path Config
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils


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
# Base Solver Class
##########################################################################################

class BaseTSPSolver(ABC):
    """Abstract base class for TSP solvers."""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def solve(self, distance_matrix: np.ndarray, pomo_size: int = None) -> tuple:
        """
        Solve TSP given distance matrix.
        
        Args:
            distance_matrix: (n, n) numpy array of distances
            pomo_size: Number of starting points for POMO-style methods (default: n)
        
        Returns:
            tour: List of node indices representing the tour
            tour_distance: Total tour distance
        """
        pass
    
    def solve_batch(self, distance_matrices: np.ndarray, pomo_size: int = None) -> tuple:
        """
        Solve TSP for a batch of instances.
        
        Args:
            distance_matrices: (batch, n, n) numpy array
            pomo_size: Number of starting points
        
        Returns:
            tours: List of tours
            tour_distances: (batch,) numpy array of distances
        """
        tours = []
        distances = []
        for dm in distance_matrices:
            tour, dist = self.solve(dm, pomo_size)
            tours.append(tour)
            distances.append(dist)
        return tours, np.array(distances)


##########################################################################################
# Greedy (Nearest Neighbor) Solver
##########################################################################################

class GreedySolver(BaseTSPSolver):
    """
    Nearest Neighbor heuristic.
    Starting from a given node, always visit the nearest unvisited node.
    """
    
    def __init__(self, start_node: int = 0):
        super().__init__("Greedy (Nearest Neighbor)")
        self.start_node = start_node
    
    def solve(self, distance_matrix: np.ndarray, pomo_size: int = None) -> tuple:
        n = len(distance_matrix)
        
        tour = self._nearest_neighbor(distance_matrix, self.start_node)
        tour_distance = self._calc_tour_distance(tour, distance_matrix)
        
        return tour, tour_distance
    
    def _nearest_neighbor(self, distance_matrix, start):
        n = len(distance_matrix)
        visited = [False] * n
        tour = [start]
        visited[start] = True
        
        current = start
        for _ in range(n - 1):
            # Find nearest unvisited node
            best_next = -1
            best_dist = float('inf')
            for j in range(n):
                if not visited[j] and distance_matrix[current, j] < best_dist:
                    best_dist = distance_matrix[current, j]
                    best_next = j
            
            tour.append(best_next)
            visited[best_next] = True
            current = best_next
        
        return tour
    
    def _calc_tour_distance(self, tour, distance_matrix):
        dist = 0.0
        for i in range(len(tour)):
            dist += distance_matrix[tour[i], tour[(i + 1) % len(tour)]]
        return dist


##########################################################################################
# POMO-style Greedy Solver
##########################################################################################

class POMOGreedySolver(BaseTSPSolver):
    """
    POMO-style Greedy: Run nearest neighbor from all starting points,
    return the best tour found.
    """
    
    def __init__(self):
        super().__init__("POMO Greedy (NN from all starts)")
    
    def solve(self, distance_matrix: np.ndarray, pomo_size: int = None) -> tuple:
        n = len(distance_matrix)
        if pomo_size is None:
            pomo_size = n
        
        best_tour = None
        best_distance = float('inf')
        
        for start in range(min(pomo_size, n)):
            tour = self._nearest_neighbor(distance_matrix, start)
            tour_distance = self._calc_tour_distance(tour, distance_matrix)
            
            if tour_distance < best_distance:
                best_distance = tour_distance
                best_tour = tour
        
        return best_tour, best_distance
    
    def _nearest_neighbor(self, distance_matrix, start):
        n = len(distance_matrix)
        visited = [False] * n
        tour = [start]
        visited[start] = True
        
        current = start
        for _ in range(n - 1):
            best_next = -1
            best_dist = float('inf')
            for j in range(n):
                if not visited[j] and distance_matrix[current, j] < best_dist:
                    best_dist = distance_matrix[current, j]
                    best_next = j
            
            tour.append(best_next)
            visited[best_next] = True
            current = best_next
        
        return tour
    
    def _calc_tour_distance(self, tour, distance_matrix):
        dist = 0.0
        for i in range(len(tour)):
            dist += distance_matrix[tour[i], tour[(i + 1) % len(tour)]]
        return dist


##########################################################################################
# 2-opt Local Search Solver
##########################################################################################

class TwoOptSolver(BaseTSPSolver):
    """
    2-opt local search.
    Starts from a greedy solution and improves it using 2-opt moves.
    """
    
    def __init__(self, max_iterations: int = 1000, use_pomo: bool = False):
        name = "2-opt" + (" POMO" if use_pomo else "")
        super().__init__(name)
        self.max_iterations = max_iterations
        self.use_pomo = use_pomo
    
    def solve(self, distance_matrix: np.ndarray, pomo_size: int = None) -> tuple:
        n = len(distance_matrix)
        
        if self.use_pomo:
            if pomo_size is None:
                pomo_size = n
            
            best_tour = None
            best_distance = float('inf')
            
            for start in range(min(pomo_size, n)):
                # Start from greedy solution
                tour = self._nearest_neighbor(distance_matrix, start)
                tour = self._two_opt(tour, distance_matrix)
                tour_distance = self._calc_tour_distance(tour, distance_matrix)
                
                if tour_distance < best_distance:
                    best_distance = tour_distance
                    best_tour = tour
            
            return best_tour, best_distance
        else:
            tour = self._nearest_neighbor(distance_matrix, 0)
            tour = self._two_opt(tour, distance_matrix)
            tour_distance = self._calc_tour_distance(tour, distance_matrix)
            return tour, tour_distance
    
    def _nearest_neighbor(self, distance_matrix, start):
        n = len(distance_matrix)
        visited = [False] * n
        tour = [start]
        visited[start] = True
        
        current = start
        for _ in range(n - 1):
            best_next = -1
            best_dist = float('inf')
            for j in range(n):
                if not visited[j] and distance_matrix[current, j] < best_dist:
                    best_dist = distance_matrix[current, j]
                    best_next = j
            
            tour.append(best_next)
            visited[best_next] = True
            current = best_next
        
        return tour
    
    def _two_opt(self, tour, distance_matrix):
        """Apply 2-opt local search until no improvement."""
        n = len(tour)
        improved = True
        iterations = 0
        
        while improved and iterations < self.max_iterations:
            improved = False
            iterations += 1
            
            for i in range(n - 1):
                for j in range(i + 2, n):
                    if j == n - 1 and i == 0:
                        continue  # Skip if it would just reverse the tour
                    
                    # Calculate delta
                    i1, i2 = tour[i], tour[i + 1]
                    j1, j2 = tour[j], tour[(j + 1) % n]
                    
                    current_cost = distance_matrix[i1, i2] + distance_matrix[j1, j2]
                    new_cost = distance_matrix[i1, j1] + distance_matrix[i2, j2]
                    
                    if new_cost < current_cost - 1e-10:
                        # Reverse the segment between i+1 and j
                        tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                        improved = True
        
        return tour
    
    def _calc_tour_distance(self, tour, distance_matrix):
        dist = 0.0
        for i in range(len(tour)):
            dist += distance_matrix[tour[i], tour[(i + 1) % len(tour)]]
        return dist


##########################################################################################
# Evaluation Functions
##########################################################################################

def load_optimal_solutions(script_dir, dataset_names=None):
    """Load pre-computed optimal solutions."""
    optimal_data = {}
    source_data = {}
    
    datasets_to_load = DATASETS
    if dataset_names:
        datasets_to_load = [d for d in DATASETS if d['name'] in dataset_names]
    
    for ds_config in datasets_to_load:
        name = ds_config['name']
        npz_path = ds_config['npz_path']
        basemap_path = ds_config.get('basemap_path', None)
        
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
        
        opt_data = np.load(optimal_path, allow_pickle=True)
        optimal_data[name] = {
            'sample_indices': opt_data['sample_indices'],
            'optimal_tours': opt_data['optimal_tours'],
            'optimal_distances_m': opt_data['optimal_distances_m'],
            'optimal_distances_norm': opt_data['optimal_distances_norm'],
        }
        
        src_data = np.load(source_path, allow_pickle=True)
        sample_indices = opt_data['sample_indices']
        
        source_data[name] = {
            'matched_node_norm': src_data['matched_node_norm'][sample_indices],
            'undirected_dist_norm': src_data['undirected_dist_norm'][sample_indices],
            'basemap_path': os.path.normpath(os.path.join(script_dir, basemap_path)) if basemap_path else None,
        }
        
        print(f'Loaded {len(sample_indices)} optimal solutions for {name}')
    
    return optimal_data, source_data


def evaluate_solver(solver: BaseTSPSolver, optimal_data: dict, source_data: dict,
                    batch_size: int = 64, pomo_size: int = 20) -> dict:
    """
    Evaluate a solver on all datasets.
    
    Returns:
        dict: {dataset_name: {'gap_percent_mean': float, ...}}
    """
    results = {}
    
    for dataset_name in optimal_data.keys():
        opt = optimal_data[dataset_name]
        src = source_data[dataset_name]
        
        num_samples = len(opt['sample_indices'])
        distance_matrices = src['undirected_dist_norm']
        optimal_distances = opt['optimal_distances_norm']
        
        solver_distances = []
        inference_times = []
        
        # Process one by one
        pbar = tqdm(range(num_samples), desc=f'{solver.name} on {dataset_name}', leave=False)
        
        for idx in pbar:
            dm = distance_matrices[idx]
            
            start_time = time.time()
            _, dist = solver.solve(dm, pomo_size)
            inference_time = time.time() - start_time
            
            solver_distances.append(dist)
            inference_times.append(inference_time)
            
            gap = (dist - optimal_distances[idx]) / optimal_distances[idx] * 100
            pbar.set_postfix({'Gap': f'{gap:.2f}%'})
        
        solver_distances = np.array(solver_distances)
        gaps = (solver_distances - optimal_distances) / optimal_distances * 100
        
        results[dataset_name] = {
            'gap_percent_mean': np.mean(gaps),
            'gap_percent_std': np.std(gaps),
            'gap_percent_min': np.min(gaps),
            'gap_percent_max': np.max(gaps),
            'solver_distance_mean': np.mean(solver_distances),
            'solver_distance_std': np.std(solver_distances),
            'optimal_distance_mean': np.mean(optimal_distances),
            'optimal_distance_std': np.std(optimal_distances),
            'num_samples': num_samples,
            'inference_time_total': sum(inference_times),
            'inference_time_per_instance': sum(inference_times) / num_samples,
            'solver_distances': solver_distances,
            'optimal_distances': optimal_distances,
            'gaps': gaps,
        }
    
    return results


def print_results(results: dict, solver_name: str):
    """Print evaluation results for a solver."""
    print(f"\n{'='*70}")
    print(f" {solver_name} - Evaluation Results")
    print(f"{'='*70}")
    
    all_gaps = []
    
    for dataset_name, metrics in results.items():
        print(f"\n{dataset_name}:")
        print(f"  Samples: {metrics['num_samples']}")
        print(f"  Gap:     {metrics['gap_percent_mean']:.2f}% ± {metrics['gap_percent_std']:.2f}%")
        print(f"           [Min: {metrics['gap_percent_min']:.2f}%, Max: {metrics['gap_percent_max']:.2f}%]")
        print(f"  Solver:  {metrics['solver_distance_mean']:.4f} ± {metrics['solver_distance_std']:.4f}")
        print(f"  Optimal: {metrics['optimal_distance_mean']:.4f} ± {metrics['optimal_distance_std']:.4f}")
        print(f"  Time:    {metrics['inference_time_per_instance']*1000:.2f} ms/instance")
        
        all_gaps.extend(metrics['gaps'].tolist())
    
    all_gaps = np.array(all_gaps)
    print(f"\n{'─'*70}")
    print(f"Overall Average Gap: {np.mean(all_gaps):.2f}% ± {np.std(all_gaps):.2f}%")
    print(f"{'='*70}")
    
    return np.mean(all_gaps)


def print_comparison_table(all_results: dict):
    """Print comparison table for all solvers."""
    if not all_results:
        return
    
    print(f"\n{'='*100}")
    print(f" Baseline Comparison Summary")
    print(f"{'='*100}")
    
    # Get dataset names from first solver
    first_solver = list(all_results.keys())[0]
    dataset_names = list(all_results[first_solver].keys())
    
    # Header
    header = f"{'Method':<35}"
    for ds in dataset_names:
        header += f" {ds:<15}"
    header += f" {'Average':<12}"
    print(header)
    print(f"{'─'*100}")
    
    # Data rows
    avg_gaps = []
    for solver_name, results in all_results.items():
        row = f"{solver_name:<35}"
        solver_avg_gaps = []
        for ds in dataset_names:
            gap = results[ds]['gap_percent_mean']
            solver_avg_gaps.append(gap)
            row += f" {gap:>6.2f}%        "
        avg = np.mean(solver_avg_gaps)
        avg_gaps.append((solver_name, avg))
        row += f" {avg:>6.2f}%"
        print(row)
    
    print(f"{'─'*100}")
    
    # Find best
    best_solver, best_gap = min(avg_gaps, key=lambda x: x[1])
    print(f"\n✓ Best Baseline: {best_solver} with average gap {best_gap:.2f}%")
    print(f"{'='*100}")


##########################################################################################
# Main
##########################################################################################

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate baseline methods for TSP',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test all baselines on 苏州
  python baselines.py --datasets 苏州
  
  # Test specific methods
  python baselines.py --datasets 苏州 --methods greedy pomo_greedy
  
Available methods:
  - greedy:      Nearest neighbor (single start)
  - pomo_greedy: Nearest neighbor from all starts, pick best
  - two_opt:     2-opt local search (single start)
  - pomo_two_opt: 2-opt from all starts, pick best
        """
    )
    
    parser.add_argument('--datasets', type=str, nargs='+', default=['苏州'],
                        help='Dataset names to evaluate on')
    parser.add_argument('--methods', type=str, nargs='+', default=None,
                        help='Methods to evaluate (default: all)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for evaluation')
    parser.add_argument('--pomo_size', type=int, default=20,
                        help='Number of starting points for POMO-style methods')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Define available solvers
    available_solvers = {
        'greedy': lambda: GreedySolver(start_node=0),
        'pomo_greedy': lambda: POMOGreedySolver(),
        'two_opt': lambda: TwoOptSolver(use_pomo=False),
        'pomo_two_opt': lambda: TwoOptSolver(use_pomo=True),
    }
    
    # Select methods
    if args.methods is None:
        methods_to_test = list(available_solvers.keys())
    else:
        methods_to_test = args.methods
        for m in methods_to_test:
            if m not in available_solvers:
                print(f"Warning: Unknown method '{m}', skipping")
    
    print(f"\n{'='*70}")
    print(f" TSP Baseline Evaluation")
    print(f"{'='*70}")
    print(f"Datasets: {args.datasets}")
    print(f"Methods:  {methods_to_test}")
    print(f"POMO size: {args.pomo_size}")
    
    # Load optimal solutions
    script_dir = os.path.dirname(os.path.abspath(__file__))
    optimal_data, source_data = load_optimal_solutions(script_dir, args.datasets)
    
    if not optimal_data:
        print("Error: No optimal solution data found.")
        return
    
    # Evaluate each solver
    all_results = {}
    
    for method_name in methods_to_test:
        if method_name not in available_solvers:
            continue
        
        print(f"\n{'─'*70}")
        print(f"Evaluating: {method_name}")
        print(f"{'─'*70}")
        
        solver = available_solvers[method_name]()
        results = evaluate_solver(
            solver, optimal_data, source_data,
            batch_size=args.batch_size,
            pomo_size=args.pomo_size
        )
        
        all_results[solver.name] = results
        print_results(results, solver.name)
    
    # Print comparison table
    print_comparison_table(all_results)


if __name__ == "__main__":
    main()
