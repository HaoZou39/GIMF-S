#!/usr/bin/env python3
"""
Use OR-Tools to solve TSP optimal solutions for benchmark datasets.

This script:
1. Loads datasets from the DATASETS configuration
2. Randomly samples 200 test cases from each dataset
3. Solves TSP optimal solutions using OR-Tools
4. Saves the optimal solutions and tour distances back to the dataset

Usage:
    python solve_optimal_ortools.py [--num_samples 200] [--time_limit 60] [--use_directed]
"""

import argparse
import os
import sys
import numpy as np
from tqdm import tqdm
import time

# OR-Tools imports
from ortools.constraint_solver import routing_enums_pb2, pywrapcp

# Try to import numba for JIT acceleration
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("Warning: numba not available, using pure Python (slower)")

##########################################################################################
# Dataset Configuration (same as train_motsp_n20_single_obj.py)
##########################################################################################

DATASETS = [
    {
        'name': '杭州',
        'npz_path': '../../../MMDataset/杭州/distance_dataset_30.318899_120.055447_5000.0.npz',
    },
    {
        'name': '上海',
        'npz_path': '../../../MMDataset/上海/distance_dataset_31.240186_121.496062_5000.0.npz',
    },
    {
        'name': '柏林',
        'npz_path': '../../../MMDataset/柏林/distance_dataset_52.516298_13.377914_5000.0.npz',
    },
    {
        'name': '鹤岗',
        'npz_path': '../../../MMDataset/鹤岗/distance_dataset_47.332394_130.278898_5000.0.npz',
    },
    {
        'name': '苏州',
        'npz_path': '../../../MMDataset/苏州/distance_dataset_test_31.298909_120.579205_5000.0.npz',
    },
]


##########################################################################################
# OR-Tools TSP Solver
##########################################################################################

def solve_tsp_ortools(distance_matrix, time_limit_seconds=60):
    """
    Solve TSP using OR-Tools.
    
    Args:
        distance_matrix: 2D numpy array of shape (n, n) with distances between nodes.
                        The distance matrix should be in integer format for OR-Tools.
        time_limit_seconds: Maximum time in seconds for the solver.
    
    Returns:
        tour: List of node indices representing the optimal tour (starting from node 0).
        tour_distance: Total distance of the optimal tour.
        status: Solver status string.
    """
    n = len(distance_matrix)
    
    # Scale distances to integers (OR-Tools requires integer distances)
    # Use a large multiplier to preserve precision
    SCALE_FACTOR = 1_000_000
    int_distance_matrix = (distance_matrix * SCALE_FACTOR).astype(np.int64)
    
    # Create the routing index manager
    manager = pywrapcp.RoutingIndexManager(n, 1, 0)  # n nodes, 1 vehicle, depot at 0
    
    # Create Routing Model
    routing = pywrapcp.RoutingModel(manager)
    
    # Create distance callback
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int_distance_matrix[from_node][to_node]
    
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    
    # Define cost of each arc
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    # Setting first solution heuristic
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    
    # Set local search metaheuristic (GUIDED_LOCAL_SEARCH for better solutions)
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    
    # Set time limit
    search_parameters.time_limit.seconds = time_limit_seconds
    
    # Solve the problem
    solution = routing.SolveWithParameters(search_parameters)
    
    if solution:
        # Extract the tour
        tour = []
        index = routing.Start(0)
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            tour.append(node)
            index = solution.Value(routing.NextVar(index))
        
        # Calculate tour distance using original (float) distance matrix
        tour_distance = 0.0
        for i in range(len(tour)):
            from_node = tour[i]
            to_node = tour[(i + 1) % len(tour)]
            tour_distance += distance_matrix[from_node][to_node]
        
        return tour, tour_distance, "OPTIMAL" if routing.status() == 1 else "FEASIBLE"
    else:
        return None, None, "NO_SOLUTION"


if NUMBA_AVAILABLE:
    @jit(nopython=True, cache=True)
    def _held_karp_dp_numba(distance_matrix):
        """
        Numba-accelerated Held-Karp DP algorithm.
        Returns (dp, parent) arrays.
        """
        n = distance_matrix.shape[0]
        num_masks = 1 << n
        INF = 1e18
        
        # dp[mask][i] = minimum distance to reach node i visiting exactly nodes in mask
        dp = np.full((num_masks, n), INF, dtype=np.float64)
        parent = np.full((num_masks, n), -1, dtype=np.int32)
        
        # Start from node 0
        dp[1, 0] = 0.0
        
        for mask in range(1, num_masks):
            for last in range(n):
                if not (mask & (1 << last)):
                    continue
                if dp[mask, last] >= INF:
                    continue
                
                for next_node in range(n):
                    if mask & (1 << next_node):
                        continue
                    new_mask = mask | (1 << next_node)
                    new_dist = dp[mask, last] + distance_matrix[last, next_node]
                    if new_dist < dp[new_mask, next_node]:
                        dp[new_mask, next_node] = new_dist
                        parent[new_mask, next_node] = last
        
        return dp, parent
    
    def solve_tsp_exact_dp(distance_matrix):
        """
        Solve TSP exactly using dynamic programming (Held-Karp algorithm).
        This is O(n^2 * 2^n), feasible for n <= 20.
        Uses Numba JIT for acceleration.
        
        Args:
            distance_matrix: 2D numpy array of shape (n, n) with distances.
        
        Returns:
            tour: List of node indices representing the optimal tour.
            tour_distance: Total distance of the optimal tour.
        """
        n = len(distance_matrix)
        INF = 1e18
        
        # Convert to float64 for numba
        dist_matrix = distance_matrix.astype(np.float64)
        
        # Run numba-accelerated DP
        dp, parent = _held_karp_dp_numba(dist_matrix)
        
        # Find the best ending node (returning to 0)
        full_mask = (1 << n) - 1
        best_dist = INF
        best_last = -1
        for last in range(n):
            total_dist = dp[full_mask, last] + distance_matrix[last, 0]
            if total_dist < best_dist:
                best_dist = total_dist
                best_last = last
        
        # Reconstruct the tour
        tour = []
        mask = full_mask
        current = best_last
        while current != -1:
            tour.append(current)
            prev = parent[mask, current]
            mask ^= (1 << current)
            current = prev
        
        tour.reverse()
        
        return tour, best_dist

else:
    # Pure Python fallback (slower)
    def solve_tsp_exact_dp(distance_matrix):
        """
        Solve TSP exactly using dynamic programming (Held-Karp algorithm).
        This is O(n^2 * 2^n), feasible for n <= 20.
        
        Args:
            distance_matrix: 2D numpy array of shape (n, n) with distances.
        
        Returns:
            tour: List of node indices representing the optimal tour.
            tour_distance: Total distance of the optimal tour.
        """
        n = len(distance_matrix)
        
        # dp[mask][i] = minimum distance to reach node i visiting exactly nodes in mask
        INF = float('inf')
        dp = [[INF] * n for _ in range(1 << n)]
        parent = [[-1] * n for _ in range(1 << n)]
        
        # Start from node 0
        dp[1][0] = 0
        
        for mask in range(1, 1 << n):
            for last in range(n):
                if not (mask & (1 << last)):
                    continue
                if dp[mask][last] == INF:
                    continue
                
                for next_node in range(n):
                    if mask & (1 << next_node):
                        continue
                    new_mask = mask | (1 << next_node)
                    new_dist = dp[mask][last] + distance_matrix[last][next_node]
                    if new_dist < dp[new_mask][next_node]:
                        dp[new_mask][next_node] = new_dist
                        parent[new_mask][next_node] = last
        
        # Find the best ending node (returning to 0)
        full_mask = (1 << n) - 1
        best_dist = INF
        best_last = -1
        for last in range(n):
            total_dist = dp[full_mask][last] + distance_matrix[last][0]
            if total_dist < best_dist:
                best_dist = total_dist
                best_last = last
        
        # Reconstruct the tour
        tour = []
        mask = full_mask
        current = best_last
        while current != -1:
            tour.append(current)
            prev = parent[mask][current]
            mask ^= (1 << current)
            current = prev
        
        tour.reverse()
        
        return tour, best_dist


##########################################################################################
# Main Processing
##########################################################################################

def process_dataset(dataset_config, num_samples=200, time_limit=60, use_directed=True, 
                    use_exact_dp=True, random_seed=42):
    """
    Process a single dataset: sample cases, solve optimal, and save results.
    
    Args:
        dataset_config: Dict with 'name' and 'npz_path'.
        num_samples: Number of samples to solve.
        time_limit: Time limit per problem for OR-Tools (ignored if use_exact_dp=True).
        use_directed: Whether to use directed (True) or undirected (False) distances.
        use_exact_dp: Whether to use exact DP algorithm (True) or OR-Tools heuristic (False).
        random_seed: Random seed for reproducibility.
    
    Returns:
        Dict with results.
    """
    name = dataset_config['name']
    npz_path = dataset_config['npz_path']
    
    # Resolve path relative to this script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.normpath(os.path.join(script_dir, npz_path))
    
    print(f"\n{'='*60}")
    print(f"Processing dataset: {name}")
    print(f"Path: {full_path}")
    print(f"{'='*60}")
    
    if not os.path.exists(full_path):
        print(f"ERROR: Dataset file not found: {full_path}")
        return None
    
    # Load dataset
    data = np.load(full_path, allow_pickle=True)
    
    # Choose distance matrix type
    dist_key = 'directed_dist_m' if use_directed else 'undirected_dist_m'
    dist_norm_key = 'directed_dist_norm' if use_directed else 'undirected_dist_norm'
    
    distance_matrices = data[dist_key]  # (10000, 20, 20)
    distance_matrices_norm = data[dist_norm_key]  # (10000, 20, 20) normalized
    input_norm = data['input_norm']  # (10000, 20, 2)
    
    total_cases = len(distance_matrices)
    print(f"Total cases in dataset: {total_cases}")
    print(f"Distance matrix type: {dist_key}")
    print(f"Problem size: {distance_matrices.shape[1]}")
    
    # Random sampling
    np.random.seed(random_seed)
    if num_samples > total_cases:
        print(f"Warning: num_samples ({num_samples}) > total_cases ({total_cases}). Using all cases.")
        num_samples = total_cases
    
    sample_indices = np.random.choice(total_cases, size=num_samples, replace=False)
    sample_indices = np.sort(sample_indices)  # Sort for consistency
    
    print(f"Sampled {num_samples} cases for optimization")
    print(f"Sample indices range: [{sample_indices.min()}, {sample_indices.max()}]")
    
    # Solve each sample
    optimal_tours = []
    optimal_distances = []
    optimal_distances_norm = []
    solve_times = []
    solve_statuses = []
    
    solver_name = "Exact DP (Held-Karp)" if use_exact_dp else "OR-Tools"
    print(f"\nSolving with {solver_name}...")
    
    for idx in tqdm(sample_indices, desc=f"Solving {name}"):
        dist_matrix = distance_matrices[idx]  # (20, 20)
        dist_matrix_norm = distance_matrices_norm[idx]  # (20, 20)
        
        start_time = time.time()
        
        if use_exact_dp:
            # Use exact DP (Held-Karp) algorithm
            tour, tour_dist = solve_tsp_exact_dp(dist_matrix)
            status = "OPTIMAL"
        else:
            # Use OR-Tools
            tour, tour_dist, status = solve_tsp_ortools(dist_matrix, time_limit)
        
        solve_time = time.time() - start_time
        
        if tour is not None:
            # Calculate normalized distance using the tour
            tour_dist_norm = 0.0
            for i in range(len(tour)):
                from_node = tour[i]
                to_node = tour[(i + 1) % len(tour)]
                tour_dist_norm += dist_matrix_norm[from_node][to_node]
            
            optimal_tours.append(tour)
            optimal_distances.append(tour_dist)
            optimal_distances_norm.append(tour_dist_norm)
            solve_times.append(solve_time)
            solve_statuses.append(status)
        else:
            # Failed to solve - use None/NaN placeholders
            optimal_tours.append(None)
            optimal_distances.append(np.nan)
            optimal_distances_norm.append(np.nan)
            solve_times.append(solve_time)
            solve_statuses.append(status)
    
    # Statistics
    valid_mask = ~np.isnan(optimal_distances)
    num_solved = np.sum(valid_mask)
    
    print(f"\nResults for {name}:")
    print(f"  Solved: {num_solved}/{num_samples}")
    
    if num_solved > 0:
        valid_distances = np.array(optimal_distances)[valid_mask]
        valid_distances_norm = np.array(optimal_distances_norm)[valid_mask]
        valid_times = np.array(solve_times)[valid_mask]
        
        print(f"  Optimal distances (meters):")
        print(f"    Mean: {np.mean(valid_distances):.2f}")
        print(f"    Std:  {np.std(valid_distances):.2f}")
        print(f"    Min:  {np.min(valid_distances):.2f}")
        print(f"    Max:  {np.max(valid_distances):.2f}")
        print(f"  Optimal distances (normalized):")
        print(f"    Mean: {np.mean(valid_distances_norm):.6f}")
        print(f"    Std:  {np.std(valid_distances_norm):.6f}")
        print(f"  Solve times (seconds):")
        print(f"    Mean: {np.mean(valid_times):.4f}")
        print(f"    Max:  {np.max(valid_times):.4f}")
    
    return {
        'name': name,
        'npz_path': full_path,
        'sample_indices': sample_indices,
        'optimal_tours': optimal_tours,
        'optimal_distances': np.array(optimal_distances),
        'optimal_distances_norm': np.array(optimal_distances_norm),
        'solve_times': np.array(solve_times),
        'solve_statuses': solve_statuses,
        'use_directed': use_directed,
        'use_exact_dp': use_exact_dp,
        'random_seed': random_seed,
    }


def save_results_to_dataset(results):
    """
    Save the optimal solution results back to the original dataset.
    
    Creates a new key in the npz file with the optimal solutions.
    """
    if results is None:
        return
    
    npz_path = results['npz_path']
    
    # Load original data
    data = dict(np.load(npz_path, allow_pickle=True))
    
    # Determine the key prefix based on distance type
    prefix = 'directed' if results['use_directed'] else 'undirected'
    
    # Create new keys for optimal solutions
    # Note: We save sparse data (only for sampled indices)
    opt_key = f'optimal_{prefix}_sample_indices'
    tour_key = f'optimal_{prefix}_tours'
    dist_key = f'optimal_{prefix}_dist_m'
    dist_norm_key = f'optimal_{prefix}_dist_norm'
    time_key = f'optimal_{prefix}_solve_times'
    
    # Convert tours to numpy array (padded with -1 for None entries)
    n_samples = len(results['optimal_tours'])
    problem_size = len(results['optimal_tours'][0]) if results['optimal_tours'][0] is not None else 20
    tours_array = np.full((n_samples, problem_size), -1, dtype=np.int32)
    for i, tour in enumerate(results['optimal_tours']):
        if tour is not None:
            tours_array[i] = tour
    
    # Update data
    data[opt_key] = results['sample_indices']
    data[tour_key] = tours_array
    data[dist_key] = results['optimal_distances']
    data[dist_norm_key] = results['optimal_distances_norm']
    data[time_key] = results['solve_times']
    
    # Save back (create backup first)
    backup_path = npz_path.replace('.npz', '_backup_before_optimal.npz')
    if not os.path.exists(backup_path):
        print(f"Creating backup: {backup_path}")
        import shutil
        shutil.copy(npz_path, backup_path)
    
    # Save updated data
    np.savez(npz_path, **data)
    print(f"Saved optimal solutions to: {npz_path}")
    print(f"  New keys added: {opt_key}, {tour_key}, {dist_key}, {dist_norm_key}, {time_key}")


def save_results_to_separate_file(results, output_dir=None):
    """
    Save the optimal solution results to a separate npz file.
    
    This is safer than modifying the original dataset.
    """
    if results is None:
        return
    
    if output_dir is None:
        output_dir = os.path.dirname(results['npz_path'])
    
    # Create output filename
    base_name = os.path.basename(results['npz_path']).replace('.npz', '')
    prefix = 'directed' if results['use_directed'] else 'undirected'
    method = 'exact_dp' if results['use_exact_dp'] else 'ortools'
    output_name = f'{base_name}_optimal_{prefix}_{method}.npz'
    output_path = os.path.join(output_dir, output_name)
    
    # Convert tours to numpy array
    n_samples = len(results['optimal_tours'])
    problem_size = len(results['optimal_tours'][0]) if results['optimal_tours'][0] is not None else 20
    tours_array = np.full((n_samples, problem_size), -1, dtype=np.int32)
    for i, tour in enumerate(results['optimal_tours']):
        if tour is not None:
            tours_array[i] = tour
    
    # Save to separate file
    np.savez(
        output_path,
        sample_indices=results['sample_indices'],
        optimal_tours=tours_array,
        optimal_distances_m=results['optimal_distances'],
        optimal_distances_norm=results['optimal_distances_norm'],
        solve_times=results['solve_times'],
        solve_statuses=np.array(results['solve_statuses'], dtype=object),
        use_directed=results['use_directed'],
        use_exact_dp=results['use_exact_dp'],
        random_seed=results['random_seed'],
        dataset_name=results['name'],
        source_npz_path=results['npz_path'],
    )
    
    print(f"Saved optimal solutions to separate file: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Solve TSP optimal solutions using OR-Tools or exact DP'
    )
    parser.add_argument('--num_samples', type=int, default=200,
                        help='Number of samples to solve from each dataset (default: 200)')
    parser.add_argument('--time_limit', type=int, default=60,
                        help='Time limit in seconds per problem for OR-Tools (default: 60)')
    parser.add_argument('--use_directed', action='store_true', default=True,
                        help='Use directed distance matrix (default: True)')
    parser.add_argument('--use_undirected', action='store_true',
                        help='Use undirected distance matrix')
    parser.add_argument('--use_ortools', action='store_true',
                        help='Use OR-Tools heuristic instead of exact DP')
    parser.add_argument('--save_to_original', action='store_true',
                        help='Save results back to original dataset (creates backup)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for sampling (default: 42)')
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                        help='Specific dataset names to process (default: all)')
    
    args = parser.parse_args()
    
    # Determine distance type
    use_directed = not args.use_undirected
    use_exact_dp = not args.use_ortools
    
    print("="*60)
    print("TSP Optimal Solution Solver")
    print("="*60)
    print(f"Configuration:")
    print(f"  Number of samples per dataset: {args.num_samples}")
    print(f"  Distance type: {'directed' if use_directed else 'undirected'}")
    print(f"  Solver: {'Exact DP (Held-Karp)' if use_exact_dp else 'OR-Tools'}")
    if not use_exact_dp:
        print(f"  Time limit per problem: {args.time_limit}s")
    print(f"  Random seed: {args.seed}")
    print(f"  Save to original dataset: {args.save_to_original}")
    
    # Filter datasets if specified
    datasets_to_process = DATASETS
    if args.datasets:
        datasets_to_process = [d for d in DATASETS if d['name'] in args.datasets]
        if not datasets_to_process:
            print(f"ERROR: No matching datasets found for: {args.datasets}")
            print(f"Available datasets: {[d['name'] for d in DATASETS]}")
            return
    
    print(f"  Datasets to process: {[d['name'] for d in datasets_to_process]}")
    
    # Process each dataset
    all_results = []
    for dataset_config in datasets_to_process:
        results = process_dataset(
            dataset_config,
            num_samples=args.num_samples,
            time_limit=args.time_limit,
            use_directed=use_directed,
            use_exact_dp=use_exact_dp,
            random_seed=args.seed,
        )
        
        if results is not None:
            # Save results
            if args.save_to_original:
                save_results_to_dataset(results)
            else:
                save_results_to_separate_file(results)
            
            all_results.append(results)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for results in all_results:
        valid_mask = ~np.isnan(results['optimal_distances'])
        num_solved = np.sum(valid_mask)
        
        if num_solved > 0:
            valid_distances_norm = results['optimal_distances_norm'][valid_mask]
            print(f"\n{results['name']}:")
            print(f"  Solved: {num_solved}/{len(results['sample_indices'])}")
            print(f"  Optimal distance (norm): {np.mean(valid_distances_norm):.6f} ± {np.std(valid_distances_norm):.6f}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
