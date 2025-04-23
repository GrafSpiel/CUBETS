#!/usr/bin/env python3
"""
Eternity II Solver - Main Entry Point

This script orchestrates the Eternity II puzzle solver pipeline, 
combining preprocessing, ZLA, local search, and SAT solving.
"""

import argparse
import os
import sys
import time
import numpy as np
import torch
import random
from datetime import datetime
import matplotlib.pyplot as plt

from src.preprocessing import PuzzlePreprocessor
from src.zla import ZLASolver
from src.local_search import LocalSearchSolver
from src.sat_solver import SATSolver
from src.utils import (
    visualize_solution, count_matching_edges, create_empty_grid,
    save_solution, load_solution, print_solution_stats
)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Eternity II Solver')
    
    # Pipeline control
    parser.add_argument('--only-zla', action='store_true', 
                        help='Only run the ZLA phase')
    parser.add_argument('--only-local-search', action='store_true',
                        help='Only run the local search phase')
    parser.add_argument('--only-sat', action='store_true',
                        help='Only run the SAT solving phase')
    parser.add_argument('--skip-zla', action='store_true',
                        help='Skip the ZLA phase')
    parser.add_argument('--skip-local-search', action='store_true',
                        help='Skip the local search phase')
    parser.add_argument('--skip-sat', action='store_true',
                        help='Skip the SAT solving phase')
    
    # Input/output
    parser.add_argument('--load-state', type=str,
                        help='Load a saved solution state')
    parser.add_argument('--save-state', type=str,
                        help='Save the solution state (auto-generated if not specified)')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Directory to save results')
    
    # ZLA options
    parser.add_argument('--zla-iterations', type=int, default=10000,
                        help='Number of ZLA iterations per thread')
    parser.add_argument('--zla-retain', type=int, default=50,
                        help='Number of top solutions to retain from ZLA')
    parser.add_argument('--zla-seed', type=int, default=None,
                        help='Random seed for ZLA')
    
    # Local search options
    parser.add_argument('--ls-iterations', type=int, default=1000000,
                        help='Maximum number of local search iterations')
    parser.add_argument('--ls-time-limit', type=int, default=3600,
                        help='Time limit for local search in seconds')
    parser.add_argument('--ls-no-improvement', type=int, default=10000,
                        help='Maximum iterations without improvement before phase switch')
    
    # SAT options
    parser.add_argument('--sat-max-region', type=int, default=20,
                        help='Maximum region size for SAT solving')
    parser.add_argument('--sat-timeout', type=int, default=300,
                        help='Timeout for SAT solver in seconds')
    
    # General options
    parser.add_argument('--use-gpu', action='store_true',
                        help='Use GPU acceleration if available')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    parser.add_argument('--seed', type=int, default=None,
                        help='Global random seed')
    
    return parser.parse_args()

def setup_environment(args):
    """Set up the environment based on arguments."""
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seeds for reproducibility
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
    
    # Check GPU availability
    if args.use_gpu and not torch.cuda.is_available():
        print("WARNING: GPU requested but not available. Using CPU instead.")
        args.use_gpu = False
    
    # Print environment info
    if args.verbose:
        print(f"Environment:")
        print(f"  Python: {sys.version.split()[0]}")
        print(f"  NumPy: {np.__version__}")
        print(f"  PyTorch: {torch.__version__}")
        print(f"  GPU Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA Version: {torch.version.cuda}")

def run_zla_phase(preprocessor, args):
    """Run the Zero Look-ahead Algorithm phase."""
    print("\n=== Running ZLA Phase ===")
    
    # Create solver
    zla_solver = ZLASolver(
        preprocessor=preprocessor,
        retain_top_k=args.zla_retain
    )
    
    # Run solver
    start_time = time.time()
    
    if args.use_gpu and torch.cuda.is_available():
        print("Using GPU acceleration for ZLA")
        solutions = zla_solver.run_gpu_zla(
            max_iterations_per_thread=args.zla_iterations,
            seed=args.zla_seed or args.seed
        )
    else:
        print("Using CPU for ZLA")
        solutions = zla_solver.run_cpu_zla(
            max_iterations=args.zla_iterations,
            seed=args.zla_seed or args.seed
        )
    
    elapsed_time = time.time() - start_time
    
    # Print statistics
    stats = zla_solver.get_solution_stats()
    print(f"ZLA completed in {elapsed_time:.2f} seconds")
    print(f"  Solutions explored: {stats['solutions_explored']}")
    print(f"  Solutions per second: {stats['solutions_per_sec']:.2f}")
    print(f"  Best score: {stats['best_score']} ({stats['best_percentage']:.2f}%)")
    
    # Visualize best solution
    if solutions:
        best_score, best_solution = solutions[0]
        vis_filename = os.path.join(args.output_dir, f"zla_best_solution.png")
        visualize_solution(
            best_solution, 
            filename=vis_filename,
            show=False,
            title=f"ZLA Best Solution - Score: {best_score}"
        )
        print(f"Best solution visualization saved to {vis_filename}")
    
    return solutions

def run_local_search_phase(preprocessor, initial_solutions, args):
    """Run the Local Search phase."""
    print("\n=== Running Local Search Phase ===")
    
    # Create solver
    ls_solver = LocalSearchSolver(
        preprocessor=preprocessor,
        use_gpu=args.use_gpu,
        max_no_improvement=args.ls_no_improvement
    )
    
    best_solution = None
    best_score = 0
    
    # Run solver on each initial solution
    for i, (score, solution) in enumerate(initial_solutions[:5]):  # Limit to top 5
        print(f"\nOptimizing solution {i+1}/{min(5, len(initial_solutions))} with initial score {score}")
        
        # Run local search
        start_time = time.time()
        improved_solution = ls_solver.run_local_search(
            initial_solution=solution,
            max_iterations=args.ls_iterations,
            time_limit=args.ls_time_limit
        )
        elapsed_time = time.time() - start_time
        
        # Evaluate solution
        improved_score = count_matching_edges(improved_solution)
        
        # Print statistics
        stats = ls_solver.get_search_stats()
        print(f"Local search completed in {elapsed_time:.2f} seconds")
        print(f"  Initial score: {score}")
        print(f"  Final score: {improved_score} ({improved_score/480*100:.2f}%)")
        print(f"  Improvement: {improved_score - score}")
        print(f"  Iterations: {stats['iterations']}")
        print(f"  Iterations per second: {stats['iterations_per_sec']:.2f}")
        
        # Update best solution
        if improved_score > best_score:
            best_score = improved_score
            best_solution = improved_solution
            
            # Visualize best solution
            vis_filename = os.path.join(args.output_dir, f"ls_best_solution.png")
            visualize_solution(
                best_solution, 
                filename=vis_filename,
                show=False,
                title=f"Local Search Best Solution - Score: {best_score}"
            )
            print(f"New best solution visualization saved to {vis_filename}")
    
    return [(best_score, best_solution)] if best_solution is not None else []

def run_sat_phase(preprocessor, solution, args):
    """Run the SAT solving phase."""
    print("\n=== Running SAT Polishing Phase ===")
    
    # Create solver
    sat_solver = SATSolver(
        preprocessor=preprocessor,
        timeout=args.sat_timeout
    )
    
    # Run solver
    start_time = time.time()
    improved_solution = sat_solver.polish_solution(
        solution=solution,
        max_region_size=args.sat_max_region
    )
    elapsed_time = time.time() - start_time
    
    # Evaluate solution
    initial_score = count_matching_edges(solution)
    improved_score = count_matching_edges(improved_solution)
    
    # Print statistics
    stats = sat_solver.get_solver_stats()
    print(f"SAT polishing completed in {elapsed_time:.2f} seconds")
    print(f"  Initial score: {initial_score}")
    print(f"  Final score: {improved_score} ({improved_score/480*100:.2f}%)")
    print(f"  Improvement: {improved_score - initial_score}")
    print(f"  Subproblems attempted: {stats['subproblems_attempted']}")
    print(f"  Subproblems solved: {stats['subproblems_solved']}")
    print(f"  Success rate: {stats['success_rate']*100:.2f}%")
    
    # Visualize solution
    if improved_score > initial_score:
        vis_filename = os.path.join(args.output_dir, f"sat_improved_solution.png")
        visualize_solution(
            improved_solution, 
            filename=vis_filename,
            show=False,
            title=f"SAT Polished Solution - Score: {improved_score}"
        )
        print(f"Improved solution visualization saved to {vis_filename}")
    
    return improved_solution, improved_score

def main():
    """Main entry point."""
    # Parse arguments
    args = parse_arguments()
    
    # Set up environment
    setup_environment(args)
    
    # Record start time
    total_start_time = time.time()
    
    # Create preprocessor
    print("\n=== Initializing Preprocessor ===")
    preprocessor = PuzzlePreprocessor()
    print(f"Preprocessing complete")
    
    # Initialize solution
    best_solution = None
    best_score = 0
    
    # Load existing solution if requested
    if args.load_state:
        best_solution = load_solution(args.load_state)
        best_score = count_matching_edges(best_solution)
        print(f"Loaded solution with score {best_score} ({best_score/480*100:.2f}%)")
    
    # Run ZLA phase
    zla_solutions = []
    if not args.skip_zla and not args.only_local_search and not args.only_sat:
        zla_solutions = run_zla_phase(preprocessor, args)
        
        # Update best solution
        if zla_solutions and (best_solution is None or zla_solutions[0][0] > best_score):
            best_score, best_solution = zla_solutions[0]
            print(f"New best solution from ZLA with score {best_score}")
    
    # Run local search phase
    if not args.skip_local_search and not args.only_zla and not args.only_sat:
        # If we have solutions from ZLA, use them
        initial_solutions = zla_solutions
        
        # If no ZLA solutions but we have a loaded solution, use it
        if not initial_solutions and best_solution is not None:
            initial_solutions = [(best_score, best_solution)]
        
        # If we still don't have any solutions, create a random one
        if not initial_solutions:
            print("No initial solutions available. Creating a random solution.")
            grid = create_empty_grid()
            preprocessor.apply_hints(grid)
            initial_score = count_matching_edges(grid)
            initial_solutions = [(initial_score, grid)]
        
        # Run local search
        ls_solutions = run_local_search_phase(preprocessor, initial_solutions, args)
        
        # Update best solution
        if ls_solutions and (best_solution is None or ls_solutions[0][0] > best_score):
            best_score, best_solution = ls_solutions[0]
            print(f"New best solution from local search with score {best_score}")
    
    # Run SAT phase
    if not args.skip_sat and not args.only_zla and not args.only_local_search:
        if best_solution is not None:
            # Run SAT polishing
            sat_solution, sat_score = run_sat_phase(preprocessor, best_solution, args)
            
            # Update best solution
            if sat_score > best_score:
                best_score = sat_score
                best_solution = sat_solution
                print(f"New best solution from SAT with score {best_score}")
        else:
            print("No solution available for SAT polishing. Skipping.")
    
    # Save final solution
    if best_solution is not None:
        if args.save_state:
            save_solution(best_solution, args.save_state)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(args.output_dir, f"final_solution_{best_score}_{timestamp}.pkl")
            save_solution(best_solution, save_path)
        
        # Create final visualization
        final_vis_path = os.path.join(args.output_dir, f"final_solution_{best_score}.png")
        visualize_solution(
            best_solution,
            filename=final_vis_path,
            show=False,
            title=f"Final Solution - Score: {best_score} ({best_score/480*100:.2f}%)"
        )
        print(f"Final solution visualization saved to {final_vis_path}")
        
        # Print detailed statistics
        print("\n=== Final Solution Statistics ===")
        print_solution_stats(best_solution)
    
    # Calculate total time
    total_time = time.time() - total_start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

if __name__ == "__main__":
    main() 