"""
Two-Phase Hyper-Heuristic Local Search Module

This module implements a sophisticated local search algorithm for refining candidate
solutions to the Eternity II puzzle, with a focus on GPU-accelerated evaluation.
"""

import torch
import numpy as np
import time
import random
import copy
import math
import sys
import os
from collections import defaultdict

# Add the parent directory to the path to import from data
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.puzzle_definition import GRID_SIZE, TOTAL_PIECES, BORDER_COLOR
from src.utils import count_matching_edges

class LocalSearchSolver:
    """
    Two-Phase Hyper-Heuristic Local Search for refining Eternity II solutions.
    
    This class implements a sophisticated local search algorithm with two phases:
    1. Phase I (Obj4): Maximize the number of fully matched 3×3 sub-blocks
    2. Phase II (Obj1): Maximize total edge matches across the grid
    """
    
    def __init__(self, preprocessor, use_gpu=None, max_no_improvement=1000):
        """
        Initialize the local search solver.
        
        Args:
            preprocessor: The preprocessor with piece data
            use_gpu: Whether to use GPU acceleration (None=auto-detect)
            max_no_improvement: Maximum iterations without improvement before switching phases
        """
        self.preprocessor = preprocessor
        self.use_gpu = torch.cuda.is_available() if use_gpu is None else use_gpu
        self.max_no_improvement = max_no_improvement
        
        # Solution tracking
        self.current_solution = None
        self.best_solution = None
        self.current_score = 0
        self.best_score = 0
        
        # Phase tracking
        self.current_phase = 1  # Start with Phase I
        
        # Statistics
        self.iterations = 0
        self.phase1_iterations = 0
        self.phase2_iterations = 0
        self.start_time = None
        self.moves_evaluated = 0
        
        # Local neighborhood parameters
        self.temperature = 1.0
        self.cooling_rate = 0.99995
        self.min_temperature = 0.001
        
        # Initialize CUDA tensors if using GPU
        if self.use_gpu:
            self._initialize_cuda_tensors()
    
    def _initialize_cuda_tensors(self):
        """Initialize CUDA tensors for GPU-based processing."""
        # Placeholder for actual CUDA initialization
        # In a real implementation, this would prepare tensors for fast neighborhood evaluation
        pass
    
    def run_local_search(self, initial_solution, max_iterations=1000000, time_limit=3600):
        """
        Run the two-phase local search algorithm.
        
        Args:
            initial_solution: Initial candidate solution (grid)
            max_iterations: Maximum number of iterations
            time_limit: Time limit in seconds
            
        Returns:
            Best solution found (grid)
        """
        self.start_time = time.time()
        self.iterations = 0
        self.phase1_iterations = 0
        self.phase2_iterations = 0
        self.moves_evaluated = 0
        
        # Initialize current and best solutions
        self.current_solution = copy.deepcopy(initial_solution)
        self.best_solution = copy.deepcopy(initial_solution)
        
        # Calculate initial scores
        initial_phase = 1  # Always start with Phase I
        self.current_phase = initial_phase
        self.current_score = self._evaluate_solution(self.current_solution, self.current_phase)
        self.best_score = self.current_score
        
        # Keep track of the original solution in case we need to revert
        original_score = self.current_score
        original_solution = copy.deepcopy(initial_solution)
        
        # Track absolute best solution (across all phases)
        absolute_best_solution = copy.deepcopy(initial_solution)
        absolute_best_score = count_matching_edges(initial_solution)  # Use the same metric as main.py
        
        # Reset temperature
        self.temperature = 1.0
        
        # Iterations without improvement
        iterations_no_improvement = 0
        
        # Main search loop
        while (self.iterations < max_iterations and 
               time.time() - self.start_time < time_limit):
            
            # Generate and evaluate a move
            new_solution, new_score, move_type = self._generate_move()
            self.moves_evaluated += 1
            
            # Decide whether to accept the move
            if self._accept_move(new_score):
                # Accept the move
                self.current_solution = new_solution
                self.current_score = new_score
                
                # Check if this is a new best solution for the current phase
                if new_score > self.best_score:
                    self.best_solution = copy.deepcopy(new_solution)
                    self.best_score = new_score
                    iterations_no_improvement = 0
                else:
                    iterations_no_improvement += 1
                
                # Check if this is a new absolute best solution by matching edges
                current_absolute_score = count_matching_edges(new_solution)
                if current_absolute_score > absolute_best_score:
                    absolute_best_solution = copy.deepcopy(new_solution)
                    absolute_best_score = current_absolute_score
                    print(f"New absolute best solution with {absolute_best_score} matching edges")
            else:
                # Reject the move
                iterations_no_improvement += 1
            
            # Update temperature
            self.temperature = max(self.min_temperature, 
                                  self.temperature * self.cooling_rate)
            
            # Check if we should switch phases
            if iterations_no_improvement >= self.max_no_improvement:
                if self.current_phase == 1:
                    # Switch to Phase II
                    print(f"Switching to Phase II after {self.iterations} iterations")
                    self.current_phase = 2
                    self.phase1_iterations = self.iterations
                    iterations_no_improvement = 0
                    self.temperature = 1.0  # Reset temperature
                    
                    # Recalculate scores for Phase II
                    self.current_score = self._evaluate_solution(self.current_solution, self.current_phase)
                    self.best_score = self.current_score
                    self.best_solution = copy.deepcopy(self.current_solution)
                else:
                    # We're already in Phase II, try perturbing the solution
                    self._large_perturbation()
                    iterations_no_improvement = 0
            
            # Increment iteration counter
            self.iterations += 1
            if self.current_phase == 1:
                self.phase1_iterations += 1
            else:
                self.phase2_iterations += 1
            
            # Occasionally print progress
            if self.iterations % 10000 == 0:
                absolute_current_score = count_matching_edges(self.current_solution)
                print(f"Iteration {self.iterations}, Best score: {self.best_score}, "
                      f"Absolute score: {absolute_current_score}, "
                      f"Current phase: {self.current_phase}, Temp: {self.temperature:.6f}")
                
                # Check if the solution has become invalid
                if absolute_current_score == 0 and absolute_best_score > 0:
                    print("Warning: Current solution has 0 matching edges. Restoring from absolute best.")
                    self.current_solution = copy.deepcopy(absolute_best_solution)
                    self.current_score = self._evaluate_solution(self.current_solution, self.current_phase)
                    self.best_solution = copy.deepcopy(self.current_solution)
                    self.best_score = self.current_score
        
        # Return the absolute best solution found
        print(f"Local search completed. Best matching edges: {absolute_best_score}")
        return absolute_best_solution
    
    def _generate_move(self):
        """
        Generate a random move based on the current solution.
        
        Returns:
            (new_solution, new_score, move_type)
        """
        # Select move type based on phase and temperature
        move_types = ["swap", "rotate", "swap_adjacent", "block_swap", "large_neighborhood"]
        
        # Weight move types based on phase and temperature
        if self.current_phase == 1:
            # In Phase I, prefer moves that affect blocks
            weights = [0.1, 0.1, 0.3, 0.3, 0.2]
        else:
            # In Phase II, more balanced
            weights = [0.2, 0.2, 0.2, 0.2, 0.2]
        
        # Adjust based on temperature
        if self.temperature < 0.1:
            # At low temperatures, prefer smaller moves
            weights = [0.3, 0.3, 0.2, 0.1, 0.1]
        
        move_type = random.choices(move_types, weights=weights)[0]
        
        # Apply the selected move type
        if move_type == "swap":
            new_solution = self._swap_move()
        elif move_type == "rotate":
            new_solution = self._rotate_move()
        elif move_type == "swap_adjacent":
            new_solution = self._swap_adjacent_move()
        elif move_type == "block_swap":
            new_solution = self._block_swap_move()
        elif move_type == "large_neighborhood":
            new_solution = self._large_neighborhood_move()
        
        # Evaluate the new solution
        new_score = self._evaluate_solution(new_solution, self.current_phase)
        
        return new_solution, new_score, move_type
    
    def _swap_move(self):
        """
        Swap two random pieces in the grid.
        
        Returns:
            New solution with pieces swapped
        """
        new_solution = copy.deepcopy(self.current_solution)
        
        # Try to find valid positions up to 100 times
        for _ in range(100):
            # Select two random positions
            pos1 = (random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1))
            pos2 = (random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1))
            
            # Don't swap hint pieces or empty positions
            if (self._is_hint_piece(pos1) or self._is_hint_piece(pos2) or
                new_solution[pos1[0]][pos1[1]] is None or 
                new_solution[pos2[0]][pos2[1]] is None):
                continue
            
            # Swap the pieces
            new_solution[pos1[0]][pos1[1]], new_solution[pos2[0]][pos2[1]] = \
                new_solution[pos2[0]][pos2[1]], new_solution[pos1[0]][pos1[1]]
            return new_solution
        
        # If we couldn't find valid pieces to swap after 100 attempts,
        # return the original solution unchanged
        return new_solution
    
    def _rotate_move(self):
        """
        Rotate a random piece in the grid.
        
        Returns:
            New solution with one piece rotated
        """
        new_solution = copy.deepcopy(self.current_solution)
        
        # Try to find a valid position up to 100 times
        for _ in range(100):
            # Select a random position
            pos = (random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1))
            
            # Don't rotate hint pieces or empty positions
            if self._is_hint_piece(pos) or new_solution[pos[0]][pos[1]] is None:
                continue
            
            # Rotate the piece
            piece_id, rotation, edges = new_solution[pos[0]][pos[1]]
            new_rotation = (rotation + 1) % 4
            rotated_edges = self.preprocessor._rotate_piece(edges, 1)  # Rotate edges by 1 position
            new_solution[pos[0]][pos[1]] = (piece_id, new_rotation, rotated_edges)
            return new_solution
        
        # If we couldn't find a valid piece to rotate after 100 attempts,
        # return the original solution unchanged
        return new_solution
    
    def _swap_adjacent_move(self):
        """
        Swap two adjacent pieces.
        
        Returns:
            New solution with adjacent pieces swapped
        """
        new_solution = copy.deepcopy(self.current_solution)
        
        # Select a random position
        row = random.randint(0, GRID_SIZE-1)
        col = random.randint(0, GRID_SIZE-1)
        
        # Select a random direction
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        dr, dc = random.choice(directions)
        
        # Calculate adjacent position
        adj_row, adj_col = row + dr, col + dc
        
        # Check if adjacent position is valid
        if 0 <= adj_row < GRID_SIZE and 0 <= adj_col < GRID_SIZE:
            # Don't swap hint pieces
            if self._is_hint_piece((row, col)) or self._is_hint_piece((adj_row, adj_col)):
                return new_solution
            
            # Swap the pieces
            new_solution[row][col], new_solution[adj_row][adj_col] = \
                new_solution[adj_row][adj_col], new_solution[row][col]
        
        return new_solution
    
    def _block_swap_move(self):
        """
        Swap two 2x2 blocks of pieces.
        
        Returns:
            New solution with blocks swapped
        """
        new_solution = copy.deepcopy(self.current_solution)
        
        # Select two random block positions (top-left corners)
        block1_row = random.randint(0, GRID_SIZE-2)
        block1_col = random.randint(0, GRID_SIZE-2)
        block2_row = random.randint(0, GRID_SIZE-2)
        block2_col = random.randint(0, GRID_SIZE-2)
        
        # Check if blocks overlap
        if abs(block1_row - block2_row) < 2 and abs(block1_col - block2_col) < 2:
            return new_solution
        
        # Check if any blocks contain hint pieces
        for r in range(2):
            for c in range(2):
                if (self._is_hint_piece((block1_row + r, block1_col + c)) or
                    self._is_hint_piece((block2_row + r, block2_col + c))):
                    return new_solution
        
        # Swap the blocks
        for r in range(2):
            for c in range(2):
                new_solution[block1_row + r][block1_col + c], new_solution[block2_row + r][block2_col + c] = \
                    new_solution[block2_row + r][block2_col + c], new_solution[block1_row + r][block1_col + c]
        
        return new_solution
    
    def _large_neighborhood_move(self):
        """
        Remove n pieces and reinsert them optimally.
        
        Returns:
            New solution with pieces reinserted
        """
        new_solution = copy.deepcopy(self.current_solution)
        
        # Select a random region size (3-5)
        region_size = random.randint(3, 5)
        
        # Select a random region position (top-left corner)
        region_row = random.randint(0, GRID_SIZE - region_size)
        region_col = random.randint(0, GRID_SIZE - region_size)
        
        # Check if region contains hint pieces
        has_hint = False
        for r in range(region_size):
            for c in range(region_size):
                if self._is_hint_piece((region_row + r, region_col + c)):
                    has_hint = True
                    break
            if has_hint:
                break
        
        if has_hint:
            # Region contains a hint piece, try a different move
            return self._swap_move()
        
        # Remove pieces from the region and store them
        removed_pieces = []
        for r in range(region_size):
            for c in range(region_size):
                piece = new_solution[region_row + r][region_col + c]
                if piece is not None:  # Only add non-None pieces
                    removed_pieces.append(piece)
                new_solution[region_row + r][region_col + c] = None
        
        # If no pieces were removed, return the original solution
        if not removed_pieces:
            return new_solution
        
        # Reinsert pieces one by one in a greedy manner
        for piece in removed_pieces:
            best_pos = None
            best_score = -1
            best_orientation = 0
            
            piece_id, orientation, edges = piece
            
            # Try all positions and orientations
            for r in range(region_size):
                for c in range(region_size):
                    row, col = region_row + r, region_col + c
                    
                    if new_solution[row][col] is not None:
                        continue  # Position already filled
                    
                    for orient in range(4):
                        rotated_edges = self.preprocessor._rotate_piece(edges, (orient - orientation) % 4)
                        
                        # Place the piece temporarily
                        new_solution[row][col] = (piece_id, orient, rotated_edges)
                        
                        # Score this placement
                        score = self._score_position(new_solution, row, col)
                        
                        # Remove the piece
                        new_solution[row][col] = None
                        
                        # Update best if better
                        if score > best_score:
                            best_score = score
                            best_pos = (row, col)
                            best_orientation = orient
            
            # Place the piece in its best position
            if best_pos:
                rotated_edges = self.preprocessor._rotate_piece(edges, (best_orientation - orientation) % 4)
                new_solution[best_pos[0]][best_pos[1]] = (piece_id, best_orientation, rotated_edges)
        
        return new_solution
    
    def _large_perturbation(self):
        """
        Apply a large perturbation to escape local optima.
        
        This modifies the current solution in place.
        """
        # Make a backup of the current solution in case we completely break it
        backup_solution = copy.deepcopy(self.current_solution)
        backup_score = self.current_score
        
        # Scramble 20% of the grid
        num_scrambles = GRID_SIZE * GRID_SIZE // 5  # 20% of grid
        
        for _ in range(num_scrambles):
            # Select two random non-hint positions that have pieces
            valid_positions = []
            for r in range(GRID_SIZE):
                for c in range(GRID_SIZE):
                    if not self._is_hint_piece((r, c)) and self.current_solution[r][c] is not None:
                        valid_positions.append((r, c))
            
            # If we don't have enough valid positions with pieces, stop scrambling
            if len(valid_positions) < 2:
                break
                
            # Select two random positions from the valid ones
            pos1, pos2 = random.sample(valid_positions, 2)
            
            # Swap the pieces
            self.current_solution[pos1[0]][pos1[1]], self.current_solution[pos2[0]][pos2[1]] = \
                self.current_solution[pos2[0]][pos2[1]], self.current_solution[pos1[0]][pos1[1]]
        
        # Recalculate current score
        self.current_score = self._evaluate_solution(self.current_solution, self.current_phase)
        
        # If the score dropped to 0, revert to the backup
        if self.current_score == 0 and backup_score > 0:
            self.current_solution = backup_solution
            self.current_score = backup_score
    
    def _accept_move(self, new_score):
        """
        Decide whether to accept a move using the ILTA criterion.
        
        Args:
            new_score: Score of the new solution
            
        Returns:
            True if the move should be accepted, False otherwise
        """
        # Always accept improving moves
        if new_score >= self.current_score:
            return True
        
        # Accept some non-improving moves based on temperature
        delta = new_score - self.current_score
        acceptance_prob = math.exp(delta / self.temperature)
        
        return random.random() < acceptance_prob
    
    def _evaluate_solution(self, solution, phase):
        """
        Evaluate a solution based on the current phase.
        
        Args:
            solution: Solution to evaluate
            phase: Current phase (1 or 2)
            
        Returns:
            Score of the solution
        """
        if phase == 1:
            # Phase I: Number of fully matched 3×3 sub-blocks
            return self._evaluate_phase1(solution)
        else:
            # Phase II: Total number of matching edges
            return self._evaluate_phase2(solution)
    
    def _evaluate_phase1(self, solution):
        """
        Phase I evaluation: Number of fully matched 3×3 sub-blocks.
        
        Args:
            solution: Solution to evaluate
            
        Returns:
            Score based on 3×3 sub-blocks
        """
        score = 0
        
        # Check all possible 3×3 blocks
        for i in range(GRID_SIZE - 2):
            for j in range(GRID_SIZE - 2):
                block_score = self._evaluate_block(solution, i, j, 3)
                if block_score == 12:  # Fully matched (12 internal edges)
                    score += 1
        
        # Also include a component of matching edges for smoother optimization
        edge_score = self._evaluate_phase2(solution)
        
        # Score is primarily number of matched blocks, with edge score as a tiebreaker
        return score * 1000 + edge_score
    
    def _evaluate_block(self, solution, row, col, size):
        """
        Evaluate a block of the specified size.
        
        Args:
            solution: Solution to evaluate
            row, col: Top-left corner of the block
            size: Size of the block
            
        Returns:
            Number of matching internal edges in the block
        """
        matches = 0
        
        # Check horizontal edges
        for r in range(row, row + size):
            for c in range(col, col + size - 1):
                if solution[r][c] is not None and solution[r][c+1] is not None:
                    _, _, edges1 = solution[r][c]
                    _, _, edges2 = solution[r][c+1]
                    if edges1[1] == edges2[3]:  # East of piece1 = West of piece2
                        matches += 1
        
        # Check vertical edges
        for r in range(row, row + size - 1):
            for c in range(col, col + size):
                if solution[r][c] is not None and solution[r+1][c] is not None:
                    _, _, edges1 = solution[r][c]
                    _, _, edges2 = solution[r+1][c]
                    if edges1[2] == edges2[0]:  # South of piece1 = North of piece2
                        matches += 1
        
        return matches
    
    def _evaluate_phase2(self, solution):
        """
        Phase II evaluation: Total number of matching edges.
        
        Args:
            solution: Solution to evaluate
            
        Returns:
            Total number of matching edges
        """
        matches = 0
        
        # Check horizontal edges
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE - 1):
                if solution[r][c] is not None and solution[r][c+1] is not None:
                    _, _, edges1 = solution[r][c]
                    _, _, edges2 = solution[r][c+1]
                    if edges1[1] == edges2[3]:  # East of piece1 = West of piece2
                        matches += 1
        
        # Check vertical edges
        for r in range(GRID_SIZE - 1):
            for c in range(GRID_SIZE):
                if solution[r][c] is not None and solution[r+1][c] is not None:
                    _, _, edges1 = solution[r][c]
                    _, _, edges2 = solution[r+1][c]
                    if edges1[2] == edges2[0]:  # South of piece1 = North of piece2
                        matches += 1
        
        # Check border edges
        for r in range(GRID_SIZE):
            # Left border
            if solution[r][0] is not None:
                _, _, edges = solution[r][0]
                if edges[3] == BORDER_COLOR:
                    matches += 1
            
            # Right border
            if solution[r][GRID_SIZE-1] is not None:
                _, _, edges = solution[r][GRID_SIZE-1]
                if edges[1] == BORDER_COLOR:
                    matches += 1
        
        for c in range(GRID_SIZE):
            # Top border
            if solution[0][c] is not None:
                _, _, edges = solution[0][c]
                if edges[0] == BORDER_COLOR:
                    matches += 1
            
            # Bottom border
            if solution[GRID_SIZE-1][c] is not None:
                _, _, edges = solution[GRID_SIZE-1][c]
                if edges[2] == BORDER_COLOR:
                    matches += 1
        
        return matches
    
    def _score_position(self, solution, row, col):
        """
        Score a single position in the grid.
        
        Args:
            solution: Current solution
            row, col: Position to score
            
        Returns:
            Score for this position (number of matching edges)
        """
        if solution[row][col] is None:
            return 0
        
        _, _, edges = solution[row][col]
        score = 0
        
        # Check north
        if row > 0 and solution[row-1][col] is not None:
            _, _, north_edges = solution[row-1][col]
            if edges[0] == north_edges[2]:
                score += 1
        elif row == 0 and edges[0] == BORDER_COLOR:
            score += 1
        
        # Check east
        if col < GRID_SIZE-1 and solution[row][col+1] is not None:
            _, _, east_edges = solution[row][col+1]
            if edges[1] == east_edges[3]:
                score += 1
        elif col == GRID_SIZE-1 and edges[1] == BORDER_COLOR:
            score += 1
        
        # Check south
        if row < GRID_SIZE-1 and solution[row+1][col] is not None:
            _, _, south_edges = solution[row+1][col]
            if edges[2] == south_edges[0]:
                score += 1
        elif row == GRID_SIZE-1 and edges[2] == BORDER_COLOR:
            score += 1
        
        # Check west
        if col > 0 and solution[row][col-1] is not None:
            _, _, west_edges = solution[row][col-1]
            if edges[3] == west_edges[1]:
                score += 1
        elif col == 0 and edges[3] == BORDER_COLOR:
            score += 1
        
        return score
    
    def _is_hint_piece(self, pos):
        """
        Check if a position contains a hint piece.
        
        Args:
            pos: (row, col) tuple
            
        Returns:
            True if the position contains a hint piece
        """
        row, col = pos
        for piece_id, hint_row, hint_col, _ in self.preprocessor.hint_pieces:
            if row == hint_row and col == hint_col:
                return True
        return False
    
    def get_search_stats(self):
        """
        Get statistics about the search process.
        
        Returns:
            Dictionary of statistics
        """
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        
        return {
            "iterations": self.iterations,
            "phase1_iterations": self.phase1_iterations,
            "phase2_iterations": self.phase2_iterations,
            "moves_evaluated": self.moves_evaluated,
            "elapsed_time": elapsed_time,
            "iterations_per_sec": self.iterations / elapsed_time if elapsed_time > 0 else 0,
            "moves_per_sec": self.moves_evaluated / elapsed_time if elapsed_time > 0 else 0,
            "best_score": self.best_score,
            "best_percentage": (self.best_score / 480) * 100 if self.best_score > 0 else 0,
            "temperature": self.temperature
        } 