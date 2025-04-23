"""
SAT/CSP Solver Module for Eternity II

This module implements SAT/CSP encodings and solvers for small subproblems 
of the Eternity II puzzle, used to polish near-complete solutions.
"""

import numpy as np
import time
from pysat.formula import CNF
from pysat.solvers import Solver
from collections import defaultdict
import sys
import os

# Add the parent directory to the path to import from data
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.puzzle_definition import GRID_SIZE, TOTAL_PIECES, BORDER_COLOR

class SATSolver:
    """
    SAT/CSP solver for Eternity II subproblems.
    
    This class implements encodings and solvers for small subproblems 
    of the Eternity II puzzle, focusing on regions with mismatched edges.
    """
    
    def __init__(self, preprocessor, timeout=300):
        """
        Initialize the SAT solver.
        
        Args:
            preprocessor: The preprocessor with piece data
            timeout: Solver timeout in seconds
        """
        self.preprocessor = preprocessor
        self.timeout = timeout
        
        # Statistics
        self.encoding_time = 0
        self.solving_time = 0
        self.subproblems_solved = 0
        self.subproblems_attempted = 0
    
    def polish_solution(self, solution, max_region_size=20):
        """
        Polish a near-complete solution by solving small subproblems.
        
        Args:
            solution: Current best solution
            max_region_size: Maximum subproblem size (in pieces)
            
        Returns:
            Improved solution
        """
        # Create a copy of the solution
        improved_solution = np.copy(solution)
        
        # Track statistics
        self.encoding_time = 0
        self.solving_time = 0
        self.subproblems_solved = 0
        self.subproblems_attempted = 0
        
        # Find regions with mismatched edges
        regions = self._find_mismatch_regions(solution)
        
        # Sort regions by size (smallest first for quicker wins)
        regions.sort(key=lambda r: len(r))
        
        # Process each region
        for region in regions:
            # Skip regions that are too large
            if len(region) > max_region_size:
                print(f"Skipping region with {len(region)} pieces (max: {max_region_size})")
                continue
            
            self.subproblems_attempted += 1
            
            # Extract the subproblem
            subproblem = self._extract_subproblem(solution, region)
            
            # Solve the subproblem
            solved_subproblem = self._solve_subproblem(subproblem)
            
            if solved_subproblem is not None:
                # Subproblem was successfully solved
                self.subproblems_solved += 1
                
                # Integrate the solution back
                improved_solution = self._integrate_subproblem(
                    improved_solution, region, solved_subproblem
                )
        
        return improved_solution
    
    def _find_mismatch_regions(self, solution):
        """
        Find regions with mismatched edges.
        
        Args:
            solution: Current solution
            
        Returns:
            List of regions (sets of positions)
        """
        # Initialize grid of mismatches
        mismatch_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
        
        # Check horizontal edges for mismatches
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE - 1):
                if solution[r][c] is not None and solution[r][c+1] is not None:
                    _, _, edges1 = solution[r][c]
                    _, _, edges2 = solution[r][c+1]
                    if edges1[1] != edges2[3]:  # East of piece1 != West of piece2
                        mismatch_grid[r, c] = True
                        mismatch_grid[r, c+1] = True
        
        # Check vertical edges for mismatches
        for r in range(GRID_SIZE - 1):
            for c in range(GRID_SIZE):
                if solution[r][c] is not None and solution[r+1][c] is not None:
                    _, _, edges1 = solution[r][c]
                    _, _, edges2 = solution[r+1][c]
                    if edges1[2] != edges2[0]:  # South of piece1 != North of piece2
                        mismatch_grid[r, c] = True
                        mismatch_grid[r+1, c] = True
        
        # Check border edges for mismatches
        for r in range(GRID_SIZE):
            # Left border
            if solution[r][0] is not None:
                _, _, edges = solution[r][0]
                if r == 0 or r == GRID_SIZE-1:  # Corner
                    if (edges[3] != BORDER_COLOR or 
                        (r == 0 and edges[0] != BORDER_COLOR) or
                        (r == GRID_SIZE-1 and edges[2] != BORDER_COLOR)):
                        mismatch_grid[r, 0] = True
                else:  # Non-corner border
                    if edges[3] != BORDER_COLOR:
                        mismatch_grid[r, 0] = True
            
            # Right border
            if solution[r][GRID_SIZE-1] is not None:
                _, _, edges = solution[r][GRID_SIZE-1]
                if r == 0 or r == GRID_SIZE-1:  # Corner
                    if (edges[1] != BORDER_COLOR or 
                        (r == 0 and edges[0] != BORDER_COLOR) or
                        (r == GRID_SIZE-1 and edges[2] != BORDER_COLOR)):
                        mismatch_grid[r, GRID_SIZE-1] = True
                else:  # Non-corner border
                    if edges[1] != BORDER_COLOR:
                        mismatch_grid[r, GRID_SIZE-1] = True
        
        for c in range(GRID_SIZE):
            # Top border
            if solution[0][c] is not None:
                _, _, edges = solution[0][c]
                if c == 0 or c == GRID_SIZE-1:  # Corner (already checked)
                    pass
                else:  # Non-corner border
                    if edges[0] != BORDER_COLOR:
                        mismatch_grid[0, c] = True
            
            # Bottom border
            if solution[GRID_SIZE-1][c] is not None:
                _, _, edges = solution[GRID_SIZE-1][c]
                if c == 0 or c == GRID_SIZE-1:  # Corner (already checked)
                    pass
                else:  # Non-corner border
                    if edges[2] != BORDER_COLOR:
                        mismatch_grid[GRID_SIZE-1, c] = True
        
        # Group mismatches into connected regions
        regions = []
        visited = np.zeros_like(mismatch_grid, dtype=bool)
        
        # Helper function for connected component DFS
        def dfs(r, c, region):
            if (r < 0 or r >= GRID_SIZE or c < 0 or c >= GRID_SIZE or
                visited[r, c] or not mismatch_grid[r, c]):
                return
            
            visited[r, c] = True
            region.add((r, c))
            
            # Check neighbors
            dfs(r+1, c, region)
            dfs(r-1, c, region)
            dfs(r, c+1, region)
            dfs(r, c-1, region)
        
        # Find all connected regions
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if mismatch_grid[r, c] and not visited[r, c]:
                    region = set()
                    dfs(r, c, region)
                    regions.append(region)
        
        return regions
    
    def _extract_subproblem(self, solution, region):
        """
        Extract a subproblem from the full solution.
        
        Args:
            solution: Full solution
            region: Set of positions in the region
            
        Returns:
            Dictionary with subproblem information
        """
        # Determine the bounding box of the region
        min_r = min(r for r, c in region)
        max_r = max(r for r, c in region)
        min_c = min(c for r, c in region)
        max_c = max(c for r, c in region)
        
        # Expand the bounding box by 1 in each direction (if possible)
        min_r = max(0, min_r - 1)
        max_r = min(GRID_SIZE - 1, max_r + 1)
        min_c = max(0, min_c - 1)
        max_c = min(GRID_SIZE - 1, max_c + 1)
        
        # Extract the subgrid
        subgrid = np.array([
            [solution[r][c] for c in range(min_c, max_c + 1)]
            for r in range(min_r, max_r + 1)
        ])
        
        # Determine fixed vs. variable pieces
        fixed_positions = set()
        variable_positions = set()
        
        for r in range(subgrid.shape[0]):
            for c in range(subgrid.shape[1]):
                abs_r, abs_c = min_r + r, min_c + c
                
                # Fixed if it's a hint piece or not in the region
                if self._is_hint_piece((abs_r, abs_c)) or (abs_r, abs_c) not in region:
                    fixed_positions.add((r, c))
                else:
                    variable_positions.add((r, c))
        
        # Extract used piece IDs
        used_pieces = set()
        for r in range(subgrid.shape[0]):
            for c in range(subgrid.shape[1]):
                if subgrid[r, c] is not None:
                    piece_id = subgrid[r, c][0]
                    used_pieces.add(piece_id)
        
        # Return the subproblem
        return {
            'subgrid': subgrid,
            'bounds': (min_r, max_r, min_c, max_c),
            'fixed_positions': fixed_positions,
            'variable_positions': variable_positions,
            'used_pieces': used_pieces
        }
    
    def _solve_subproblem(self, subproblem):
        """
        Solve a subproblem using SAT.
        
        Args:
            subproblem: Subproblem information
            
        Returns:
            Solved subgrid or None if no solution found
        """
        # Extract subproblem information
        subgrid = subproblem['subgrid']
        bounds = subproblem['bounds']
        fixed_positions = subproblem['fixed_positions']
        variable_positions = subproblem['variable_positions']
        used_pieces = subproblem['used_pieces']
        
        # Create a set of candidate pieces (unused from the full grid)
        candidate_pieces = set()
        for pos in variable_positions:
            r, c = pos
            if subgrid[r, c] is not None:
                piece_id = subgrid[r, c][0]
                candidate_pieces.add(piece_id)
        
        # Encode the problem
        start_time = time.time()
        cnf, var_map = self._encode_subproblem(subproblem)
        self.encoding_time += time.time() - start_time
        
        # Solve the problem
        start_time = time.time()
        assignment = self._solve_cnf(cnf)
        self.solving_time += time.time() - start_time
        
        if assignment is None:
            # No solution found
            return None
        
        # Extract the solution
        solved_subgrid = np.copy(subgrid)
        inv_var_map = {v: k for k, v in var_map.items()}
        
        for var, value in enumerate(assignment, 1):
            if value > 0 and var in inv_var_map:
                pos, piece_id, orientation = inv_var_map[var]
                r, c = pos
                
                # Get the rotated piece
                rotated_piece = self.preprocessor._rotate_piece(
                    self.preprocessor.piece_defs[piece_id], orientation
                )
                
                # Update the subgrid
                solved_subgrid[r, c] = (piece_id, orientation, rotated_piece)
        
        return solved_subgrid
    
    def _encode_subproblem(self, subproblem):
        """
        Encode a subproblem as a SAT formula.
        
        Args:
            subproblem: Subproblem information
            
        Returns:
            (CNF formula, variable mapping)
        """
        # Extract subproblem information
        subgrid = subproblem['subgrid']
        bounds = subproblem['bounds']
        fixed_positions = subproblem['fixed_positions']
        variable_positions = subproblem['variable_positions']
        used_pieces = subproblem['used_pieces']
        
        min_r, max_r, min_c, max_c = bounds
        height, width = subgrid.shape
        
        # Create variable mapping
        var_map = {}  # Maps (pos, piece_id, orientation) to variable ID
        var_counter = 1
        
        # Create variables for each position, piece, and orientation
        for r, c in variable_positions:
            # Get the current piece at this position (if any)
            current_piece = subgrid[r, c]
            current_piece_id = current_piece[0] if current_piece is not None else None
            
            # Try all possible pieces for this position
            for piece_id in range(TOTAL_PIECES):
                # Skip pieces that are used in fixed positions
                if piece_id != current_piece_id and piece_id in used_pieces:
                    continue
                
                # Try all orientations
                for orientation in range(4):
                    # Create variable
                    var_map[((r, c), piece_id, orientation)] = var_counter
                    var_counter += 1
        
        # Create CNF formula
        cnf = CNF()
        
        # Constraint 1: Each variable position must have exactly one piece
        for r, c in variable_positions:
            # Collect variables for this position
            pos_vars = []
            for piece_id in range(TOTAL_PIECES):
                for orientation in range(4):
                    if ((r, c), piece_id, orientation) in var_map:
                        pos_vars.append(var_map[((r, c), piece_id, orientation)])
            
            # At least one piece at this position
            cnf.append(pos_vars)
            
            # At most one piece at this position
            for i in range(len(pos_vars)):
                for j in range(i + 1, len(pos_vars)):
                    cnf.append([-pos_vars[i], -pos_vars[j]])
        
        # Constraint 2: Each piece can be used at most once
        for piece_id in range(TOTAL_PIECES):
            # Skip pieces that are used in fixed positions
            if piece_id in used_pieces and any(
                subgrid[r, c] is not None and subgrid[r, c][0] == piece_id
                for r, c in fixed_positions
            ):
                continue
            
            # Collect variables for this piece
            piece_vars = []
            for r, c in variable_positions:
                for orientation in range(4):
                    if ((r, c), piece_id, orientation) in var_map:
                        piece_vars.append(var_map[((r, c), piece_id, orientation)])
            
            # At most one placement for this piece
            for i in range(len(piece_vars)):
                for j in range(i + 1, len(piece_vars)):
                    cnf.append([-piece_vars[i], -piece_vars[j]])
        
        # Constraint 3: Matching edges between adjacent positions
        for r, c in variable_positions:
            # North edge
            self._add_edge_constraints(cnf, var_map, subgrid, (r, c), (r-1, c), 0, 2, fixed_positions, variable_positions)
            
            # East edge
            self._add_edge_constraints(cnf, var_map, subgrid, (r, c), (r, c+1), 1, 3, fixed_positions, variable_positions)
            
            # South edge
            self._add_edge_constraints(cnf, var_map, subgrid, (r, c), (r+1, c), 2, 0, fixed_positions, variable_positions)
            
            # West edge
            self._add_edge_constraints(cnf, var_map, subgrid, (r, c), (r, c-1), 3, 1, fixed_positions, variable_positions)
        
        # Constraint 4: Border edges
        for r, c in variable_positions:
            abs_r, abs_c = min_r + r, min_c + c
            
            # North border
            if abs_r == 0:
                for piece_id in range(TOTAL_PIECES):
                    for orientation in range(4):
                        if ((r, c), piece_id, orientation) in var_map:
                            rotated_piece = self.preprocessor._rotate_piece(
                                self.preprocessor.piece_defs[piece_id], orientation
                            )
                            if rotated_piece[0] != BORDER_COLOR:
                                cnf.append([-var_map[((r, c), piece_id, orientation)]])
            
            # East border
            if abs_c == GRID_SIZE - 1:
                for piece_id in range(TOTAL_PIECES):
                    for orientation in range(4):
                        if ((r, c), piece_id, orientation) in var_map:
                            rotated_piece = self.preprocessor._rotate_piece(
                                self.preprocessor.piece_defs[piece_id], orientation
                            )
                            if rotated_piece[1] != BORDER_COLOR:
                                cnf.append([-var_map[((r, c), piece_id, orientation)]])
            
            # South border
            if abs_r == GRID_SIZE - 1:
                for piece_id in range(TOTAL_PIECES):
                    for orientation in range(4):
                        if ((r, c), piece_id, orientation) in var_map:
                            rotated_piece = self.preprocessor._rotate_piece(
                                self.preprocessor.piece_defs[piece_id], orientation
                            )
                            if rotated_piece[2] != BORDER_COLOR:
                                cnf.append([-var_map[((r, c), piece_id, orientation)]])
            
            # West border
            if abs_c == 0:
                for piece_id in range(TOTAL_PIECES):
                    for orientation in range(4):
                        if ((r, c), piece_id, orientation) in var_map:
                            rotated_piece = self.preprocessor._rotate_piece(
                                self.preprocessor.piece_defs[piece_id], orientation
                            )
                            if rotated_piece[3] != BORDER_COLOR:
                                cnf.append([-var_map[((r, c), piece_id, orientation)]])
        
        return cnf, var_map
    
    def _add_edge_constraints(self, cnf, var_map, subgrid, pos1, pos2, edge1, edge2, fixed_positions, variable_positions):
        """
        Add edge matching constraints between two positions.
        
        Args:
            cnf: CNF formula
            var_map: Variable mapping
            subgrid: Subgrid
            pos1, pos2: Positions to constrain
            edge1, edge2: Edges to match (0=north, 1=east, 2=south, 3=west)
            fixed_positions: Set of fixed positions
            variable_positions: Set of variable positions
        """
        r1, c1 = pos1
        r2, c2 = pos2
        
        # Check if both positions are valid
        if not (0 <= r1 < subgrid.shape[0] and 0 <= c1 < subgrid.shape[1] and
                0 <= r2 < subgrid.shape[0] and 0 <= c2 < subgrid.shape[1]):
            return
        
        # Handle different position status
        if pos1 in fixed_positions and pos2 in fixed_positions:
            # Both positions are fixed, no constraint needed
            return
        
        elif pos1 in fixed_positions:
            # pos1 is fixed, constrain pos2
            fixed_piece = subgrid[r1, c1]
            if fixed_piece is None:
                return
            
            fixed_id, fixed_orientation, fixed_edges = fixed_piece
            fixed_color = fixed_edges[edge1]
            
            # For each piece that could go at pos2, it must match the fixed color
            for piece_id in range(TOTAL_PIECES):
                for orientation in range(4):
                    if ((r2, c2), piece_id, orientation) in var_map:
                        rotated_piece = self.preprocessor._rotate_piece(
                            self.preprocessor.piece_defs[piece_id], orientation
                        )
                        if rotated_piece[edge2] != fixed_color:
                            cnf.append([-var_map[((r2, c2), piece_id, orientation)]])
        
        elif pos2 in fixed_positions:
            # pos2 is fixed, constrain pos1
            fixed_piece = subgrid[r2, c2]
            if fixed_piece is None:
                return
            
            fixed_id, fixed_orientation, fixed_edges = fixed_piece
            fixed_color = fixed_edges[edge2]
            
            # For each piece that could go at pos1, it must match the fixed color
            for piece_id in range(TOTAL_PIECES):
                for orientation in range(4):
                    if ((r1, c1), piece_id, orientation) in var_map:
                        rotated_piece = self.preprocessor._rotate_piece(
                            self.preprocessor.piece_defs[piece_id], orientation
                        )
                        if rotated_piece[edge1] != fixed_color:
                            cnf.append([-var_map[((r1, c1), piece_id, orientation)]])
        
        else:
            # Both positions are variable, add pairwise constraints
            # For any two pieces placed at pos1 and pos2, their edges must match
            for pid1 in range(TOTAL_PIECES):
                for orient1 in range(4):
                    if ((r1, c1), pid1, orient1) not in var_map:
                        continue
                    
                    rotated_piece1 = self.preprocessor._rotate_piece(
                        self.preprocessor.piece_defs[pid1], orient1
                    )
                    color1 = rotated_piece1[edge1]
                    
                    for pid2 in range(TOTAL_PIECES):
                        for orient2 in range(4):
                            if ((r2, c2), pid2, orient2) not in var_map:
                                continue
                            
                            rotated_piece2 = self.preprocessor._rotate_piece(
                                self.preprocessor.piece_defs[pid2], orient2
                            )
                            color2 = rotated_piece2[edge2]
                            
                            if color1 != color2:
                                cnf.append([
                                    -var_map[((r1, c1), pid1, orient1)],
                                    -var_map[((r2, c2), pid2, orient2)]
                                ])
    
    def _solve_cnf(self, cnf):
        """
        Solve a CNF formula.
        
        Args:
            cnf: CNF formula
            
        Returns:
            Assignment or None if unsatisfiable
        """
        # Create solver
        solver = Solver(name='glucose4')
        
        # Add clauses
        for clause in cnf:
            solver.add_clause(clause)
        
        # Set timeout
        if self.timeout > 0:
            solver.set_timeout(self.timeout)
        
        # Solve
        if solver.solve():
            return solver.get_model()
        else:
            return None
    
    def _integrate_subproblem(self, solution, region, solved_subproblem):
        """
        Integrate a solved subproblem back into the full solution.
        
        Args:
            solution: Full solution
            region: Region of the subproblem
            solved_subproblem: Solved subproblem
            
        Returns:
            Updated solution
        """
        # Create a copy of the solution
        updated_solution = np.copy(solution)
        
        # Update the region with the solved subproblem
        min_r = min(r for r, c in region)
        min_c = min(c for r, c in region)
        
        for r in range(solved_subproblem.shape[0]):
            for c in range(solved_subproblem.shape[1]):
                abs_r, abs_c = min_r - 1 + r, min_c - 1 + c
                
                # Skip if outside the grid
                if not (0 <= abs_r < GRID_SIZE and 0 <= abs_c < GRID_SIZE):
                    continue
                
                # Skip fixed pieces (outside the region)
                if (abs_r, abs_c) not in region and not (
                    # Skip hint pieces
                    solved_subproblem[r, c] is not None and
                    any(solved_subproblem[r, c][0] == pid 
                        for pid, _, _, _ in self.preprocessor.hint_pieces)
                ):
                    continue
                
                # Update the position
                updated_solution[abs_r][abs_c] = solved_subproblem[r, c]
        
        return updated_solution
    
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
    
    def get_solver_stats(self):
        """
        Get statistics about the solver process.
        
        Returns:
            Dictionary of statistics
        """
        return {
            "encoding_time": self.encoding_time,
            "solving_time": self.solving_time,
            "total_time": self.encoding_time + self.solving_time,
            "subproblems_attempted": self.subproblems_attempted,
            "subproblems_solved": self.subproblems_solved,
            "success_rate": (self.subproblems_solved / self.subproblems_attempted 
                            if self.subproblems_attempted > 0 else 0)
        } 