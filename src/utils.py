"""
Utilities Module for Eternity II Solver

This module provides visualization and helper functions for the Eternity II solver.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import os
import sys
import pickle
from datetime import datetime

# Add the parent directory to the path to import from data
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.puzzle_definition import GRID_SIZE, EDGE_COLORS, PIECE_SIZE, BORDER_COLOR

def visualize_solution(solution, filename=None, show=True, title=None):
    """
    Visualize a solution to the Eternity II puzzle.
    
    Args:
        solution: The solution grid
        filename: If provided, save the visualization to this file
        show: Whether to display the visualization
        title: Optional title for the visualization
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Set up the figure
    ax.set_xlim(0, GRID_SIZE * PIECE_SIZE)
    ax.set_ylim(0, GRID_SIZE * PIECE_SIZE)
    ax.set_aspect('equal')
    
    # Draw grid lines
    for i in range(GRID_SIZE + 1):
        ax.axhline(i * PIECE_SIZE, color='black', linewidth=0.5)
        ax.axvline(i * PIECE_SIZE, color='black', linewidth=0.5)
    
    # Draw pieces
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            piece = solution[row][col]
            if piece is not None:
                piece_id, _, edges = piece
                
                # Coordinates
                x = col * PIECE_SIZE
                y = (GRID_SIZE - row - 1) * PIECE_SIZE  # Flip y-axis for visualization
                
                # Draw piece background
                rect = patches.Rectangle((x, y), PIECE_SIZE, PIECE_SIZE, 
                                         facecolor='white', edgecolor='black',
                                         linewidth=0.5)
                ax.add_patch(rect)
                
                # Draw piece ID
                ax.text(x + PIECE_SIZE/2, y + PIECE_SIZE/2, str(piece_id),
                        horizontalalignment='center', verticalalignment='center',
                        fontsize=8)
                
                # Draw edges
                edge_width = 5
                
                # North edge
                north_color = EDGE_COLORS[edges[0]] if edges[0] < len(EDGE_COLORS) else (0, 0, 0)
                north = patches.Rectangle((x, y + PIECE_SIZE - edge_width), PIECE_SIZE, edge_width,
                                          facecolor=tuple(c/255 for c in north_color), 
                                          edgecolor='none')
                ax.add_patch(north)
                
                # East edge
                east_color = EDGE_COLORS[edges[1]] if edges[1] < len(EDGE_COLORS) else (0, 0, 0)
                east = patches.Rectangle((x + PIECE_SIZE - edge_width, y), edge_width, PIECE_SIZE,
                                         facecolor=tuple(c/255 for c in east_color), 
                                         edgecolor='none')
                ax.add_patch(east)
                
                # South edge
                south_color = EDGE_COLORS[edges[2]] if edges[2] < len(EDGE_COLORS) else (0, 0, 0)
                south = patches.Rectangle((x, y), PIECE_SIZE, edge_width,
                                          facecolor=tuple(c/255 for c in south_color), 
                                          edgecolor='none')
                ax.add_patch(south)
                
                # West edge
                west_color = EDGE_COLORS[edges[3]] if edges[3] < len(EDGE_COLORS) else (0, 0, 0)
                west = patches.Rectangle((x, y), edge_width, PIECE_SIZE,
                                         facecolor=tuple(c/255 for c in west_color), 
                                         edgecolor='none')
                ax.add_patch(west)
    
    # Set title
    if title:
        ax.set_title(title)
    else:
        # Count matching edges
        matching_edges = count_matching_edges(solution)
        ax.set_title(f'Eternity II Solution - {matching_edges}/480 matching edges ({matching_edges/480*100:.1f}%)')
    
    # Turn off axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Save figure if filename provided
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    
    # Show figure if requested
    if show:
        plt.show()
    else:
        plt.close(fig)

def count_matching_edges(solution):
    """
    Count the number of matching edges in a solution.
    
    Args:
        solution: The solution grid
        
    Returns:
        Number of matching edges
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
            if edges[3] == BORDER_COLOR:  # West edge is border
                matches += 1
        
        # Right border
        if solution[r][GRID_SIZE-1] is not None:
            _, _, edges = solution[r][GRID_SIZE-1]
            if edges[1] == BORDER_COLOR:  # East edge is border
                matches += 1
    
    for c in range(GRID_SIZE):
        # Top border
        if solution[0][c] is not None:
            _, _, edges = solution[0][c]
            if edges[0] == BORDER_COLOR:  # North edge is border
                matches += 1
        
        # Bottom border
        if solution[GRID_SIZE-1][c] is not None:
            _, _, edges = solution[GRID_SIZE-1][c]
            if edges[2] == BORDER_COLOR:  # South edge is border
                matches += 1
    
    return matches

def create_empty_grid():
    """
    Create an empty grid for the puzzle.
    
    Returns:
        Empty grid of size GRID_SIZE x GRID_SIZE
    """
    return np.full((GRID_SIZE, GRID_SIZE), None)

def save_solution(solution, filename=None):
    """
    Save a solution to a file.
    
    Args:
        solution: The solution grid
        filename: Filename to save to (default: auto-generate based on score and timestamp)
    """
    if filename is None:
        # Generate filename based on score and timestamp
        score = count_matching_edges(solution)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"solution_score{score}_{timestamp}.pkl"
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    
    # Save solution
    with open(filename, 'wb') as f:
        pickle.dump(solution, f)
    
    print(f"Solution saved to {filename}")

def load_solution(filename):
    """
    Load a solution from a file.
    
    Args:
        filename: Filename to load from
        
    Returns:
        Loaded solution grid
    """
    with open(filename, 'rb') as f:
        solution = pickle.load(f)
    
    print(f"Solution loaded from {filename}")
    return solution

def print_solution_stats(solution):
    """
    Print statistics about a solution.
    
    Args:
        solution: The solution grid
    """
    matching_edges = count_matching_edges(solution)
    total_edges = 4 * GRID_SIZE * (GRID_SIZE - 1)  # Internal edges + border edges
    
    # Count missing pieces
    missing_pieces = sum(1 for r in range(GRID_SIZE) for c in range(GRID_SIZE) if solution[r][c] is None)
    
    print(f"Solution Statistics:")
    print(f"  Matching Edges: {matching_edges}/{total_edges} ({matching_edges/total_edges*100:.1f}%)")
    print(f"  Missing Pieces: {missing_pieces}/{GRID_SIZE*GRID_SIZE}")
    
    # Count horizontal and vertical matches separately
    horizontal_matches = 0
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE-1):
            if solution[r][c] is not None and solution[r][c+1] is not None:
                _, _, edges1 = solution[r][c]
                _, _, edges2 = solution[r][c+1]
                if edges1[1] == edges2[3]:  # East of piece1 = West of piece2
                    horizontal_matches += 1
    
    vertical_matches = 0
    for r in range(GRID_SIZE-1):
        for c in range(GRID_SIZE):
            if solution[r][c] is not None and solution[r+1][c] is not None:
                _, _, edges1 = solution[r][c]
                _, _, edges2 = solution[r+1][c]
                if edges1[2] == edges2[0]:  # South of piece1 = North of piece2
                    vertical_matches += 1
    
    border_matches = matching_edges - horizontal_matches - vertical_matches
    
    print(f"  Horizontal Matches: {horizontal_matches}/{GRID_SIZE*(GRID_SIZE-1)}")
    print(f"  Vertical Matches: {vertical_matches}/{GRID_SIZE*(GRID_SIZE-1)}")
    print(f"  Border Matches: {border_matches}/{2*GRID_SIZE*2}")
    
    # Check for duplicate pieces
    used_pieces = {}
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if solution[r][c] is not None:
                piece_id = solution[r][c][0]
                if piece_id in used_pieces:
                    used_pieces[piece_id].append((r, c))
                else:
                    used_pieces[piece_id] = [(r, c)]
    
    duplicates = {pid: positions for pid, positions in used_pieces.items() if len(positions) > 1}
    if duplicates:
        print(f"  Duplicate Pieces Found:")
        for pid, positions in duplicates.items():
            print(f"    Piece {pid} used at positions: {positions}")
    else:
        print(f"  No duplicate pieces found.")

def time_function(func, *args, **kwargs):
    """
    Time the execution of a function.
    
    Args:
        func: Function to time
        *args, **kwargs: Arguments to pass to the function
        
    Returns:
        (result, elapsed_time)
    """
    start_time = time.time()
    result = func(*args, **kwargs)
    elapsed_time = time.time() - start_time
    
    return result, elapsed_time 