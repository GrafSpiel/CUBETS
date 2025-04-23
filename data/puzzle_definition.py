"""
Eternity II Puzzle Definition

This file contains the constants and definitions for the Eternity II puzzle,
including board size, number of colors, and hint pieces.
"""
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Puzzle dimensions
GRID_SIZE = 16
TOTAL_PIECES = 256
TOTAL_EDGES = 4 * GRID_SIZE * (GRID_SIZE - 1)  # 480 internal edges
NUM_COLORS = 22

# Special edge color for border pieces (typically color 0)
BORDER_COLOR = 0

# Fixed hint pieces in the format (piece_id, row, col, orientation)
# Orientation: 0=North, 1=East, 2=South, 3=West rotation from original
HINT_PIECES = [
    # Center piece (1-indexed in the original puzzle, converted to 0-indexed)
    (138, 7, 7, 0),  
    
    # Other hint pieces (using example values - replace with actual hints)
    (5, 0, 8, 2),    # Top row
    (92, 8, 15, 1),  # Right side
    (147, 15, 8, 0), # Bottom row
    (184, 8, 0, 3),  # Left side
]

# Define the actual piece definitions
# Each piece is defined as a tuple of (north, east, south, west) edge colors
# This is a placeholder - the actual definitions would be loaded from a file
def load_piece_definitions(filename=None):
    """
    Load or generate the piece definitions for the puzzle.
    
    In a real implementation, this would load from a file.
    For this example, we'll generate synthetic pieces.
    """
    if filename and os.path.exists(filename):
        # Load from file
        import numpy as np
        return np.load(filename)
    
    # Generate synthetic piece definitions for demonstration
    import random
    random.seed(42)  # For reproducibility
    
    pieces = []
    
    # Corner pieces (4) - have two border edges
    for _ in range(4):
        # Different arrangements of the two border edges
        edges = [
            BORDER_COLOR, 
            BORDER_COLOR, 
            random.randint(1, NUM_COLORS-1),
            random.randint(1, NUM_COLORS-1)
        ]
        # Randomize non-border edge positions
        if random.choice([True, False]):
            edges = [edges[3], edges[0], edges[1], edges[2]]
        else:
            edges = [edges[2], edges[3], edges[0], edges[1]]
        pieces.append(tuple(edges))
    
    # Border pieces (56) - have one border edge
    for _ in range(56):
        border_pos = random.randint(0, 3)
        edges = [random.randint(1, NUM_COLORS-1) for _ in range(4)]
        edges[border_pos] = BORDER_COLOR
        pieces.append(tuple(edges))
    
    # Interior pieces (196)
    for _ in range(196):
        edges = [random.randint(1, NUM_COLORS-1) for _ in range(4)]
        pieces.append(tuple(edges))
    
    # Ensure we have the exact number of pieces
    assert len(pieces) == TOTAL_PIECES
    return pieces

# Image rendering constants (for visualization)
PIECE_SIZE = 50  # Pixels per piece
EDGE_COLORS = [
    (0, 0, 0),       # Black for border (color 0)
    (255, 0, 0),     # Red
    (0, 255, 0),     # Green
    (0, 0, 255),     # Blue
    (255, 255, 0),   # Yellow
    (255, 0, 255),   # Magenta
    (0, 255, 255),   # Cyan
    (128, 0, 0),     # Maroon
    (0, 128, 0),     # Dark Green
    (0, 0, 128),     # Navy
    (128, 128, 0),   # Olive
    (128, 0, 128),   # Purple
    (0, 128, 128),   # Teal
    (255, 165, 0),   # Orange
    (255, 192, 203), # Pink
    (165, 42, 42),   # Brown
    (240, 230, 140), # Khaki
    (46, 139, 87),   # Sea Green
    (106, 90, 205),  # Slate Blue
    (220, 20, 60),   # Crimson
    (184, 134, 11),  # Dark Goldenrod
    (178, 34, 34),   # Firebrick
] 