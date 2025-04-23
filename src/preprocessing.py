"""
Preprocessing Module for Eternity II Solver

This module handles the preprocessing of pieces and creation of efficient data structures 
for the solver algorithms. It includes bitwise encoding of pieces, edge color indexing,
and creation of lookup tables for fast matching.
"""

import numpy as np
import torch
from collections import defaultdict
import sys
import os

# Add the parent directory to the path to import from data
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.puzzle_definition import (
    GRID_SIZE, TOTAL_PIECES, NUM_COLORS, BORDER_COLOR, 
    HINT_PIECES, load_piece_definitions
)

class PuzzlePreprocessor:
    """
    Handles preprocessing of pieces and creation of efficient data structures.
    """
    
    def __init__(self, piece_defs=None):
        """
        Initialize the preprocessor with piece definitions.
        
        Args:
            piece_defs: List of pieces as (north, east, south, west) color tuples.
                        If None, loads the default pieces.
        """
        self.piece_defs = piece_defs or load_piece_definitions()
        self.hint_pieces = HINT_PIECES
        
        # Initialize data structures
        self.piece_orientations = []  # All pieces in all orientations
        self.piece_encodings = []     # Bitwise encodings of all pieces
        self.edge_color_piece_map = defaultdict(list)  # Maps (side, color) to list of valid (piece_id, orientation)
        self.used_pieces_bitset = 0   # Bitset to track used pieces
        
        # CUDA tensors for GPU-based solution
        self.cuda_piece_encodings = None
        self.cuda_edge_color_maps = None
        
        # Preprocess the pieces
        self._preprocess_pieces()
        
    def _preprocess_pieces(self):
        """Preprocess all pieces and create lookup tables."""
        # Generate all piece orientations
        for piece_id, piece in enumerate(self.piece_defs):
            for rot in range(4):  # 4 possible rotations
                # Apply rotation to the piece edges
                rotated_piece = self._rotate_piece(piece, rot)
                
                # Encode the piece as a 32-bit integer 
                # (5 bits per edge for color, 8 bits for piece ID)
                encoding = self._encode_piece(piece_id, rotated_piece)
                
                self.piece_orientations.append((piece_id, rot, rotated_piece))
                self.piece_encodings.append(encoding)
                
                # Create lookup by edge color and side
                for side, color in enumerate(rotated_piece):
                    self.edge_color_piece_map[(side, color)].append((piece_id, rot))
        
        # Convert to numpy arrays for faster processing
        self.piece_encodings = np.array(self.piece_encodings, dtype=np.uint32)
        
        # Initialize CUDA tensors if CUDA is available
        if torch.cuda.is_available():
            self._initialize_cuda_tensors()
    
    def _rotate_piece(self, piece, rotation):
        """
        Rotate a piece by the specified number of 90-degree rotations.
        
        Args:
            piece: Tuple of (north, east, south, west) edge colors
            rotation: Number of 90-degree clockwise rotations (0-3)
            
        Returns:
            Tuple of rotated edge colors
        """
        return tuple(piece[(-rotation) % 4:] + piece[:(-rotation) % 4])
    
    def _encode_piece(self, piece_id, piece):
        """
        Encode a piece as a 32-bit integer.
        
        Format: [piece_id(8 bits) | north(5 bits) | east(5 bits) | south(5 bits) | west(5 bits)]
        
        Args:
            piece_id: ID of the piece (0-255)
            piece: Tuple of (north, east, south, west) edge colors
            
        Returns:
            32-bit encoding of the piece
        """
        assert 0 <= piece_id < 256, "Piece ID must be in range 0-255"
        encoding = piece_id << 20  # Piece ID in the top 8 bits
        
        # Encode each edge color in 5 bits
        for i, color in enumerate(piece):
            assert 0 <= color < 32, f"Color {color} exceeds 5-bit range"
            # Shift based on position (north=15, east=10, south=5, west=0)
            encoding |= color << (15 - i * 5)
            
        return encoding
    
    def _decode_piece(self, encoding):
        """
        Decode a 32-bit piece encoding back to piece_id and edge colors.
        
        Args:
            encoding: 32-bit encoded piece
            
        Returns:
            Tuple of (piece_id, (north, east, south, west))
        """
        piece_id = (encoding >> 20) & 0xFF
        north = (encoding >> 15) & 0x1F
        east = (encoding >> 10) & 0x1F
        south = (encoding >> 5) & 0x1F
        west = encoding & 0x1F
        
        return piece_id, (north, east, south, west)
    
    def _initialize_cuda_tensors(self):
        """Initialize CUDA tensors for GPU-based processing."""
        # Convert piece encodings to CUDA tensor
        self.cuda_piece_encodings = torch.tensor(
            self.piece_encodings, dtype=torch.int32, device='cuda'
        )
        
        # Create edge color mapping tensors for fast lookup on GPU
        # For each (side, color) combination, create a tensor of valid piece indices
        max_matches = max(len(pieces) for pieces in self.edge_color_piece_map.values())
        
        # Create a tensor of shape [4, NUM_COLORS, max_matches, 2]
        # Each entry contains (piece_id, orientation)
        edge_color_maps = -np.ones((4, NUM_COLORS, max_matches, 2), dtype=np.int32)
        
        for (side, color), pieces in self.edge_color_piece_map.items():
            for i, (piece_id, orientation) in enumerate(pieces):
                if i < max_matches:
                    edge_color_maps[side, color, i, 0] = piece_id
                    edge_color_maps[side, color, i, 1] = orientation
        
        self.cuda_edge_color_maps = torch.tensor(
            edge_color_maps, dtype=torch.int32, device='cuda'
        )
        
        # Create a tensor to track the count of valid pieces for each (side, color)
        edge_color_counts = np.zeros((4, NUM_COLORS), dtype=np.int32)
        
        for (side, color), pieces in self.edge_color_piece_map.items():
            edge_color_counts[side, color] = len(pieces)
        
        self.cuda_edge_color_counts = torch.tensor(
            edge_color_counts, dtype=torch.int32, device='cuda'
        )
    
    def get_valid_pieces(self, side, color):
        """
        Get all pieces that have the specified color on the specified side.
        
        Args:
            side: Side of the piece (0=north, 1=east, 2=south, 3=west)
            color: Color ID
            
        Returns:
            List of (piece_id, orientation) tuples
        """
        return self.edge_color_piece_map.get((side, color), [])
    
    def get_matching_side(self, side):
        """
        Get the opposite side that needs to match.
        
        Args:
            side: Side of the piece (0=north, 1=east, 2=south, 3=west)
            
        Returns:
            The opposite side (0=north, 1=east, 2=south, 3=west)
        """
        return (side + 2) % 4
    
    def mark_piece_used(self, piece_id):
        """
        Mark a piece as used in the bitset.
        
        Args:
            piece_id: ID of the piece to mark as used
        """
        self.used_pieces_bitset |= (1 << piece_id)
    
    def mark_piece_unused(self, piece_id):
        """
        Mark a piece as unused in the bitset.
        
        Args:
            piece_id: ID of the piece to mark as unused
        """
        self.used_pieces_bitset &= ~(1 << piece_id)
    
    def is_piece_used(self, piece_id):
        """
        Check if a piece is marked as used.
        
        Args:
            piece_id: ID of the piece to check
            
        Returns:
            True if the piece is used, False otherwise
        """
        return bool(self.used_pieces_bitset & (1 << piece_id))
    
    def reset_used_pieces(self):
        """Reset the used pieces bitset."""
        self.used_pieces_bitset = 0
    
    def apply_hints(self, grid):
        """
        Apply the hint pieces to the grid and mark them as used.
        
        Args:
            grid: 2D grid to place the hints in
            
        Returns:
            Updated grid with hints placed
        """
        for piece_id, row, col, orientation in self.hint_pieces:
            rotated_piece = self._rotate_piece(self.piece_defs[piece_id], orientation)
            grid[row, col] = (piece_id, orientation, rotated_piece)
            self.mark_piece_used(piece_id)
        
        return grid 