# Eternity II Solver

A multi-strategy approach to solving the Eternity II puzzle using a combination of preprocessing, GPU-accelerated search, hyper-heuristic optimization, and exact SAT solving.

## Overview

This project implements a comprehensive solver for the Eternity II puzzle with these components:

1. **Preprocessing & Data Structures**: Efficient encoding of pieces and edge colors with bitwise operations
2. **Zero Look-ahead Instantiation (ZLA)**: Massively parallel GPU-based candidate solution generation
3. **Two-Phase Hyper-Heuristic Search**: Optimization using sub-block and edge-matching objectives
4. **SAT/CSP Polishing**: Exact solving for small subregions to complete near-solutions

## Requirements

- Python 3.8+
- CUDA Toolkit 11.0+
- PyTorch 1.9+
- NumPy
- PySAT

## Hardware Target

- CPU: 8+ cores (optimized for Ryzen 7 5800H)
- GPU: NVIDIA GPU with CUDA support (optimized for RTX 3060)
- RAM: 16GB+

## Usage

```bash
# Run the full solver pipeline
python main.py

# Run specific components
python main.py --only-zla  # Only run ZLA phase
python main.py --only-local-search  # Only run local search phase
python main.py --only-sat  # Only run SAT solving phase

# Load a partial solution and continue
python main.py --load-state saved_state.pkl
```

## Project Structure

- `data/`: Puzzle definition and hint pieces
- `src/`: Source code
  - `preprocessing.py`: Data structure setup and preprocessing
  - `zla.py`: Zero look-ahead algorithm implementation
  - `local_search.py`: Two-phase hyper-heuristic local search
  - `sat_solver.py`: SAT/CSP encoding and solving
  - `utils.py`: Helper functions and visualization
- `main.py`: Main entry point orchestrating the solution process

## Performance

On a system with Ryzen 7 5800H and RTX 3060:
- ZLA phase: ~500M configurations explored per hour
- Local search: ~10M moves evaluated per second
- SAT solving: ~60 seconds for 20-piece subproblems 