# Scale-Adaptive ICP with PCA Initialization

This project implements the **Scale-Adaptive Iterative Closest Point (ICP)** algorithm based on Sahillioglu & Kavan (2021), enhanced with a **Principal Component Analysis (PCA)** initialization step to handle arbitrary initial rotations and scales.

## Overview

The standard ICP algorithm assumes roughly aligned models and fails when there are significant differences in scale or orientation. This implementation solves these issues by:

1.  **Global Initialization (PCA)**: Roughly aligns the source and target models by matching their centroids and principal axes (eigenvectors of covariance). It estimates an initial uniform scale factor based on the ratio of the eigenvalues.
2.  **Scale-Adaptive ICP Refinement**: Iteratively optimizes for:
    *   **Correspondence**: Nearest neighbor search using KD-Trees.
    *   **Rotation**: Optimal rigid rotation (SVD).
    *   **Scale & Translation**: Solved jointly via a $4 \times 4$ linear system to minimize alignment error.

## Features

*   **Robust to Initial Scale**: Can recover scale differences from 0.5x to 2000x.
*   **Robust to Initial Rotation**: Handles full global rotation using PCA alignment.
*   **Efficient**: Uses `scipy.spatial.cKDTree` for fast neighbor queries and `numpy` for vectorized linear algebra.
*   **Library**: Built with **Open3D** (for mesh IO) and **NumPy/SciPy** (for math).

## Installation

Ensure you have Python installed, then install the dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Synthetic Test
Verifies the algorithm by taking a model, applying a random transformation (Scale 0.5-2000x, random Rotation, Translation), and recovering the original pose.

```bash
python3 tests/test_synthetic.py
```
**Expected Output:** `Final RMSE (Point-to-Point): 0.00000000` (Success)

### 2. Real Experiment
Aligns `example_models/to_fit.obj` (Source) to `example_models/main.obj` (Target).

```bash
python3 tests/run_experiment.py
```
**Output:** Saves the aligned model to `aligned_result.obj`.

## Implementation Details

### PCA Initialization (`src/icp.py`)
To handle global registration:
1.  Compute centroids $\mu_s, \mu_t$ and covariance matrices $C_s, C_t$.
2.  **Scale Estimate**: $s = \sqrt{\lambda_{t,0} / \lambda_{s,0}}$ (ratio of largest eigenvalues).
    *   *Note*: Covariances are normalized by point count to ensure scale is density-invariant.
3.  **Rotation Estimate**: Matches eigenvectors $U_s$ to $U_t$. Tests 4 sign permutations to resolve the 180-degree ambiguity, picking the one that minimizes point-to-point distance on a subsample.

### Scale-Adaptive Loop (`src/icp.py`)
Iterates until convergence:
1.  **Match**: Find nearest neighbor $q_i$ for each $p_i$.
2.  **Rotate**: Compute $R$ minimizing $\sum \| R p_i - q_i \|^2$.
3.  **Scale & Translate**: Solve linear system $A x = b$ for $s, t$ where $x = [s, t_x, t_y, t_z]^T$.

## File Structure

*   `src/icp.py`: Core algorithm implementation.
*   `src/utils.py`: Helper functions for Mesh IO.
*   `tests/`: Scripts for validation and experiments.
