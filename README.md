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

## API Usage

The Python API provides access to the raw points and the transformation parameters (`s`, `R`, `t`), allowing you to recover the full transformation matrix or invert it.

```python
from src.icp import ScaleAdaptiveICP

# 1. Global Initialization (Returns coarse aligned points + PCA params)
source_init, pca_params = ScaleAdaptiveICP.pca_align(source_points, target_points)

# 2. Iterative Refinement (Returns final points + ICP params)
icp = ScaleAdaptiveICP(max_iterations=100)
aligned_source, icp_params = icp(source_init, target_points)

# 3. Get Total Forward Transform (Source -> Target)
# T_total = T_icp(T_pca(x))
total_forward = ScaleAdaptiveICP.compose_transforms(icp_params, pca_params)
print(f"Scale: {total_forward['s']}")
print(f"Rotation: \n{total_forward['R']}")
print(f"Translation: {total_forward['t']}")

# 4. Get Inverse Transform (Target -> Source)
# Useful if you aligned Sparse->Dense but need the Dense->Sparse transform
s_inv, R_inv, t_inv = ScaleAdaptiveICP.invert_transform(
    total_forward['s'], total_forward['R'], total_forward['t']
)
```

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

## Best Practices

**Target Selection**: Always set the **denser** point cloud as the **Target** (Fixed) and the **sparser** cloud as the **Source** (Moving).
*   **Reason**: ICP minimizes the distance from Source points to the nearest Target surface. If the Target is sparse, Source points map to discrete points rather than a continuous surface, leading to poor convergence and "clumping."
*   **Inversion**: If you need to move the dense cloud, perform the alignment as Sparse $\to$ Dense first, then compute the inverse transformation and apply it to the dense cloud.

## Comparison with Original C++ Implementation

While based on the mathematical core of the original C++ code, this Python implementation introduces several architectural differences:

1.  **Efficiency**: This implementation uses `scipy.spatial.cKDTree` by default, ensuring $O(N \log M)$ performance for nearest neighbor queries. The original C++ reference implementation defaults to brute-force $O(N \cdot M)$ search (with an experimental KD-tree option), which can be slower for dense meshes like the ones used in this project (~175k points).
2.  **Global Initialization**: The original Scale-Adaptive ICP is a local optimizer, requiring rough initial alignment. We have added a **PCA Initialization** step to handle arbitrary initial rotations and scales automatically, making the tool robust to global registration challenges.
3.  **Scale Strategy**: The original paper employs a "1-to-1 correspondence" heuristic to prevent the source model from shrinking when initially much smaller than the target. Our approach solves this by normalizing the scale via PCA covariance matrices *before* starting the ICP loop, rendering the heuristic largely unnecessary for these cases.
4.  **Input Formats**: Supports standard mesh formats (`.obj`, `.ply`, etc.) via Open3D, whereas the original implementation processes raw `.xyz` point clouds.

## Authors & Reference

**Implementation Author:** Igor Zubrycki

This repository contains a partial implementation (focusing on the core algorithm and PCA initialization) of the method described in:

> **Scale-Adaptive ICP**  
> Yusuf Sahillioğlu and Ladislav Kavan  
> *Graphical Models* 116 (2021) 101113  
> DOI: [10.1016/j.gmod.2021.101113](https://doi.org/10.1016/j.gmod.2021.101113)

Original project page: [http://www.ceng.metu.edu.tr/~ys/pubs/ScaleAdaptiveICP.zip](http://www.ceng.metu.edu.tr/~ys/pubs/ScaleAdaptiveICP.zip)
