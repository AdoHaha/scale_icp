import sys
import os
import math
import numpy as np
from scipy.spatial.transform import Rotation as SciRot

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import load_mesh_as_points
from src.icp import ScaleAdaptiveICP

def generate_random_transform():
    # Random uniform scale between 0.5 and 2000.0
    scale = np.random.uniform(0.5, 2000.0)
    
    # Random full rotation (Global Registration challenge)
    rot = SciRot.random().as_matrix()
    
    # Random translation
    trans = np.random.uniform(-100.0, 100.0, size=(3,))
    
    return scale, rot, trans

def apply_transform(points, s, R, t):
    # points: (N, 3)
    # transformed = s * (R @ points^T)^T + t
    #             = s * points @ R^T + t
    return s * np.dot(points, R.T) + t

def main():
    print("Initializing Synthetic Test (Global Registration + Scale)...")
    
    # Path to a mesh
    mesh_path = 'example_models/to_fit.obj'
    if not os.path.exists(mesh_path):
        print(f"Error: {mesh_path} not found.")
        return

    # Load original points
    print("Loading mesh...")
    original_verts, _ = load_mesh_as_points(mesh_path)
    
    # Center the mesh at the origin to ensure rotation doesn't throw it far away
    # This is critical for ICP which assumes rough initial alignment (or at least overlap)
    centroid = np.mean(original_verts, axis=0)
    target_points = original_verts - centroid
    print(f"Centered mesh. Centroid shifted from {centroid} to {np.mean(target_points, axis=0)}")
    
    # Downsample for speed if needed, but 778 points is small enough (from previous run output)
    print(f"Number of points: {target_points.shape[0]}")
    
    # 2. Generate Source by applying random transform to original points
    s_gt, R_gt, t_gt = generate_random_transform()
    
    print("\nGround Truth Transform applied to create Source:")
    print(f"Scale: {s_gt:.4f}")
    print(f"Translation: {t_gt}")
    print(f"Rotation (det): {np.linalg.det(R_gt):.4f}")
    
    source_points = apply_transform(target_points, s_gt, R_gt, t_gt)
    
    # 3. Global Initialization using PCA
    print("\nRunning PCA Initialization...")
    source_init = ScaleAdaptiveICP.pca_align(source_points, target_points)
    
    # Check initial error after PCA
    # Just for info
    icp = ScaleAdaptiveICP(max_iterations=100, tolerance=1e-7)
    
    # 4. Run ICP to refine alignment
    print("Running Scale-Adaptive ICP refinement...")
    aligned_source = icp(source_init, target_points)
    
    # 5. Compute Error
    diff = aligned_source - target_points
    mse = np.mean(np.sum(diff**2, axis=1))
    rmse = math.sqrt(mse)
    
    print(f"\nFinal RMSE (Point-to-Point): {rmse:.8f}")
    
    if rmse < 1e-4:
        print("SUCCESS: Recovery is near perfect!")
    else:
        print("WARNING: Recovery error is high.")

if __name__ == "__main__":
    # Seed for reproducibility
    np.random.seed(42)
    main()
