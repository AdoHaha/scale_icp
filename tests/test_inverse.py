import sys
import os
import math
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import load_mesh_as_points, save_mesh
from src.icp import ScaleAdaptiveICP

def main():
    print("Initializing Inverse Scale-Adaptive ICP experiment...")
    print("Goal: Align dense 'main.obj' (Source) to sparse 'to_fit.obj' (Target)")

    # Paths
    source_path = 'example_models/main.obj'    # Dense moving model
    target_path = 'example_models/to_fit.obj'  # Sparse fixed model
    output_path = 'aligned_inverse.obj'

    if not os.path.exists(source_path) or not os.path.exists(target_path):
        print(f"Error: Example models not found.")
        return

    # Load data
    print("Loading meshes...")
    source_verts, source_faces = load_mesh_as_points(source_path)
    target_verts, _ = load_mesh_as_points(target_path)
    
    print(f"Source points (Moving): {source_verts.shape[0]}")
    print(f"Target points (Fixed):  {target_verts.shape[0]}")

    # Initialize ICP
    icp = ScaleAdaptiveICP(max_iterations=50, tolerance=1e-6)

    # Run Registration
    print("\nStarting Scale-Adaptive ICP registration...")
    
    print("Performing Global PCA Initialization...")
    source_coarse, pca_params = ScaleAdaptiveICP.pca_align(source_verts, target_verts)
    
    # Check coarse error
    coarse_matched, coarse_squared_dists = icp.find_correspondences(source_coarse, target_verts)
    coarse_rmse = math.sqrt(np.mean(coarse_squared_dists))
    print(f"RMSE after PCA Initialization: {coarse_rmse:.6f}")
    
    print("Performing Iterative Refinement...")
    aligned_verts, icp_params = icp(source_coarse, target_verts)
    
    # Calculate final error (RMSE to nearest neighbor)
    final_matched, squared_dists = icp.find_correspondences(aligned_verts, target_verts)
    mse = np.mean(squared_dists)
    rmse = math.sqrt(mse)
    
    print(f"\nRegistration finished.")
    print(f"Final RMSE (to sparse target): {rmse:.6f}")

    # Save result
    save_mesh(output_path, aligned_verts, source_faces)
    print(f"Saved aligned dense mesh to {output_path}")

if __name__ == "__main__":
    main()
