import sys
import os
import math
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import load_mesh_as_points, save_mesh
from src.icp import ScaleAdaptiveICP

def main():
    print("Initializing Scale-Adaptive ICP experiment...")

    # Paths
    source_path = 'example_models/to_fit.obj' # The one moving
    target_path = 'example_models/main.obj'   # The fixed one
    output_path = 'aligned_result.obj'

    if not os.path.exists(source_path) or not os.path.exists(target_path):
        print(f"Error: Example models not found at {source_path} or {target_path}")
        return

    # Load data
    print("Loading meshes...")
    # verts: (N, 3), faces: (F, 3)
    source_verts, source_faces = load_mesh_as_points(source_path)
    target_verts, _ = load_mesh_as_points(target_path)
    
    print(f"Source points: {source_verts.shape[0]}")
    print(f"Target points: {target_verts.shape[0]}")

    # Initialize ICP
    icp = ScaleAdaptiveICP(max_iterations=50, tolerance=1e-6)

    # Run Registration
    print("Starting Scale-Adaptive ICP registration...")
    
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
    
    print(f"Registration finished.")
    print(f"Final RMSE: {rmse:.6f}")

    # Save result
    save_mesh(output_path, aligned_verts, source_faces)
    print(f"Saved aligned mesh to {output_path}")

if __name__ == "__main__":
    main()