import sys
import os
import math
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import load_mesh_as_points, save_mesh
from src.icp import ScaleAdaptiveICP

def estimate_similarity(source, target):
    """
    Estimates s, R, t such that s * (source @ R.T) + t ~= target
    Assumes perfect correspondence (points are matched by index).
    """
    # 1. Centroids
    mu_s = np.mean(source, axis=0)
    mu_t = np.mean(target, axis=0)
    
    src_c = source - mu_s
    tgt_c = target - mu_t
    
    # 2. Scale
    # Ratio of RMS distances from centroid
    ss_s = np.sum(src_c**2)
    ss_t = np.sum(tgt_c**2)
    s = np.sqrt(ss_t / ss_s)
    
    # 3. Rotation
    # min || s * src_c @ R.T - tgt_c ||
    # Equivalent to min || src_c @ R.T - tgt_c/s ||
    # Procrustes
    H = np.dot(src_c.T, tgt_c)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)
        
    # 4. Translation
    # t = mu_t - s * (mu_s @ R.T)
    t = mu_t - s * np.dot(mu_s, R.T)
    
    return s, R, t

def apply_similarity(points, s, R, t):
    return s * np.dot(points, R.T) + t

def main():
    print("Experiment: Comparing 'Direct' vs 'Swapped & Inverted' strategies")
    
    # Paths
    dense_path = 'example_models/main.obj'
    sparse_path = 'example_models/to_fit.obj'
    
    if not os.path.exists(dense_path) or not os.path.exists(sparse_path):
        print("Models not found.")
        return

    # Load
    print("Loading meshes...")
    dense_verts, dense_faces = load_mesh_as_points(dense_path)
    sparse_verts, sparse_faces = load_mesh_as_points(sparse_path) # We need faces to save result
    
    print(f"Dense (main): {dense_verts.shape[0]} points")
    print(f"Sparse (to_fit): {sparse_verts.shape[0]} points")
    
    icp = ScaleAdaptiveICP(max_iterations=50, tolerance=1e-6)

    # --- Strategy 1: Direct (Dense -> Sparse) ---
    print("\n--- Strategy 1: Direct (Dense -> Sparse) ---")
    print("Initializing...")
    dense_init, _ = ScaleAdaptiveICP.pca_align(dense_verts, sparse_verts)
    print("Refining...")
    aligned_direct, _ = icp(dense_init, sparse_verts)
    
    # Error: Direct RMSE (Dense points to nearest Sparse neighbor)
    matched, sq_dists = icp.find_correspondences(aligned_direct, sparse_verts)
    rmse_direct = math.sqrt(np.mean(sq_dists))
    print(f"Direct RMSE: {rmse_direct:.6f}")
    save_mesh('result_direct.obj', aligned_direct, dense_faces)

    # --- Strategy 2: Swapped (Sparse -> Dense) ---
    print("\n--- Strategy 2: Swapped (Sparse -> Dense) ---")
    print("Initializing...")
    sparse_init, _ = ScaleAdaptiveICP.pca_align(sparse_verts, dense_verts)
    print("Refining...")
    aligned_swapped, _ = icp(sparse_init, dense_verts)
    
    # Check alignment of swapped
    matched_s, sq_dists_s = icp.find_correspondences(aligned_swapped, dense_verts)
    rmse_swapped_forward = math.sqrt(np.mean(sq_dists_s))
    print(f"Swapped Forward RMSE (Sparse -> Dense surface): {rmse_swapped_forward:.6f}")
    
    # --- Inversion ---
    print("Computing Inverse Transform...")
    # Recover T that mapped original sparse_verts -> aligned_swapped
    s, R, t = estimate_similarity(sparse_verts, aligned_swapped)
    
    print(f"Recovered Forward Transform: s={s:.4f}")
    
    s_inv = 1.0 / s
    R_inv = R.T  
    t_inv = -s_inv * np.dot(t, R)
    
    # Apply T_inv to Dense (main)
    dense_inverted = apply_similarity(dense_verts, s_inv, R_inv, t_inv)
    
    # Error: Inverted RMSE
    matched_i, sq_dists_i = icp.find_correspondences(dense_inverted, sparse_verts)
    rmse_inverted = math.sqrt(np.mean(sq_dists_i))
    print(f"Inverted RMSE (Dense -> Sparse): {rmse_inverted:.6f}")
    
    save_mesh('result_inverted.obj', dense_inverted, dense_faces)
    
    print("\n--- Summary ---")
    print(f"Direct RMSE:   {rmse_direct:.6f}")
    print(f"Inverted RMSE: {rmse_inverted:.6f}")
    
    if rmse_inverted < rmse_direct:
        print("Conclusion: Swapping strategy IS better.")
    else:
        print("Conclusion: Direct strategy is better (or metrics are misleading due to sparsity).")

if __name__ == "__main__":
    main()