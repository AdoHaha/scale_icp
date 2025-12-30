import sys
import os
import math
import numpy as np
import torch
from scipy.spatial.transform import Rotation as SciRot

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import load_mesh_as_points
from src.icp import ScaleAdaptiveICP

def get_device():
    device = os.environ.get("SCALE_ICP_DEVICE")
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_random_transform():
    # Random uniform scale between 0.5 and 2000.0
    scale = np.random.uniform(0.5, 2000.0)
    
    # Random full rotation
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
    print("Initializing Parameter Verification Test...")
    device = get_device()
    print(f"Using device: {device}")
    
    # Use a simple cube or similar if available, or just use the to_fit model
    mesh_path = 'example_models/to_fit.obj'
    if not os.path.exists(mesh_path):
        print(f"Error: {mesh_path} not found.")
        return

    # Load original points
    print("Loading mesh...")
    original_verts, _ = load_mesh_as_points(mesh_path)
    
    # Center points
    centroid = np.mean(original_verts, axis=0)
    centered_verts = original_verts - centroid
    
    # Create Source by undersampling (take 50% of points)
    n_points = centered_verts.shape[0]
    indices = np.random.choice(n_points, n_points // 2, replace=False)
    source_points = centered_verts[indices]
    
    # Create Target by applying random transform to ALL original points
    s_gt, R_gt, t_gt = generate_random_transform()
    target_points = apply_transform(centered_verts, s_gt, R_gt, t_gt)
    
    print(f"Source points: {source_points.shape[0]}")
    print(f"Target points: {target_points.shape[0]}")
    
    print(f"\nGround Truth Transform (Source subset -> Target):")
    print(f"Scale: {s_gt:.4f}")
    
    # Run Alignment
    print("\nRunning Alignment (PCA + ICP)...")
    
    # 1. PCA Init
    source_t = torch.as_tensor(source_points, device=device, dtype=torch.float32)
    target_t = torch.as_tensor(target_points, device=device, dtype=torch.float32)
    source_init, pca_params = ScaleAdaptiveICP.pca_align(
        source_t, target_t, device=device
    )
    
    # 2. ICP Refine
    icp = ScaleAdaptiveICP(max_iterations=100, tolerance=1e-7, device=device)
    aligned_source, icp_params = icp(source_init, target_t)
    
    # 3. Compose Params (PCA + ICP)
    # Total Transform T = T_icp(T_pca(x))
    total_params = ScaleAdaptiveICP.compose_transforms(icp_params, pca_params)
    
    print("\nRecovered Transform (Source -> Target):")
    total_s_val = total_params["s"].item() if torch.is_tensor(total_params["s"]) else total_params["s"]
    print(f"Scale: {total_s_val:.4f}")
    
    # Verify Forward (Source -> Target) using params
    # manually apply params to source_points
    total_s = total_params["s"]
    total_R = total_params["R"]
    total_t = total_params["t"]
    manual_aligned = total_s * (source_t @ total_R.transpose(0, 1)) + total_t
    target_subset = target_t[torch.as_tensor(indices, device=device)]
    diff_fwd = manual_aligned - target_subset
    rmse_fwd = math.sqrt((diff_fwd * diff_fwd).sum(dim=1).mean().item())
    print(f"Forward RMSE (using params, subset check): {rmse_fwd:.8f}")
    
    # 4. Invert Params
    s_inv, R_inv, t_inv = ScaleAdaptiveICP.invert_transform(
        total_params["s"], total_params["R"], total_params["t"]
    )
    
    print("\nInverted Transform (Target -> Source):")
    s_inv_val = s_inv.item() if torch.is_tensor(s_inv) else s_inv
    print(f"Scale: {s_inv_val:.6f} (Expected: {1/s_gt:.6f})")
    
    # Verify Inverse (Target -> Source) using params
    # Apply inverted params to target_points, should match the full centered_verts
    centered_t = torch.as_tensor(centered_verts, device=device, dtype=torch.float32)
    manual_inverse = s_inv * (target_t @ R_inv.transpose(0, 1)) + t_inv
    diff_inv = manual_inverse - centered_t
    rmse_inv = math.sqrt((diff_inv * diff_inv).sum(dim=1).mean().item())
    print(f"Inverse RMSE (using inverted params, full mesh check): {rmse_inv:.8f}")
    
    if rmse_fwd < 1e-4 and rmse_inv < 1e-4:
        print("\nSUCCESS: Both forward and inverse parameters work perfectly.")
    else:
        print("\nFAILURE: Parameter verification failed.")

if __name__ == "__main__":
    np.random.seed(42)
    main()
