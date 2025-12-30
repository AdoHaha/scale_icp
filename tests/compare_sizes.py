import open3d as o3d
import numpy as np
import os

def compare_meshes():
    target_path = 'example_models/main.obj'
    result_path = 'aligned_result.obj'
    
    if not os.path.exists(target_path) or not os.path.exists(result_path):
        print("Files missing.")
        return

    m_tgt = o3d.io.read_triangle_mesh(target_path)
    m_res = o3d.io.read_triangle_mesh(result_path)
    
    def get_info(mesh, name):
        v = np.asarray(mesh.vertices)
        if v.size == 0:
            return f"{name}: No vertices"
        bbox = mesh.get_axis_aligned_bounding_box()
        extent = bbox.get_extent()
        return {
            "name": name,
            "points": v.shape[0],
            "min": bbox.min_bound,
            "max": bbox.max_bound,
            "size": extent,
            "diagonal": np.linalg.norm(extent)
        }

    i_tgt = get_info(m_tgt, "Target (main.obj)")
    i_res = get_info(m_res, "Result (aligned_result.obj)")
    
    for info in [i_tgt, i_res]:
        print(f"\n--- {info['name']} ---")
        print(f"Points:   {info['points']}")
        print(f"Size:     {info['size']}")
        print(f"Diagonal: {info['diagonal']:.4f}")
        print(f"File Size: {os.path.getsize(target_path if 'Target' in info['name'] else result_path) / 1024:.2f} KB")

if __name__ == "__main__":
    compare_meshes()
