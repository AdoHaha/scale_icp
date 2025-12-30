import numpy as np
import os
import trimesh

def compare_meshes():
    target_path = 'example_models/main.obj'
    result_path = 'aligned_result.obj'
    
    if not os.path.exists(target_path) or not os.path.exists(result_path):
        print("Files missing.")
        return

    m_tgt = trimesh.load(target_path, process=False)
    m_res = trimesh.load(result_path, process=False)
    
    def get_info(mesh, name):
        if isinstance(mesh, trimesh.Scene):
            if len(mesh.geometry) == 0:
                return f"{name}: No vertices"
            mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
        v = np.asarray(mesh.vertices) if mesh.vertices is not None else np.zeros((0, 3))
        if v.size == 0:
            return f"{name}: No vertices"
        mins = v.min(axis=0)
        maxs = v.max(axis=0)
        extent = maxs - mins
        return {
            "name": name,
            "points": v.shape[0],
            "min": mins,
            "max": maxs,
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
