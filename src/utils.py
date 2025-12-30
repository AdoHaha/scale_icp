import numpy as np
import trimesh


def _ensure_trimesh(mesh_or_scene):
    if isinstance(mesh_or_scene, trimesh.Scene):
        if len(mesh_or_scene.geometry) == 0:
            return None
        meshes = list(mesh_or_scene.geometry.values())
        return trimesh.util.concatenate(meshes)
    return mesh_or_scene

def load_mesh_as_points(file_path):
    """
    Loads an OBJ file and returns the vertices and faces.
    
    Args:
        file_path (str): Path to the .obj file.
        
    Returns:
        numpy.ndarray: Array of shape (N, 3) containing vertices.
        numpy.ndarray: Array of shape (F, 3) containing faces (indices).
    """
    mesh = trimesh.load(file_path, process=False)
    mesh = _ensure_trimesh(mesh)
    if mesh is None or mesh.vertices is None or len(mesh.vertices) == 0:
        raise ValueError(f"Mesh at {file_path} has no vertices.")

    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces) if mesh.faces is not None else np.zeros((0, 3))
    
    return verts, faces

def save_mesh(file_path, verts, faces):
    """
    Saves a mesh to an OBJ file.
    
    Args:
        file_path (str): Output path.
        verts (numpy.ndarray): Shape (N, 3).
        faces (numpy.ndarray): Shape (F, 3).
    """
    if faces is None:
        faces = np.zeros((0, 3), dtype=np.int64)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh.export(file_path)


def load_points(file_path):
    """
    Loads a point cloud from common formats (.npy, .npz, .txt/.xyz, mesh files).
    Returns:
        numpy.ndarray: Array of shape (N, 3).
    """
    if file_path.endswith(".npy"):
        return np.load(file_path)
    if file_path.endswith(".npz"):
        data = np.load(file_path)
        if "points" not in data:
            raise ValueError("Expected key 'points' in npz file.")
        return data["points"]
    if file_path.endswith(".txt") or file_path.endswith(".xyz"):
        return np.loadtxt(file_path)

    mesh = trimesh.load(file_path, process=False)
    mesh = _ensure_trimesh(mesh)
    if mesh is None or mesh.vertices is None or len(mesh.vertices) == 0:
        raise ValueError(f"Mesh at {file_path} has no vertices.")
    return np.asarray(mesh.vertices)


def save_points(file_path, points):
    """
    Saves a point cloud to .npy or .xyz (text).
    """
    if file_path.endswith(".npy"):
        np.save(file_path, points)
        return
    if file_path.endswith(".xyz") or file_path.endswith(".txt"):
        np.savetxt(file_path, points)
        return
    raise ValueError("Unsupported output format for points.")
