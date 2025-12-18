import open3d as o3d
import numpy as np

def load_mesh_as_points(file_path):
    """
    Loads an OBJ file and returns the vertices and faces.
    
    Args:
        file_path (str): Path to the .obj file.
        
    Returns:
        numpy.ndarray: Array of shape (N, 3) containing vertices.
        numpy.ndarray: Array of shape (F, 3) containing faces (indices).
    """
    mesh = o3d.io.read_triangle_mesh(file_path)
    if not mesh.has_vertices():
        raise ValueError(f"Mesh at {file_path} has no vertices.")
        
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    
    return verts, faces

def save_mesh(file_path, verts, faces):
    """
    Saves a mesh to an OBJ file.
    
    Args:
        file_path (str): Output path.
        verts (numpy.ndarray): Shape (N, 3).
        faces (numpy.ndarray): Shape (F, 3).
    """
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    if faces is not None and faces.size > 0:
        mesh.triangles = o3d.utility.Vector3iVector(faces)
    
    o3d.io.write_triangle_mesh(file_path, mesh)