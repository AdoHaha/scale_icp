import numpy as np
from scipy.spatial import cKDTree

class ScaleAdaptiveICP:
    def __init__(self, max_iterations=20, tolerance=1e-5):
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def find_correspondences(self, source_points, target_points):
        """
        Finds the nearest neighbor in target for each point in source.
        Args:
            source_points: (N, 3) numpy array
            target_points: (M, 3) numpy array
        Returns:
            matched_target_points: (N, 3) corresponding points from target
            distances: (N,) squared L2 distances
        """
        # Build KDTree on target points
        # For efficiency, if target doesn't change, we could cache this.
        # But in ICP target is usually static.
        tree = cKDTree(target_points)
        
        # Query
        dists, indices = tree.query(source_points, k=1)
        
        # Gather points
        matched_target_points = target_points[indices]
        
        # dists from cKDTree are Euclidean distances, we want squared for consistency/checking
        squared_dists = dists ** 2
        
        return matched_target_points, squared_dists

    def compute_rotation(self, source, target):
        """
        Computes optimal rotation R that aligns centered source to centered target.
        Args:
            source: (N, 3)
            target: (N, 3) corresponding points
        Returns:
            R: (3, 3) rotation matrix
        """
        # Compute centroids
        source_mean = np.mean(source, axis=0) # (3,)
        target_mean = np.mean(target, axis=0) # (3,)
        
        # Center the points
        p_c = source - source_mean
        q_c = target - target_mean
        
        # Covariance matrix H = P_c^T @ Q_c
        H = np.dot(p_c.T, q_c) # (3, 3)
        
        # SVD
        U, S, Vt = np.linalg.svd(H)
        
        # R = V @ U^T
        # numpy svd returns Vt which is V^T. 
        # So V = Vt.T
        R = np.dot(Vt.T, U.T)
        
        # Handle reflection
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = np.dot(Vt.T, U.T)
            
        return R

    def compute_scale_translation(self, rotated_source, target):
        """
        Solves Eq. 6 for scale s and translation t.
        Args:
            rotated_source (p'): (N, 3) source points after rotation
            target (q): (N, 3) corresponding target points
        Returns:
            s: scalar scale
            t: (3,) translation vector
        """
        p_prime = rotated_source # (N, 3)
        q = target # (N, 3)
        n = p_prime.shape[0]
        
        # sum(p'^T p')
        # This is sum of squared norms of p_prime
        sum_p_sq = np.sum(p_prime * p_prime)
        
        # c vector = sum(p')
        c = np.sum(p_prime, axis=0) # (3,)
        c0, c1, c2 = c
        
        # Construct A
        A = np.zeros((4, 4))
        A[0, 0] = sum_p_sq
        A[0, 1:] = c
        A[1:, 0] = c
        
        A[1, 1] = n
        A[2, 2] = n
        A[3, 3] = n
        
        # Construct b
        # sum(p'^T q) -> sum of dot products
        sum_pq = np.sum(p_prime * q)
        
        # d vector = sum(q)
        d = np.sum(q, axis=0) # (3,)
        
        b = np.zeros(4)
        b[0] = sum_pq
        b[1:] = d
        
        # Solve Ax = b
        try:
            x = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            # Fallback
            x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            
        s = x[0]
        t = x[1:]
        
        return s, t

    @staticmethod
    def pca_align(source, target):
        """
        Computes a coarse alignment (Rotation, Scale, Translation) using PCA.
        This handles global rotation and initial scale estimation.
        """
        # 1. Centroids
        mu_s = np.mean(source, axis=0)
        mu_t = np.mean(target, axis=0)
        
        # Center data
        src_c = source - mu_s
        tgt_c = target - mu_t
        
        # 2. Covariance and Eigendecomposition
        # Cov = (1/N) * X.T @ X
        Cov_s = np.dot(src_c.T, src_c) / len(source)
        Cov_t = np.dot(tgt_c.T, tgt_c) / len(target)
        
        # SVD returns U, S, Vt. U are eigenvectors, S eigenvalues (variances).
        U_s, S_s, _ = np.linalg.svd(Cov_s)
        U_t, S_t, _ = np.linalg.svd(Cov_t)
        
        # 3. Estimate Scale
        # Ratio of spread along principal axis (sqrt of eigenvalues)
        # S contains eigenvalues squared? No, for Cov=X.T@X, eigenvalues are variance*N.
        # We need sqrt ratio for scale.
        # Handle division by zero
        if S_s[0] < 1e-8:
            scale_init = 1.0
        else:
            scale_init = np.sqrt(S_t[0] / S_s[0])
            
        # 4. Estimate Rotation with ambiguity check
        # R = U_t @ M @ U_s.T
        # We test 4 sign combinations for M that preserve det(R)=1 (proper rotation)
        # ( +,+,+ ), ( +,-,- ), ( -,+,- ), ( -,-,+ )
        possible_signs = [
            np.diag([1, 1, 1]),
            np.diag([1, -1, -1]),
            np.diag([-1, 1, -1]),
            np.diag([-1, -1, 1])
        ]
        
        best_error = float('inf')
        best_transform = None # (s, R, t)
        
        # Subsample for speed if N is large
        N = source.shape[0]
        indices = np.random.choice(N, min(N, 1000), replace=False)
        sub_src = source[indices]
        sub_tgt = target[indices]
        
        # Pre-scale source for checking
        sub_src_scaled_centered = (sub_src - mu_s) * scale_init
        
        # We need to find R such that: scale * (source-mu_s) @ R.T + mu_t ~ target
        # So we align the *centered* versions.
        
        for M in possible_signs:
            R_candidate = np.dot(U_t, np.dot(M, U_s.T))
            
            # Rotate
            # rotated = centered @ R.T
            rotated_c = np.dot(sub_src_scaled_centered, R_candidate.T)
            
            # Translate to target center
            aligned = rotated_c + mu_t
            
            # Simple error metric: distance to nearest neighbor in target
            # Ideally we check against corresponding features, but we don't have them.
            # We assume PCA axes align semantically (major axis to major axis).
            # We just sum distances to centroids or check overlap. 
            # A simple check is checking distance to nearest neighbor in sub_tgt.
            # But PCA alignment assumes axes match.
            # Let's use simple Euclidean distance between transformed sub_src and sub_tgt 
            # Assuming indices roughly correspond? NO, indices don't correspond globally.
            # We can use KDTree for error, but that's expensive inside this loop?
            # Actually, just checking if the bounding box aligns is often enough.
            # Let's use a quick KDTree on the subsample.
            
            tree = cKDTree(sub_tgt)
            dists, _ = tree.query(aligned, k=1)
            error = np.mean(dists)
            
            if error < best_error:
                best_error = error
                best_R = R_candidate
        
        # Final Coarse Transform
        # result = scale * (source - mu_s) @ R.T + mu_t
        #        = scale * source @ R.T - scale * mu_s @ R.T + mu_t
        # new_source = scale * source @ R.T + (mu_t - scale * mu_s @ R.T)
        
        # Let's return the transformed points directly
        s = scale_init
        R = best_R
        t = mu_t - s * np.dot(mu_s, R.T)
        
        transformed_source = s * np.dot(source, R.T) + t
        return transformed_source

    def __call__(self, source_points, target_points):
        """
        Runs the Scale-Adaptive ICP.
        Args:
            source_points: (N, 3) numpy array
            target_points: (M, 3) numpy array
        Returns:
            transformed_source: (N, 3) numpy array
        """
        current_source = source_points.copy()
        
        prev_error = float('inf')
        
        for i in range(self.max_iterations):
            # 1. Correspondences
            matched_target, squared_dists = self.find_correspondences(current_source, target_points)
            
            error = np.mean(squared_dists)
            if abs(prev_error - error) < self.tolerance:
                break
            prev_error = error
            
            # 2. Optimal Rotation
            R = self.compute_rotation(current_source, matched_target)
            
            # Rotate current points
            # (N, 3) @ (3, 3)^T
            rotated_source = np.dot(current_source, R.T)
            
            # 3. Optimal Scale and Translation
            s, t = self.compute_scale_translation(rotated_source, matched_target)
            
            # 4. Update
            # new_source = s * rotated_source + t
            current_source = s * rotated_source + t
            
        return current_source