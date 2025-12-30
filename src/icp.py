import numpy as np
import torch

try:
    from knn_standalone import knn_points as _knn_points_standalone
except Exception:
    _knn_points_standalone = None

class ScaleAdaptiveICP:
    def __init__(self, max_iterations=20, tolerance=1e-5, device="cpu", knn_backend="auto"):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but torch.cuda.is_available() is False.")
        self.knn_backend = self._resolve_knn_backend(knn_backend)

    @staticmethod
    def _resolve_knn_backend(knn_backend):
        if knn_backend == "auto":
            return "knn_standalone" if _knn_points_standalone is not None else "torch"
        if knn_backend == "knn_standalone":
            if _knn_points_standalone is None:
                raise RuntimeError("knn_standalone is not available.")
            return "knn_standalone"
        if knn_backend == "torch":
            return "torch"
        raise ValueError(f"Unknown knn_backend: {knn_backend}")

    def _as_tensor(self, x):
        if torch.is_tensor(x):
            return x.to(device=self.device, dtype=torch.float32), True
        return torch.as_tensor(x, device=self.device, dtype=torch.float32), False

    @staticmethod
    def _to_numpy(x):
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return x

    def _knn_points(self, p1, p2, K=1, norm=2, return_nn=False):
        if p1.ndim == 2:
            p1 = p1.unsqueeze(0)
        if p2.ndim == 2:
            p2 = p2.unsqueeze(0)

        if self.knn_backend == "knn_standalone":
            out = _knn_points_standalone(p1, p2, K=K, norm=norm, return_nn=return_nn)
            dists = out.dists
            idx = out.idx
            nn = out.knn
        else:
            dists = torch.cdist(p1, p2, p=norm)
            dists, idx = torch.topk(dists, k=K, dim=2, largest=False, sorted=True)
            if norm == 2:
                dists = dists * dists
            nn = None
            if return_nn:
                _, _, D = p2.shape
                idx_expanded = idx[:, :, :, None].expand(-1, -1, -1, D)
                nn = p2[:, :, None].expand(-1, -1, K, -1).gather(1, idx_expanded)

        if return_nn:
            return dists.squeeze(0), idx.squeeze(0), nn.squeeze(0)
        return dists.squeeze(0), idx.squeeze(0), None

    def find_correspondences(self, source_points, target_points):
        """
        Finds the nearest neighbor in target for each point in source.
        Args:
            source_points: (N, 3) numpy array or torch Tensor
            target_points: (M, 3) numpy array or torch Tensor
        Returns:
            matched_target_points: (N, 3) corresponding points from target
            distances: (N,) squared L2 distances
        """
        src, src_is_torch = self._as_tensor(source_points)
        tgt, tgt_is_torch = self._as_tensor(target_points)
        return_torch = src_is_torch or tgt_is_torch

        dists, _, nn = self._knn_points(src, tgt, K=1, norm=2, return_nn=True)
        matched_target_points = nn[:, 0, :]
        squared_dists = dists[:, 0]

        if return_torch:
            return matched_target_points, squared_dists
        return self._to_numpy(matched_target_points), self._to_numpy(squared_dists)

    def compute_rotation(self, source, target):
        """
        Computes optimal rotation R that aligns centered source to centered target.
        Args:
            source: (N, 3) numpy array or torch Tensor
            target: (N, 3) corresponding points
        Returns:
            R: (3, 3) rotation matrix
        """
        src, src_is_torch = self._as_tensor(source)
        tgt, tgt_is_torch = self._as_tensor(target)
        return_torch = src_is_torch or tgt_is_torch

        source_mean = src.mean(dim=0)
        target_mean = tgt.mean(dim=0)
        p_c = src - source_mean
        q_c = tgt - target_mean
        H = p_c.transpose(0, 1) @ q_c
        U, _, Vh = torch.linalg.svd(H)
        R = Vh.transpose(0, 1) @ U.transpose(0, 1)

        if torch.det(R) < 0:
            Vh[-1, :] *= -1
            R = Vh.transpose(0, 1) @ U.transpose(0, 1)

        if return_torch:
            return R
        return self._to_numpy(R)

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
        p_prime, p_is_torch = self._as_tensor(rotated_source)
        q, q_is_torch = self._as_tensor(target)
        return_torch = p_is_torch or q_is_torch

        n = p_prime.shape[0]
        sum_p_sq = (p_prime * p_prime).sum()
        c = p_prime.sum(dim=0)
        A = torch.zeros((4, 4), device=p_prime.device, dtype=p_prime.dtype)
        A[0, 0] = sum_p_sq
        A[0, 1:] = c
        A[1:, 0] = c
        A[1, 1] = n
        A[2, 2] = n
        A[3, 3] = n

        sum_pq = (p_prime * q).sum()
        d = q.sum(dim=0)
        b = torch.zeros(4, device=p_prime.device, dtype=p_prime.dtype)
        b[0] = sum_pq
        b[1:] = d

        try:
            x = torch.linalg.solve(A, b)
        except RuntimeError:
            x = torch.linalg.lstsq(A, b).solution

        s = x[0]
        t = x[1:]

        if return_torch:
            return s, t
        return float(s.item()), self._to_numpy(t)

    @staticmethod
    def pca_align(source, target, device=None, pca_device=None, output_device=None):
        """
        Computes a coarse alignment (Rotation, Scale, Translation) using PCA.
        This handles global rotation and initial scale estimation.
        """
        if pca_device is None:
            if device is not None:
                pca_device = device
            elif torch.is_tensor(source):
                pca_device = source.device
            else:
                pca_device = "cpu"
        if output_device is None:
            if device is not None:
                output_device = device
            elif torch.is_tensor(source):
                output_device = source.device
            else:
                output_device = pca_device

        pca_device = torch.device(pca_device)
        output_device = torch.device(output_device)

        return_torch = (
            output_device is not None
            or device is not None
            or torch.is_tensor(source)
            or torch.is_tensor(target)
        )

        if torch.is_tensor(source):
            src = source.to(dtype=torch.float32, device=pca_device)
        else:
            src = torch.as_tensor(source, dtype=torch.float32, device=pca_device)
        if torch.is_tensor(target):
            tgt = target.to(dtype=torch.float32, device=pca_device)
        else:
            tgt = torch.as_tensor(target, dtype=torch.float32, device=pca_device)
        device = pca_device

        mu_s = src.mean(dim=0)
        mu_t = tgt.mean(dim=0)
        src_c = src - mu_s
        tgt_c = tgt - mu_t
        Cov_s = src_c.transpose(0, 1) @ src_c / src.shape[0]
        Cov_t = tgt_c.transpose(0, 1) @ tgt_c / tgt.shape[0]
        U_s, S_s, _ = torch.linalg.svd(Cov_s)
        U_t, S_t, _ = torch.linalg.svd(Cov_t)

        scale_init = torch.tensor(1.0, device=device, dtype=src.dtype)
        if S_s[0] > 1e-8:
            scale_init = torch.sqrt(S_t[0] / S_s[0])

        possible_signs = [
            torch.diag(torch.tensor([1.0, 1.0, 1.0], device=device)),
            torch.diag(torch.tensor([1.0, -1.0, -1.0], device=device)),
            torch.diag(torch.tensor([-1.0, 1.0, -1.0], device=device)),
            torch.diag(torch.tensor([-1.0, -1.0, 1.0], device=device)),
        ]
        det_fix = torch.diag(torch.tensor([1.0, 1.0, -1.0], device=device))

        N_s = src.shape[0]
        N_t = tgt.shape[0]
        sample_s = min(N_s, 1000)
        sample_t = min(N_t, 1000)
        idx_s_np = np.random.choice(N_s, sample_s, replace=False)
        idx_t_np = np.random.choice(N_t, sample_t, replace=False)
        idx_s = torch.as_tensor(idx_s_np, device=device)
        idx_t = torch.as_tensor(idx_t_np, device=device)
        sub_src = src.index_select(0, idx_s)
        sub_tgt = tgt.index_select(0, idx_t)
        sub_src_scaled_centered = (sub_src - mu_s) * scale_init

        best_error = None
        best_R = None
        for M in possible_signs:
            R_candidate = U_t @ (M @ U_s.transpose(0, 1))
            if torch.det(R_candidate) < 0:
                R_candidate = R_candidate @ det_fix
            aligned = sub_src_scaled_centered @ R_candidate.transpose(0, 1) + mu_t
            dists = torch.cdist(aligned.unsqueeze(0), sub_tgt.unsqueeze(0), p=2)
            min_dists, _ = torch.min(dists, dim=2)
            error = min_dists.mean()
            if best_error is None or error < best_error:
                best_error = error
                best_R = R_candidate

        s = scale_init
        R = best_R
        t = mu_t - s * (mu_s @ R.transpose(0, 1))
        transformed_source = s * (src @ R.transpose(0, 1)) + t

        if output_device != device:
            transformed_source = transformed_source.to(device=output_device)
            s = s.to(device=output_device)
            R = R.to(device=output_device)
            t = t.to(device=output_device)

        if return_torch:
            return transformed_source, {"s": s, "R": R, "t": t}
        return (
            transformed_source.detach().cpu().numpy(),
            {
                "s": float(s.item()) if torch.is_tensor(s) else s,
                "R": R.detach().cpu().numpy(),
                "t": t.detach().cpu().numpy(),
            },
        )

    @staticmethod
    def compose_transforms(trans2, trans1):
        """
        Composes two transforms: T2(T1(x)).
        Args:
            trans2: dict {'s', 'R', 't'}
            trans1: dict {'s', 'R', 't'}
        Returns:
            dict {'s', 'R', 't'} representing T_composed(x) = T2(T1(x))
        """
        s1, R1, t1 = trans1["s"], trans1["R"], trans1["t"]
        s2, R2, t2 = trans2["s"], trans2["R"], trans2["t"]

        use_torch = any(torch.is_tensor(x) for x in [s1, R1, t1, s2, R2, t2])
        if use_torch:
            s1 = s1 if torch.is_tensor(s1) else torch.tensor(s1, dtype=torch.float32)
            s2 = s2 if torch.is_tensor(s2) else torch.tensor(s2, dtype=torch.float32)
            R1 = R1 if torch.is_tensor(R1) else torch.as_tensor(R1, dtype=torch.float32)
            R2 = R2 if torch.is_tensor(R2) else torch.as_tensor(R2, dtype=torch.float32)
            t1 = t1 if torch.is_tensor(t1) else torch.as_tensor(t1, dtype=torch.float32)
            t2 = t2 if torch.is_tensor(t2) else torch.as_tensor(t2, dtype=torch.float32)
            s_new = s2 * s1
            R_new = R2 @ R1
            t_new = s2 * (R2 @ t1) + t2
            return {"s": s_new, "R": R_new, "t": t_new}

        s_new = s2 * s1
        R_new = np.dot(R2, R1)
        t_new = s2 * np.dot(R2, t1) + t2
        return {"s": s_new, "R": R_new, "t": t_new}

    @staticmethod
    def invert_transform(s, R, t):
        """
        Computes the inverse transformation parameters.
        Forward:  y = s * (R @ x) + t
        Inverse:  x = (1/s) * R.T @ (y - t)
                    = (1/s) * R.T @ y - (1/s) * R.T @ t
        Returns:
            s_inv, R_inv, t_inv
        """
        if torch.is_tensor(s) or torch.is_tensor(R) or torch.is_tensor(t):
            s = s if torch.is_tensor(s) else torch.tensor(s, dtype=torch.float32)
            R = R if torch.is_tensor(R) else torch.as_tensor(R, dtype=torch.float32)
            t = t if torch.is_tensor(t) else torch.as_tensor(t, dtype=torch.float32)
            s_inv = 1.0 / s
            R_inv = R.transpose(0, 1)
            t_inv = -s_inv * (R.transpose(0, 1) @ t)
            return s_inv, R_inv, t_inv

        s_inv = 1.0 / s
        R_inv = R.T
        t_inv = -s_inv * np.dot(R.T, t)
        return s_inv, R_inv, t_inv

    def __call__(self, source_points, target_points):
        """
        Runs the Scale-Adaptive ICP.
        Args:
            source_points: (N, 3) numpy array
            target_points: (M, 3) numpy array
        Returns:
            transformed_source: (N, 3) numpy array
            transforms: dict containing 's', 'R', 't' of the total transformation
        """
        src, src_is_torch = self._as_tensor(source_points)
        tgt, tgt_is_torch = self._as_tensor(target_points)
        return_torch = src_is_torch or tgt_is_torch

        current_source = src.clone()
        total_s = torch.tensor(1.0, device=src.device, dtype=src.dtype)
        total_R = torch.eye(3, device=src.device, dtype=src.dtype)
        total_t = torch.zeros(3, device=src.device, dtype=src.dtype)

        prev_error = float("inf")
        for _ in range(self.max_iterations):
            matched_target, squared_dists = self.find_correspondences(
                current_source, tgt
            )
            if not torch.is_tensor(squared_dists):
                squared_dists = torch.as_tensor(
                    squared_dists, device=src.device, dtype=src.dtype
                )

            error = squared_dists.mean().item()
            if abs(prev_error - error) < self.tolerance:
                break
            prev_error = error

            R = self.compute_rotation(current_source, matched_target)
            if not torch.is_tensor(R):
                R = torch.as_tensor(R, device=src.device, dtype=src.dtype)
            rotated_source = current_source @ R.transpose(0, 1)
            s, t = self.compute_scale_translation(rotated_source, matched_target)
            if not torch.is_tensor(s):
                s = torch.tensor(s, device=src.device, dtype=src.dtype)
            if not torch.is_tensor(t):
                t = torch.as_tensor(t, device=src.device, dtype=src.dtype)
            current_source = s * rotated_source + t

            total_s = s * total_s
            total_R = R @ total_R
            total_t = s * (R @ total_t) + t

        if return_torch:
            return current_source, {"s": total_s, "R": total_R, "t": total_t}
        return (
            self._to_numpy(current_source),
            {
                "s": float(total_s.item()),
                "R": self._to_numpy(total_R),
                "t": self._to_numpy(total_t),
            },
        )
