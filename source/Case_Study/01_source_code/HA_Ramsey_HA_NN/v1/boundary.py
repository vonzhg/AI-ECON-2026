"""
Admissible Boundary Learning Module (5D Version).
Implements α-Shape filtering and Geometric Buffer in full state space.

State Space: s = (K, a^e, a^u, c^e, c^u) ∈ R^5

The boundary is learned by:
1. Normalizing all 5 dimensions to [0, 1].
2. Computing Delaunay triangulation on admissible points.
3. Filtering simplices with circumradius R <= 1/α.
4. Providing a 'is_admissible' check that includes a buffer zone.
"""

import numpy as np
from scipy.spatial import Delaunay, cKDTree
import torch

class AlphaBoundary:
    def __init__(self, config):
        self.alpha = config['boundary']['alpha_param']
        self.buffer = config['boundary']['buffer_percent']

        # Storage for the learned geometric structures
        self.admissible_points = None   # Raw points (N, 5)
        self.delaunay = None            # Scipy Delaunay object
        self.alpha_simplices = None     # Set of indices of valid simplices
        self.tree = None                # KDTree for efficient buffer queries

        # Normalization Bounds (5 Dimensions)
        sb = config['state_bounds']
        # We assume c_e and c_u share the same bounds c_min/c_max
        self.mins = np.array([
            sb['K_min'], sb['a_min'], sb['a_min'], sb['c_min'], sb['c_min']
        ])
        self.maxs = np.array([
            sb['K_max'], sb['a_max'], sb['a_max'], sb['c_max'], sb['c_max']
        ])
        self.ranges = self.maxs - self.mins

    def _normalize(self, points):
        """
        Normalize points to [0, 1] for consistent distance calculation.
        Args:
            points: Numpy array (N, 5)
        """
        # Add epsilon to ranges to avoid division by zero
        return (points - self.mins) / (self.ranges + 1e-8)

    def _compute_circumradius_nd(self, simplex_points):
        """
        Compute the circumradius of a simplex in N dimensions.

        For a simplex with vertices v0, ..., vn, the circumradius R is
        calculated by solving the linear system for the circumcenter C:
            |C - vi|^2 = R^2  for all i=0...n

        This reduces to solving A*x = b where:
        A = 2 * (vi - v0)^T
        b = |vi|^2 - |v0|^2
        C = x + v0
        """
        # Shift vertices so v0 is at origin to simplify calculation
        v0 = simplex_points[0]
        vectors = simplex_points[1:] - v0  # Shape (n, n)

        # Matrix A (rows are 2 * vectors)
        A = 2 * vectors

        # Vector b (|v|^2)
        b = np.sum(vectors**2, axis=1)

        try:
            # Solve for relative center x
            x = np.linalg.solve(A, b)
            # R is just the norm of x (distance from v0 to Center)
            return np.linalg.norm(x)
        except np.linalg.LinAlgError:
            # Collinear or degenerate simplex -> Infinite radius
            return np.inf

    def _filter_alpha_simplices(self):
        """
        Filter Delaunay simplices to create α-complex.
        Retain only simplices with circumradius R ≤ 1/α (in normalized space).
        """
        if self.delaunay is None or self.alpha <= 0:
            return

        max_radius = 1.0 / self.alpha
        valid_simplices = []

        # Retrieve all simplices (N_simplices, 6) for 5D (5+1 vertices)
        # However, Delaunay dimension depends on input data rank.
        # Ideally it is N+1 vertices.

        for i, simplex in enumerate(self.delaunay.simplices):
            # Get the vertices for this simplex
            # self.admissible_points has shape (N_points, 5)
            points = self.admissible_points[simplex]

            # Normalize points before measuring size!
            norm_points = self._normalize(points)

            # Compute circumradius in normalized space
            radius = self._compute_circumradius_nd(norm_points)

            # Filter condition
            if radius <= max_radius:
                valid_simplices.append(i)

        self.alpha_simplices = set(valid_simplices)

        # Logging
        total = len(self.delaunay.simplices)
        kept = len(self.alpha_simplices)
        if total > 0:
            print(f"   α-Shape (5D): Kept {kept}/{total} simplices ({100*kept/total:.1f}%)")

    def update(self, states, scores, threshold=0.9):
        """
        Update the boundary estimate using new training data.

        Args:
            states: Tensor (Batch, 5) - (K, ae, au, ce, cu)
            scores: Tensor (Batch, 1) - Admissibility scores
            threshold: float - Cutoff for considering a point 'admissible'
        """
        s_np = states.detach().cpu().numpy()
        scores_np = scores.detach().cpu().numpy()

        # 1. Select High-Scoring Points
        mask = (scores_np > threshold).flatten()

        # CRITICAL CHANGE: Keep all 5 dimensions!
        new_points = s_np[mask, :]

        if len(new_points) < 50:
            # Need at least N+1 points to form a simplex in N dimensions
            # 50 is a safe lower bound to avoid degenerate hulls
            print(f"   Warning: Not enough admissible points ({len(new_points)}) to update boundary.")
            return

        # 2. Reservoir Sampling (Memory Management)
        # We don't want the Delaunay calculation to explode with 1M points.
        max_points = 10000

        if self.admissible_points is None:
            self.admissible_points = new_points
        else:
            combined = np.vstack([self.admissible_points, new_points])
            if len(combined) > max_points:
                # Randomly downsample
                indices = np.random.choice(len(combined), max_points, replace=False)
                self.admissible_points = combined[indices]
            else:
                self.admissible_points = combined

        # 3. Recompute Geometry
        try:
            # A. Delaunay Triangulation (Expensive step)
            self.delaunay = Delaunay(self.admissible_points)

            # B. Filter for Alpha Shape
            self._filter_alpha_simplices()

            # C. Build KDTree for Buffer queries (using normalized points)
            norm_points = self._normalize(self.admissible_points)
            self.tree = cKDTree(norm_points)

        except Exception as e:
            print(f"   Boundary update failed: {e}")
            # If Qhull fails (e.g., coplanar points), we keep the old boundary
            pass

    def is_admissible(self, query_states):
        """
        Check if query states are admissible.

        Logic:
        1. Exact Check: Is the point inside a valid α-simplex?
        2. Buffer Check: Is the point within distance 'buffer' of a valid point?

        Args:
            query_states: Tensor (Batch, 5)

        Returns:
            Boolean Tensor (Batch, 1)
        """
        if self.delaunay is None:
            # If no boundary exists yet, assume everything is valid (exploration)
            # or nothing is valid. Usually exploration is better.
            return torch.ones(query_states.shape[0], 1, dtype=torch.bool, device=query_states.device)

        points = query_states.detach().cpu().numpy()
        N = len(points)

        # --- 1. Exact Simplex Check ---
        # find_simplex returns -1 if point is outside the Convex Hull
        simplex_ids = self.delaunay.find_simplex(points)

        # Check if the found simplex is part of the Alpha Complex
        is_inside = np.zeros(N, dtype=bool)

        if self.alpha_simplices is not None:
            for i in range(N):
                s_id = simplex_ids[i]
                if s_id != -1 and s_id in self.alpha_simplices:
                    is_inside[i] = True
        else:
            # If no alpha filtering (alpha <= 0), convex hull determines inside
            is_inside = (simplex_ids != -1)

        # --- 2. Geometric Buffer Check ---
        # For points strictly outside, check distance to the cloud
        outside_indices = np.where(~is_inside)[0]

        if len(outside_indices) > 0 and self.tree is not None:
            outside_pts = points[outside_indices]

            # Query in NORMALIZED space to respect variable scales
            norm_outside = self._normalize(outside_pts)

            # k=1 gives distance to nearest neighbor
            dists, _ = self.tree.query(norm_outside, k=1)

            # Mark as valid if within buffer distance
            # Buffer is a percentage of the normalized hypercube size (approx)
            in_buffer = dists < self.buffer

            # Update the main mask
            is_inside[outside_indices] = in_buffer

        return torch.tensor(is_inside, dtype=torch.bool, device=query_states.device).unsqueeze(1)

    def get_boundary_stats(self):
        """Return simple stats for logging."""
        return {
            'n_points': len(self.admissible_points) if self.admissible_points is not None else 0,
            'n_simplices': len(self.delaunay.simplices) if self.delaunay else 0,
            'n_alpha_simplices': len(self.alpha_simplices) if self.alpha_simplices else 0,
        }