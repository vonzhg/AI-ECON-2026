"""
Admissible Boundary Learning Module.
Includes proper Alpha-Shape filtering and Geometric Buffer.

The α-shape (α-convex hull) is constructed by:
1. Computing the Delaunay triangulation
2. Filtering simplices to retain only those with circumradius ≤ 1/α
3. This allows representation of non-convex regions unlike the standard convex hull
"""

import numpy as np
from scipy.spatial import Delaunay, cKDTree
import torch

class AlphaBoundary:
    def __init__(self, config):
        self.alpha = config['boundary']['alpha_param']
        self.buffer = config['boundary']['buffer_percent']
        self.admissible_points = None
        self.delaunay = None
        self.alpha_simplices = None  # Indices of simplices in α-complex
        self.tree = None  # KDTree for buffer queries

        # Store bounds for normalization
        sb = config['state_bounds']
        self.mins = np.array([sb['K_min'], sb['a_min'], sb['a_min']])
        self.ranges = np.array([
            sb['K_max'] - sb['K_min'],
            sb['a_max'] - sb['a_min'],
            sb['a_max'] - sb['a_min']
        ])

    def _normalize(self, points):
        """Normalize points to [0, 1] for consistent distance calculation."""
        return (points - self.mins) / (self.ranges + 1e-8)

    def _compute_circumradius(self, simplex_points):
        """
        Compute circumradius of a tetrahedron (3D simplex).
        
        For a tetrahedron with vertices p0, p1, p2, p3:
        The circumradius R = |a||b||c| / (4 * Volume * 6)
        where a, b, c are edge lengths and Volume is the tetrahedron volume.
        
        Using the formula: R = |det(M)| / (4 * Volume)
        where M is constructed from vertex coordinates.
        """
        if len(simplex_points) != 4:
            return np.inf
        
        # Get vertices
        p0, p1, p2, p3 = simplex_points
        
        # Compute edge vectors from p0
        v1 = p1 - p0
        v2 = p2 - p0
        v3 = p3 - p0
        
        # Volume = |det([v1, v2, v3])| / 6
        det = np.dot(v1, np.cross(v2, v3))
        volume = abs(det) / 6.0
        
        if volume < 1e-12:
            return np.inf
        
        # Compute circumradius using formula
        # R = |p1-p0||p2-p0||p3-p0| / (6 * Volume) * correction factor
        # More accurate: use the circumsphere formula
        
        # Build the system for circumcenter
        # The circumcenter satisfies: |c - pi|^2 = R^2 for all i
        # This gives us: 2(p1-p0)·c = |p1|^2 - |p0|^2, etc.
        
        A = 2 * np.array([v1, v2, v3])
        b = np.array([
            np.dot(p1, p1) - np.dot(p0, p0),
            np.dot(p2, p2) - np.dot(p0, p0),
            np.dot(p3, p3) - np.dot(p0, p0)
        ])
        
        try:
            circumcenter = np.linalg.solve(A, b)
            circumradius = np.linalg.norm(circumcenter - p0)
            return circumradius
        except np.linalg.LinAlgError:
            return np.inf

    def _filter_alpha_simplices(self):
        """
        Filter Delaunay simplices to create α-complex.
        
        Retain only simplices with circumradius ≤ 1/α.
        This creates the α-shape which can represent non-convex regions.
        """
        if self.delaunay is None or self.alpha <= 0:
            return
        
        max_radius = 1.0 / self.alpha
        valid_simplices = []
        
        for i, simplex in enumerate(self.delaunay.simplices):
            # Get the 4 vertices of this tetrahedron
            points = self.admissible_points[simplex]
            
            # Normalize points for consistent radius calculation
            norm_points = self._normalize(points)
            
            # Compute circumradius
            radius = self._compute_circumradius(norm_points)
            
            # Keep if radius ≤ 1/α
            if radius <= max_radius:
                valid_simplices.append(i)
        
        self.alpha_simplices = set(valid_simplices)
        
        # Debug info
        total = len(self.delaunay.simplices)
        kept = len(self.alpha_simplices)
        if total > 0:
            print(f"   α-shape: kept {kept}/{total} simplices ({100*kept/total:.1f}%)")

    def update(self, states, scores, threshold=0.9):
        """
        Update the α-shape boundary with new admissible points.
        
        Steps:
        1. Filter points where A(s) > threshold
        2. Accumulate admissible points (with reservoir sampling for memory)
        3. Build Delaunay triangulation
        4. Filter to α-complex (circumradius ≤ 1/α)
        5. Build KDTree for buffer distance queries
        """
        s_np = states.detach().cpu().numpy()
        scores_np = scores.detach().cpu().numpy()

        # 1. Filter Admissible Points: P_adm = {(K', a'^e, a'^u) : A(s) > τ_high}
        mask = (scores_np > threshold).flatten()
        good_points = s_np[mask, :3]  # Only (K, a_e, a_u) dimensions

        if len(good_points) < 10:
            print(f"   Warning: Only {len(good_points)} admissible points found")
            return

        # 2. Update Storage with reservoir sampling
        if self.admissible_points is None:
            self.admissible_points = good_points
        else:
            # Limit memory: keep at most 10000 points
            if self.admissible_points.shape[0] > 10000:
                indices = np.random.choice(self.admissible_points.shape[0], 5000, replace=False)
                self.admissible_points = self.admissible_points[indices]
            self.admissible_points = np.vstack([self.admissible_points, good_points])

        # 3. Build Spatial Structures
        try:
            # Delaunay triangulation
            self.delaunay = Delaunay(self.admissible_points)
            
            # 4. Filter to α-complex
            self._filter_alpha_simplices()

            # 5. KDTree for buffer distance checks (on normalized points)
            norm_points = self._normalize(self.admissible_points)
            self.tree = cKDTree(norm_points)

        except Exception as e:
            print(f"   Boundary update failed: {e}")

    def _is_in_alpha_complex(self, simplex_indices):
        """
        Check if points are in simplices that belong to the α-complex.
        
        For points inside the Delaunay triangulation, verify their containing
        simplex is part of the filtered α-shape.
        """
        if self.alpha_simplices is None:
            # No α-filtering applied, accept all
            return simplex_indices >= 0
        
        result = np.zeros(len(simplex_indices), dtype=bool)
        for i, idx in enumerate(simplex_indices):
            if idx >= 0 and idx in self.alpha_simplices:
                result[i] = True
        return result

    def is_admissible(self, query_states):
        """
        Check if states are inside the α-shape OR within buffer distance.
        
        Two-stage check:
        A. Exact membership: point is inside a simplex of the α-complex
        B. Buffer membership: point is within buffer distance of admissible points
        
        Returns:
            Boolean tensor of shape (N, 1)
        """
        if self.delaunay is None:
            # No boundary learned yet - accept all points
            return torch.ones(query_states.shape[0], 1, dtype=torch.bool)

        points = query_states[:, :3].detach().cpu().numpy()

        # A. Exact Check: Inside α-complex simplex
        simplex_indices = self.delaunay.find_simplex(points)
        
        # Check if the containing simplex is in the α-complex
        is_inside = self._is_in_alpha_complex(simplex_indices)

        # B. Buffer Check: For points outside α-complex
        outside_indices = np.where(~is_inside)[0]

        if len(outside_indices) > 0 and self.tree is not None:
            outside_points = points[outside_indices]
            norm_outside = self._normalize(outside_points)

            # Query distance to nearest admissible point
            dists, _ = self.tree.query(norm_outside, k=1)

            # Check if distance is within buffer (in normalized space)
            in_buffer = dists < self.buffer

            # Update results
            is_inside[outside_indices] = in_buffer

        return torch.tensor(is_inside, dtype=torch.bool).unsqueeze(1)

    def get_boundary_stats(self):
        """Return statistics about the learned boundary."""
        stats = {
            'n_points': 0 if self.admissible_points is None else len(self.admissible_points),
            'n_simplices': 0,
            'n_alpha_simplices': 0,
            'alpha': self.alpha,
            'buffer': self.buffer
        }
        
        if self.delaunay is not None:
            stats['n_simplices'] = len(self.delaunay.simplices)
        
        if self.alpha_simplices is not None:
            stats['n_alpha_simplices'] = len(self.alpha_simplices)
        
        return stats
