"""
Admissible Boundary Learning Module (5D Version).
Implements α-Shape filtering, Geometric Buffer, and Projection to Admissible Set.

State Space: s = (K, a^e, a^u, c^e, c^u) ∈ R^5

Key Methods:
- update(): Learn boundary from admissible samples
- is_admissible(): Check if point is inside α-shape
- project_to_admissible(): Project point to nearest admissible point (NEW)

The boundary is learned by:
1. Normalizing all 5 dimensions to [0, 1].
2. Computing Delaunay triangulation on admissible points.
3. Filtering simplices with circumradius R <= 1/α.
4. Providing projection via KDTree nearest-neighbor lookup.
"""

import numpy as np
from scipy.spatial import Delaunay, cKDTree
import torch


class AlphaBoundary:
    """
    Manages the admissible region S_α using α-shapes in 5D.
    
    The admissible set is the region of state space from which a competitive
    equilibrium can be sustained. It is learned iteratively from simulation data.
    
    IMPORTANT: All geometric operations (distance, projection) are performed
    in NORMALIZED space [0,1]^5 to handle different scales of variables.
    
    SAMPLING METHODS:
    - "uniform": Sample from full hypercube (original method)
    - "expanded_shape": Expand α-shape outward from centroid, sample within
    - "gaussian": Perturb admissible points with Gaussian noise
    """
    
    def __init__(self, config):
        """
        Initialize boundary with configuration parameters.
        
        Args:
            config: Dictionary containing 'boundary' and 'state_bounds' sections
        """
        self.alpha = config['boundary']['alpha_param']
        self.buffer = config['boundary']['buffer_percent']
        
        # Sampling configuration
        self.sampling_method = config['boundary'].get('sampling_method', 'uniform')
        self.expansion_percent = config['boundary'].get('expansion_percent', 0.20)

        # Storage for learned geometric structures
        self.admissible_points = None   # Raw points in original space (N, 5)
        self.admissible_points_norm = None  # Normalized points (N, 5) - for KDTree
        self.delaunay = None            # Scipy Delaunay object
        self.alpha_simplices = None     # Set of indices of valid α-simplices
        self.tree = None                # KDTree for projection queries (normalized space)

        # Normalization bounds for all 5 dimensions
        # Order: K, a^e, a^u, c^e, c^u
        sb = config['state_bounds']
        self.mins = np.array([
            sb['K_min'], sb['a_min'], sb['a_min'], sb['c_min'], sb['c_min']
        ])
        self.maxs = np.array([
            sb['K_max'], sb['a_max'], sb['a_max'], sb['c_max'], sb['c_max']
        ])
        self.ranges = self.maxs - self.mins
        
        # Prevent division by zero
        self.ranges = np.where(self.ranges < 1e-8, 1.0, self.ranges)

    # ==================== SAMPLING METHODS ====================
    
    def sample_candidates(self, num_samples, device=None):
        """
        Sample candidate states using the configured sampling method.
        
        This is the main entry point for sampling. It dispatches to the
        appropriate method based on self.sampling_method.
        
        Args:
            num_samples: Number of samples to generate
            device: Torch device (optional, for returning tensors)
            
        Returns:
            Tensor (num_samples, 5) of candidate states
        """
        import torch
        
        # Check if we have enough admissible points for non-uniform methods
        have_admissible = (self.admissible_points is not None and 
                          len(self.admissible_points) >= 50)
        
        # Select sampling method
        if self.sampling_method == 'uniform' or not have_admissible:
            if not have_admissible and self.sampling_method != 'uniform':
                print(f"   Note: Falling back to uniform sampling "
                      f"(insufficient admissible points)")
            samples = self._sample_uniform(num_samples)
            
        elif self.sampling_method == 'expanded_shape':
            samples = self._sample_expanded_shape(num_samples, self.expansion_percent)
            
        elif self.sampling_method == 'gaussian':
            samples = self._sample_gaussian(num_samples, self.expansion_percent)
            
        else:
            print(f"   Warning: Unknown sampling method '{self.sampling_method}', "
                  f"using uniform")
            samples = self._sample_uniform(num_samples)
        
        # Convert to tensor
        samples_tensor = torch.tensor(samples, dtype=torch.float32)
        if device is not None:
            samples_tensor = samples_tensor.to(device)
            
        return samples_tensor
    
    def _sample_uniform(self, num_samples):
        """
        Method 0: Sample uniformly from the full hypercube.
        
        This is the original method. Simple but wasteful when the α-shape
        is much smaller than the hypercube.
        
        Args:
            num_samples: Number of samples
            
        Returns:
            Numpy array (num_samples, 5)
        """
        samples = np.random.uniform(
            self.mins, 
            self.maxs, 
            size=(num_samples, 5)
        )
        return samples
    
    def _sample_expanded_shape(self, num_samples, expansion_percent):
        """
        Method A: Sample from α-shape expanded outward from centroid.
        
        Algorithm:
        1. Compute centroid of admissible points
        2. For each point, compute direction vector from centroid
        3. Create expanded points by moving along direction by expansion_percent
        4. Sample by interpolating between original and expanded points
        
        This approximates expanding the α-shape boundary outward while
        maintaining the overall shape structure.
        
        TODO (Future Enhancement): 
        Implement true boundary normal expansion (Option 3):
        - Identify boundary simplices (4D facets on the α-shape surface)
        - Compute outward normal vectors for each boundary facet
        - Move boundary vertices along their normals
        - Reconstruct expanded α-shape
        This would give more accurate boundary expansion but is computationally
        expensive in 5D.
        
        Args:
            num_samples: Number of samples
            expansion_percent: How much to expand (e.g., 0.20 = 20%)
            
        Returns:
            Numpy array (num_samples, 5)
        """
        # Compute centroid of admissible points
        centroid = self.admissible_points.mean(axis=0)  # (5,)
        
        # Direction vectors from centroid to each admissible point
        directions = self.admissible_points - centroid  # (N, 5)
        
        # Expanded points: move each point outward by expansion_percent
        # New position = original + expansion_percent * direction
        expanded_points = self.admissible_points + expansion_percent * directions
        
        # Clip expanded points to hypercube bounds
        expanded_points = np.clip(expanded_points, self.mins, self.maxs)
        
        # Sampling strategy:
        # 1. Randomly select base points from admissible_points
        # 2. Interpolate between original and expanded position
        # This samples the "shell" between α-shape and expanded boundary
        
        n_admissible = len(self.admissible_points)
        indices = np.random.randint(0, n_admissible, size=num_samples)
        
        # Interpolation factor t ∈ [0, 1]
        # t=0 gives original point, t=1 gives expanded point
        # Using uniform [0, 1] samples the full expanded region
        t = np.random.uniform(0, 1, size=(num_samples, 1))
        
        # Interpolate: sample = (1-t) * original + t * expanded
        original_selected = self.admissible_points[indices]
        expanded_selected = expanded_points[indices]
        samples = (1 - t) * original_selected + t * expanded_selected
        
        # Final clip to ensure within bounds (should already be, but safety check)
        samples = np.clip(samples, self.mins, self.maxs)
        
        return samples
    
    def _sample_gaussian(self, num_samples, expansion_percent):
        """
        Method B: Sample by adding Gaussian noise to admissible points.
        
        Algorithm:
        1. Randomly select base points from admissible_points (with replacement)
        2. Add Gaussian noise with σ proportional to expansion_percent × range
        3. Clip to hypercube bounds
        
        This naturally samples more densely near the admissible region,
        with density falling off with distance.
        
        Args:
            num_samples: Number of samples
            expansion_percent: Controls noise scale (σ = expansion_percent × range)
            
        Returns:
            Numpy array (num_samples, 5)
        """
        # Randomly select base points (with replacement)
        n_admissible = len(self.admissible_points)
        indices = np.random.randint(0, n_admissible, size=num_samples)
        base_points = self.admissible_points[indices]  # (num_samples, 5)
        
        # Compute standard deviation for each dimension
        # σ = expansion_percent × range of dimension
        sigma = expansion_percent * self.ranges  # (5,)
        
        # Generate Gaussian noise
        noise = np.random.normal(0, sigma, size=(num_samples, 5))
        
        # Add noise to base points
        samples = base_points + noise
        
        # Clip to hypercube bounds
        samples = np.clip(samples, self.mins, self.maxs)
        
        return samples

    # ==================== NORMALIZATION ====================

    def _normalize(self, points):
        """
        Normalize points to [0, 1]^5 for consistent distance calculation.
        
        This is CRITICAL because state variables have different scales:
        - K ∈ [0.01, 1.5]
        - a ∈ [0, 2]
        - c ∈ [0.005, 1.2]
        
        Without normalization, Euclidean distance would be dominated by 
        variables with larger ranges.
        
        Args:
            points: Numpy array (N, 5) in original coordinates
            
        Returns:
            Numpy array (N, 5) in [0, 1]^5
        """
        return (points - self.mins) / self.ranges

    def _denormalize(self, points_norm):
        """
        Convert normalized points back to original coordinates.
        
        Args:
            points_norm: Numpy array (N, 5) in [0, 1]^5
            
        Returns:
            Numpy array (N, 5) in original coordinates
        """
        return points_norm * self.ranges + self.mins

    def _compute_circumradius_nd(self, simplex_points):
        """
        Compute the circumradius of a simplex in N dimensions.

        For a simplex with vertices v0, ..., vn, the circumradius R is
        the radius of the unique sphere passing through all vertices.
        
        Method: Solve linear system for circumcenter, then compute distance.
        
        Args:
            simplex_points: Numpy array (n+1, n) of simplex vertices
            
        Returns:
            float: Circumradius (inf if degenerate/collinear)
        """
        v0 = simplex_points[0]
        vectors = simplex_points[1:] - v0  # (n, n) matrix

        # System: 2 * (vi - v0) · (C - v0) = |vi - v0|^2
        A = 2 * vectors
        b = np.sum(vectors**2, axis=1)

        try:
            x = np.linalg.solve(A, b)
            return np.linalg.norm(x)
        except np.linalg.LinAlgError:
            # Degenerate simplex (collinear points)
            return np.inf

    def _filter_alpha_simplices(self):
        """
        Filter Delaunay simplices to create α-complex.
        
        The α-shape is a subset of the Delaunay triangulation where we
        only keep simplices with circumradius R ≤ 1/α.
        
        Larger α → smaller circumradius threshold → tighter boundary
        Smaller α → larger circumradius threshold → looser boundary
        """
        if self.delaunay is None or self.alpha <= 0:
            return

        max_radius = 1.0 / self.alpha
        valid_simplices = []

        for i, simplex in enumerate(self.delaunay.simplices):
            # Get vertices for this simplex (in original coordinates)
            points = self.admissible_points[simplex]
            
            # IMPORTANT: Normalize before computing circumradius!
            # This ensures α parameter has consistent meaning across dimensions.
            norm_points = self._normalize(points)
            radius = self._compute_circumradius_nd(norm_points)

            if radius <= max_radius:
                valid_simplices.append(i)

        self.alpha_simplices = set(valid_simplices)
        # Also store as numpy array for vectorized is_admissible lookup
        self._alpha_simplices_array = np.array(list(self.alpha_simplices), dtype=np.int64)

        # Logging
        total = len(self.delaunay.simplices)
        kept = len(self.alpha_simplices)
        if total > 0:
            print(f"   α-Shape (5D): Kept {kept}/{total} simplices "
                  f"({100*kept/total:.1f}%), α={self.alpha}, max_R={max_radius:.3f}")

    def update(self, states, scores, threshold=0.9):
        """
        Update the boundary estimate using new training data.

        This implements the "Level 1: Domain Consistency" step from the document.
        Points with high admissibility scores are added to the boundary.

        Args:
            states: Tensor (Batch, 5) - state vectors (K, a^e, a^u, c^e, c^u)
            scores: Tensor (Batch, 1) - admissibility scores ∈ [0, 1]
            threshold: float - cutoff for considering a point 'admissible'
        """
        s_np = states.detach().cpu().numpy()
        scores_np = scores.detach().cpu().numpy()

        # 1. Select high-scoring points
        mask = (scores_np > threshold).flatten()
        new_points = s_np[mask, :]  # Keep all 5 dimensions

        if len(new_points) < 50:
            # Need sufficient points for meaningful geometry in 5D
            # Minimum: 6 points for a 5-simplex, but we need more for robustness
            print(f"   Warning: Only {len(new_points)} admissible points "
                  f"(need ≥50 to update boundary)")
            return

        # 2. Reservoir sampling to manage memory
        # Delaunay in 5D can have O(n^3) simplices, so we cap point count
        max_points = 10000

        if self.admissible_points is None:
            self.admissible_points = new_points
        else:
            combined = np.vstack([self.admissible_points, new_points])
            if len(combined) > max_points:
                # Random subsampling (could use importance sampling instead)
                indices = np.random.choice(len(combined), max_points, replace=False)
                self.admissible_points = combined[indices]
            else:
                self.admissible_points = combined

        # 3. Recompute geometric structures
        try:
            # A. Delaunay triangulation (this is the expensive step)
            # FIX: Use qhull_options="QJ" to "joggle" inputs.
            # This prevents "topology error" and "wide merge" crashes when data
            # is degenerate (coplanar) or extremely close, which happens as
            # the model converges to the equilibrium manifold.
            self.delaunay = Delaunay(self.admissible_points, qhull_options="QJ")

            # B. Filter for α-shape
            self._filter_alpha_simplices()

            # C. Build KDTree for projection queries
            # IMPORTANT: KDTree uses NORMALIZED points for distance calculations
            self.admissible_points_norm = self._normalize(self.admissible_points)
            self.tree = cKDTree(self.admissible_points_norm)

        except Exception as e:
            print(f"   Boundary update failed: {e}")
            # Keep old boundary if Qhull fails (e.g., near-coplanar points)

    def is_admissible(self, query_states):
        """
        Check if query states are inside the admissible region.

        Two-stage check:
        1. Exact: Is the point inside a valid α-simplex?
        2. Buffer: Is the point within 'buffer' distance of any admissible point?

        Args:
            query_states: Tensor (Batch, 5)

        Returns:
            Boolean Tensor (Batch, 1) - True if admissible
        """
        if self.delaunay is None:
            # No boundary learned yet → assume all points valid (exploration phase)
            return torch.ones(query_states.shape[0], 1,
                            dtype=torch.bool, device=query_states.device)

        points = query_states.detach().cpu().numpy()
        N = len(points)

        # --- Stage 1: Exact Simplex Check ---
        # find_simplex returns -1 if outside convex hull
        simplex_ids = self.delaunay.find_simplex(points)

        if self.alpha_simplices is not None:
            # VECTORIZED: Check if the containing simplex is part of the α-complex
            # Convert alpha_simplices set to numpy array for vectorized lookup
            valid_simplex_mask = (simplex_ids != -1)
            is_inside = np.zeros(N, dtype=bool)

            if valid_simplex_mask.any():
                # Only check simplices that are valid (not -1)
                valid_indices = np.where(valid_simplex_mask)[0]
                valid_simplex_ids = simplex_ids[valid_indices]
                # Vectorized membership check using numpy
                is_in_alpha = np.isin(valid_simplex_ids, self._alpha_simplices_array)
                is_inside[valid_indices] = is_in_alpha
        else:
            # No α-filtering: use full convex hull
            is_inside = (simplex_ids != -1)

        # --- Stage 2: Buffer Check for Points Outside α-Shape ---
        outside_indices = np.where(~is_inside)[0]

        if len(outside_indices) > 0 and self.tree is not None:
            outside_pts = points[outside_indices]

            # Query in NORMALIZED space
            norm_outside = self._normalize(outside_pts)
            dists, _ = self.tree.query(norm_outside, k=1)

            # Mark as valid if within buffer distance
            # Buffer is fraction of normalized hypercube diagonal
            in_buffer = dists < self.buffer
            is_inside[outside_indices] = in_buffer

        return torch.tensor(is_inside, dtype=torch.bool,
                          device=query_states.device).unsqueeze(1)

    def project_to_admissible(self, query_states):
        """
        Project states to the nearest admissible point.

        FAST VERSION: Uses only KDTree distance check, skips expensive Delaunay lookup.
        Points within 'buffer' distance of an admissible point are considered inside.

        Args:
            query_states: Tensor (Batch, 5) - states to project

        Returns:
            projected_states: Tensor (Batch, 5) - projected states
            distances: Tensor (Batch, 1) - projection distances (normalized)
            was_inside: Tensor (Batch, 1) - True if already inside (no projection needed)
        """
        device = query_states.device
        N = query_states.shape[0]

        if self.tree is None or self.admissible_points is None:
            # No boundary yet → no projection needed
            return (query_states.clone(),
                    torch.zeros(N, 1, device=device),
                    torch.ones(N, 1, dtype=torch.bool, device=device))

        # Single CPU transfer
        points = query_states.detach().cpu().numpy()

        # Normalize for KDTree query
        norm_points = self._normalize(points)

        # Query KDTree - get distance and nearest index for ALL points
        dists_norm, nearest_indices = self.tree.query(norm_points, k=1)

        # Points within buffer distance are considered "inside"
        is_inside = dists_norm < self.buffer

        # Initialize outputs
        projected = points.copy()
        distances = np.zeros(N, dtype=np.float32)

        # Project points that are outside
        outside_mask = ~is_inside
        if outside_mask.any():
            projected[outside_mask] = self.admissible_points[nearest_indices[outside_mask]]
            distances[outside_mask] = dists_norm[outside_mask]

        # Convert back to tensors
        projected_tensor = torch.tensor(projected, dtype=torch.float32, device=device)
        distances_tensor = torch.tensor(distances, dtype=torch.float32, device=device).unsqueeze(1)
        was_inside_tensor = torch.tensor(is_inside, dtype=torch.bool, device=device).unsqueeze(1)

        return projected_tensor, distances_tensor, was_inside_tensor

    def distance_to_admissible(self, query_states):
        """
        Compute distance from query states to the admissible set.

        Returns 0 for points inside S_α, positive distance for points outside.
        Distance is computed in NORMALIZED space.

        Args:
            query_states: Tensor (Batch, 5)

        Returns:
            distances: Tensor (Batch, 1) - distance to S_α (0 if inside)
        """
        _, distances, was_inside = self.project_to_admissible(query_states)

        # Points inside have distance 0
        distances = torch.where(was_inside, torch.zeros_like(distances), distances)

        return distances

    def get_boundary_stats(self):
        """Return statistics about the current boundary for logging."""
        return {
            'n_points': len(self.admissible_points) if self.admissible_points is not None else 0,
            'n_simplices': len(self.delaunay.simplices) if self.delaunay else 0,
            'n_alpha_simplices': len(self.alpha_simplices) if self.alpha_simplices else 0,
        }

    def save(self, filepath):
        """Save boundary state to file."""
        import pickle
        state = {
            'admissible_points': self.admissible_points,
            'alpha': self.alpha,
            'buffer': self.buffer,
            'mins': self.mins,
            'maxs': self.maxs,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        print(f"   Boundary saved to {filepath}")

    def load(self, filepath):
        """Load boundary state from file and rebuild geometric structures."""
        import pickle
        import os

        if not os.path.exists(filepath):
            print(f"   Boundary file not found: {filepath}")
            return False

        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        self.admissible_points = state['admissible_points']
        self.alpha = state['alpha']
        self.buffer = state['buffer']
        self.mins = state['mins']
        self.maxs = state['maxs']
        self.ranges = self.maxs - self.mins

        # Rebuild geometric structures
        if self.admissible_points is not None and len(self.admissible_points) >= 50:
            try:
                # FIX: Use qhull_options="QJ" here as well for consistency when reloading
                self.delaunay = Delaunay(self.admissible_points, qhull_options="QJ")
                self._filter_alpha_simplices()
                self.admissible_points_norm = self._normalize(self.admissible_points)
                self.tree = cKDTree(self.admissible_points_norm)
                print(f"   Boundary loaded from {filepath} "
                      f"({len(self.admissible_points)} points)")
                return True
            except Exception as e:
                print(f"   Failed to rebuild boundary: {e}")
                return False

        return False