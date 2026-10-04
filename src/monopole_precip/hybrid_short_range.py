"""Short-range operator for the scale-separated transport formulation.

The physical concentration is split as

    C = C_LR + C_short

with

    G_short(r) = erfc(alpha*r)/(4*pi*r),  alpha = 1/ell.

For a periodic box the short operator used at particle surfaces is

    A = B - c0 1 1^T,     c0 = 1/(4 alpha^2 V),

where B contains the finite-radius self term and sparse erfc pair
interactions.  The rank-one term enforces the same zero-mean convention
as the Ewald split without making the sparse matrix dense.

This module deliberately contains no precipitation thermodynamics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import erfc
from scipy.sparse import csc_matrix, diags
from scipy.sparse.linalg import splu

from .utilities import as_box_lengths, minimum_image_displacement


@dataclass
class HybridShortRangeOperator:
    """Sparse short-range particle interaction operator.

    Parameters
    ----------
    positions : (N, 3) array_like
        Particle centres in nm. Positions are wrapped into the periodic box.
    box : float or (3,) array_like
        Periodic box lengths in nm.
    alpha : float
        Matching parameter in 1/nm. The matching length is ell = 1/alpha.
    cutoff : float, optional
        Numerical truncation radius in nm. Default is 3/alpha.
    image_shell : int, optional
        Number of periodic image shells included in the regular self term.
        For boxes much larger than ell, zero is normally sufficient.
    """

    positions: np.ndarray
    box: np.ndarray
    alpha: float
    cutoff: float
    image_shell: int
    pair_i: np.ndarray
    pair_j: np.ndarray
    pair_values: np.ndarray
    pair_matrix: csc_matrix
    self_regular: float
    zero_mean_constant: float

    def __init__(
        self,
        positions,
        box,
        alpha,
        cutoff=None,
        image_shell=0,
    ):
        self.box = as_box_lengths(box).astype(float)
        self.alpha = float(alpha)
        if not np.isfinite(self.alpha) or self.alpha <= 0.0:
            raise ValueError("alpha must be finite and positive.")

        self.cutoff = (
            3.0 / self.alpha if cutoff is None else float(cutoff)
        )
        if not np.isfinite(self.cutoff) or self.cutoff <= 0.0:
            raise ValueError("cutoff must be finite and positive.")

        self.image_shell = int(image_shell)
        if self.image_shell < 0:
            raise ValueError("image_shell must be non-negative.")

        self.update_positions(positions)

    @property
    def n_particles(self):
        return int(self.positions.shape[0])

    @property
    def matching_length(self):
        return 1.0 / self.alpha

    def _regular_self_term(self):
        # Regular r -> 0 contribution of erfc(alpha r)/(4 pi r).
        value = -self.alpha / (2.0 * np.pi**1.5)

        if self.image_shell == 0:
            return float(value)

        vals = np.arange(
            -self.image_shell, self.image_shell + 1, dtype=int
        )
        shifts = np.array(
            np.meshgrid(vals, vals, vals, indexing="ij")
        ).reshape(3, -1).T
        shifts = shifts[np.any(shifts != 0, axis=1)]
        image_vectors = shifts * self.box[None, :]
        distance = np.linalg.norm(image_vectors, axis=1)

        value += np.sum(
            erfc(self.alpha * distance)
            / (4.0 * np.pi * distance)
        )
        return float(value)

    def update_positions(self, positions):
        """Rebuild the sparse neighbour graph for a new particle set."""
        pos = np.asarray(positions, dtype=float)
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError(
                f"positions must have shape (N,3), got {pos.shape}."
            )
        if np.any(~np.isfinite(pos)):
            raise ValueError("positions contain non-finite values.")

        self.positions = np.mod(pos, self.box[None, :])
        n = self.positions.shape[0]

        volume = float(np.prod(self.box))
        self.zero_mean_constant = (
            1.0 / (4.0 * self.alpha**2 * volume)
        )
        self.self_regular = self._regular_self_term()

        if n == 0:
            self.pair_i = np.zeros(0, dtype=int)
            self.pair_j = np.zeros(0, dtype=int)
            self.pair_values = np.zeros(0, dtype=float)
            self.pair_matrix = csc_matrix((0, 0), dtype=float)
            return

        tree = cKDTree(self.positions, boxsize=self.box)
        pairs = np.asarray(list(tree.query_pairs(self.cutoff)), dtype=int)
        if pairs.size == 0:
            pairs = np.empty((0, 2), dtype=int)

        if len(pairs):
            dr = minimum_image_displacement(
                self.positions[pairs[:, 0]] - self.positions[pairs[:, 1]],
                self.box,
            )
            distance = np.linalg.norm(dr, axis=1)
            gij = (
                erfc(self.alpha * distance)
                / (4.0 * np.pi * distance)
            )
            self.pair_i = pairs[:, 0]
            self.pair_j = pairs[:, 1]
            self.pair_values = gij
            rows = np.concatenate([self.pair_i, self.pair_j])
            cols = np.concatenate([self.pair_j, self.pair_i])
            data = np.concatenate([gij, gij])
            self.pair_matrix = csc_matrix(
                (data, (rows, cols)), shape=(n, n)
            )
        else:
            self.pair_i = np.zeros(0, dtype=int)
            self.pair_j = np.zeros(0, dtype=int)
            self.pair_values = np.zeros(0, dtype=float)
            self.pair_matrix = csc_matrix((n, n), dtype=float)

    def factor(self, radii):
        """Factor the current finite-radius operator.

        Returns a factor object providing solve(rhs) and
        solve_transpose(rhs).  Re-factor when radii change.
        """
        return HybridShortRangeFactor(self, radii)

    def dense_matrix(self, radii):
        """Dense A matrix, intended only for validation/tests."""
        radii = np.asarray(radii, dtype=float)
        if radii.shape != (self.n_particles,):
            raise ValueError("radii have incompatible shape.")
        diag = (
            1.0 / (4.0 * np.pi * radii)
            + self.self_regular
        )
        B = self.pair_matrix.toarray()
        B[np.diag_indices_from(B)] += diag
        return B - self.zero_mean_constant * np.ones_like(B)

    def evaluate_at_points(self, points, Q):
        """Evaluate the short-range monopole field at arbitrary points.

        Parameters
        ----------
        points : (M,3) array_like
            Candidate/evaluation positions.
        Q : (S,N) or (N,) array_like
            Existing-particle monopole strengths.

        Returns
        -------
        values : (S,M) or (M,) ndarray
            Sum_j Q_j G_short(x-x_j), using the same cutoff and
            zero-mean rank-one constant as the particle operator.

        Notes
        -----
        Candidate points must be distinct from existing particle centres.
        """
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(
                f"points must have shape (M,3), got {pts.shape}."
            )
        if np.any(~np.isfinite(pts)):
            raise ValueError("points contain non-finite values.")
        pts = np.mod(pts, self.box[None, :])

        q = np.asarray(Q, dtype=float)
        one_species = q.ndim == 1
        if one_species:
            q = q[None, :]
        if q.ndim != 2 or q.shape[1] != self.n_particles:
            raise ValueError(
                "Q must have shape (S,N) or (N,), matching the operator."
            )
        if np.any(~np.isfinite(q)):
            raise ValueError("Q contains non-finite values.")

        out = np.zeros((q.shape[0], len(pts)), dtype=float)

        if self.n_particles:
            tree = cKDTree(self.positions, boxsize=self.box)
            neighbours = tree.query_ball_point(pts, r=self.cutoff)

            for p, js in enumerate(neighbours):
                if not js:
                    continue
                js = np.asarray(js, dtype=int)
                dr = minimum_image_displacement(
                    pts[p][None, :] - self.positions[js],
                    self.box,
                )
                distance = np.linalg.norm(dr, axis=1)
                if np.any(distance <= 1.0e-14):
                    raise ValueError(
                        "evaluate_at_points is singular at a particle centre."
                    )
                gij = (
                    erfc(self.alpha * distance)
                    / (4.0 * np.pi * distance)
                )
                out[:, p] = q[:, js] @ gij

            out -= (
                self.zero_mean_constant
                * np.sum(q, axis=1)[:, None]
            )

        return out[0] if one_species else out

    def neighbour_counts(self):
        return np.diff(self.pair_matrix.tocsr().indptr)


class HybridShortRangeFactor:
    """Sparse LU + Sherman-Morrison factorization of A."""

    def __init__(self, operator, radii):
        self.operator = operator
        self.radii = np.asarray(radii, dtype=float).copy()
        n = operator.n_particles
        self.shape = (n, n)
        if self.radii.shape != (n,):
            raise ValueError(
                f"radii must have shape ({n},), got {self.radii.shape}."
            )
        if np.any(~np.isfinite(self.radii)) or np.any(self.radii <= 0.0):
            raise ValueError("radii must be finite and positive.")

        diag = (
            1.0 / (4.0 * np.pi * self.radii)
            + operator.self_regular
        )
        B = operator.pair_matrix + diags(
            diag, offsets=0, format="csc"
        )
        self._lu = splu(B)

        self._ones = np.ones(n, dtype=float)
        c0 = operator.zero_mean_constant

        self._z = self._lu.solve(self._ones)
        self._denom = 1.0 - c0 * np.dot(self._ones, self._z)

        self._zT = self._lu.solve(self._ones, trans="T")
        self._denomT = 1.0 - c0 * np.dot(self._ones, self._zT)

        tiny = 1.0e-12
        if abs(self._denom) < tiny or abs(self._denomT) < tiny:
            raise RuntimeError(
                "Hybrid short-range Sherman-Morrison denominator "
                "is nearly singular."
            )

    def diagonal(self):
        """Return the diagonal of the full short-range operator A.

        A = B - c0 1 1^T, so every diagonal entry contains the
        rank-one contribution -c0 in addition to the finite-radius
        self term stored in B.
        """
        return (
            1.0 / (4.0 * np.pi * self.radii)
            + self.operator.self_regular
            - self.operator.zero_mean_constant
        )

    def solve(self, rhs):
        """Apply A^{-1} to one vector or a matrix of RHS columns."""
        rhs = np.asarray(rhs, dtype=float)
        x0 = self._lu.solve(rhs)
        c0 = self.operator.zero_mean_constant
        if rhs.ndim == 1:
            correction = np.dot(self._ones, x0)
            return x0 + c0 * self._z * correction / self._denom

        correction = self._ones @ x0
        return (
            x0
            + c0
            * self._z[:, None]
            * correction[None, :]
            / self._denom
        )

    def solve_transpose(self, rhs):
        """Apply A^{-T}; needed by the analytic Jacobian transpose."""
        rhs = np.asarray(rhs, dtype=float)
        x0 = self._lu.solve(rhs, trans="T")
        c0 = self.operator.zero_mean_constant
        if rhs.ndim == 1:
            correction = np.dot(self._ones, x0)
            return x0 + c0 * self._zT * correction / self._denomT

        correction = self._ones @ x0
        return (
            x0
            + c0
            * self._zT[:, None]
            * correction[None, :]
            / self._denomT
        )
