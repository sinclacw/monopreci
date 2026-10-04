"""Explicitly coupled ternary monopole interface solver.

At fixed particle radii, solve simultaneously for both interfacial
concentration fields, C_Mg^s and C_Si^s.  For N particles the nonlinear
unknown is a vector of length 2N.  Each residual evaluation performs two
linear monopole solves and returns:

  1. N Gibbs--Thomson solubility-product residuals;
  2. N equal-interface-velocity residuals.

The physical concentrations are represented by bounded logistic variables,
so trial values remain strictly between zero and the precipitate composition.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import time
from scipy.optimize import least_squares, brentq
from scipy.sparse import csc_matrix, diags, bmat
from scipy.sparse.linalg import LinearOperator, gmres, splu
from scipy.linalg import lu_factor, lu_solve


KB = 1.380649e-23


@dataclass
class CoupledTernarySolveResult:
    """Species-indexed result from the coupled interface solve.

    The current nonlinear kernel still solves the two-solute Mg-Si problem,
    but the stored result state is generic in the number of diffusing
    solutes. This separates the result representation from the current
    two-solute implementation and establishes the array convention required
    by the later arbitrary-S solver.

    Shapes
    ------
    Q, c_surf, dRdt_species : (S, N)
    cbar                    : (S,)
    dRdt                    : (N,)
    """

    solute_names: tuple[str, ...]
    Q: np.ndarray
    cbar: np.ndarray
    c_surf: np.ndarray

    dRdt: np.ndarray
    dRdt_species: np.ndarray

    success: bool
    message: str

    nfev: int
    cost: float

    max_thermodynamic_residual: float
    max_relative_velocity_mismatch: float
    optimality: float

    final_lambda: float | None = None
    continuation_history: list[dict] | None = None

    nonlinear_solver_used: str | None = None
    dense_fallback_used: bool = False

    def __post_init__(self):
        """Normalize and validate the species-indexed result arrays."""
        self.solute_names = tuple(self.solute_names)
        self.Q = np.asarray(self.Q, dtype=float)
        self.cbar = np.asarray(self.cbar, dtype=float)
        self.c_surf = np.asarray(self.c_surf, dtype=float)
        self.dRdt = np.asarray(self.dRdt, dtype=float)
        self.dRdt_species = np.asarray(
            self.dRdt_species,
            dtype=float,
        )

        n_solutes = len(self.solute_names)

        if n_solutes == 0:
            raise ValueError("solute_names must contain at least one species.")
        if len(set(self.solute_names)) != n_solutes:
            raise ValueError("solute_names must be unique.")

        if self.cbar.shape != (n_solutes,):
            raise ValueError(
                "cbar must have shape (S,), where S=len(solute_names)."
            )
        if self.Q.ndim != 2 or self.Q.shape[0] != n_solutes:
            raise ValueError(
                "Q must have shape (S, N), where S=len(solute_names)."
            )
        if self.c_surf.shape != self.Q.shape:
            raise ValueError("c_surf must have the same shape as Q.")
        if self.dRdt_species.shape != self.Q.shape:
            raise ValueError(
                "dRdt_species must have the same shape as Q."
            )

        n_particles = self.Q.shape[1]
        if self.dRdt.shape != (n_particles,):
            raise ValueError(
                "dRdt must have shape (N,), matching the particle axis."
            )

    def _legacy_species_index(self, name: str) -> int:
        """Find a legacy species row by name rather than fixed position."""
        try:
            return self.solute_names.index(name)
        except ValueError as exc:
            raise AttributeError(
                f"Legacy {name} view is unavailable because this result "
                f"contains species {self.solute_names!r}."
            ) from exc

    # Temporary Mg/Si compatibility views. The generic arrays above are
    # the source of truth; these can disappear once all callers use them.
    @property
    def Q_mg(self) -> np.ndarray:
        return self.Q[self._legacy_species_index("Mg")]

    @property
    def Q_si(self) -> np.ndarray:
        return self.Q[self._legacy_species_index("Si")]

    @property
    def cbar_mg(self) -> float:
        return float(self.cbar[self._legacy_species_index("Mg")])

    @property
    def cbar_si(self) -> float:
        return float(self.cbar[self._legacy_species_index("Si")])

    @property
    def c_surf_mg(self) -> np.ndarray:
        return self.c_surf[self._legacy_species_index("Mg")]

    @property
    def c_surf_si(self) -> np.ndarray:
        return self.c_surf[self._legacy_species_index("Si")]

    @property
    def dRdt_mg(self) -> np.ndarray:
        return self.dRdt_species[self._legacy_species_index("Mg")]

    @property
    def dRdt_si(self) -> np.ndarray:
        return self.dRdt_species[self._legacy_species_index("Si")]


def log_curvature_solubility_product(
    R, T, aa, bb, vatp, gamma, x=5, y=6, cc=0.0,
    use_xy_factor=True,
):
    """Return log(K_R) in site-fraction form."""
    R = np.asarray(R, dtype=float)
    if np.any(R <= 0.0):
        raise ValueError("All active radii must be positive.")
    if T <= 0.0 or vatp <= 0.0 or gamma < 0.0:
        raise ValueError("Require T>0, vatp>0, and gamma>=0.")
    if x <= 0 or y <= 0:
        raise ValueError("Stoichiometric coefficients must be positive.")

    log_k_inf = np.log(10.0) * (bb - aa / T + cc / T**2)
    stoich = (x + y) if use_xy_factor else 1.0
    return log_k_inf + stoich * 2.0 * gamma * vatp / (R * KB * T)


def _build_monopole_operator(
    R,
    G0,
    Gpair,
    use_preconditioner=True,
):
    R = np.asarray(R, dtype=float)
    Gpair = np.asarray(Gpair, dtype=float)

    n = R.size


    if Gpair.shape != (n, n):
        raise ValueError(
            f"Gpair must have shape {(n, n)}, "
            f"got {Gpair.shape}."
        )

    if np.any(~np.isfinite(R)) or np.any(R <= 0.0):
        raise ValueError(
            "R must contain finite positive values."
        )

    if not np.isfinite(G0):
        raise ValueError(
            "G0 must be finite."
        )

    if np.any(~np.isfinite(Gpair)):
        raise ValueError(
            "Gpair contains non-finite values."
        )

    diag = (
        1.0 / (4.0 * np.pi * R)
        + float(G0)
    )

    def matvec(q):
        return diag * q + Gpair @ q

    A = LinearOperator((n, n), matvec=matvec, dtype=float)
    M = None
    if use_preconditioner:
        if np.any(np.abs(diag) <= np.finfo(float).tiny):
            raise ValueError("Zero diagonal encountered in monopole operator.")
        M = LinearOperator((n, n), matvec=lambda q: q / diag, dtype=float)
    return A, M


def _solve_q(A, M, rhs, q0, rtol, atol, maxiter, restart):
    q, info = gmres(
        A, np.asarray(rhs, dtype=float), x0=q0, M=M,
        rtol=rtol, atol=atol, maxiter=maxiter, restart=restart,
    )
    if info != 0:
        raise RuntimeError(f"GMRES failed to converge, info={info}.")
    return q


def _stable_sigmoid(z):
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0.0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def _bounded_from_z(z, upper, margin):
    """Map R -> (margin*upper, (1-margin)*upper)."""
    lo = margin * upper
    hi = (1.0 - margin) * upper
    return lo + (hi - lo) * _stable_sigmoid(z)

def _bounded_derivative_from_z(z, upper, margin):
    """
    Derivative of _bounded_from_z with respect to z.
    If
        c(z) = lo + (hi - lo) * sigmoid(z),
    then
        dc/dz = (hi - lo) * sigmoid(z) * (1 - sigmoid(z)).
    """
    lo = margin * upper
    hi = (1.0 - margin) * upper

    s = _stable_sigmoid(z)

    return (hi - lo) * s * (1.0 - s)


def _z_from_bounded(c, upper, margin):
    lo = margin * upper
    hi = (1.0 - margin) * upper
    f = (np.asarray(c, dtype=float) - lo) / (hi - lo)
    f = np.clip(f, 1e-12, 1.0 - 1e-12)
    return np.log(f / (1.0 - f))


def _velocity(R, Q, D, cp, cs, denominator_margin):
    dc = cp - cs
    floor = denominator_margin * cp
    if np.any(dc <= floor):
        raise ValueError("Interfacial concentration is too close to precipitate composition.")
    return -D * Q / (4.0 * np.pi * R**2 * dc)

def _velocity_directional_derivative(
    R,
    Q,
    D,
    cp,
    cs,
    delta_Q,
    delta_cs,
    denominator_margin,
):
    """
    Directional derivative of the interfacial velocity.

    For

        v = -D Q / [4 pi R^2 (cp - cs)],

    return the perturbation delta_v caused by delta_Q and delta_cs.
    """

    dc = cp - cs
    floor = denominator_margin * cp

    if np.any(dc <= floor):
        raise ValueError(
            "Interfacial concentration is too close to "
            "precipitate composition."
        )

    prefactor = -D / (4.0 * np.pi * R**2)

    delta_v = prefactor * (
        delta_Q / dc
        + Q * delta_cs / dc**2
    )

    return delta_v

def _equal_velocity_residual_species(
    velocity_species,
    velocity_floor,
):
    """Return the S-1 equal-velocity residual blocks.

    Species 0 is the reference species.  For s = 1,...,S-1,

        f_s = (v_0 - v_s) / sqrt(v_0^2 + v_s^2 + v_floor^2).

    The returned array has shape (S-1, N).  For S=1 it has shape
    (0, N), so the nonlinear system contains only the thermodynamic
    constraint.
    """
    velocity_species = np.asarray(velocity_species, dtype=float)
    if velocity_species.ndim != 2:
        raise ValueError("velocity_species must have shape (S,N).")

    n_solutes, n_particles = velocity_species.shape
    if n_solutes < 1:
        raise ValueError("At least one diffusing solute is required.")

    if n_solutes == 1:
        return np.empty((0, n_particles), dtype=float)

    v0 = velocity_species[0]
    f_kin = np.empty((n_solutes - 1, n_particles), dtype=float)

    for s in range(1, n_solutes):
        vs = velocity_species[s]
        vscale = np.sqrt(v0**2 + vs**2 + velocity_floor**2)
        f_kin[s - 1] = (v0 - vs) / vscale

    return f_kin


def _equal_velocity_directional_derivative_species(
    velocity_species,
    delta_velocity_species,
    velocity_floor,
):
    """Directional derivative of the S-1 equal-velocity residuals."""
    velocity_species = np.asarray(velocity_species, dtype=float)
    delta_velocity_species = np.asarray(
        delta_velocity_species, dtype=float
    )

    if velocity_species.ndim != 2:
        raise ValueError("velocity_species must have shape (S,N).")
    if delta_velocity_species.shape != velocity_species.shape:
        raise ValueError(
            "delta_velocity_species must match velocity_species."
        )

    n_solutes, n_particles = velocity_species.shape
    if n_solutes < 1:
        raise ValueError("At least one diffusing solute is required.")

    if n_solutes == 1:
        return np.empty((0, n_particles), dtype=float)

    v0 = velocity_species[0]
    dv0 = delta_velocity_species[0]
    delta_f_kin = np.empty(
        (n_solutes - 1, n_particles), dtype=float
    )

    for s in range(1, n_solutes):
        vs = velocity_species[s]
        dvs = delta_velocity_species[s]

        vscale = np.sqrt(v0**2 + vs**2 + velocity_floor**2)
        velocity_difference = v0 - vs
        delta_velocity_difference = dv0 - dvs
        delta_vscale = (v0 * dv0 + vs * dvs) / vscale

        delta_f_kin[s - 1] = (
            delta_velocity_difference / vscale
            - velocity_difference * delta_vscale / vscale**2
        )

    return delta_f_kin


def _equal_velocity_transpose_action_species(
    velocity_species,
    weight_kin,
    velocity_floor,
):
    """Transpose action from kinetic residual weights to velocities."""
    velocity_species = np.asarray(velocity_species, dtype=float)
    weight_kin = np.asarray(weight_kin, dtype=float)

    if velocity_species.ndim != 2:
        raise ValueError("velocity_species must have shape (S,N).")

    n_solutes, n_particles = velocity_species.shape
    expected_shape = (max(n_solutes - 1, 0), n_particles)
    if weight_kin.shape != expected_shape:
        raise ValueError(
            f"weight_kin must have shape {expected_shape}."
        )

    weight_velocity_species = np.zeros_like(velocity_species)

    if n_solutes == 1:
        return weight_velocity_species

    v0 = velocity_species[0]

    for s in range(1, n_solutes):
        vs = velocity_species[s]
        vscale = np.sqrt(v0**2 + vs**2 + velocity_floor**2)
        velocity_difference = v0 - vs

        df_dv0 = (
            1.0 / vscale
            - velocity_difference * v0 / vscale**3
        )
        df_dvs = (
            -1.0 / vscale
            - velocity_difference * vs / vscale**3
        )

        w = weight_kin[s - 1]
        weight_velocity_species[0] += w * df_dv0
        weight_velocity_species[s] += w * df_dvs

    return weight_velocity_species


class ParticlePhaseThermodynamics:
    """Dispatch one thermodynamic constraint per particle by phase ID.

    This adapter presents the same residual/JVP/VJP interface as a single
    phase thermodynamic model while allowing different particles in one
    globally coupled solve to use different phase models and phase
    properties.  Phase names have no numerical meaning.
    """

    def __init__(
        self,
        *,
        phase_id,
        models,
        precipitate_atomic_volumes,
        gammas,
        stoichiometric_weights,
    ):
        self.phase_id = np.asarray(phase_id, dtype=np.int32)
        self.models = tuple(models)
        self.precipitate_atomic_volumes = np.asarray(
            precipitate_atomic_volumes, dtype=float
        )
        self.gammas = np.asarray(gammas, dtype=float)
        self.stoichiometric_weights = np.asarray(
            stoichiometric_weights, dtype=float
        )

        if self.phase_id.ndim != 1:
            raise ValueError("phase_id must have shape (N,).")
        p = len(self.models)
        if p < 1:
            raise ValueError("At least one phase thermodynamic model is required.")
        if self.precipitate_atomic_volumes.shape != (p,):
            raise ValueError(
                "precipitate_atomic_volumes must have shape (P,)."
            )
        if self.gammas.shape != (p,):
            raise ValueError("gammas must have shape (P,).")
        if self.stoichiometric_weights.ndim != 2:
            raise ValueError(
                "stoichiometric_weights must have shape (P,S)."
            )
        if self.stoichiometric_weights.shape[0] != p:
            raise ValueError(
                "stoichiometric_weights first dimension must equal P."
            )
        if np.any(self.phase_id < 0) or np.any(self.phase_id >= p):
            raise ValueError("phase_id contains an invalid phase index.")

    def _validate_state(self, c_surf, radius):
        c_surf = np.asarray(c_surf, dtype=float)
        radius = np.asarray(radius, dtype=float)
        s = self.stoichiometric_weights.shape[1]
        n = self.phase_id.size
        if c_surf.shape != (s, n):
            raise ValueError(
                f"c_surf must have shape ({s}, {n}), got {c_surf.shape}."
            )
        if radius.shape != (n,):
            raise ValueError(
                f"radius must have shape ({n},), got {radius.shape}."
            )
        return c_surf, radius

    def residual(
        self, *, c_surf, radius, temperature, matrix_atomic_volume,
    ):
        c_surf, radius = self._validate_state(c_surf, radius)
        result = np.empty(self.phase_id.size, dtype=float)
        for p, model in enumerate(self.models):
            idx = np.flatnonzero(self.phase_id == p)
            if idx.size == 0:
                continue
            result[idx] = model.residual(
                c_surf=c_surf[:, idx],
                radius=radius[idx],
                temperature=temperature,
                matrix_atomic_volume=matrix_atomic_volume,
                precipitate_atomic_volume=float(
                    self.precipitate_atomic_volumes[p]
                ),
                gamma=float(self.gammas[p]),
                stoichiometric_weights=self.stoichiometric_weights[p],
            )
        return result

    def directional_derivative(
        self, *, c_surf, delta_c_surf, radius, temperature,
        matrix_atomic_volume,
    ):
        c_surf, radius = self._validate_state(c_surf, radius)
        delta_c_surf = np.asarray(delta_c_surf, dtype=float)
        if delta_c_surf.shape != c_surf.shape:
            raise ValueError("delta_c_surf must have shape (S,N).")
        result = np.empty(self.phase_id.size, dtype=float)
        for p, model in enumerate(self.models):
            idx = np.flatnonzero(self.phase_id == p)
            if idx.size == 0:
                continue
            result[idx] = model.directional_derivative(
                c_surf=c_surf[:, idx],
                delta_c_surf=delta_c_surf[:, idx],
                radius=radius[idx],
                temperature=temperature,
                matrix_atomic_volume=matrix_atomic_volume,
                precipitate_atomic_volume=float(
                    self.precipitate_atomic_volumes[p]
                ),
                gamma=float(self.gammas[p]),
                stoichiometric_weights=self.stoichiometric_weights[p],
            )
        return result

    def transpose_contribution(
        self, *, weight_residual, c_surf, radius, temperature,
        matrix_atomic_volume,
    ):
        c_surf, radius = self._validate_state(c_surf, radius)
        weight_residual = np.asarray(weight_residual, dtype=float)
        if weight_residual.shape != (self.phase_id.size,):
            raise ValueError("weight_residual must have shape (N,).")
        result = np.zeros_like(c_surf)
        for p, model in enumerate(self.models):
            idx = np.flatnonzero(self.phase_id == p)
            if idx.size == 0:
                continue
            local = model.transpose_contribution(
                weight_residual=weight_residual[idx],
                c_surf=c_surf[:, idx],
                radius=radius[idx],
                temperature=temperature,
                matrix_atomic_volume=matrix_atomic_volume,
                precipitate_atomic_volume=float(
                    self.precipitate_atomic_volumes[p]
                ),
                gamma=float(self.gammas[p]),
                stoichiometric_weights=self.stoichiometric_weights[p],
            )
            local = np.asarray(local, dtype=float)
            expected = (c_surf.shape[0], idx.size)
            if local.shape != expected:
                raise RuntimeError(
                    "Phase thermodynamic transpose contribution returned "
                    f"shape {local.shape}; expected {expected}."
                )
            result[:, idx] = local
        return result


def _ternary_residual_directional_derivative(
    *,
    z,
    delta_z,
    n,
    cp_species,
    concentration_margin,
    linear_solver,
    matrix_integrals,
    matrix_volume,
    Ainv_ones,
    R,
    D_species,
    Q_species,
    velocity_species,
    velocity_floor,
    thermodynamics,
    T,
    vatm,
    prescribed_background_mode=False,
):
    """Return J @ delta_z using species-indexed internal state.

    The residual contains one thermodynamic constraint and S-1
    equal-velocity constraints per particle.
    """
    z = np.asarray(z, dtype=float)
    delta_z = np.asarray(delta_z, dtype=float).reshape(-1)
    cp_species = np.asarray(cp_species, dtype=float)
    D_species = np.asarray(D_species, dtype=float)
    Q_species = np.asarray(Q_species, dtype=float)
    velocity_species = np.asarray(velocity_species, dtype=float)

    n_solutes = D_species.size
    if n_solutes < 1:
        raise ValueError("At least one diffusing solute is required.")

    if cp_species.shape == (n_solutes,):
        cp_particle = np.broadcast_to(
            cp_species[:, None],
            (n_solutes, n),
        )
    elif cp_species.shape == (n_solutes, n):
        cp_particle = cp_species
    else:
        raise ValueError(
            "cp_species must have shape (S,) or (S,N)."
        )

    if z.shape != (n_solutes * n,):
        raise ValueError("z must have shape (S*N,).")
    if delta_z.shape != (n_solutes * n,):
        raise ValueError("delta_z must have shape (S*N,).")
    if D_species.shape != (n_solutes,):
        raise ValueError("D_species must have shape (S,).")
    if Q_species.shape != (n_solutes, n):
        raise ValueError("Q_species must have shape (S,N).")
    if velocity_species.shape != (n_solutes, n):
        raise ValueError("velocity_species must have shape (S,N).")

    z_species = z.reshape(n_solutes, n)
    delta_z_species = delta_z.reshape(n_solutes, n)

    c_surf_species = np.empty_like(z_species)
    dc_surf_dz_species = np.empty_like(z_species)
    delta_c_surf_species = np.empty_like(z_species)

    for s in range(n_solutes):
        c_surf_species[s] = _bounded_from_z(
            z_species[s],
            cp_particle[s],
            concentration_margin,
        )
        dc_surf_dz_species[s] = _bounded_derivative_from_z(
            z_species[s],
            cp_particle[s],
            concentration_margin,
        )
        delta_c_surf_species[s] = (
            dc_surf_dz_species[s] * delta_z_species[s]
        )

    if prescribed_background_mode:
        # C_LR(R_i) is fixed during one interface solve:
        #     Q = A^{-1}(c_surf - C_LR)
        # so
        #     delta Q = A^{-1} delta c_surf.
        delta_Q_species = np.empty_like(delta_c_surf_species)
        for s in range(n_solutes):
            delta_Q_species[s] = linear_solver.solve(
                delta_c_surf_species[s]
            )
    else:
        _, delta_Q_species = (
            _mass_conserving_monopole_directional_derivative_species(
                linear_solver=linear_solver,
                delta_c_surf=delta_c_surf_species,
                matrix_integrals=matrix_integrals,
                matrix_volume=matrix_volume,
                Ainv_ones=Ainv_ones,
            )
        )

    delta_velocity_species = np.empty_like(Q_species)
    for s in range(n_solutes):
        delta_velocity_species[s] = _velocity_directional_derivative(
            R,
            Q_species[s],
            D_species[s],
            cp_particle[s],
            c_surf_species[s],
            delta_Q_species[s],
            delta_c_surf_species[s],
            concentration_margin,
        )

    delta_f_therm = thermodynamics.directional_derivative(
        c_surf=c_surf_species,
        delta_c_surf=delta_c_surf_species,
        radius=R,
        temperature=T,
        matrix_atomic_volume=vatm,
    )

    delta_f_kin_species = (
        _equal_velocity_directional_derivative_species(
            velocity_species,
            delta_velocity_species,
            velocity_floor,
        )
    )

    return np.concatenate((delta_f_therm, delta_f_kin_species.reshape(-1)))



def _ternary_residual_transpose_action(
    *,
    z,
    weight_residual,
    n,
    cp_species,
    concentration_margin,
    linear_solver,
    matrix_integrals,
    matrix_volume,
    Ainv_ones,
    R,
    D_species,
    Q_species,
    velocity_species,
    velocity_floor,
    thermodynamics,
    T,
    vatm,
    prescribed_background_mode=False,
):
    """Apply J.T using species-indexed internal state.

    The residual contains one thermodynamic constraint and S-1
    equal-velocity constraints per particle.
    """
    z = np.asarray(z, dtype=float)
    weight_residual = np.asarray(
        weight_residual, dtype=float
    ).reshape(-1)
    cp_species = np.asarray(cp_species, dtype=float)
    D_species = np.asarray(D_species, dtype=float)
    Q_species = np.asarray(Q_species, dtype=float)
    velocity_species = np.asarray(velocity_species, dtype=float)

    n_solutes = D_species.size
    if n_solutes < 1:
        raise ValueError("At least one diffusing solute is required.")

    if cp_species.shape == (n_solutes,):
        cp_particle = np.broadcast_to(
            cp_species[:, None],
            (n_solutes, n),
        )
    elif cp_species.shape == (n_solutes, n):
        cp_particle = cp_species
    else:
        raise ValueError(
            "cp_species must have shape (S,) or (S,N)."
        )

    if z.shape != (n_solutes * n,):
        raise ValueError("z must have shape (S*N,).")
    if weight_residual.shape != (n_solutes * n,):
        raise ValueError("weight_residual must have shape (S*N,).")
    if D_species.shape != (n_solutes,):
        raise ValueError("D_species must have shape (S,).")
    if Q_species.shape != (n_solutes, n):
        raise ValueError("Q_species must have shape (S,N).")
    if velocity_species.shape != (n_solutes, n):
        raise ValueError("velocity_species must have shape (S,N).")

    z_species = z.reshape(n_solutes, n)
    c_surf_species = np.empty_like(z_species)
    dc_surf_dz_species = np.empty_like(z_species)

    for s in range(n_solutes):
        c_surf_species[s] = _bounded_from_z(
            z_species[s],
            cp_particle[s],
            concentration_margin,
        )
        dc_surf_dz_species[s] = _bounded_derivative_from_z(
            z_species[s],
            cp_particle[s],
            concentration_margin,
        )

    weight_therm = weight_residual[:n]
    weight_kin_species = weight_residual[n:].reshape(
        n_solutes - 1, n
    )

    weight_c_species = thermodynamics.transpose_contribution(
        weight_residual=weight_therm,
        c_surf=c_surf_species,
        radius=R,
        temperature=T,
        matrix_atomic_volume=vatm,
    )
    if weight_c_species.shape != (n_solutes, n):
        raise RuntimeError(
            "Phase thermodynamic transpose contribution returned "
            f"shape {weight_c_species.shape}; expected "
            f"{(n_solutes, n)}."
        )
    weight_c_species = np.asarray(
        weight_c_species, dtype=float
    ).copy()

    weight_velocity_species = (
        _equal_velocity_transpose_action_species(
            velocity_species,
            weight_kin_species,
            velocity_floor,
        )
    )

    # Velocity transpose: contributions to Q and directly to c_surf.
    weight_Q_species = np.empty_like(Q_species)

    for s in range(n_solutes):
        dc_s = cp_particle[s] - c_surf_species[s]
        floor_s = concentration_margin * cp_particle[s]

        if np.any(dc_s <= floor_s):
            raise ValueError(
                "Interfacial concentration is too close to "
                f"precipitate composition for species index {s}."
            )

        prefactor_s = -D_species[s] / (4.0 * np.pi * R**2)

        weight_Q_species[s] = (
            weight_velocity_species[s]
            * prefactor_s
            / dc_s
        )

        weight_c_species[s] += (
            weight_velocity_species[s]
            * prefactor_s
            * Q_species[s]
            / dc_s**2
        )

    # Q -> c_surf through the active monopole map.
    if prescribed_background_mode:
        # Q = A^{-1}(c_surf - C_LR), with C_LR fixed here.
        for s in range(n_solutes):
            weight_c_species[s] += linear_solver.solve_transpose(
                weight_Q_species[s]
            )
    else:
        weight_c_species += (
            _mass_conserving_monopole_transpose_action_species(
                linear_solver=linear_solver,
                weight_Q=weight_Q_species,
                matrix_integrals=matrix_integrals,
                matrix_volume=matrix_volume,
                Ainv_ones=Ainv_ones,
            )
        )

    # c_surf -> z through the bounded transform.
    weight_z_species = (
        dc_surf_dz_species * weight_c_species
    )

    return weight_z_species.reshape(-1)



def _validate_optional_vector(value, n, name):
    """
    Validate an optional one-dimensional warm-start vector.

    Returns
    -------
    None or ndarray with shape (n,)
    """
    if value is None:
        return None

    value = np.asarray(value, dtype=float)

    if value.shape != (n,):
        raise ValueError(
            f"{name} must have shape {(n,)}, got {value.shape}."
        )

    if np.any(~np.isfinite(value)):
        raise ValueError(
            f"{name} contains non-finite values."
        )

    return value.copy()


class _DenseMonopoleLinearSolver:
    """Small adapter around SciPy dense LU for the monopole operator.

    Earlier implementations passed ``(lu, piv)`` directly through the nonlinear
    residual/Jacobian machinery and called ``lu_solve`` at each use site.
    Linear solves are now routed through the two methods below:

        solve(rhs)             -> A^{-1} rhs
        solve_transpose(rhs)   -> A^{-T} rhs

    The direct formulation uses the same dense matrix and SciPy LU
    factorization.  The adapter removes the dense-LU assumption from the
    thermodynamic/Jacobian kernels, allowing the scale-separated short-range
    factor to supply the same two-method API.
    """

    def __init__(self, A):
        A = np.asarray(A, dtype=float)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("A must be a square two-dimensional array.")
        if np.any(~np.isfinite(A)):
            raise ValueError("A contains non-finite values.")
        self.shape = A.shape
        self._diagonal = np.diag(A).copy()
        self._lu_and_piv = lu_factor(A, check_finite=False)

    def diagonal(self):
        """Return the diagonal of the represented monopole operator A."""
        return self._diagonal.copy()

    def solve(self, rhs):
        rhs = np.asarray(rhs, dtype=float)
        return lu_solve(
            self._lu_and_piv,
            rhs,
            check_finite=False,
        )

    def solve_transpose(self, rhs):
        rhs = np.asarray(rhs, dtype=float)
        return lu_solve(
            self._lu_and_piv,
            rhs,
            trans=1,
            check_finite=False,
        )


def _mass_conserving_monopoles_species(
    *,
    linear_solver,
    c_surf,
    matrix_integrals,
    matrix_volume,
    particle_volume,
    M_total,
    cp,
    Ainv_ones,
    fixed_green_inventory=None,
):
    """Solve mass-conserving monopoles for an arbitrary species axis.

    Parameters
    ----------
    c_surf : ndarray, shape (S, N)
        Trial interface concentrations.
    M_total : ndarray, shape (S,)
        Conserved total inventories.
    cp : ndarray, shape (S,) or (S, N)
        Precipitate concentrations.  A one-dimensional value is shared by
        all particles; a two-dimensional value is particle-specific.
    fixed_green_inventory : ndarray, shape (S,), optional
        Inventory carried by monopoles outside this coupled solve.

    Returns
    -------
    cbar : ndarray, shape (S,)
        Mass-conserving mean matrix concentrations.
    Q : ndarray, shape (S, N)
        Species-indexed monopole strengths.

    Notes
    -----
    All species share the same geometric monopole matrix A.  The LU solve
    therefore uses all species as simultaneous right-hand sides.  Species
    remain independent except through the common geometry.
    """
    c_surf = np.asarray(c_surf, dtype=float)
    matrix_integrals = np.asarray(matrix_integrals, dtype=float)
    Ainv_ones = np.asarray(Ainv_ones, dtype=float)
    M_total = np.asarray(M_total, dtype=float)
    cp = np.asarray(cp, dtype=float)

    if c_surf.ndim != 2:
        raise ValueError("c_surf must have shape (S, N).")

    n_solutes, n_particles = c_surf.shape
    if n_solutes < 1:
        raise ValueError("At least one diffusing solute is required.")

    if M_total.shape != (n_solutes,):
        raise ValueError("M_total must have shape (S,).")
    if cp.shape == (n_solutes,):
        particle_inventory = cp * particle_volume
    elif cp.shape == (n_solutes, n_particles):
        particle_volumes = np.asarray(particle_volume, dtype=float)
        if particle_volumes.shape != (n_particles,):
            raise ValueError(
                "When cp has shape (S,N), particle_volume must contain "
                "the N individual particle volumes."
            )
        particle_inventory = cp @ particle_volumes
    else:
        raise ValueError("cp must have shape (S,) or (S,N).")
    if matrix_integrals.shape != (n_particles,):
        raise ValueError(
            "matrix_integrals must have shape (Nparticles,)."
        )
    if Ainv_ones.shape != (n_particles,):
        raise ValueError("Ainv_ones must have shape (Nparticles,).")

    if fixed_green_inventory is None:
        fixed_green_inventory = np.zeros(n_solutes, dtype=float)
    else:
        fixed_green_inventory = np.asarray(
            fixed_green_inventory, dtype=float
        )
        if fixed_green_inventory.shape != (n_solutes,):
            raise ValueError(
                "fixed_green_inventory must have shape (S,)."
            )

    for name, values in (
        ("c_surf", c_surf),
        ("matrix_integrals", matrix_integrals),
        ("Ainv_ones", Ainv_ones),
        ("M_total", M_total),
        ("cp", cp),
        ("fixed_green_inventory", fixed_green_inventory),
    ):
        if np.any(~np.isfinite(values)):
            raise ValueError(f"{name} contains non-finite values.")

    matrix_volume = float(matrix_volume)

    if not np.isfinite(matrix_volume) or matrix_volume <= 0.0:
        raise ValueError("matrix_volume must be finite and positive.")

    if cp.ndim == 1:
        particle_volume = float(particle_volume)
        if not np.isfinite(particle_volume) or particle_volume < 0.0:
            raise ValueError(
                "particle_volume must be finite and non-negative."
            )
    else:
        particle_volume = np.asarray(particle_volume, dtype=float)
        if (
            particle_volume.shape != (n_particles,)
            or np.any(~np.isfinite(particle_volume))
            or np.any(particle_volume < 0.0)
        ):
            raise ValueError(
                "particle_volume must contain N finite non-negative "
                "particle volumes when cp is particle-specific."
            )

    # A p_s = c_surf_s.  scipy expects right-hand sides in columns,
    # hence the transpose into/out of the LU solve.
    p = linear_solver.solve(c_surf.T).T

    denominator = (
        matrix_volume
        - float(np.dot(matrix_integrals, Ainv_ones))
    )

    if not np.isfinite(denominator):
        raise RuntimeError(
            "The mass-conservation denominator is non-finite."
        )
    if denominator <= 0.0:
        raise RuntimeError(
            "The mass-conservation denominator must be positive, "
            f"but is {denominator:.16e}."
        )

    # Preserve the original per-species dot-product evaluation order.
    green_from_solved = np.asarray(
        [
            float(np.dot(matrix_integrals, p_s))
            for p_s in p
        ],
        dtype=float,
    )

    cbar = (
        M_total
        - particle_inventory
        - fixed_green_inventory
        - green_from_solved
    ) / denominator

    if np.any(~np.isfinite(cbar)):
        raise RuntimeError(
            "The calculated mean matrix concentrations are non-finite."
        )

    Q = p - cbar[:, None] * Ainv_ones[None, :]

    if np.any(~np.isfinite(Q)):
        raise RuntimeError(
            "The calculated monopole strengths are non-finite."
        )

    return cbar, Q



def _mass_conserving_monopole_directional_derivative_species(
    *,
    linear_solver,
    delta_c_surf,
    matrix_integrals,
    matrix_volume,
    Ainv_ones,
):
    """Directional derivative of the mass-conserving map for S species.

    Parameters
    ----------
    delta_c_surf : ndarray, shape (S, N)

    Returns
    -------
    delta_cbar : ndarray, shape (S,)
    delta_Q : ndarray, shape (S, N)

    Notes
    -----
    Species are solved independently, matching the previous two separate
    Mg/Si calls and preserving their floating-point evaluation order.
    """
    delta_c_surf = np.asarray(delta_c_surf, dtype=float)
    matrix_integrals = np.asarray(matrix_integrals, dtype=float)
    Ainv_ones = np.asarray(Ainv_ones, dtype=float)

    if delta_c_surf.ndim != 2:
        raise ValueError("delta_c_surf must have shape (S, N).")

    n_solutes, n_particles = delta_c_surf.shape
    if n_solutes < 1:
        raise ValueError("At least one diffusing solute is required.")
    if matrix_integrals.shape != (n_particles,):
        raise ValueError(
            "matrix_integrals must have shape (Nparticles,)."
        )
    if Ainv_ones.shape != (n_particles,):
        raise ValueError("Ainv_ones must have shape (Nparticles,).")

    denominator = (
        matrix_volume
        - float(np.dot(matrix_integrals, Ainv_ones))
    )

    if not np.isfinite(denominator):
        raise RuntimeError(
            "The mass-conservation denominator is non-finite."
        )
    if denominator <= 0.0:
        raise RuntimeError(
            "The mass-conservation denominator must be positive."
        )

    delta_cbar = np.empty(n_solutes, dtype=float)
    delta_Q = np.empty_like(delta_c_surf)

    for s in range(n_solutes):
        delta_p_s = linear_solver.solve(delta_c_surf[s])

        delta_cbar_s = (
            -float(np.dot(matrix_integrals, delta_p_s))
            / denominator
        )

        delta_cbar[s] = delta_cbar_s
        delta_Q[s] = (
            delta_p_s
            - delta_cbar_s * Ainv_ones
        )

    return delta_cbar, delta_Q



def _mass_conserving_monopole_transpose_action_species(
    *,
    linear_solver,
    weight_Q,
    matrix_integrals,
    matrix_volume,
    Ainv_ones,
):
    """Apply the transpose mass-conserving map for S species.

    weight_Q has shape (S,N); the returned concentration weight has the
    same shape.  Species are solved independently to preserve the previous
    two-call floating-point evaluation order.
    """
    weight_Q = np.asarray(weight_Q, dtype=float)
    matrix_integrals = np.asarray(matrix_integrals, dtype=float)
    Ainv_ones = np.asarray(Ainv_ones, dtype=float)

    if weight_Q.ndim != 2:
        raise ValueError("weight_Q must have shape (S,N).")

    n_solutes, n_particles = weight_Q.shape
    if n_solutes < 1:
        raise ValueError("At least one diffusing solute is required.")
    if matrix_integrals.shape != (n_particles,):
        raise ValueError(
            "matrix_integrals must have shape (Nparticles,)."
        )
    if Ainv_ones.shape != (n_particles,):
        raise ValueError("Ainv_ones must have shape (Nparticles,).")

    denominator = (
        matrix_volume
        - float(np.dot(matrix_integrals, Ainv_ones))
    )
    if not np.isfinite(denominator):
        raise RuntimeError(
            "The mass-conservation denominator is non-finite."
        )
    if denominator <= 0.0:
        raise RuntimeError(
            "The mass-conservation denominator must be positive."
        )

    weight_c = np.empty_like(weight_Q)

    for s in range(n_solutes):
        weight_p_s = (
            weight_Q[s]
            + matrix_integrals
            * (
                float(np.dot(Ainv_ones, weight_Q[s]))
                / denominator
            )
        )

        weight_c[s] = linear_solver.solve_transpose(weight_p_s)

    return weight_c



def solve_interface_coupled(
    *,
    R,
    G0,
    Gpair,
    solute_names,
    cbar,
    matrix_integrals,
    matrix_volume,
    particle_volume,
    M_total,
    cp,
    D,
    T,
    vatm,
    thermodynamics=None,
    fixed_green_inventory=None,
    c_surf_init=None,
    Q_init=None,
    gmres_rtol=1e-10,
    gmres_atol=0.0,
    gmres_maxiter=200,
    gmres_restart=50,
    use_preconditioner=True,
    nonlinear_xtol=1e-8,
    nonlinear_ftol=1e-8,
    nonlinear_gtol=1e-8,
    nonlinear_max_nfev=200,
    concentration_margin=1e-10,
    velocity_floor=0.1,
    velocity_absolute_tolerance=None,
    thermodynamic_tolerance=1e-7,
    velocity_mismatch_tolerance=1e-6,
    nonlinear_solver="lsmr",
    use_lsmr_block_preconditioner=True,
    use_lsmr_neighbor_preconditioner=True,
    lsmr_neighbor_preconditioner_k=12,
    lsmr_block_preconditioner_rcond=1.0e-12,
    diagnostic_positions=None,
    diagnostic_box=None,
    _neighbor_preconditioner_cache=None,
    linear_solver=None,
    background_concentration=None,
    strict=True,
    nonlinear_diagnostics=False,
    lsmr_preconditioner_refresh_nfev=6,
    lsmr_tr_maxiter=100,
    lsmr_max_refresh_chunks=8,
):
    """Solve the coupled interface problem for an arbitrary species axis.

    The nonlinear unknown is ``c_surf`` with shape ``(S, N)`` represented
    internally by bounded variables.  For the current line-compound
    thermodynamic model there is one thermodynamic constraint per particle
    and ``S - 1`` equal-velocity constraints per particle.

    ``solute_names`` defines the ordering of every species-indexed input and
    output.  No species name has special numerical meaning in this solver.
    """

    R = np.asarray(R, dtype=float)

    if R.ndim != 1:
        raise ValueError(
            f"R must be one-dimensional, got shape {R.shape}."
        )

    n = R.size

    nonlinear_diagnostics = bool(nonlinear_diagnostics)

    if lsmr_preconditioner_refresh_nfev is None:
        lsmr_preconditioner_refresh_nfev = 0
    lsmr_preconditioner_refresh_nfev = int(
        lsmr_preconditioner_refresh_nfev
    )
    if lsmr_preconditioner_refresh_nfev < 0:
        raise ValueError(
            "lsmr_preconditioner_refresh_nfev must be >= 0 or None."
        )

    if lsmr_tr_maxiter is None:
        lsmr_tr_maxiter = 0
    lsmr_tr_maxiter = int(lsmr_tr_maxiter)
    if lsmr_tr_maxiter < 0:
        raise ValueError("lsmr_tr_maxiter must be >= 0 or None.")

    if lsmr_max_refresh_chunks is None:
        lsmr_max_refresh_chunks = 0
    lsmr_max_refresh_chunks = int(lsmr_max_refresh_chunks)
    if lsmr_max_refresh_chunks < 0:
        raise ValueError(
            "lsmr_max_refresh_chunks must be >= 0 or None."
        )

    _nd = {
        "evaluate_calls": 0,
        "evaluate_wall": 0.0,
        "jacobian_builds": 0,
        "jacobian_build_wall": 0.0,
        "jac_matvec_calls": 0,
        "jac_matvec_wall": 0.0,
        "jac_rmatvec_calls": 0,
        "jac_rmatvec_wall": 0.0,
        "linear_solve_calls": 0,
        "linear_solve_rhs": 0,
        "linear_solve_wall": 0.0,
        "linear_solve_transpose_calls": 0,
        "linear_solve_transpose_rhs": 0,
        "linear_solve_transpose_wall": 0.0,
        "least_squares_wall": 0.0,
    }

    solute_names = tuple(str(name) for name in solute_names)
    if len(solute_names) < 1:
        raise ValueError("At least one diffusing solute is required.")
    if len(set(solute_names)) != len(solute_names):
        raise ValueError("solute_names must contain unique names.")

    n_solutes = len(solute_names)
    kernel_solute_names = solute_names

    cbar_input_species = np.asarray(cbar, dtype=float)
    M_total_species = np.asarray(M_total, dtype=float)
    cp_species = np.asarray(cp, dtype=float)
    D_species = np.asarray(D, dtype=float)

    for name, values in (
        ("cbar", cbar_input_species),
        ("M_total", M_total_species),
        ("D", D_species),
    ):
        if values.shape != (n_solutes,):
            raise ValueError(
                f"{name} must have shape ({n_solutes},), got {values.shape}."
            )
        if np.any(~np.isfinite(values)):
            raise ValueError(f"{name} contains non-finite values.")

    if cp_species.shape == (n_solutes,):
        cp_particle = np.broadcast_to(
            cp_species[:, None],
            (n_solutes, n),
        )
        cp_is_particle_specific = False
    elif cp_species.shape == (n_solutes, n):
        cp_particle = cp_species
        cp_is_particle_specific = True
    else:
        raise ValueError(
            "cp must have shape "
            f"({n_solutes},) or ({n_solutes}, {n}); got {cp_species.shape}."
        )
    if np.any(~np.isfinite(cp_particle)):
        raise ValueError("cp contains non-finite values.")

    # Optional prescribed long-range background sampled at particles.
    # In this mode ``cbar`` is only the legacy/diagnostic species scalar
    # supplied alongside the particle-resolved LR background.  Because the
    # LR field is an Ewald-like decomposition component, its particle-sampled
    # mean may legitimately be zero or negative.  Retain the strict positive
    # cbar requirement only for the legacy non-prescribed-background solve.
    prescribed_background_mode = background_concentration is not None
    if (
        not prescribed_background_mode
        and np.any(cbar_input_species <= 0.0)
    ):
        raise ValueError("cbar must contain finite positive values.")
    if np.any(M_total_species <= 0.0):
        raise ValueError("M_total must contain finite positive values.")
    if np.any(cp_particle <= 0.0):
        raise ValueError("cp must contain finite positive values.")
    if np.any(D_species <= 0.0):
        raise ValueError("D must contain finite positive values.")

    # Optional prescribed long-range background sampled at particles.
    # Shape is species x particles.  When present, global matrix inventory
    # is owned by the external transient field rather than solved here via
    # the legacy scalar cbar mass-conservation map.
    if prescribed_background_mode:
        background_species = np.asarray(
            background_concentration, dtype=float
        )
        if background_species.shape != (n_solutes, n):
            raise ValueError(
                "background_concentration must have shape "
                f"{(n_solutes, n)}, got {background_species.shape}."
            )
        if np.any(~np.isfinite(background_species)):
            raise ValueError(
                "background_concentration contains non-finite values."
            )
        # In scale-separated mode this is the smooth long-range component of
        # an Ewald-like decomposition, not the physical concentration by
        # itself.  C_LR may be locally zero or negative; the physical
        # interfacial concentrations remain bounded positive nonlinear
        # variables and the total field is C_LR + C_short.
        # A dedicated hybrid prescribed-background neighbour
        # preconditioner is constructed below when the injected linear
        # solver exposes the HybridShortRangeFactor operator.
    else:
        background_species = None

    for name, value in {
        "vatm": vatm,
    }.items():
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive.")

    matrix_integrals = np.asarray(matrix_integrals, dtype=float)
    if matrix_integrals.shape != (n,):
        raise ValueError(
            "matrix_integrals must have shape "
            f"({n},), got {matrix_integrals.shape}."
        )
    if np.any(~np.isfinite(matrix_integrals)):
        raise ValueError("matrix_integrals contains non-finite values.")

    matrix_volume = float(matrix_volume)

    if not np.isfinite(matrix_volume) or matrix_volume <= 0.0:
        raise ValueError("matrix_volume must be finite and positive.")

    if cp_is_particle_specific:
        particle_volume = np.asarray(particle_volume, dtype=float)
        if (
            particle_volume.shape != (n,)
            or np.any(~np.isfinite(particle_volume))
            or np.any(particle_volume < 0.0)
        ):
            raise ValueError(
                "particle_volume must have shape (N,) when cp is "
                "particle-specific."
            )
    else:
        particle_volume = float(particle_volume)
        if not np.isfinite(particle_volume) or particle_volume < 0.0:
            raise ValueError(
                "particle_volume must be finite and non-negative."
            )

    if fixed_green_inventory is None:
        fixed_green_inventory_species = np.zeros(
            n_solutes, dtype=float
        )
    else:
        fixed_green_inventory_species = np.asarray(
            fixed_green_inventory, dtype=float
        )
        if fixed_green_inventory_species.shape != (n_solutes,):
            raise ValueError(
                "fixed_green_inventory must have shape "
                f"({n_solutes},), got "
                f"{fixed_green_inventory_species.shape}."
            )
        if np.any(~np.isfinite(fixed_green_inventory_species)):
            raise ValueError(
                "fixed_green_inventory contains non-finite values."
            )

    # Backward compatibility / explicit absolute velocity tolerance.
    if velocity_absolute_tolerance is None:
        velocity_absolute_tolerance = velocity_floor
    velocity_absolute_tolerance = float(
        velocity_absolute_tolerance
    )

    if n == 0:
        e = np.zeros(0, dtype=float)
        return CoupledTernarySolveResult(
            solute_names=kernel_solute_names,
            Q=np.zeros((n_solutes, 0), dtype=float),
            cbar=cbar_input_species.copy(),
            c_surf=np.zeros((n_solutes, 0), dtype=float),
            dRdt=e.copy(),
            dRdt_species=np.zeros((n_solutes, 0), dtype=float),
            success=True,
            message="No active particles.",
            nfev=0,
            cost=0.0,
            max_thermodynamic_residual=0.0,
            max_relative_velocity_mismatch=0.0,
            optimality=0.0,
        )

    if np.any(~np.isfinite(R)) or np.any(R <= 0.0):
        raise ValueError("R must contain finite positive active radii.")

    if thermodynamics is None:
        raise ValueError(
            "A phase thermodynamic model must be supplied through thermodynamics."
        )

    if not (0.0 < concentration_margin < 0.1):
        raise ValueError("concentration_margin must lie in (0, 0.1).")

    # --------------------------------------------------------------
    # Monopole linear operator.
    #
    # If no solver is supplied, use the direct dense path.
    # An injected short-range factor must provide solve() and solve_transpose().
    # The mass-conserving cbar algebra is intentionally unchanged here.
    # --------------------------------------------------------------
    if linear_solver is None:
        diag = (
            1.0 / (4.0 * np.pi * R)
            + float(G0)
        )
        A_dense = (
            np.diag(diag)
            + np.asarray(Gpair, dtype=float)
        )
        active_linear_solver = _DenseMonopoleLinearSolver(
            A_dense
        )
    else:
        if not callable(getattr(linear_solver, "solve", None)):
            raise TypeError(
                "linear_solver must provide a callable solve(rhs) method."
            )
        if not callable(
            getattr(linear_solver, "solve_transpose", None)
        ):
            raise TypeError(
                "linear_solver must provide a callable "
                "solve_transpose(rhs) method."
            )
        if not callable(getattr(linear_solver, "diagonal", None)):
            raise TypeError(
                "linear_solver must provide a callable diagonal() method."
            )

        solver_shape = getattr(linear_solver, "shape", None)
        if solver_shape is not None and tuple(solver_shape) != (n, n):
            raise ValueError(
                "linear_solver.shape must match the active particle "
                f"operator shape {(n, n)}, got {solver_shape}."
            )

        active_linear_solver = linear_solver

    hybrid_preconditioner_factor = active_linear_solver
    hybrid_preconditioner_operator = getattr(
        hybrid_preconditioner_factor, "operator", None
    )
    hybrid_neighbor_available = bool(
        hybrid_preconditioner_operator is not None
        and hasattr(hybrid_preconditioner_operator, "pair_matrix")
        and hasattr(hybrid_preconditioner_operator, "self_regular")
        and hasattr(hybrid_preconditioner_operator, "zero_mean_constant")
        and hasattr(hybrid_preconditioner_factor, "_z")
        and hasattr(hybrid_preconditioner_factor, "_zT")
        and hasattr(hybrid_preconditioner_factor, "_denom")
    )

    if (
        prescribed_background_mode
        and use_lsmr_neighbor_preconditioner
        and not hybrid_neighbor_available
    ):
        raise ValueError(
            "Prescribed-background neighbour preconditioning requires "
            "an injected hybrid short-range factor exposing its operator."
        )

    if nonlinear_diagnostics:
        class _TimedLinearSolver:
            def __init__(self, base):
                self._base = base
                self.shape = getattr(base, "shape", None)

            @staticmethod
            def _rhs_count(rhs):
                arr = np.asarray(rhs)
                if arr.ndim <= 1:
                    return 1
                return int(np.prod(arr.shape[1:]))

            def solve(self, rhs):
                _nd["linear_solve_calls"] += 1
                _nd["linear_solve_rhs"] += self._rhs_count(rhs)
                t0 = time.perf_counter()
                try:
                    return self._base.solve(rhs)
                finally:
                    _nd["linear_solve_wall"] += time.perf_counter() - t0

            def solve_transpose(self, rhs):
                _nd["linear_solve_transpose_calls"] += 1
                _nd["linear_solve_transpose_rhs"] += self._rhs_count(rhs)
                t0 = time.perf_counter()
                try:
                    return self._base.solve_transpose(rhs)
                finally:
                    _nd["linear_solve_transpose_wall"] += (
                        time.perf_counter() - t0
                    )

            def diagonal(self):
                return self._base.diagonal()

        active_linear_solver = _TimedLinearSolver(active_linear_solver)

    Ainv_ones = np.asarray(
        active_linear_solver.solve(
            np.ones(n, dtype=float)
        ),
        dtype=float,
    )

    if Ainv_ones.shape != (n,):
        raise ValueError(
            "linear_solver.solve() returned the wrong shape for "
            f"A^{{-1}}1: expected {(n,)}, got {Ainv_ones.shape}."
        )
    if np.any(~np.isfinite(Ainv_ones)):
        raise RuntimeError("A^{-1}1 contains non-finite values.")

    active_operator_diagonal = np.asarray(
        active_linear_solver.diagonal(),
        dtype=float,
    )
    if active_operator_diagonal.shape != (n,):
        raise ValueError(
            "linear_solver.diagonal() returned the wrong shape: "
            f"expected {(n,)}, got {active_operator_diagonal.shape}."
        )
    if np.any(~np.isfinite(active_operator_diagonal)):
        raise RuntimeError(
            "linear_solver.diagonal() contains non-finite values."
        )
    if np.any(np.abs(active_operator_diagonal) <= np.finfo(float).tiny):
        raise RuntimeError(
            "linear_solver.diagonal() contains a zero or underflow-scale "
            "entry, so the block preconditioner cannot be formed."
        )

    def default_guess(cbar_value, cp_values):
        """Return a physically admissible cold-start surface concentration.

        ``cp_values`` may be either a scalar-equivalent value or an
        N-particle array (for particle-specific phase compositions).  Start
        from the matrix concentration and clip particle-by-particle to the
        same admissible bounds used by the bounded nonlinear variables.
        """
        cp_values = np.asarray(cp_values, dtype=float)
        lower = concentration_margin * cp_values * 10.0
        upper = (1.0 - 10.0 * concentration_margin) * cp_values
        return np.clip(np.full(n, float(cbar_value), dtype=float), lower, upper)

    if c_surf_init is None:
        c_surf0_species = np.empty((n_solutes, n), dtype=float)
        for sidx in range(n_solutes):
            c_surf0_species[sidx] = default_guess(
                cbar_input_species[sidx],
                cp_particle[sidx],
            )
    else:
        c_surf0_species = np.asarray(c_surf_init, dtype=float)
        if c_surf0_species.shape != (n_solutes, n):
            raise ValueError(
                "c_surf_init must have shape "
                f"({n_solutes}, {n}), got {c_surf0_species.shape}."
            )
        if np.any(~np.isfinite(c_surf0_species)):
            raise ValueError(
                "c_surf_init contains non-finite values."
            )

    if np.any(c_surf0_species <= 0.0):
        raise ValueError("c_surf_init must be strictly positive.")
    if np.any(c_surf0_species >= cp_particle):
        raise ValueError(
            "c_surf_init must remain below the particle-specific "
            "precipitate concentration."
        )

    # Q_init is retained as part of the generic warm-start state interface.
    # As in the legacy implementation, the current nonlinear unknowns are
    # surface concentrations, so Q is recomputed from mass conservation.
    if Q_init is not None:
        Q_init_array = np.asarray(Q_init, dtype=float)
        if Q_init_array.shape != (n_solutes, n):
            raise ValueError(
                "Q_init must have shape "
                f"({n_solutes}, {n}), got {Q_init_array.shape}."
            )
        if np.any(~np.isfinite(Q_init_array)):
            raise ValueError("Q_init contains non-finite values.")

    z0 = np.concatenate(
        [
            _z_from_bounded(
                c_surf0_species[sidx],
                cp_particle[sidx],
                concentration_margin,
            )
            for sidx in range(n_solutes)
        ]
    )

    last = {}

    def _evaluate_impl(z):
        # Species-indexed nonlinear state.  The residual has one
        # thermodynamic constraint plus S-1 equal-velocity constraints
        # per particle.
        n_solutes = len(kernel_solute_names)
        if n_solutes < 1:
            raise ValueError("At least one diffusing solute is required.")

        z_species = np.asarray(z, dtype=float).reshape(n_solutes, n)
        c_surf_species = np.empty_like(z_species)
        for s in range(n_solutes):
            c_surf_species[s] = _bounded_from_z(
                z_species[s],
                cp_particle[s],
                concentration_margin,
            )

        # The mass-conserving monopole kernel is still the existing
        # two-solute implementation.  Pack its outputs immediately into
        # species-axis arrays so the remainder of evaluate() no longer owns
        # separate Mg/Si state.
        if prescribed_background_mode:
            # Scale-separated relation:
            #
            #   A_short Q_s = c_R,s - C_LR,s(R_i)
            #
            # The transient LR field owns the global inventory.  The scalar
            # cbar returned below is diagnostic only in this mode.
            Q_species = np.empty_like(c_surf_species)
            for s in range(n_solutes):
                Q_species[s] = active_linear_solver.solve(
                    c_surf_species[s] - background_species[s]
                )
            cbar_species = np.mean(background_species, axis=1)
        else:
            cbar_species, Q_species = (
                _mass_conserving_monopoles_species(
                    linear_solver=active_linear_solver,
                    c_surf=c_surf_species,
                    matrix_integrals=matrix_integrals,
                    matrix_volume=matrix_volume,
                    particle_volume=particle_volume,
                    M_total=M_total_species,
                    cp=cp_species,
                    Ainv_ones=Ainv_ones,
                    fixed_green_inventory=(
                        fixed_green_inventory_species
                    ),
                )
            )

        velocity_species = np.empty_like(Q_species)
        for s in range(n_solutes):
            velocity_species[s] = _velocity(
                R,
                Q_species[s],
                D_species[s],
                cp_particle[s],
                c_surf_species[s],
                concentration_margin,
            )

        f_therm = thermodynamics.residual(
            c_surf=c_surf_species,
            radius=R,
            temperature=T,
            matrix_atomic_volume=vatm,
        )

        f_kin_species = _equal_velocity_residual_species(
            velocity_species,
            velocity_floor,
        )

        last.update(
            Q=Q_species,
            cbar=cbar_species,
            c_surf=c_surf_species,
            velocity_species=velocity_species,
            f_therm=f_therm,
            f_kin_species=f_kin_species,
        )

        return np.concatenate(
            (f_therm, f_kin_species.reshape(-1))
        )

    def evaluate(z):
        _nd["evaluate_calls"] += 1
        t0 = time.perf_counter()
        try:
            return _evaluate_impl(z)
        finally:
            _nd["evaluate_wall"] += time.perf_counter() - t0


    def analytic_jacobian(z):

        _nd["jacobian_builds"] += 1
        _jac_build_t0 = time.perf_counter()

        n_solutes = len(kernel_solute_names)
        

        # Make sure the state-dependent quantities correspond to this z.
        evaluate(z)

        Q_species = last["Q"]
        velocity_species = last["velocity_species"]

        def _matvec_impl(delta_z):

            return _ternary_residual_directional_derivative(
                z=z,
                delta_z=delta_z,
                n=n,
                cp_species=cp_species,
                concentration_margin=concentration_margin,
                linear_solver=active_linear_solver,
                matrix_integrals=matrix_integrals,
                matrix_volume=matrix_volume,
                Ainv_ones=Ainv_ones,
                R=R,
                D_species=D_species,
                Q_species=Q_species,
                velocity_species=velocity_species,
                velocity_floor=velocity_floor,
                thermodynamics=thermodynamics,
                T=T,
                vatm=vatm,
                prescribed_background_mode=prescribed_background_mode,
            )

        def _rmatvec_impl(weight_residual):

            return _ternary_residual_transpose_action(
                z=z,
                weight_residual=weight_residual,
                n=n,
                cp_species=cp_species,
                concentration_margin=concentration_margin,
                linear_solver=active_linear_solver,
                matrix_integrals=matrix_integrals,
                matrix_volume=matrix_volume,
                Ainv_ones=Ainv_ones,
                R=R,
                D_species=D_species,
                Q_species=Q_species,
                velocity_species=velocity_species,
                velocity_floor=velocity_floor,
                thermodynamics=thermodynamics,
                T=T,
                vatm=vatm,
                prescribed_background_mode=prescribed_background_mode,
            )

        def matvec(delta_z):
            _nd["jac_matvec_calls"] += 1
            t0 = time.perf_counter()
            try:
                return _matvec_impl(delta_z)
            finally:
                _nd["jac_matvec_wall"] += time.perf_counter() - t0

        def rmatvec(weight_residual):
            _nd["jac_rmatvec_calls"] += 1
            t0 = time.perf_counter()
            try:
                return _rmatvec_impl(weight_residual)
            finally:
                _nd["jac_rmatvec_wall"] += time.perf_counter() - t0

        _nd["jacobian_build_wall"] += (
            time.perf_counter() - _jac_build_t0
        )

        return LinearOperator(
            shape=(n_solutes * n, n_solutes * n),
            matvec=matvec,
            rmatvec=rmatvec,
            dtype=float,
        )

    # ------------------------------------------------------------
    # Fixed right preconditioner for the matrix-free LSMR nonlinear
    # solve.  Two variants are supported:
    #
    #   1. the original particle-by-particle 2x2 block Jacobi;
    #   2. a sparse nearest-neighbour preconditioner that retains
    #      the strongly localized inter-particle monopole response.
    #
    # The latter is the default when use_lsmr_neighbor_preconditioner
    # is True.  We solve in transformed variables
    #
    #     z = z0 + P u,
    #
    # so LSMR sees J P.
    # ------------------------------------------------------------

    preconditioner_blocks = None
    preconditioner_lu = None
    preconditioner_woodbury_Y = None
    preconditioner_woodbury_Z = None
    preconditioner_woodbury_U = None
    preconditioner_woodbury_V = None
    preconditioner_woodbury_K = None
    preconditioner_kind = "none"
    n_solutes = len(kernel_solute_names)

    hybrid_neighbor_B_sparse = None
    hybrid_neighbor_setup_wall = 0.0

    if (
        prescribed_background_mode
        and use_lsmr_neighbor_preconditioner
        and hybrid_neighbor_available
        and n > 1
    ):
        _hybrid_neighbor_t0 = time.perf_counter()

        k_hybrid = int(lsmr_neighbor_preconditioner_k)
        if k_hybrid < 1:
            raise ValueError(
                "lsmr_neighbor_preconditioner_k must be >= 1."
            )
        k_hybrid = min(k_hybrid, n - 1)

        pair_csc = hybrid_preconditioner_operator.pair_matrix.tocsc()
        pair_csr = pair_csc.tocsr()
        c0_hybrid = float(
            hybrid_preconditioner_operator.zero_mean_constant
        )
        diag_B = (
            1.0 / (4.0 * np.pi * R)
            + float(hybrid_preconditioner_operator.self_regular)
        )

        # Build a sparse local approximation to B^{-1}, where
        #
        #     A = B - c0 1 1^T.
        #
        # The global Sherman--Morrison contribution to A^{-1} is added
        # analytically as a low-rank correction to the Jacobian below.
        # For each source particle j, retain j plus the k strongest
        # short-range neighbours and solve the exact local B subproblem.
        rows_B = []
        cols_B = []
        data_B = []

        for j in range(n):
            col_start = pair_csc.indptr[j]
            col_end = pair_csc.indptr[j + 1]
            nbr_idx = pair_csc.indices[col_start:col_end]
            nbr_val = pair_csc.data[col_start:col_end]

            if nbr_idx.size > k_hybrid:
                keep = np.argpartition(
                    np.abs(nbr_val), -k_hybrid
                )[-k_hybrid:]
                nbr_idx = nbr_idx[keep]

            local_set = np.concatenate(
                (np.array([j], dtype=int), nbr_idx.astype(int))
            )
            # Defensive uniqueness: the sparse pair matrix should not contain
            # diagonal entries, but this keeps the local ordering well-defined.
            _, unique_pos = np.unique(local_set, return_index=True)
            local_set = local_set[np.sort(unique_pos)]

            # Put j first because the RHS is e_j.
            if local_set[0] != j:
                local_set = np.concatenate(
                    ([j], local_set[local_set != j])
                )

            B_SS = pair_csr[
                local_set[:, None], local_set
            ].toarray()
            B_SS[np.diag_indices_from(B_SS)] += diag_B[local_set]

            rhs = np.zeros(local_set.size, dtype=float)
            rhs[0] = 1.0

            try:
                response = np.linalg.solve(B_SS, rhs)
            except np.linalg.LinAlgError:
                response = np.linalg.lstsq(
                    B_SS, rhs, rcond=None
                )[0]

            rows_B.extend(local_set.tolist())
            cols_B.extend([j] * local_set.size)
            data_B.extend(response.tolist())

        hybrid_neighbor_B_sparse = csc_matrix(
            (np.asarray(data_B, dtype=float),
             (np.asarray(rows_B, dtype=int),
              np.asarray(cols_B, dtype=int))),
            shape=(n, n),
        )

        hybrid_neighbor_setup_wall = (
            time.perf_counter() - _hybrid_neighbor_t0
        )

        if nonlinear_diagnostics:
            avg_col_nnz = (
                hybrid_neighbor_B_sparse.nnz / max(n, 1)
            )
            print(
                "[hybrid-neighbor-Binv] "
                f"N={n} k={k_hybrid} "
                f"nnz={hybrid_neighbor_B_sparse.nnz} "
                f"avg_col_nnz={avg_col_nnz:.2f} "
                f"setup_wall={hybrid_neighbor_setup_wall:.3f}s"
            )

    def _build_hybrid_neighbor_preconditioner(z_ref):
        """Build sparse + global-rank-one hybrid Jacobian preconditioner.

        The local sparse part uses a k-neighbour approximation to B^{-1}.
        The exact global Sherman--Morrison term in

            A^{-1} = B^{-1} + beta z zT^T

        is propagated analytically into the velocity rows.  This produces a
        sparse Jacobian plus at most S rank-one updates, inverted with a
        Woodbury correction around the sparse LU.
        """
        if hybrid_neighbor_B_sparse is None:
            raise RuntimeError(
                "Hybrid neighbour B^{-1} response has not been constructed."
            )

        t0_build = time.perf_counter()
        evaluate(z_ref)

        c_surf_ref = np.asarray(last["c_surf"], dtype=float)
        Q_ref = np.asarray(last["Q"], dtype=float)
        velocity_ref = np.asarray(
            last["velocity_species"], dtype=float
        )

        z_ref_species = np.asarray(
            z_ref, dtype=float
        ).reshape(n_solutes, n)

        dc_surf_dz_ref = np.empty_like(z_ref_species)
        for s in range(n_solutes):
            dc_surf_dz_ref[s] = _bounded_derivative_from_z(
                z_ref_species[s],
                cp_particle[s],
                concentration_margin,
            )

        denom_species = cp_particle - c_surf_ref
        pref_species = (
            -D_species[:, None]
            / (4.0 * np.pi * R[None, :]**2)
        )

        thermo_dfdz = np.empty((n_solutes, n), dtype=float)
        for s in range(n_solutes):
            delta_c = np.zeros_like(c_surf_ref)
            delta_c[s] = dc_surf_dz_ref[s]
            thermo_dfdz[s] = thermodynamics.directional_derivative(
                c_surf=c_surf_ref,
                delta_c_surf=delta_c,
                radius=R,
                temperature=T,
                matrix_atomic_volume=vatm,
            )

        kinetic_dfdv = np.zeros(
            (max(n_solutes - 1, 0), n_solutes, n),
            dtype=float,
        )
        if n_solutes > 1:
            v0 = velocity_ref[0]
            for s in range(1, n_solutes):
                vs = velocity_ref[s]
                vscale = np.sqrt(
                    v0**2 + vs**2 + velocity_floor**2
                )
                velocity_difference = v0 - vs
                kinetic_dfdv[s - 1, 0] = (
                    1.0 / vscale
                    - velocity_difference * v0 / vscale**3
                )
                kinetic_dfdv[s - 1, s] = (
                    -1.0 / vscale
                    - velocity_difference * vs / vscale**3
                )

        # Sparse local Jacobian: only the k-neighbour B^{-1} response.
        DV_species_local = []
        for s in range(n_solutes):
            Dcs_s = diags(dc_surf_dz_ref[s])
            DV_s = (
                diags(pref_species[s] / denom_species[s])
                @ hybrid_neighbor_B_sparse
                @ Dcs_s
                + diags(
                    pref_species[s]
                    * Q_ref[s]
                    * dc_surf_dz_ref[s]
                    / denom_species[s]**2
                )
            )
            DV_species_local.append(DV_s)

        block_rows = [
            [diags(thermo_dfdz[s]) for s in range(n_solutes)]
        ]
        for r in range(n_solutes - 1):
            block_rows.append([
                diags(kinetic_dfdv[r, s]) @ DV_species_local[s]
                for s in range(n_solutes)
            ])

        Jlocal = bmat(block_rows, format="csc")

        try:
            lu = splu(Jlocal)
            regularized = False
        except RuntimeError:
            scale = max(
                1.0,
                float(np.max(np.abs(Jlocal.data)))
                if Jlocal.nnz else 1.0,
            )
            Jlocal = Jlocal + diags(
                np.full(
                    n_solutes * n,
                    1.0e-12 * scale,
                )
            )
            lu = splu(Jlocal)
            regularized = True

        # Exact global Sherman--Morrison response of the physical hybrid A.
        c0 = float(hybrid_preconditioner_operator.zero_mean_constant)
        zB = np.asarray(hybrid_preconditioner_factor._z, dtype=float)
        zBT = np.asarray(hybrid_preconditioner_factor._zT, dtype=float)
        denom_sm = float(hybrid_preconditioner_factor._denom)
        beta_sm = c0 / denom_sm

        if zB.shape != (n,) or zBT.shape != (n,):
            raise RuntimeError(
                "Hybrid Sherman--Morrison vectors have unexpected shape."
            )

        # J = Jlocal + U V^T. There is one rank-one column per diffusing
        # species, arising only in the kinetic residual rows.
        rank = n_solutes if n_solutes > 1 else 0
        U = np.zeros((n_solutes * n, rank), dtype=float)
        V = np.zeros((n_solutes * n, rank), dtype=float)

        for s in range(rank):
            p_s = (
                pref_species[s]
                / denom_species[s]
                * zB
            )
            q_s = beta_sm * dc_surf_dz_ref[s] * zBT

            # Variable block for species s.
            V[s * n:(s + 1) * n, s] = q_s

            # Thermodynamic residual block receives no monopole-response
            # contribution. Each kinetic residual block receives its local
            # d f / d v_s multiplier times p_s.
            for r in range(n_solutes - 1):
                row0 = (r + 1) * n
                U[row0:row0 + n, s] = (
                    kinetic_dfdv[r, s] * p_s
                )

        if rank > 0:
            Y = lu.solve(U)
            Z = lu.solve(V, trans="T")
            K = np.eye(rank, dtype=float) + V.T @ Y
            # Do not form an explicit inverse; K is only S x S and is solved
            # directly on every preconditioner action.
            cond_K = float(np.linalg.cond(K))
        else:
            Y = np.zeros((n_solutes * n, 0), dtype=float)
            Z = np.zeros((n_solutes * n, 0), dtype=float)
            K = np.zeros((0, 0), dtype=float)
            cond_K = 1.0

        build_wall = time.perf_counter() - t0_build

        if nonlinear_diagnostics:
            print(
                "[hybrid-neighbor-globalSM-preconditioner] "
                f"N={n} k={int(lsmr_neighbor_preconditioner_k)} "
                f"Jnnz={Jlocal.nnz} rank={rank} "
                f"sm_beta={beta_sm:.3e} condK={cond_K:.3e} "
                f"build_wall={build_wall:.3f}s "
                f"regularized={regularized}"
            )

        return lu, U, V, Y, Z, K


    if (
        nonlinear_solver == "lsmr"
        and use_lsmr_block_preconditioner
    ):

        hybrid_prescribed_neighbor = bool(
            prescribed_background_mode
            and use_lsmr_neighbor_preconditioner
            and hybrid_neighbor_B_sparse is not None
        )

        if hybrid_prescribed_neighbor:
            (
                preconditioner_lu,
                preconditioner_woodbury_U,
                preconditioner_woodbury_V,
                preconditioner_woodbury_Y,
                preconditioner_woodbury_Z,
                preconditioner_woodbury_K,
            ) = _build_hybrid_neighbor_preconditioner(z0)
            preconditioner_kind = "neighbor_global_sm"

        evaluate(z0)

        c_surf_ref = np.asarray(last["c_surf"], dtype=float)
        Q_ref = np.asarray(last["Q"], dtype=float)
        velocity_ref = np.asarray(
            last["velocity_species"], dtype=float
        )

        if c_surf_ref.shape != (n_solutes, n):
            raise RuntimeError(
                "Preconditioner c_surf state has unexpected shape."
            )
        if Q_ref.shape != (n_solutes, n):
            raise RuntimeError(
                "Preconditioner Q state has unexpected shape."
            )
        if velocity_ref.shape != (n_solutes, n):
            raise RuntimeError(
                "Preconditioner velocity state has unexpected shape."
            )

        z0_species = np.asarray(z0, dtype=float).reshape(
            n_solutes, n
        )
        dc_surf_dz_ref = np.empty_like(z0_species)
        for s in range(n_solutes):
            dc_surf_dz_ref[s] = _bounded_derivative_from_z(
                z0_species[s],
                cp_particle[s],
                concentration_margin,
            )

        denom_species = (
            cp_particle - c_surf_ref
        )
        pref_species = (
            -D_species[:, None]
            / (4.0 * np.pi * R[None, :]**2)
        )

        # Thermodynamic row of the approximate Jacobian.  Obtain it
        # through the phase model rather than assuming a particular
        # solubility-product algebra.
        thermo_dfdz = np.empty((n_solutes, n), dtype=float)
        for s in range(n_solutes):
            delta_c = np.zeros_like(c_surf_ref)
            delta_c[s] = dc_surf_dz_ref[s]
            thermo_dfdz[s] = thermodynamics.directional_derivative(
                c_surf=c_surf_ref,
                delta_c_surf=delta_c,
                radius=R,
                temperature=T,
                matrix_atomic_volume=vatm,
            )

        # Derivatives of each equal-velocity residual with respect to
        # the species velocities.  Shape: (S-1, S, N).
        kinetic_dfdv = np.zeros(
            (max(n_solutes - 1, 0), n_solutes, n),
            dtype=float,
        )
        if n_solutes > 1:
            v0 = velocity_ref[0]
            for s in range(1, n_solutes):
                vs = velocity_ref[s]
                vscale = np.sqrt(
                    v0**2 + vs**2 + velocity_floor**2
                )
                velocity_difference = v0 - vs
                kinetic_dfdv[s - 1, 0] = (
                    1.0 / vscale
                    - velocity_difference * v0 / vscale**3
                )
                kinetic_dfdv[s - 1, s] = (
                    -1.0 / vscale
                    - velocity_difference * vs / vscale**3
                )

        if (
            use_lsmr_neighbor_preconditioner
            and n > 1
            and not hybrid_prescribed_neighbor
        ):
            # ----------------------------------------------------
            # Sparse local approximation to B = dQ/dc.
            #
            # This geometric mass-conserving response is common to
            # every species.  Species dependence enters through the
            # bounded concentration transform and velocity law below.
            # ----------------------------------------------------
            k = int(lsmr_neighbor_preconditioner_k)
            if k < 1:
                raise ValueError(
                    "lsmr_neighbor_preconditioner_k must be >= 1."
                )
            k = min(k, n - 1)

            Gpair_arr = np.asarray(Gpair, dtype=float)
            A_dense_local = np.diag(diag) + Gpair_arr
            neighbor_sets = None

            use_position_neighbors = (
                diagnostic_positions is not None
                and diagnostic_box is not None
            )

            # Particle-neighbor topology is purely geometric when explicit
            # positions are available.  It is unchanged throughout one
            # interaction-continuation sequence, although G0/Gpair and the
            # state-dependent preconditioner coefficients change with lambda.
            if (
                use_position_neighbors
                and _neighbor_preconditioner_cache is not None
            ):
                cached_sets = _neighbor_preconditioner_cache.get(
                    "position_neighbor_sets"
                )
                if (
                    cached_sets is not None
                    and _neighbor_preconditioner_cache.get("n") == n
                    and _neighbor_preconditioner_cache.get("k") == k
                ):
                    neighbor_sets = cached_sets

            if use_position_neighbors and neighbor_sets is None:
                pos_pre = np.asarray(
                    diagnostic_positions, dtype=float
                )
                box_pre = np.asarray(diagnostic_box, dtype=float)

                if pos_pre.shape != (n, 3):
                    raise ValueError(
                        "diagnostic_positions must have shape (N, 3)."
                    )

                if box_pre.shape == (3,):
                    box_lengths_pre = box_pre
                elif box_pre.shape == (3, 2):
                    box_lengths_pre = (
                        box_pre[:, 1] - box_pre[:, 0]
                    )
                else:
                    raise ValueError(
                        "diagnostic_box must have shape (3,) or (3,2)."
                    )

                neighbor_sets = []
                for j in range(n):
                    dr = pos_pre - pos_pre[j]
                    dr -= box_lengths_pre * np.round(
                        dr / box_lengths_pre
                    )
                    distance = np.linalg.norm(dr, axis=1)
                    distance[j] = np.inf
                    nbr = np.argpartition(distance, k)[:k]
                    neighbor_sets.append(
                        np.concatenate(([j], nbr))
                    )

                if _neighbor_preconditioner_cache is not None:
                    _neighbor_preconditioner_cache[
                        "position_neighbor_sets"
                    ] = neighbor_sets
                    _neighbor_preconditioner_cache["n"] = n
                    _neighbor_preconditioner_cache["k"] = k

            elif not use_position_neighbors:
                # Do not cache Gpair-ranked neighbors: at lambda=0 Gpair is
                # identically zero, so that ranking is not a valid topology
                # for the later continuation solves.
                neighbor_sets = []
                for j in range(n):
                    strength = np.abs(
                        Gpair_arr[:, j]
                    ).copy()
                    strength[j] = -np.inf
                    nbr = np.argpartition(
                        strength, -k
                    )[-k:]
                    neighbor_sets.append(
                        np.concatenate(([j], nbr))
                    )

            B_local = np.zeros((n, n), dtype=float)

            for j in range(n):
                local_set = neighbor_sets[j]
                A_SS = A_dense_local[
                    np.ix_(local_set, local_set)
                ]
                rhs = np.zeros(local_set.size, dtype=float)
                rhs[0] = 1.0

                try:
                    response = np.linalg.solve(A_SS, rhs)
                except np.linalg.LinAlgError:
                    response = np.linalg.lstsq(
                        A_SS, rhs, rcond=None
                    )[0]

                B_local[local_set, j] = response

            mass_denominator = (
                matrix_volume
                - float(np.dot(matrix_integrals, Ainv_ones))
            )
            if abs(mass_denominator) > np.finfo(float).tiny:
                local_Ainv_ones = (
                    B_local @ np.ones(n, dtype=float)
                )
                local_I_Ainv = matrix_integrals @ B_local
                B_local += np.outer(
                    local_Ainv_ones, local_I_Ainv
                ) / mass_denominator

            B_sparse = np.zeros_like(B_local)
            for j in range(n):
                local_set = neighbor_sets[j]
                B_sparse[local_set, j] = B_local[
                    local_set, j
                ]

            B_sparse = csc_matrix(B_sparse)

            # Species-specific dv_s/dz_s approximate operators.
            DV_species = []
            for s in range(n_solutes):
                Dcs_s = diags(dc_surf_dz_ref[s])
                DV_s = (
                    diags(
                        pref_species[s] / denom_species[s]
                    )
                    @ B_sparse
                    @ Dcs_s
                    + diags(
                        pref_species[s]
                        * Q_ref[s]
                        * dc_surf_dz_ref[s]
                        / denom_species[s]**2
                    )
                )
                DV_species.append(DV_s)

            # Block rows: thermodynamic row followed by S-1
            # equal-velocity rows.  Block columns are species.
            block_rows = [
                [diags(thermo_dfdz[s])
                 for s in range(n_solutes)]
            ]
            for r in range(n_solutes - 1):
                block_rows.append([
                    (
                        diags(kinetic_dfdv[r, s])
                        @ DV_species[s]
                    )
                    for s in range(n_solutes)
                ])

            Japprox = bmat(block_rows, format="csc")

            try:
                preconditioner_lu = splu(Japprox)
            except RuntimeError:
                scale = max(
                    1.0,
                    float(np.max(np.abs(Japprox.data)))
                    if Japprox.nnz else 1.0,
                )
                Japprox = Japprox + diags(
                    np.full(
                        n_solutes * n,
                        1.0e-12 * scale,
                    )
                )
                preconditioner_lu = splu(Japprox)

            preconditioner_kind = "neighbor"

        elif not hybrid_prescribed_neighbor:
            # ----------------------------------------------------
            # Particle block-Jacobi fallback, now S x S.
            # ----------------------------------------------------
            dqdc_local = 1.0 / active_operator_diagonal

            dv_dz_local = np.empty(
                (n_solutes, n), dtype=float
            )
            for s in range(n_solutes):
                dv_dz_local[s] = (
                    pref_species[s]
                    * (
                        dqdc_local
                        * dc_surf_dz_ref[s]
                        / denom_species[s]
                        + Q_ref[s]
                        * dc_surf_dz_ref[s]
                        / denom_species[s]**2
                    )
                )

            preconditioner_blocks = np.empty(
                (n, n_solutes, n_solutes),
                dtype=float,
            )

            for i in range(n):
                Jloc = np.empty(
                    (n_solutes, n_solutes),
                    dtype=float,
                )
                Jloc[0, :] = thermo_dfdz[:, i]

                for r in range(n_solutes - 1):
                    Jloc[r + 1, :] = (
                        kinetic_dfdv[r, :, i]
                        * dv_dz_local[:, i]
                    )

                U, svals, Vt = np.linalg.svd(
                    Jloc, full_matrices=False
                )
                sigma_max = float(svals[0])

                if (
                    not np.isfinite(sigma_max)
                    or sigma_max <= 0.0
                ):
                    preconditioner_blocks[i] = np.eye(
                        n_solutes
                    )
                    continue

                sigma_floor = (
                    lsmr_block_preconditioner_rcond
                    * sigma_max
                )
                svals_safe = np.maximum(
                    svals, sigma_floor
                )
                preconditioner_blocks[i] = (
                    Vt.T
                    @ np.diag(1.0 / svals_safe)
                    @ U.T
                )

            preconditioner_kind = "block"


    def _rebuild_block_preconditioner(z_ref):
        """Rebuild the same particle-local SxS block-Jacobi preconditioner."""
        evaluate(z_ref)

        c_surf_ref = np.asarray(last["c_surf"], dtype=float)
        Q_ref = np.asarray(last["Q"], dtype=float)
        velocity_ref = np.asarray(
            last["velocity_species"], dtype=float
        )

        z_ref_species = np.asarray(
            z_ref, dtype=float
        ).reshape(n_solutes, n)

        dc_surf_dz_ref = np.empty_like(z_ref_species)
        for s in range(n_solutes):
            dc_surf_dz_ref[s] = _bounded_derivative_from_z(
                z_ref_species[s],
                cp_particle[s],
                concentration_margin,
            )

        denom_species = cp_particle - c_surf_ref
        pref_species = (
            -D_species[:, None]
            / (4.0 * np.pi * R[None, :]**2)
        )

        thermo_dfdz = np.empty(
            (n_solutes, n),
            dtype=float,
        )
        for s in range(n_solutes):
            delta_c = np.zeros_like(c_surf_ref)
            delta_c[s] = dc_surf_dz_ref[s]
            thermo_dfdz[s] = thermodynamics.directional_derivative(
                c_surf=c_surf_ref,
                delta_c_surf=delta_c,
                radius=R,
                temperature=T,
                matrix_atomic_volume=vatm,
            )

        kinetic_dfdv = np.zeros(
            (max(n_solutes - 1, 0), n_solutes, n),
            dtype=float,
        )
        if n_solutes > 1:
            v0 = velocity_ref[0]
            for s in range(1, n_solutes):
                vs = velocity_ref[s]
                vscale = np.sqrt(
                    v0**2 + vs**2 + velocity_floor**2
                )
                velocity_difference = v0 - vs
                kinetic_dfdv[s - 1, 0] = (
                    1.0 / vscale
                    - velocity_difference * v0 / vscale**3
                )
                kinetic_dfdv[s - 1, s] = (
                    -1.0 / vscale
                    - velocity_difference * vs / vscale**3
                )

        dqdc_local = 1.0 / active_operator_diagonal

        dv_dz_local = np.empty(
            (n_solutes, n), dtype=float
        )
        for s in range(n_solutes):
            dv_dz_local[s] = (
                pref_species[s]
                * (
                    dqdc_local
                    * dc_surf_dz_ref[s]
                    / denom_species[s]
                    + Q_ref[s]
                    * dc_surf_dz_ref[s]
                    / denom_species[s]**2
                )
            )

        blocks = np.empty(
            (n, n_solutes, n_solutes),
            dtype=float,
        )

        for i in range(n):
            Jloc = np.empty(
                (n_solutes, n_solutes),
                dtype=float,
            )
            Jloc[0, :] = thermo_dfdz[:, i]

            for r in range(n_solutes - 1):
                Jloc[r + 1, :] = (
                    kinetic_dfdv[r, :, i]
                    * dv_dz_local[:, i]
                )

            U, svals, Vt = np.linalg.svd(
                Jloc, full_matrices=False
            )
            sigma_max = float(svals[0])

            if (
                not np.isfinite(sigma_max)
                or sigma_max <= 0.0
            ):
                blocks[i] = np.eye(n_solutes)
                continue

            sigma_floor = (
                lsmr_block_preconditioner_rcond
                * sigma_max
            )
            svals_safe = np.maximum(
                svals, sigma_floor
            )
            blocks[i] = (
                Vt.T
                @ np.diag(1.0 / svals_safe)
                @ U.T
            )

        return blocks


    def apply_lsmr_preconditioner(u):
        """Apply the fixed right preconditioner P."""
        u = np.asarray(u, dtype=float).reshape(-1)

        if preconditioner_kind == "neighbor_global_sm":
            x0 = preconditioner_lu.solve(u)
            if preconditioner_woodbury_K.shape[0] == 0:
                return x0
            coeff = np.linalg.solve(
                preconditioner_woodbury_K,
                preconditioner_woodbury_V.T @ x0,
            )
            return x0 - preconditioner_woodbury_Y @ coeff

        if preconditioner_kind == "neighbor":
            return preconditioner_lu.solve(u)

        if preconditioner_blocks is None:
            return u.copy()

        u_species = u.reshape(n_solutes, n)
        out_species = np.empty_like(u_species)

        for i in range(n):
            out_species[:, i] = (
                preconditioner_blocks[i]
                @ u_species[:, i]
            )

        return out_species.reshape(-1)

    def apply_lsmr_preconditioner_transpose(v):
        """Apply P.T for the transpose Jacobian action."""
        v = np.asarray(v, dtype=float).reshape(-1)

        if preconditioner_kind == "neighbor_global_sm":
            y0 = preconditioner_lu.solve(v, trans="T")
            if preconditioner_woodbury_K.shape[0] == 0:
                return y0
            coeff = np.linalg.solve(
                preconditioner_woodbury_K.T,
                preconditioner_woodbury_U.T @ y0,
            )
            return y0 - preconditioner_woodbury_Z @ coeff

        if preconditioner_kind == "neighbor":
            return preconditioner_lu.solve(v, trans="T")

        if preconditioner_blocks is None:
            return v.copy()

        v_species = v.reshape(n_solutes, n)
        out_species = np.empty_like(v_species)

        for i in range(n):
            out_species[:, i] = (
                preconditioner_blocks[i].T
                @ v_species[:, i]
            )

        return out_species.reshape(-1)


    def z_from_preconditioned_variables(u):

        return (
            z0
            + apply_lsmr_preconditioner(u)
        )


    def evaluate_preconditioned(u):

        z = z_from_preconditioned_variables(u)

        return evaluate(z)


    def analytic_jacobian_preconditioned(u):

        z = z_from_preconditioned_variables(u)

        J = analytic_jacobian(z)

        def matvec(delta_u):

            return J.matvec(
                apply_lsmr_preconditioner(
                    delta_u
                )
            )

        def rmatvec(weight_residual):

            return (
                apply_lsmr_preconditioner_transpose(
                    J.rmatvec(
                        weight_residual
                    )
                )
            )

        return LinearOperator(
            shape=(n_solutes * n, n_solutes * n),
            matvec=matvec,
            rmatvec=rmatvec,
            dtype=float,
        )


    ## Original using exact solve but SLOW
    # opt = least_squares(
    #     evaluate, z0, method="trf",
    #     xtol=nonlinear_xtol, ftol=nonlinear_ftol, gtol=nonlinear_gtol,
    #     max_nfev=nonlinear_max_nfev, x_scale="jac",
    #)

    # Approximate solve using lsmr - much faster and testing shows good results vs above
    #opt = least_squares(
    #    evaluate, z0, method="trf",jac="2-point",tr_solver="lsmr",
    #    xtol=nonlinear_xtol, ftol=nonlinear_ftol, gtol=nonlinear_gtol,
    #    max_nfev=nonlinear_max_nfev, x_scale="jac",
    #)

    _least_squares_t0 = time.perf_counter()

    if nonlinear_solver == "lsmr":

        if use_lsmr_block_preconditioner:

            use_refresh = bool(
                prescribed_background_mode
                and lsmr_preconditioner_refresh_nfev > 0
            )

            if not use_refresh:
                u0 = np.zeros_like(
                    z0,
                    dtype=float,
                )

                opt = least_squares(
                    evaluate_preconditioned,
                    u0,
                    method="trf",
                    jac=analytic_jacobian_preconditioned,
                    tr_solver="lsmr",
                    tr_options=(
                        {"maxiter": lsmr_tr_maxiter}
                        if lsmr_tr_maxiter > 0
                        else {}
                    ),
                    xtol=nonlinear_xtol,
                    ftol=nonlinear_ftol,
                    gtol=nonlinear_gtol,
                    max_nfev=nonlinear_max_nfev,
                    x_scale=1.0,
                )

                opt.x = (
                    z_from_preconditioned_variables(
                        opt.x
                    )
                )

            else:
                total_nfev = 0
                total_njev = 0
                restart_index = 0
                z_current = np.asarray(
                    z0, dtype=float
                ).copy()
                opt = None

                while total_nfev < nonlinear_max_nfev:
                    z0 = z_current.copy()

                    if restart_index > 0:
                        if (
                            prescribed_background_mode
                            and use_lsmr_neighbor_preconditioner
                            and hybrid_neighbor_B_sparse is not None
                        ):
                            (
                                preconditioner_lu,
                                preconditioner_woodbury_U,
                                preconditioner_woodbury_V,
                                preconditioner_woodbury_Y,
                                preconditioner_woodbury_Z,
                                preconditioner_woodbury_K,
                            ) = _build_hybrid_neighbor_preconditioner(z0)
                            preconditioner_kind = "neighbor_global_sm"
                        else:
                            preconditioner_blocks = (
                                _rebuild_block_preconditioner(z0)
                            )
                            preconditioner_kind = "block"

                    remaining_nfev = (
                        nonlinear_max_nfev - total_nfev
                    )
                    chunk_budget = min(
                        lsmr_preconditioner_refresh_nfev,
                        remaining_nfev,
                    )

                    u0 = np.zeros_like(
                        z0,
                        dtype=float,
                    )

                    jmv_before = _nd["jac_matvec_calls"]
                    jrmv_before = _nd["jac_rmatvec_calls"]
                    solve_before = _nd["linear_solve_calls"]
                    solveT_before = (
                        _nd["linear_solve_transpose_calls"]
                    )
                    chunk_t0 = time.perf_counter()

                    opt_chunk = least_squares(
                        evaluate_preconditioned,
                        u0,
                        method="trf",
                        jac=analytic_jacobian_preconditioned,
                        tr_solver="lsmr",
                        tr_options=(
                            {"maxiter": lsmr_tr_maxiter}
                            if lsmr_tr_maxiter > 0
                            else {}
                        ),
                        xtol=nonlinear_xtol,
                        ftol=nonlinear_ftol,
                        gtol=nonlinear_gtol,
                        max_nfev=chunk_budget,
                        x_scale=1.0,
                    )

                    chunk_wall = (
                        time.perf_counter() - chunk_t0
                    )

                    z_current = (
                        z_from_preconditioned_variables(
                            opt_chunk.x
                        )
                    )

                    total_nfev += int(opt_chunk.nfev)
                    total_njev += int(
                        getattr(opt_chunk, "njev", 0) or 0
                    )
                    opt = opt_chunk

                    if nonlinear_diagnostics:
                        print(
                            "[nonlinear-refresh] "
                            f"restart={restart_index} "
                            f"lsmr_maxiter={lsmr_tr_maxiter} "
                            f"chunk_nfev={int(opt_chunk.nfev)} "
                            f"chunk_njev={getattr(opt_chunk, 'njev', None)} "
                            f"status={getattr(opt_chunk, 'status', None)} "
                            f"success={bool(opt_chunk.success)} "
                            f"wall={chunk_wall:.3f}s "
                            f"Jmv={_nd['jac_matvec_calls'] - jmv_before} "
                            f"Jrmv={_nd['jac_rmatvec_calls'] - jrmv_before} "
                            f"solve={_nd['linear_solve_calls'] - solve_before} "
                            f"solveT={_nd['linear_solve_transpose_calls'] - solveT_before}"
                        )

                    if opt_chunk.success:
                        break

                    if getattr(opt_chunk, "status", 0) < 0:
                        break

                    if int(opt_chunk.nfev) <= 0:
                        break

                    restart_index += 1

                    if (
                        lsmr_max_refresh_chunks > 0
                        and restart_index >= lsmr_max_refresh_chunks
                    ):
                        if nonlinear_diagnostics:
                            print(
                                "[nonlinear-refresh-limit] "
                                f"chunks={restart_index} "
                                f"nfev={total_nfev} "
                                "status=candidate_reject"
                            )
                        break

                if opt is None:
                    raise RuntimeError(
                        "Restarted LSMR solve did not execute."
                    )

                opt.x = z_current
                opt.nfev = total_nfev
                opt.njev = total_njev

        else:

            opt = least_squares(
                evaluate,
                z0,
                method="trf",
                jac=analytic_jacobian,
                tr_solver="lsmr",
                tr_options=(
                    {"maxiter": lsmr_tr_maxiter}
                    if lsmr_tr_maxiter > 0
                    else {}
                ),
                xtol=nonlinear_xtol,
                ftol=nonlinear_ftol,
                gtol=nonlinear_gtol,
                max_nfev=nonlinear_max_nfev,
                x_scale=1.0,
            )


    elif nonlinear_solver == "dense":

        opt = least_squares(
             evaluate,
             z0,
             method="trf",
             xtol=nonlinear_xtol,
             ftol=nonlinear_ftol,
             gtol=nonlinear_gtol,
             max_nfev=nonlinear_max_nfev,
             x_scale="jac",
        )

    else:

        raise ValueError(
            "nonlinear_solver must be 'lsmr' or 'dense'."
        )



    _nd["least_squares_wall"] += (
        time.perf_counter() - _least_squares_t0
    )

    evaluate(opt.x)

    # The legacy path owns global matrix inventory and therefore must satisfy
    # the scalar mass-conservation identity below.  In prescribed-background
    # mode, global inventory is owned by the external transient
    # long-range field, so applying this legacy check is both meaningless and
    # incorrect.
    if not prescribed_background_mode:
        if cp_is_particle_specific:
            particle_inventory_check = cp_particle @ particle_volume
        else:
            particle_inventory_check = cp_species * particle_volume

        M_check_species = (
            last["cbar"] * matrix_volume
            + last["Q"] @ matrix_integrals
            + fixed_green_inventory_species
            + particle_inventory_check
        )

        mass_relative_error = (
            M_check_species - M_total_species
        ) / M_total_species

        mass_tolerance = 1.0e-12
        if np.any(np.abs(mass_relative_error) > mass_tolerance):
            worst = int(np.argmax(np.abs(mass_relative_error)))
            raise RuntimeError(
                "Mass-conserving monopole solution failed for species "
                f"{kernel_solute_names[worst]!r}: relative inventory error="
                f"{mass_relative_error[worst]:.16e}."
            )

    max_therm = float(np.max(np.abs(last["f_therm"])))

    f_kin_species = np.asarray(
        last["f_kin_species"], dtype=float
    )
    if f_kin_species.size:
        max_vel = float(np.max(np.abs(f_kin_species)))
    else:
        max_vel = 0.0

    velocity_species = np.asarray(
        last["velocity_species"], dtype=float
    )

    if n_solutes > 1:
        velocity_error = np.abs(
            velocity_species[0:1, :] - velocity_species[1:, :]
        )
        velocity_scale = np.maximum(
            np.abs(velocity_species[0:1, :]),
            np.abs(velocity_species[1:, :]),
        )
        velocity_allowance = (
            velocity_absolute_tolerance
            + velocity_mismatch_tolerance * velocity_scale
        )
        scaled_velocity_error = (
            velocity_error
            / np.maximum(
                velocity_allowance,
                np.finfo(float).tiny,
            )
        )

        # Interfaces whose species velocities are all below the
        # absolute velocity resolution are effectively stationary.
        # The integrator subsequently sets their common dR/dt to zero,
        # so do not reject an otherwise converged solution merely
        # because numerically tiny species velocities differ relative
        # to one another.
        stationary_velocity_mask = (
            velocity_scale <= velocity_absolute_tolerance
        )

        max_scaled_vel_raw = float(
            np.max(scaled_velocity_error)
        )

        scaled_velocity_error_effective = (
            scaled_velocity_error.copy()
        )
        scaled_velocity_error_effective[
            stationary_velocity_mask
        ] = 0.0

        max_scaled_vel = float(
            np.max(scaled_velocity_error_effective)
        )

        # Diagnostic for the worst velocity-consistency comparison.
        # This is intentionally printed only when the raw acceptance
        # criterion would fail, so normal runs remain quiet.
        if max_scaled_vel_raw > 1.0:
            worst_flat = int(np.argmax(scaled_velocity_error))
            worst_cmp, worst_particle = np.unravel_index(
                worst_flat, scaled_velocity_error.shape
            )
            worst_species = worst_cmp + 1

            print("\n[velocity acceptance diagnostic]")
            print(f"  particle index          = {worst_particle}")
            print(
                f"  v[{kernel_solute_names[0]}]                = "
                f"{velocity_species[0, worst_particle]:.16e}"
            )
            print(
                f"  v[{kernel_solute_names[worst_species]}]                = "
                f"{velocity_species[worst_species, worst_particle]:.16e}"
            )
            print(
                f"  velocity mismatch       = "
                f"{velocity_error[worst_cmp, worst_particle]:.16e}"
            )
            print(
                f"  velocity scale          = "
                f"{velocity_scale[worst_cmp, worst_particle]:.16e}"
            )
            print(
                f"  absolute tolerance      = "
                f"{velocity_absolute_tolerance:.16e}"
            )
            print(
                f"  relative contribution   = "
                f"{velocity_mismatch_tolerance * velocity_scale[worst_cmp, worst_particle]:.16e}"
            )
            print(
                f"  total allowance         = "
                f"{velocity_allowance[worst_cmp, worst_particle]:.16e}"
            )
            print(
                f"  stationary comparison   = "
                f"{bool(stationary_velocity_mask[worst_cmp, worst_particle])}"
            )
            print(
                f"  raw scaled error        = "
                f"{scaled_velocity_error[worst_cmp, worst_particle]:.16e}"
            )
    else:
        max_scaled_vel_raw = 0.0
        max_scaled_vel = 0.0

    success = bool(
        opt.success
        and max_therm <= thermodynamic_tolerance
        and max_scaled_vel <= 1.0
    )

    message = str(opt.message)

    if nonlinear_diagnostics:
        print(
            "[nonlinear-inner] "
            f"N={n} prescribed_bg={prescribed_background_mode} "
            f"refresh={lsmr_preconditioner_refresh_nfev} "
            f"lsmr_maxiter={lsmr_tr_maxiter} "
            f"max_chunks={lsmr_max_refresh_chunks} "
            f"success={success} opt_success={bool(opt.success)} "
            f"status={getattr(opt, 'status', None)} "
            f"nfev={int(opt.nfev)} njev={getattr(opt, 'njev', None)} "
            f"ls_wall={_nd['least_squares_wall']:.3f}s "
            f"eval={_nd['evaluate_calls']}/{_nd['evaluate_wall']:.3f}s "
            f"Jbuild={_nd['jacobian_builds']}/{_nd['jacobian_build_wall']:.3f}s "
            f"Jmv={_nd['jac_matvec_calls']}/{_nd['jac_matvec_wall']:.3f}s "
            f"Jrmv={_nd['jac_rmatvec_calls']}/{_nd['jac_rmatvec_wall']:.3f}s"
        )
        print(
            "[nonlinear-inner-linear] "
            f"solve_calls={_nd['linear_solve_calls']} "
            f"solve_rhs={_nd['linear_solve_rhs']} "
            f"solve_wall={_nd['linear_solve_wall']:.3f}s "
            f"solveT_calls={_nd['linear_solve_transpose_calls']} "
            f"solveT_rhs={_nd['linear_solve_transpose_rhs']} "
            f"solveT_wall={_nd['linear_solve_transpose_wall']:.3f}s "
            f"cost={float(opt.cost):.3e} "
            f"optimality={float(opt.optimality):.3e} "
            f"max_therm={max_therm:.3e} "
            f"max_scaled_vel={max_scaled_vel:.3e}"
        )

    if not success:
        message = (
            "Coupled interface solve failed acceptance checks: "
            f"{opt.message}; "
            f"max thermodynamic residual={max_therm:.3e}; "
            f"max relative velocity residual={max_vel:.3e}; "
            f"max scaled velocity error={max_scaled_vel:.3e}; "
            f"raw max scaled velocity error={max_scaled_vel_raw:.3e}."
        )

        if strict:
            raise RuntimeError(message)

    # All species describe the same moving interface.  Once the kinetic
    # constraints are satisfied, their arithmetic mean is the common
    # interface velocity.  For S=2 this is the legacy 0.5*(v0 + v1).
    dRdt = np.mean(velocity_species, axis=0)

    return CoupledTernarySolveResult(
        solute_names=kernel_solute_names,
        Q=np.asarray(last["Q"], dtype=float).copy(),
        cbar=np.asarray(last["cbar"], dtype=float).copy(),
        c_surf=np.asarray(last["c_surf"], dtype=float).copy(),
        dRdt=dRdt.copy(),
        dRdt_species=velocity_species.copy(),
        success=success,
        message=message,
        nfev=int(opt.nfev),
        cost=float(opt.cost),
        max_thermodynamic_residual=max_therm,
        max_relative_velocity_mismatch=max_vel,
        optimality=float(opt.optimality),
        final_lambda=None,
        continuation_history=None,
    )


def solve_ternary_interface_coupled(
    *,
    R,
    G0,
    Gpair,
    cbar_mg,
    cbar_si,
    matrix_integrals,
    matrix_volume,
    particle_volume,
    M_total_mg,
    M_total_si,
    cp_mg,
    cp_si,
    D_mg,
    D_si,
    T,
    vatm,
    vatp,
    gamma,
    thermodynamics=None,
    stoichiometric_weights=None,
    fixed_green_inventory_mg=0.0,
    fixed_green_inventory_si=0.0,
    c_surf_mg_init=None,
    c_surf_si_init=None,
    Q_mg_init=None,
    Q_si_init=None,
    **kwargs,
):
    """Compatibility wrapper for the legacy Mg/Si coupled-solver API.

    New numerical code should call ``solve_interface_coupled`` directly
    with species-indexed arrays.  This wrapper preserves compatibility with
    existing two-solute Mg/Si call sites.
    """
    n = np.asarray(R, dtype=float).size

    if c_surf_mg_init is None and c_surf_si_init is None:
        c_surf_init = None
    elif c_surf_mg_init is not None and c_surf_si_init is not None:
        c_surf_init = np.vstack(
            (
                _validate_optional_vector(
                    c_surf_mg_init, n, "c_surf_mg_init"
                ),
                _validate_optional_vector(
                    c_surf_si_init, n, "c_surf_si_init"
                ),
            )
        )
    else:
        raise ValueError(
            "c_surf_mg_init and c_surf_si_init must either both be "
            "supplied or both be None."
        )

    if Q_mg_init is None and Q_si_init is None:
        Q_init = None
    elif Q_mg_init is not None and Q_si_init is not None:
        Q_init = np.vstack(
            (
                _validate_optional_vector(
                    Q_mg_init, n, "Q_mg_init"
                ),
                _validate_optional_vector(
                    Q_si_init, n, "Q_si_init"
                ),
            )
        )
    else:
        raise ValueError(
            "Q_mg_init and Q_si_init must either both be supplied "
            "or both be None."
        )

    bound_thermodynamics = ParticlePhaseThermodynamics(
        phase_id=np.zeros(n, dtype=np.int32),
        models=(thermodynamics,),
        precipitate_atomic_volumes=np.asarray((vatp,), dtype=float),
        gammas=np.asarray((gamma,), dtype=float),
        stoichiometric_weights=np.asarray(
            stoichiometric_weights, dtype=float
        )[None, :],
    )

    return solve_interface_coupled(
        R=R,
        G0=G0,
        Gpair=Gpair,
        solute_names=("Mg", "Si"),
        cbar=np.asarray((cbar_mg, cbar_si), dtype=float),
        matrix_integrals=matrix_integrals,
        matrix_volume=matrix_volume,
        particle_volume=particle_volume,
        M_total=np.asarray(
            (M_total_mg, M_total_si), dtype=float
        ),
        cp=np.asarray((cp_mg, cp_si), dtype=float),
        D=np.asarray((D_mg, D_si), dtype=float),
        T=T,
        vatm=vatm,
        thermodynamics=bound_thermodynamics,
        fixed_green_inventory=np.asarray(
            (
                fixed_green_inventory_mg,
                fixed_green_inventory_si,
            ),
            dtype=float,
        ),
        c_surf_init=c_surf_init,
        Q_init=Q_init,
        **kwargs,
    )

def initialize_ternary_interfaces_kwn(
    *,
    R,
    cbar_mg,
    cbar_si,
    cp_mg,
    cp_si,
    D_mg,
    D_si,
    T,
    vatm,
    vatp,
    gamma,
    thermodynamics,
    stoichiometric_weights,
    concentration_margin=1e-10,
    bracket_points=500,
    root_xtol=1e-13,
    root_rtol=1e-12,
):
    """
    Construct interfacial concentration guesses by solving the
    isolated-particle ternary KWN problem independently for each radius.

    For an isolated spherical particle,

        dR/dt =
            D / R * (cbar - csurf) / (cp - csurf).

    The Mg and Si growth rates are constrained to be equal, while the
    surface concentrations satisfy the curvature-modified solubility
    product.

    Returns
    -------
    c_surf_mg : ndarray, shape (N,)
    c_surf_si : ndarray, shape (N,)
    dRdt      : ndarray, shape (N,)
    """
    R = np.asarray(R, dtype=float)

    if R.ndim != 1:
        raise ValueError("R must be one-dimensional.")

    if np.any(~np.isfinite(R)) or np.any(R <= 0.0):
        raise ValueError(
            "All active radii must be finite and positive."
        )

    if not (0.0 < concentration_margin < 0.1):
        raise ValueError(
            "concentration_margin must lie in (0, 0.1)."
        )

    if bracket_points < 10:
        raise ValueError(
            "bracket_points must be at least 10."
        )

    if thermodynamics is None:
        raise ValueError(
            "A phase thermodynamic model must be supplied through "
            "thermodynamics."
        )

    stoichiometric_weights = np.asarray(
        stoichiometric_weights,
        dtype=float,
    )
    if stoichiometric_weights.shape != (2,):
        raise ValueError(
            "The current isolated two-solute KWN initializer requires "
            "stoichiometric_weights with shape (2,)."
        )
    if (
        np.any(~np.isfinite(stoichiometric_weights))
        or np.any(stoichiometric_weights <= 0.0)
    ):
        raise ValueError(
            "stoichiometric_weights must contain finite positive values."
        )

    equilibrium = thermodynamics.prepare_equilibrium(
        radius=R,
        temperature=T,
        matrix_atomic_volume=vatm,
        precipitate_atomic_volume=vatp,
        gamma=gamma,
        stoichiometric_weights=stoichiometric_weights,
    )

    n = len(R)

    c_surf_mg = np.empty(n, dtype=float)
    c_surf_si = np.empty(n, dtype=float)
    dRdt = np.empty(n, dtype=float)

    margin = concentration_margin

    for i in range(n):
        Ri = R[i]

        def si_from_mg(ci_mg):
            """Obtain species-1 concentration from cached phase equilibrium."""
            return equilibrium.dependent_scalar(
                particle_index=i,
                independent_concentration=ci_mg,
                independent_index=0,
                dependent_index=1,
            )

        def velocity_difference(ci_mg):
            ci_si = si_from_mg(ci_mg)

            v_mg = (
                D_mg / Ri
                * (cbar_mg - ci_mg)
                / (cp_mg - ci_mg)
            )

            v_si = (
                D_si / Ri
                * (cbar_si - ci_si)
                / (cp_si - ci_si)
            )

            return v_mg - v_si

        # Determine the Mg interval associated with
        #
        #   margin*cp_si < CiSi < (1-margin)*cp_si
        #
        # while also enforcing the Mg bounds.
        ci_si_upper = (1.0 - margin) * cp_si
        ci_si_lower = margin * cp_si

        ci_mg_lower_from_si = equilibrium.dependent_scalar(
            particle_index=i,
            independent_concentration=ci_si_upper,
            independent_index=1,
            dependent_index=0,
        )

        ci_mg_upper_from_si = equilibrium.dependent_scalar(
            particle_index=i,
            independent_concentration=ci_si_lower,
            independent_index=1,
            dependent_index=0,
        )

        ci_mg_lower = max(
            ci_mg_lower_from_si,
            margin * cp_mg,
        )

        ci_mg_upper = min(
            ci_mg_upper_from_si,
            (1.0 - margin) * cp_mg,
        )

        if not (
            np.isfinite(ci_mg_lower)
            and np.isfinite(ci_mg_upper)
            and ci_mg_lower < ci_mg_upper
        ):
            raise RuntimeError(
                "No physically admissible isolated KWN interval "
                f"for particle {i}, R={Ri:.8e}. "
                f"Computed Mg interval "
                f"[{ci_mg_lower:.8e}, {ci_mg_upper:.8e}]."
            )

        # A logarithmic scan is much more robust than assuming the
        # endpoints themselves bracket the root.
        grid = np.geomspace(
            ci_mg_lower,
            ci_mg_upper,
            bracket_points,
        )

        fgrid = np.array(
            [velocity_difference(value) for value in grid],
            dtype=float,
        )

        finite = np.isfinite(fgrid)

        if not np.all(finite):
            grid = grid[finite]
            fgrid = fgrid[finite]

        if len(grid) < 2:
            raise RuntimeError(
                "Too few finite KWN bracket points for "
                f"particle {i}, R={Ri:.8e}."
            )

        # Accept an exact sampled root if one occurs.
        exact = np.where(fgrid == 0.0)[0]

        if exact.size > 0:
            ci_mg = grid[exact[0]]

        else:
            sign_changes = np.where(
                np.signbit(fgrid[:-1])
                != np.signbit(fgrid[1:])
            )[0]

            if sign_changes.size == 0:
                j_best = int(np.argmin(np.abs(fgrid)))

                raise RuntimeError(
                    "Could not bracket the isolated ternary KWN root "
                    f"for particle {i}, R={Ri:.8e}. "
                    f"Smallest sampled velocity residual was "
                    f"{fgrid[j_best]:.8e} at "
                    f"CiMg={grid[j_best]:.8e}."
                )

            j = sign_changes[0]

            ci_mg = brentq(
                velocity_difference,
                grid[j],
                grid[j + 1],
                xtol=root_xtol,
                rtol=root_rtol,
                maxiter=300,
            )

        ci_si = si_from_mg(ci_mg)

        v_mg = (
            D_mg / Ri
            * (cbar_mg - ci_mg)
            / (cp_mg - ci_mg)
        )

        v_si = (
            D_si / Ri
            * (cbar_si - ci_si)
            / (cp_si - ci_si)
        )

        c_surf_mg[i] = ci_mg
        c_surf_si[i] = ci_si
        dRdt[i] = 0.5 * (v_mg + v_si)

    return c_surf_mg, c_surf_si, dRdt

def _solve_interface_with_continuation_once(
    *,
    R,
    G0,
    Gpair,
    solute_names,
    cbar,
    matrix_integrals,
    matrix_volume,
    particle_volume,
    M_total,
    cp,
    D,
    T,
    vatm,
    thermodynamics=None,
    fixed_green_inventory=None,
    c_surf_isolated_init=None,
    initial_lambda_step=0.20,
    minimum_lambda_step=1e-3,
    maximum_lambda_step=0.50,
    lambda_growth_factor=1.5,
    lambda_shrink_factor=0.5,
    max_continuation_attempts=100,
    verbose=False,
    nonlinear_solver="lsmr",
    **solver_kwargs,
):
    """Solve the species-indexed interface problem by interaction continuation.

    The regular periodic interaction is introduced as

        G0(lambda)    = lambda * G0
        Gpair(lambda) = lambda * Gpair,

    while the singular self term remains present.  All composition,
    diffusivity, inventory, surface-concentration, and monopole states carry
    an explicit species axis.  No species name has numerical significance.

    ``c_surf_isolated_init`` is an optional ``(S, N)`` warm start for the
    lambda=0 solve.  Compatibility wrappers may construct it with a
    chemistry-specific isolated-particle initializer; the continuation
    algorithm itself is chemistry independent.
    """
    R = np.asarray(R, dtype=float)
    Gpair = np.asarray(Gpair, dtype=float)
    matrix_integrals = np.asarray(matrix_integrals, dtype=float)
    solute_names = tuple(str(name) for name in solute_names)

    if R.ndim != 1:
        raise ValueError("R must be one-dimensional.")

    n = R.size
    s = len(solute_names)
    if s < 1:
        raise ValueError("At least one diffusing solute is required.")
    if len(set(solute_names)) != s:
        raise ValueError("solute_names must contain unique names.")

    if Gpair.shape != (n, n):
        raise ValueError(
            f"Gpair must have shape {(n, n)}, got {Gpair.shape}."
        )
    if matrix_integrals.shape != (n,):
        raise ValueError(
            "matrix_integrals must have shape "
            f"({n},), got {matrix_integrals.shape}."
        )
    if np.any(~np.isfinite(matrix_integrals)):
        raise ValueError("matrix_integrals contains non-finite values.")

    cbar = np.asarray(cbar, dtype=float)
    M_total = np.asarray(M_total, dtype=float)
    cp = np.asarray(cp, dtype=float)
    D = np.asarray(D, dtype=float)
    if fixed_green_inventory is None:
        fixed_green_inventory = np.zeros(s, dtype=float)
    else:
        fixed_green_inventory = np.asarray(
            fixed_green_inventory, dtype=float
        )

    for name, values in (
        ("cbar", cbar),
        ("M_total", M_total),
        ("D", D),
        ("fixed_green_inventory", fixed_green_inventory),
    ):
        if values.shape != (s,):
            raise ValueError(
                f"{name} must have shape ({s},), got {values.shape}."
            )
        if np.any(~np.isfinite(values)):
            raise ValueError(f"{name} contains non-finite values.")

    # Precipitate composition may be shared by all particles, (S,), or
    # particle-specific, (S,N).  Continuation does not alter particle
    # identity/composition, so the same array is forwarded unchanged to
    # every lambda solve.
    if cp.shape not in ((s,), (s, n)):
        raise ValueError(
            "cp must have shape "
            f"({s},) or ({s}, {n}), got {cp.shape}."
        )
    if np.any(~np.isfinite(cp)):
        raise ValueError("cp contains non-finite values.")

    if c_surf_isolated_init is not None:
        c_surf_isolated_init = np.asarray(
            c_surf_isolated_init, dtype=float
        )
        if c_surf_isolated_init.shape != (s, n):
            raise ValueError(
                "c_surf_isolated_init must have shape "
                f"({s}, {n}), got {c_surf_isolated_init.shape}."
            )
        if np.any(~np.isfinite(c_surf_isolated_init)):
            raise ValueError(
                "c_surf_isolated_init contains non-finite values."
            )

    if not (0.0 < initial_lambda_step <= 1.0):
        raise ValueError("initial_lambda_step must lie in (0, 1].")
    if not (0.0 < minimum_lambda_step <= initial_lambda_step):
        raise ValueError(
            "minimum_lambda_step must be positive and no larger "
            "than initial_lambda_step."
        )
    if not (initial_lambda_step <= maximum_lambda_step <= 1.0):
        raise ValueError(
            "maximum_lambda_step must lie between "
            "initial_lambda_step and 1."
        )
    if lambda_growth_factor <= 1.0:
        raise ValueError("lambda_growth_factor must be greater than 1.")
    if not (0.0 < lambda_shrink_factor < 1.0):
        raise ValueError("lambda_shrink_factor must lie in (0, 1).")
    if max_continuation_attempts < 1:
        raise ValueError("max_continuation_attempts must be positive.")

    matrix_integrals_isolated = -0.5 * R**2

    def matrix_integrals_at_lambda(lam):
        lam = float(lam)
        if not np.isfinite(lam):
            raise ValueError("Continuation lambda must be finite.")
        if lam < 0.0 or lam > 1.0:
            raise ValueError("Continuation lambda must lie in [0, 1].")
        return matrix_integrals_isolated + lam * (
            matrix_integrals - matrix_integrals_isolated
        )

    common_solver_kwargs = dict(
        solute_names=solute_names,
        cbar=cbar,
        matrix_volume=matrix_volume,
        particle_volume=particle_volume,
        M_total=M_total,
        cp=cp,
        D=D,
        T=T,
        vatm=vatm,
        thermodynamics=thermodynamics,
        fixed_green_inventory=fixed_green_inventory,
    )

    clean_kwargs = dict(solver_kwargs)
    # Continuation owns its warm starts and topology cache.
    for key in (
        "c_surf_init",
        "Q_init",
        "_neighbor_preconditioner_cache",
    ):
        clean_kwargs.pop(key, None)

    neighbor_preconditioner_cache = {}

    if n == 0:
        result = solve_interface_coupled(
            R=R,
            G0=G0,
            Gpair=Gpair,
            matrix_integrals=matrix_integrals,
            nonlinear_solver=nonlinear_solver,
            **common_solver_kwargs,
            **clean_kwargs,
        )
        result.final_lambda = 1.0
        result.continuation_history = []
        return result

    current = solve_interface_coupled(
        R=R,
        G0=0.0,
        Gpair=np.zeros_like(Gpair),
        matrix_integrals=matrix_integrals_at_lambda(0.0),
        c_surf_init=c_surf_isolated_init,
        Q_init=None,
        nonlinear_solver=nonlinear_solver,
        _neighbor_preconditioner_cache=neighbor_preconditioner_cache,
        **common_solver_kwargs,
        **clean_kwargs,
    )

    history = [{
        "lambda": 0.0,
        "step": 0.0,
        "accepted": True,
        "nfev": current.nfev,
        "cost": current.cost,
        "max_thermodynamic_residual":
            current.max_thermodynamic_residual,
        "max_relative_velocity_mismatch":
            current.max_relative_velocity_mismatch,
    }]

    if verbose:
        print(
            "Accepted lambda=0.000000 "
            f"(nfev={current.nfev}, "
            f"therm={current.max_thermodynamic_residual:.3e}, "
            f"vel={current.max_relative_velocity_mismatch:.3e})"
        )

    lam = 0.0
    step = initial_lambda_step
    attempts = 0

    while lam < 1.0 - 1e-14 and attempts < max_continuation_attempts:
        attempts += 1
        trial_lam = min(1.0, lam + step)

        try:
            trial = solve_interface_coupled(
                R=R,
                G0=trial_lam * G0,
                Gpair=trial_lam * Gpair,
                matrix_integrals=matrix_integrals_at_lambda(trial_lam),
                c_surf_init=current.c_surf,
                Q_init=current.Q,
                nonlinear_solver=nonlinear_solver,
                _neighbor_preconditioner_cache=neighbor_preconditioner_cache,
                **common_solver_kwargs,
                **clean_kwargs,
            )
        except (RuntimeError, ValueError) as exc:
            history.append({
                "lambda": trial_lam,
                "step": step,
                "accepted": False,
                "error": str(exc),
            })
            if verbose:
                print(
                    f"Rejected lambda={trial_lam:.6f}; "
                    f"reducing step from {step:.6g}."
                )
            step *= lambda_shrink_factor
            if step < minimum_lambda_step:
                raise RuntimeError(
                    "Interaction continuation failed at "
                    f"lambda={lam:.8f}. Required continuation step "
                    "is below minimum_lambda_step="
                    f"{minimum_lambda_step:.3e}."
                ) from exc
            continue

        current = trial
        lam = trial_lam
        history.append({
            "lambda": lam,
            "step": step,
            "accepted": True,
            "nfev": current.nfev,
            "cost": current.cost,
            "max_thermodynamic_residual":
                current.max_thermodynamic_residual,
            "max_relative_velocity_mismatch":
                current.max_relative_velocity_mismatch,
        })

        if verbose:
            print(
                f"Accepted lambda={lam:.6f} "
                f"(nfev={current.nfev}, "
                f"therm={current.max_thermodynamic_residual:.3e}, "
                f"vel={current.max_relative_velocity_mismatch:.3e})"
            )

        step = min(maximum_lambda_step, step * lambda_growth_factor)

    if lam < 1.0 - 1e-14:
        raise RuntimeError(
            "Maximum number of interaction continuation attempts "
            "was exceeded before reaching the full interaction "
            f"strength. Final lambda={lam:.8f}, attempts={attempts}."
        )

    current.final_lambda = lam
    current.continuation_history = history
    return current


def solve_interface_with_continuation(
    *,
    R,
    G0,
    Gpair,
    solute_names,
    cbar,
    matrix_integrals,
    matrix_volume,
    particle_volume,
    M_total,
    cp,
    D,
    T,
    vatm,
    thermodynamics=None,
    fixed_green_inventory=None,
    c_surf_isolated_init=None,
    initial_lambda_step=0.20,
    minimum_lambda_step=1e-3,
    maximum_lambda_step=0.50,
    lambda_growth_factor=1.5,
    lambda_shrink_factor=0.5,
    max_continuation_attempts=100,
    verbose=False,
    solver_preference="auto",
    **solver_kwargs,
):
    """Species-indexed continuation with LSMR-to-dense fallback."""
    if solver_preference not in ("auto", "dense"):
        raise ValueError("solver_preference must be 'auto' or 'dense'.")

    continuation_kwargs = dict(
        R=R,
        G0=G0,
        Gpair=Gpair,
        solute_names=solute_names,
        cbar=cbar,
        matrix_integrals=matrix_integrals,
        matrix_volume=matrix_volume,
        particle_volume=particle_volume,
        M_total=M_total,
        cp=cp,
        D=D,
        T=T,
        vatm=vatm,
        thermodynamics=thermodynamics,
        fixed_green_inventory=fixed_green_inventory,
        c_surf_isolated_init=c_surf_isolated_init,
        initial_lambda_step=initial_lambda_step,
        minimum_lambda_step=minimum_lambda_step,
        maximum_lambda_step=maximum_lambda_step,
        lambda_growth_factor=lambda_growth_factor,
        lambda_shrink_factor=lambda_shrink_factor,
        max_continuation_attempts=max_continuation_attempts,
        verbose=verbose,
        **solver_kwargs,
    )

    if solver_preference == "dense":
        result = _solve_interface_with_continuation_once(
            nonlinear_solver="dense", **continuation_kwargs
        )
        result.nonlinear_solver_used = "dense"
        result.dense_fallback_used = False
        return result

    try:
        result = _solve_interface_with_continuation_once(
            nonlinear_solver="lsmr", **continuation_kwargs
        )
        result.nonlinear_solver_used = "lsmr"
        result.dense_fallback_used = False
        return result
    except RuntimeError as lsmr_error:
        print(
            "LSMR continuation failed; "
            "retrying entire continuation with dense solver."
        )
        try:
            result = _solve_interface_with_continuation_once(
                nonlinear_solver="dense", **continuation_kwargs
            )
            result.nonlinear_solver_used = "dense"
            result.dense_fallback_used = True
            return result
        except RuntimeError as dense_error:
            raise RuntimeError(
                "Both LSMR and dense interaction continuation failed.\n"
                f"LSMR failure: {lsmr_error}\n"
                f"Dense failure: {dense_error}"
            ) from dense_error


def solve_ternary_interface_with_continuation(
    *,
    R,
    G0,
    Gpair,
    cbar_mg,
    cbar_si,
    matrix_integrals,
    matrix_volume,
    particle_volume,
    M_total_mg,
    M_total_si,
    fixed_green_inventory_mg=0.0,
    fixed_green_inventory_si=0.0,
    initial_lambda_step=0.20,
    minimum_lambda_step=1e-3,
    maximum_lambda_step=0.50,
    lambda_growth_factor=1.5,
    lambda_shrink_factor=0.5,
    max_continuation_attempts=100,
    verbose=False,
    solver_preference="auto",
    **solver_kwargs,
):
    """Compatibility wrapper for the legacy Mg/Si continuation API.

    The continuation implementation is species indexed.  This wrapper
    translates the two-solute Mg/Si compatibility interface and preserves
    its isolated KWN initialization exactly.
    """
    required_initializer_names = (
        "cp_mg",
        "cp_si",
        "D_mg",
        "D_si",
        "T",
        "vatm",
        "vatp",
        "gamma",
        "thermodynamics",
        "stoichiometric_weights",
    )
    missing = [
        name for name in required_initializer_names if name not in solver_kwargs
    ]
    if np.asarray(R, dtype=float).size > 0 and missing:
        raise ValueError(
            "Missing parameters required for KWN initialization: "
            + ", ".join(missing)
        )

    clean_kwargs = dict(solver_kwargs)
    # Legacy continuation ignored externally supplied warm starts because it
    # constructs a consistent lambda=0 state itself.
    for key in (
        "c_surf_mg_init",
        "c_surf_si_init",
        "Q_mg_init",
        "Q_si_init",
        "_neighbor_preconditioner_cache",
    ):
        clean_kwargs.pop(key, None)

    cp_mg = clean_kwargs.pop("cp_mg")
    cp_si = clean_kwargs.pop("cp_si")
    D_mg = clean_kwargs.pop("D_mg")
    D_si = clean_kwargs.pop("D_si")
    T = clean_kwargs.pop("T")
    vatm = clean_kwargs.pop("vatm")
    vatp = clean_kwargs.pop("vatp")
    gamma = clean_kwargs.pop("gamma")
    thermodynamics = clean_kwargs.pop("thermodynamics")
    stoichiometric_weights = clean_kwargs.pop("stoichiometric_weights")

    R_array = np.asarray(R, dtype=float)
    c_surf_isolated_init = None
    if R_array.size > 0:
        initializer_kwargs = dict(
            R=R_array,
            cbar_mg=cbar_mg,
            cbar_si=cbar_si,
            cp_mg=cp_mg,
            cp_si=cp_si,
            D_mg=D_mg,
            D_si=D_si,
            T=T,
            vatm=vatm,
            vatp=vatp,
            gamma=gamma,
            thermodynamics=thermodynamics,
            stoichiometric_weights=stoichiometric_weights,
        )
        if "concentration_margin" in clean_kwargs:
            initializer_kwargs["concentration_margin"] = clean_kwargs[
                "concentration_margin"
            ]
        csmg0, cssi0, _ = initialize_ternary_interfaces_kwn(
            **initializer_kwargs
        )
        c_surf_isolated_init = np.vstack((csmg0, cssi0))

    bound_thermodynamics = ParticlePhaseThermodynamics(
        phase_id=np.zeros(R_array.size, dtype=np.int32),
        models=(thermodynamics,),
        precipitate_atomic_volumes=np.asarray((vatp,), dtype=float),
        gammas=np.asarray((gamma,), dtype=float),
        stoichiometric_weights=np.asarray(
            stoichiometric_weights, dtype=float
        )[None, :],
    )

    return solve_interface_with_continuation(
        R=R,
        G0=G0,
        Gpair=Gpair,
        solute_names=("Mg", "Si"),
        cbar=np.asarray((cbar_mg, cbar_si), dtype=float),
        matrix_integrals=matrix_integrals,
        matrix_volume=matrix_volume,
        particle_volume=particle_volume,
        M_total=np.asarray((M_total_mg, M_total_si), dtype=float),
        cp=np.asarray((cp_mg, cp_si), dtype=float),
        D=np.asarray((D_mg, D_si), dtype=float),
        T=T,
        vatm=vatm,
        thermodynamics=bound_thermodynamics,
        fixed_green_inventory=np.asarray(
            (fixed_green_inventory_mg, fixed_green_inventory_si),
            dtype=float,
        ),
        c_surf_isolated_init=c_surf_isolated_init,
        initial_lambda_step=initial_lambda_step,
        minimum_lambda_step=minimum_lambda_step,
        maximum_lambda_step=maximum_lambda_step,
        lambda_growth_factor=lambda_growth_factor,
        lambda_shrink_factor=lambda_shrink_factor,
        max_continuation_attempts=max_continuation_attempts,
        verbose=verbose,
        solver_preference=solver_preference,
        **clean_kwargs,
    )

def initialize_interfaces_kwn(
    *,
    R,
    cbar,
    cp,
    D,
    T,
    vatm,
    vatp,
    gamma,
    thermodynamics,
    stoichiometric_weights,
    concentration_margin=1e-10,
    bracket_points=500,
    root_xtol=1e-13,
    root_rtol=1e-12,
):
    """Species-indexed isolated-particle KWN interface initializer.

    Parameters use the canonical species axis: ``cbar``, ``cp``, ``D``, and
    ``stoichiometric_weights`` all have shape ``(S,)``.  The returned surface
    concentration has shape ``(S, N)`` and the returned radius velocity has
    shape ``(N,)``.

    For ``S == 2`` this function deliberately dispatches to the established
    two-solute initializer so existing Mg-Si calculations retain their exact
    regression path.  For other ``S`` values the isolated equations are solved
    independently for each particle: one thermodynamic constraint plus
    ``S-1`` equal-interface-velocity constraints.
    """
    R = np.asarray(R, dtype=float)
    cbar = np.asarray(cbar, dtype=float)
    cp = np.asarray(cp, dtype=float)
    D = np.asarray(D, dtype=float)
    weights = np.asarray(stoichiometric_weights, dtype=float)

    if R.ndim != 1:
        raise ValueError("R must be one-dimensional.")
    if np.any(~np.isfinite(R)) or np.any(R <= 0.0):
        raise ValueError("All active radii must be finite and positive.")

    if cbar.ndim != 1 or cbar.size == 0:
        raise ValueError("cbar must have shape (S,) with S >= 1.")
    n_solutes = cbar.size
    for name, values in (("cp", cp), ("D", D), ("stoichiometric_weights", weights)):
        if values.shape != (n_solutes,):
            raise ValueError(
                f"{name} must have shape ({n_solutes},); got {values.shape}."
            )
        if np.any(~np.isfinite(values)):
            raise ValueError(f"{name} contains non-finite values.")

    if np.any(cp <= 0.0):
        raise ValueError("cp must contain positive values.")
    if np.any(D <= 0.0):
        raise ValueError("D must contain positive values.")
    if np.any(weights <= 0.0):
        raise ValueError("stoichiometric_weights must contain positive values.")
    if not (0.0 < concentration_margin < 0.1):
        raise ValueError("concentration_margin must lie in (0, 0.1).")
    if thermodynamics is None:
        raise ValueError("thermodynamics must be supplied.")

    # Preserve the exact established two-solute initialization path.
    if n_solutes == 2:
        cs0, cs1, dRdt = initialize_ternary_interfaces_kwn(
            R=R,
            cbar_mg=float(cbar[0]),
            cbar_si=float(cbar[1]),
            cp_mg=float(cp[0]),
            cp_si=float(cp[1]),
            D_mg=float(D[0]),
            D_si=float(D[1]),
            T=T,
            vatm=vatm,
            vatp=vatp,
            gamma=gamma,
            thermodynamics=thermodynamics,
            stoichiometric_weights=weights,
            concentration_margin=concentration_margin,
            bracket_points=bracket_points,
            root_xtol=root_xtol,
            root_rtol=root_rtol,
        )
        return np.vstack((cs0, cs1)), np.asarray(dRdt, dtype=float)

    n = R.size
    c_surf = np.empty((n_solutes, n), dtype=float)
    dRdt = np.empty(n, dtype=float)
    margin = float(concentration_margin)
    lower = margin * cp
    upper = (1.0 - margin) * cp

    def _logistic(z):
        # Stable logistic map from R to the bounded physical interval.
        z = np.asarray(z, dtype=float)
        sig = np.empty_like(z)
        pos = z >= 0.0
        sig[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        ez = np.exp(z[~pos])
        sig[~pos] = ez / (1.0 + ez)
        return lower + (upper - lower) * sig

    # A bounded starting point based on the matrix composition is usually
    # closer to the physical branch than an arbitrary mid-interval value.
    frac0 = np.clip(
        (np.clip(cbar, lower, upper) - lower) / (upper - lower),
        1e-12,
        1.0 - 1e-12,
    )
    z0 = np.log(frac0 / (1.0 - frac0))

    for i, Ri in enumerate(R):
        def residual(z):
            cs = _logistic(z)
            thermo = np.asarray(
                thermodynamics.residual(
                    c_surf=cs[:, None],
                    radius=np.asarray([Ri], dtype=float),
                    temperature=T,
                    matrix_atomic_volume=vatm,
                    precipitate_atomic_volume=vatp,
                    gamma=gamma,
                    stoichiometric_weights=weights,
                ),
                dtype=float,
            )
            if thermo.shape != (1,):
                raise RuntimeError(
                    "Thermodynamic model returned an unexpected residual shape "
                    f"{thermo.shape}; expected (1,)."
                )

            velocities = (
                D / Ri * (cbar - cs) / (cp - cs)
            )
            if n_solutes == 1:
                return thermo

            scale = np.maximum.reduce(
                (
                    np.abs(velocities[0]) * np.ones(n_solutes - 1),
                    np.abs(velocities[1:]),
                    np.full(n_solutes - 1, 1e-30),
                )
            )
            kinetic = (velocities[0] - velocities[1:]) / scale
            return np.concatenate((thermo, kinetic))

        opt = least_squares(
            residual,
            z0,
            xtol=root_xtol,
            ftol=root_xtol,
            gtol=root_xtol,
            max_nfev=max(200, 20 * n_solutes),
        )
        cs = _logistic(opt.x)
        res = residual(opt.x)
        if (
            not opt.success
            or np.any(~np.isfinite(cs))
            or np.any(~np.isfinite(res))
            or np.max(np.abs(res)) > 1e-7
        ):
            raise RuntimeError(
                "Could not solve the isolated species-indexed KWN state for "
                f"particle {i}, R={Ri:.8e}; success={opt.success}, "
                f"max residual={np.max(np.abs(res)):.3e}."
            )

        velocities = D / Ri * (cbar - cs) / (cp - cs)
        c_surf[:, i] = cs
        dRdt[i] = float(np.mean(velocities))

    return c_surf, dRdt

