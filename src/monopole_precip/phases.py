from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

import numpy as np


class PhaseThermodynamics(Protocol):
    """Interface implemented by phase-specific thermodynamic models."""

    def residual(
        self,
        *,
        c_surf: np.ndarray,
        radius: np.ndarray,
        temperature: float,
        matrix_atomic_volume: float,
        precipitate_atomic_volume: float,
        gamma: float,
        stoichiometric_weights: np.ndarray,
    ) -> np.ndarray:
        ...

    def prepare_equilibrium(
        self,
        *,
        radius: np.ndarray,
        temperature: float,
        matrix_atomic_volume: float,
        precipitate_atomic_volume: float,
        gamma: float,
        stoichiometric_weights: np.ndarray,
    ):
        """Precompute radius-dependent equilibrium data for hot-loop use."""
        ...

    def dependent_concentration(
        self,
        *,
        independent_concentration: np.ndarray,
        independent_index: int,
        dependent_index: int,
        radius: np.ndarray,
        temperature: float,
        matrix_atomic_volume: float,
        precipitate_atomic_volume: float,
        gamma: float,
        stoichiometric_weights: np.ndarray,
    ) -> np.ndarray:
        """Solve the phase-equilibrium constraint for one concentration."""
        ...

    def directional_derivative(
        self,
        *,
        c_surf: np.ndarray,
        delta_c_surf: np.ndarray,
        radius: np.ndarray,
        temperature: float,
        matrix_atomic_volume: float,
        precipitate_atomic_volume: float,
        gamma: float,
        stoichiometric_weights: np.ndarray,
    ) -> np.ndarray:
        ...

    def transpose_contribution(
        self,
        *,
        weight_residual: np.ndarray,
        c_surf: np.ndarray,
        radius: np.ndarray,
        temperature: float,
        matrix_atomic_volume: float,
        precipitate_atomic_volume: float,
        gamma: float,
        stoichiometric_weights: np.ndarray,
    ) -> np.ndarray:
        ...


@dataclass(frozen=True)
class PreparedSolubilityProductEquilibrium:
    """Cached solubility-product equilibrium for a fixed particle set.

    This object is deliberately lightweight because ``dependent_scalar`` is
    used inside the isolated-particle KWN root-search hot loop.
    """

    log_KR: np.ndarray
    matrix_atomic_volume: float
    stoichiometric_weights: np.ndarray

    def dependent_scalar(
        self,
        *,
        particle_index: int,
        independent_concentration: float,
        independent_index: int,
        dependent_index: int,
    ) -> float:
        """Return one dependent concentration using cached equilibrium data."""
        vatm = self.matrix_atomic_volume
        weights = self.stoichiometric_weights
        nu_ind = weights[independent_index]
        nu_dep = weights[dependent_index]

        log_c_dep = (
            self.log_KR[particle_index]
            - nu_ind * np.log(independent_concentration * vatm)
        ) / nu_dep - np.log(vatm)

        return float(np.exp(log_c_dep))


@dataclass(frozen=True)
class SolubilityProductThermodynamics:
    """Generic line-compound solubility-product thermodynamics.

    The planar equilibrium product is

        log10 K(T) = bb - aa/T + cc/T^2

    and the interface equilibrium residual is

        [nu_A ln(X_A) + sum_s nu_s ln(c_s v_atm) - ln K_R] / nu_total,

    where X_A = 1 - sum_s(c_s v_atm) is the implicit solvent site fraction,
    nu_A is ``solvent_stoichiometric_weight``, and
    nu_total = nu_A + sum_s nu_s.

    Setting ``solvent_stoichiometric_weight=0`` preserves the established
    transported-solute-only Mg-Si formulation exactly.
    """

    aa: float
    bb: float
    cc: float = 0.0
    use_xy_factor: bool = True
    solvent_stoichiometric_weight: float = 0.0

    def log_curvature_equilibrium_product(
        self,
        *,
        radius: np.ndarray,
        temperature: float,
        precipitate_atomic_volume: float,
        gamma: float,
        stoichiometric_weights: np.ndarray,
    ) -> np.ndarray:
        radius = np.asarray(radius, dtype=float)
        weights = np.asarray(stoichiometric_weights, dtype=float)

        if radius.ndim != 1:
            raise ValueError("radius must be one-dimensional.")
        if np.any(~np.isfinite(radius)) or np.any(radius <= 0.0):
            raise ValueError("All radii must be finite and positive.")
        if not np.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("temperature must be finite and positive.")
        if (
            not np.isfinite(precipitate_atomic_volume)
            or precipitate_atomic_volume <= 0.0
        ):
            raise ValueError(
                "precipitate_atomic_volume must be finite and positive."
            )
        if not np.isfinite(gamma) or gamma < 0.0:
            raise ValueError("gamma must be finite and non-negative.")
        if (
            weights.ndim != 1
            or weights.size == 0
            or np.any(~np.isfinite(weights))
            or np.any(weights <= 0.0)
        ):
            raise ValueError(
                "stoichiometric_weights must be a positive 1D array."
            )

        log10_K = (
            float(self.bb)
            - float(self.aa) / float(temperature)
            + float(self.cc) / float(temperature) ** 2
        )
        log_K = np.log(10.0) * log10_K

        # Preserve the established Mg-Si arithmetic exactly when no
        # implicit solvent participates in the line-compound stoichiometry.
        solvent_weight = float(self.solvent_stoichiometric_weight)
        if not np.isfinite(solvent_weight) or solvent_weight < 0.0:
            raise ValueError(
                "solvent_stoichiometric_weight must be finite and non-negative."
            )
        nu_sum = float(np.sum(weights))
        if solvent_weight == 0.0:
            curvature_factor = nu_sum if self.use_xy_factor else 1.0
        else:
            nu_total = solvent_weight + nu_sum
            curvature_factor = nu_total if self.use_xy_factor else 1.0

        k_B = 1.380649e-23
        curvature = (
            curvature_factor
            * 2.0
            * float(gamma)
            * float(precipitate_atomic_volume)
            / (k_B * float(temperature) * radius)
        )

        return log_K + curvature

    def residual(
        self,
        *,
        c_surf: np.ndarray,
        radius: np.ndarray,
        temperature: float,
        matrix_atomic_volume: float,
        precipitate_atomic_volume: float,
        gamma: float,
        stoichiometric_weights: np.ndarray,
    ) -> np.ndarray:
        c_surf = np.asarray(c_surf, dtype=float)
        weights = np.asarray(stoichiometric_weights, dtype=float)

        if c_surf.ndim != 2:
            raise ValueError(
                "c_surf must have shape (n_species, n_particles)."
            )
        if weights.shape != (c_surf.shape[0],):
            raise ValueError(
                "stoichiometric_weights must match c_surf species dimension."
            )
        if np.any(~np.isfinite(c_surf)) or np.any(c_surf <= 0.0):
            raise ValueError(
                "All interfacial concentrations must be finite and positive."
            )
        if (
            not np.isfinite(matrix_atomic_volume)
            or matrix_atomic_volume <= 0.0
        ):
            raise ValueError(
                "matrix_atomic_volume must be finite and positive."
            )

        log_KR = self.log_curvature_equilibrium_product(
            radius=radius,
            temperature=temperature,
            precipitate_atomic_volume=precipitate_atomic_volume,
            gamma=gamma,
            stoichiometric_weights=weights,
        )

        solvent_weight = float(self.solvent_stoichiometric_weight)

        # Exact compatibility path for the established Mg-Si formulation.
        if solvent_weight == 0.0:
            return (
                np.sum(
                    weights[:, None]
                    * np.log(c_surf * float(matrix_atomic_volume)),
                    axis=0,
                )
                - log_KR
            ) / np.sum(weights)

        X = c_surf * float(matrix_atomic_volume)
        X_solvent = 1.0 - np.sum(X, axis=0)
        if np.any(~np.isfinite(X_solvent)) or np.any(X_solvent <= 0.0):
            raise ValueError(
                "Implicit solvent site fraction must remain finite and positive."
            )

        nu_total = solvent_weight + float(np.sum(weights))
        return (
            np.sum(weights[:, None] * np.log(X), axis=0)
            + solvent_weight * np.log(X_solvent)
            - log_KR
        ) / nu_total

    def prepare_equilibrium(
        self,
        *,
        radius: np.ndarray,
        temperature: float,
        matrix_atomic_volume: float,
        precipitate_atomic_volume: float,
        gamma: float,
        stoichiometric_weights: np.ndarray,
    ) -> PreparedSolubilityProductEquilibrium:
        """Precompute equilibrium data once for an isolated KWN solve."""
        radius = np.asarray(radius, dtype=float)
        weights = np.asarray(stoichiometric_weights, dtype=float)

        if (
            not np.isfinite(matrix_atomic_volume)
            or matrix_atomic_volume <= 0.0
        ):
            raise ValueError(
                "matrix_atomic_volume must be finite and positive."
            )
        if (
            weights.ndim != 1
            or weights.size == 0
            or np.any(~np.isfinite(weights))
            or np.any(weights <= 0.0)
        ):
            raise ValueError(
                "stoichiometric_weights must be a positive 1D array."
            )

        log_KR = self.log_curvature_equilibrium_product(
            radius=radius,
            temperature=temperature,
            precipitate_atomic_volume=precipitate_atomic_volume,
            gamma=gamma,
            stoichiometric_weights=weights,
        )

        return PreparedSolubilityProductEquilibrium(
            log_KR=np.asarray(log_KR, dtype=float),
            matrix_atomic_volume=float(matrix_atomic_volume),
            stoichiometric_weights=weights.copy(),
        )

    def dependent_concentration(
        self,
        *,
        independent_concentration: np.ndarray,
        independent_index: int,
        dependent_index: int,
        radius: np.ndarray,
        temperature: float,
        matrix_atomic_volume: float,
        precipitate_atomic_volume: float,
        gamma: float,
        stoichiometric_weights: np.ndarray,
    ) -> np.ndarray:
        """Return one interface concentration from the solubility product.

        This is an exact algebraic inversion of the same thermodynamic
        residual used by ``residual``.  It is useful for constructing an
        isolated-particle KWN starting state without embedding a particular
        phase chemistry in the growth solver.
        """
        independent_concentration = np.asarray(
            independent_concentration, dtype=float
        )
        radius = np.asarray(radius, dtype=float)
        weights = np.asarray(stoichiometric_weights, dtype=float)

        if weights.ndim != 1 or weights.size < 2:
            raise ValueError(
                "dependent_concentration requires at least two active "
                "stoichiometric weights."
            )
        if independent_index == dependent_index:
            raise ValueError(
                "independent_index and dependent_index must differ."
            )
        if not (0 <= independent_index < weights.size):
            raise IndexError("independent_index is outside the species array.")
        if not (0 <= dependent_index < weights.size):
            raise IndexError("dependent_index is outside the species array.")
        if np.any(weights <= 0.0):
            raise ValueError(
                "All active stoichiometric weights must be positive."
            )
        if np.any(independent_concentration <= 0.0):
            raise ValueError(
                "independent_concentration must be positive."
            )

        # This algebraic helper currently handles the two-active-species,
        # transported-solute-only line-compound case used by the legacy KWN
        # initializer. With an implicit solvent the equilibrium equation is
        # no longer algebraically separable in this form.
        if float(self.solvent_stoichiometric_weight) != 0.0:
            raise NotImplementedError(
                "dependent_concentration does not support an implicit solvent; "
                "use the generic nonlinear interface initializer instead."
            )
        if weights.size != 2:
            raise NotImplementedError(
                "Algebraic dependent-concentration initialization is "
                "currently implemented for two active species."
            )

        log_KR = self.log_curvature_equilibrium_product(
            radius=radius,
            temperature=temperature,
            precipitate_atomic_volume=precipitate_atomic_volume,
            gamma=gamma,
            stoichiometric_weights=weights,
        )

        nu_ind = weights[independent_index]
        nu_dep = weights[dependent_index]

        log_c_dep = (
            log_KR
            - nu_ind
            * np.log(
                independent_concentration * float(matrix_atomic_volume)
            )
        ) / nu_dep - np.log(float(matrix_atomic_volume))

        return np.exp(log_c_dep)

    def directional_derivative(
        self,
        *,
        c_surf: np.ndarray,
        delta_c_surf: np.ndarray,
        radius: np.ndarray,
        temperature: float,
        matrix_atomic_volume: float,
        precipitate_atomic_volume: float,
        gamma: float,
        stoichiometric_weights: np.ndarray,
    ) -> np.ndarray:
        c_surf = np.asarray(c_surf, dtype=float)
        delta_c_surf = np.asarray(delta_c_surf, dtype=float)
        weights = np.asarray(stoichiometric_weights, dtype=float)

        if delta_c_surf.shape != c_surf.shape:
            raise ValueError(
                "delta_c_surf must have the same shape as c_surf."
            )
        if weights.shape != (c_surf.shape[0],):
            raise ValueError(
                "stoichiometric_weights must match c_surf species dimension."
            )

        solvent_weight = float(self.solvent_stoichiometric_weight)

        # Exact compatibility path for the established Mg-Si formulation.
        if solvent_weight == 0.0:
            return np.sum(
                weights[:, None] * delta_c_surf / c_surf,
                axis=0,
            ) / np.sum(weights)

        vatm = float(matrix_atomic_volume)
        X_solvent = 1.0 - vatm * np.sum(c_surf, axis=0)
        if np.any(~np.isfinite(X_solvent)) or np.any(X_solvent <= 0.0):
            raise ValueError(
                "Implicit solvent site fraction must remain finite and positive."
            )

        nu_total = solvent_weight + float(np.sum(weights))
        transported_term = np.sum(
            weights[:, None] * delta_c_surf / c_surf,
            axis=0,
        )
        solvent_term = (
            -solvent_weight
            * vatm
            * np.sum(delta_c_surf, axis=0)
            / X_solvent
        )
        return (transported_term + solvent_term) / nu_total

    def transpose_contribution(
        self,
        *,
        weight_residual: np.ndarray,
        c_surf: np.ndarray,
        radius: np.ndarray,
        temperature: float,
        matrix_atomic_volume: float,
        precipitate_atomic_volume: float,
        gamma: float,
        stoichiometric_weights: np.ndarray,
    ) -> np.ndarray:
        weight_residual = np.asarray(weight_residual, dtype=float)
        c_surf = np.asarray(c_surf, dtype=float)
        weights = np.asarray(stoichiometric_weights, dtype=float)

        if c_surf.ndim != 2:
            raise ValueError(
                "c_surf must have shape (n_species, n_particles)."
            )
        if weight_residual.shape != (c_surf.shape[1],):
            raise ValueError(
                "weight_residual must have shape (n_particles,)."
            )
        if weights.shape != (c_surf.shape[0],):
            raise ValueError(
                "stoichiometric_weights must match c_surf species dimension."
            )

        solvent_weight = float(self.solvent_stoichiometric_weight)

        # Exact compatibility path for the established Mg-Si formulation.
        if solvent_weight == 0.0:
            return (
                weights[:, None]
                * weight_residual[None, :]
                / (np.sum(weights) * c_surf)
            )

        vatm = float(matrix_atomic_volume)
        X_solvent = 1.0 - vatm * np.sum(c_surf, axis=0)
        if np.any(~np.isfinite(X_solvent)) or np.any(X_solvent <= 0.0):
            raise ValueError(
                "Implicit solvent site fraction must remain finite and positive."
            )

        nu_total = solvent_weight + float(np.sum(weights))
        derivative = (
            weights[:, None] / c_surf
            - solvent_weight * vatm / X_solvent[None, :]
        ) / nu_total
        return derivative * weight_residual[None, :]


@dataclass(frozen=True)
class SoluteSpecies:
    name: str
    D0: float
    Qdiff: float


@dataclass(frozen=True)
class PrecipitatePhase:
    name: str
    stoichiometry: Mapping[str, float]
    atomic_volume: float
    gamma: float
    thermodynamics: PhaseThermodynamics
    gb_barrier_factor: float | None = None

    def resolved_gb_barrier_factor(self, fallback: float = 1.0) -> float:
        """Return this phase's GB CNT barrier factor.

        ``None`` preserves legacy behavior by using the run-level fallback.
        """
        value = fallback if self.gb_barrier_factor is None else self.gb_barrier_factor
        value = float(value)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"Phase {self.name!r} GB barrier factor must be positive and finite."
            )
        return value

    def atomic_fraction(self, species: str) -> float:
        total = sum(self.stoichiometry.values())
        if total <= 0.0:
            raise ValueError(
                f"Phase {self.name!r} has non-positive stoichiometric total."
            )
        return self.stoichiometry.get(species, 0.0) / total

    def number_concentration(self, species: str) -> float:
        return self.atomic_fraction(species) / self.atomic_volume

    def diffusing_stoichiometric_weights(
        self,
        solute_names: tuple[str, ...],
    ) -> np.ndarray:
        """Return phase stoichiometric coefficients in solver species order."""
        weights = np.asarray(
            [self.stoichiometry.get(name, 0.0) for name in solute_names],
            dtype=float,
        )

        if np.any(weights < 0.0):
            raise ValueError(
                f"Phase {self.name!r} has negative stoichiometric coefficients."
            )

        active = weights > 0.0
        if not np.any(active):
            raise ValueError(
                f"Phase {self.name!r} contains none of the diffusing solutes "
                f"{solute_names!r}."
            )

        return weights

    # Temporary compatibility accessors for the current run_case/growth bridge.
    # They allow the thermodynamic parameters to move into the model now without
    # forcing the entire lower-level solver API to change in the same commit.
    @property
    def solubility_aa(self) -> float:
        return float(getattr(self.thermodynamics, "aa"))

    @property
    def solubility_bb(self) -> float:
        return float(getattr(self.thermodynamics, "bb"))

    @property
    def solubility_cc(self) -> float:
        return float(getattr(self.thermodynamics, "cc", 0.0))


@dataclass(frozen=True)
class AlloySystem:
    solvent: str
    matrix_atomic_volume: float
    matrix_lattice_parameter: float
    solutes: tuple[SoluteSpecies, ...]
    phases: tuple[PrecipitatePhase, ...]
