"""Periodic FFT long-range diffusion field for scale-separated transport.

The long-range field is an auxiliary smooth concentration field.  It is
not itself constrained to equal the thermodynamic precipitate-interface
concentration.  Interface matching is supplied by

    c_surf = C_LR(particle) + A_short Q.

Diffusion is advanced spectrally with the exact propagator

    C_hat(t+dt) = exp(-D k^2 dt) C_hat(t),

so the long-range diffusion solve has no CFL stability restriction.

Particle inventory changes are deposited conservatively with periodic CIC
and then Gaussian-filtered with the same alpha used in the short/long split.
"""

from __future__ import annotations

import numpy as np

from .utilities import as_box_lengths


class PeriodicLongRangeField:
    """Smooth periodic long-range concentration field."""

    def __init__(
        self,
        *,
        box,
        grid_shape,
        alpha,
        diffusivities,
        initial_concentrations,
        positions=None,
    ):
        self.box = as_box_lengths(box).astype(float)
        self.grid_shape = np.asarray(grid_shape, dtype=int)
        if self.grid_shape.shape == ():
            self.grid_shape = np.repeat(self.grid_shape, 3)
        if self.grid_shape.shape != (3,) or np.any(self.grid_shape < 2):
            raise ValueError(
                "grid_shape must be an integer or a length-3 array >= 2."
            )

        self.alpha = float(alpha)
        if not np.isfinite(self.alpha) or self.alpha <= 0.0:
            raise ValueError("alpha must be finite and positive.")

        self.diffusivities = np.asarray(diffusivities, dtype=float)
        self.initial_concentrations = np.asarray(
            initial_concentrations, dtype=float
        )
        if (
            self.diffusivities.ndim != 1
            or self.initial_concentrations.shape
            != self.diffusivities.shape
        ):
            raise ValueError(
                "diffusivities and initial_concentrations must have "
                "the same one-dimensional shape."
            )
        if np.any(self.diffusivities <= 0.0):
            raise ValueError("diffusivities must be positive.")

        self.n_species = self.diffusivities.size
        self.spacing = self.box / self.grid_shape
        self.cell_volume = float(np.prod(self.spacing))
        self.volume = float(np.prod(self.box))

        k_axes = [
            2.0 * np.pi * np.fft.fftfreq(
                int(n), d=float(dx)
            )
            for n, dx in zip(self.grid_shape, self.spacing)
        ]
        KX, KY, KZ = np.meshgrid(*k_axes, indexing="ij")
        self.k2 = KX*KX + KY*KY + KZ*KZ
        self.gaussian_filter_k = np.exp(
            -self.k2 / (4.0 * self.alpha**2)
        )

        shape = (self.n_species,) + tuple(self.grid_shape)
        self.field = np.empty(shape, dtype=float)
        for s, c0 in enumerate(self.initial_concentrations):
            self.field[s].fill(float(c0))

        # The freshly initialized LR field contains only the zero Fourier mode.
        # This flag enables exact no-op fast paths for pure diffusion before any
        # spatially varying particle source has been applied.
        self._is_uniform = True

        self.positions = np.empty((0, 3), dtype=float)
        self._cic = []
        if positions is not None:
            self.update_positions(positions)

    @property
    def matching_length(self):
        return 1.0 / self.alpha

    @property
    def is_uniform(self):
        """True while the LR field is known to contain only its zero mode."""
        return bool(self._is_uniform)

    def set_diffusivities(self, diffusivities):
        """Update species diffusivities without changing the concentration field.

        Thermal-history segments change D(T), but the physical long-range
        concentration field must remain continuous across segment boundaries.
        The spectral propagator is constructed from self.diffusivities on each
        call to diffuse(), so updating this vector is sufficient and does not
        reinitialize or otherwise alter the field.
        """
        diffusivities = np.asarray(diffusivities, dtype=float)
        if diffusivities.shape != (self.n_species,):
            raise ValueError(
                "diffusivities must have shape "
                f"({self.n_species},); got {diffusivities.shape}."
            )
        if np.any(~np.isfinite(diffusivities)) or np.any(diffusivities <= 0.0):
            raise ValueError("diffusivities must contain finite positive values.")
        self.diffusivities = diffusivities.copy()

    def copy_field(self):
        return self.field.copy()

    def set_field(self, field):
        field = np.asarray(field, dtype=float)
        if field.shape != self.field.shape:
            raise ValueError(
                f"field must have shape {self.field.shape}, "
                f"got {field.shape}."
            )
        if np.any(~np.isfinite(field)):
            raise ValueError("field contains non-finite values.")
        self.field[...] = field
        # Keep the uniform-field fast path correct after externally setting a
        # field.  This matters for prescribed nonuniform LR initial conditions.
        axes = tuple(range(1, self.field.ndim))
        means = np.mean(self.field, axis=axes)
        reshape = (self.n_species,) + (1,) * (self.field.ndim - 1)
        scale = np.maximum(1.0, np.max(np.abs(self.field), axis=axes))
        max_dev = np.max(
            np.abs(self.field - means.reshape(reshape)),
            axis=axes,
        )
        self._is_uniform = bool(
            np.all(max_dev <= 32.0 * np.finfo(float).eps * scale)
        )

    def set_cosine_initial_condition(
        self,
        *,
        amplitudes,
        axis=0,
        mode=1,
    ):
        """Set a zero-mean periodic cosine perturbation on the LR field.

        For species s,

            C_s(r,0) = cbar_s * [1 + A_s * phi(r)],

        where phi is a cosine mode along one box axis.  The discrete-grid mean
        of phi is explicitly removed before use, so the matrix inventory of
        every species is unchanged to roundoff.

        amplitudes are relative (fractional) amplitudes, one per species.
        axis may be 0/1/2 or 'x'/'y'/'z'.  mode is a positive integer.
        """
        axis_map = {"x": 0, "y": 1, "z": 2}
        if isinstance(axis, str):
            key = axis.lower()
            if key not in axis_map:
                raise ValueError("axis must be 'x', 'y', 'z', 0, 1, or 2.")
            axis = axis_map[key]
        axis = int(axis)
        if axis not in (0, 1, 2):
            raise ValueError("axis must be 'x', 'y', 'z', 0, 1, or 2.")

        mode = int(mode)
        if mode < 1:
            raise ValueError("mode must be a positive integer.")

        amplitudes = np.asarray(amplitudes, dtype=float)
        if amplitudes.shape != (self.n_species,):
            raise ValueError(
                f"amplitudes must have shape ({self.n_species},); "
                f"got {amplitudes.shape}."
            )
        if np.any(~np.isfinite(amplitudes)):
            raise ValueError("amplitudes must be finite.")

        n = int(self.grid_shape[axis])
        coord = np.arange(n, dtype=float) * float(self.spacing[axis])
        phase = 2.0 * np.pi * mode * coord / float(self.box[axis])
        phi_1d = np.cos(phase)
        # Explicitly enforce zero discrete mean.  For an FFT-compatible integer
        # mode this is already zero to roundoff, but this guarantees inventory.
        phi_1d -= np.mean(phi_1d)

        shape = [1, 1, 1]
        shape[axis] = n
        phi = phi_1d.reshape(shape)

        field = np.empty_like(self.field)
        for s, (c0, amp) in enumerate(
            zip(self.initial_concentrations, amplitudes)
        ):
            field[s] = float(c0) * (1.0 + float(amp) * phi)

        if np.any(field < 0.0):
            raise ValueError(
                "Cosine initial condition produces negative concentration; "
                "reduce the requested amplitude."
            )

        self.set_field(field)

    def update_positions(self, positions):
        """Rebuild periodic CIC stencils after particle creation/removal."""
        pos = np.asarray(positions, dtype=float)
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError(
                f"positions must have shape (N,3), got {pos.shape}."
            )
        self.positions = np.mod(pos, self.box[None, :])
        self._cic = self._build_cic(self.positions)

    def _build_cic(self, points):
        if len(points) == 0:
            return []

        u = points / self.spacing[None, :]
        base_floor = np.floor(u)
        i0 = base_floor.astype(int)
        frac = u - base_floor

        stencils = []
        nx, ny, nz = map(int, self.grid_shape)

        for ox in (0, 1):
            wx = (1.0-frac[:, 0]) if ox == 0 else frac[:, 0]
            ix = (i0[:, 0] + ox) % nx
            for oy in (0, 1):
                wy = (1.0-frac[:, 1]) if oy == 0 else frac[:, 1]
                iy = (i0[:, 1] + oy) % ny
                for oz in (0, 1):
                    wz = (1.0-frac[:, 2]) if oz == 0 else frac[:, 2]
                    iz = (i0[:, 2] + oz) % nz
                    stencils.append((ix, iy, iz, wx*wy*wz))
        return stencils

    def sample_at_particles(self):
        """Return C_LR at particle centres with shape (S,N)."""
        n = len(self.positions)
        out = np.zeros((self.n_species, n), dtype=float)
        for ix, iy, iz, weight in self._cic:
            out += (
                self.field[:, ix, iy, iz]
                * weight[None, :]
            )
        return out

    def sample_at_points(self, points):
        """Return C_LR at arbitrary periodic points with shape (S,M).

        Parameters
        ----------
        points : array_like, shape (M, 3) or (3,)
            Coordinates in nm. Points are wrapped periodically into the
            simulation box before CIC interpolation.
        """
        points = np.asarray(points, dtype=float)

        if points.ndim == 1:
            if points.shape != (3,):
                raise ValueError(
                    "A one-dimensional points array must have shape (3,)."
                )
            points = points[None, :]

        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(
                f"points must have shape (M,3), got {points.shape}."
            )

        if np.any(~np.isfinite(points)):
            raise ValueError("points contains non-finite coordinates.")

        points = np.mod(points, self.box[None, :])
        cic = self._build_cic(points)

        out = np.zeros((self.n_species, len(points)), dtype=float)
        for ix, iy, iz, weight in cic:
            out += (
                self.field[:, ix, iy, iz]
                * weight[None, :]
            )
        return out

    def deposit_inventory(self, particle_inventory):
        """CIC-deposit particle inventories onto the concentration grid.

        Parameters
        ----------
        particle_inventory : (N,) or (S,N) ndarray
            Inventory in atoms. The returned grid has concentration units
            atoms/nm^3 and preserves the supplied total inventory exactly
            to floating-point roundoff.
        """
        values = np.asarray(particle_inventory, dtype=float)
        n = len(self.positions)

        one_species = values.ndim == 1
        if one_species:
            values = values[None, :]

        if values.ndim != 2 or values.shape[1] != n:
            raise ValueError(
                f"particle_inventory must have trailing shape ({n},)."
            )

        rho = np.zeros(
            (values.shape[0],) + tuple(self.grid_shape),
            dtype=float,
        )
        for ix, iy, iz, weight in self._cic:
            for s in range(values.shape[0]):
                np.add.at(
                    rho[s],
                    (ix, iy, iz),
                    weight * values[s] / self.cell_volume,
                )
        return rho[0] if one_species else rho

    def gaussian_smooth(self, density):
        """Apply W_alpha in Fourier space."""
        density = np.asarray(density, dtype=float)
        if density.shape == tuple(self.grid_shape):
            return np.fft.ifftn(
                np.fft.fftn(density) * self.gaussian_filter_k
            ).real

        if density.ndim == 4 and density.shape[1:] == tuple(self.grid_shape):
            out = np.empty_like(density)
            for s in range(density.shape[0]):
                out[s] = np.fft.ifftn(
                    np.fft.fftn(density[s])
                    * self.gaussian_filter_k
                ).real
            return out

        raise ValueError("density has incompatible grid shape.")

    def diffuse(self, dt):
        """Advance pure diffusion exactly in Fourier space.

        This operation is unconditionally stable with respect to diffusion;
        it introduces no CFL timestep limit.
        """
        dt = float(dt)
        if not np.isfinite(dt) or dt < 0.0:
            raise ValueError("dt must be finite and non-negative.")
        if dt == 0.0 or self._is_uniform:
            # Diffusion leaves a spatially uniform periodic field unchanged.
            return

        for s, D in enumerate(self.diffusivities):
            propagator = np.exp(-D * self.k2 * dt)
            self.field[s] = np.fft.ifftn(
                np.fft.fftn(self.field[s]) * propagator
            ).real

    def apply_particle_inventory_change(
        self,
        delta_particle_inventory,
        *,
        sign=-1.0,
    ):
        """Apply a conservative smoothed particle/matrix inventory transfer.

        delta_particle_inventory is the change of precipitate inventory.
        With the default sign=-1, precipitate uptake is removed from the
        matrix LR field and precipitate dissolution is returned to it.
        """
        values = np.asarray(delta_particle_inventory, dtype=float)
        if values.shape != (self.n_species, len(self.positions)):
            raise ValueError(
                "delta_particle_inventory must have shape "
                f"({self.n_species},{len(self.positions)})."
            )

        # With no particle inventory transfer there is no source term.  Avoid
        # allocating full-grid zero arrays and performing FFTs of zeros.
        if values.size == 0 or not np.any(values):
            return

        rho = self.deposit_inventory(values)
        rho_smooth = self.gaussian_smooth(rho)

        # C_LR is the smooth component of the hybrid decomposition and is not
        # required to remain positive pointwise.  Apply the conservative
        # inventory transfer without clipping or positivity modification.
        self.field += float(sign) * rho_smooth
        self._is_uniform = False

    def advance_with_particle_inventory_change(
        self,
        dt,
        delta_particle_inventory,
    ):
        """Second-order symmetric diffusion/source update.

        half diffusion -> conservative inventory transfer -> half diffusion.

        Diffusion itself is exact and unconditionally stable.  Timestep
        selection remains controlled by precipitation/coupling accuracy.
        """
        self.diffuse(0.5 * float(dt))
        self.apply_particle_inventory_change(delta_particle_inventory)
        self.diffuse(0.5 * float(dt))

    def matrix_inventory(self):
        """Integral of each LR concentration field over the periodic box."""
        if self._is_uniform:
            return self.field[(slice(None),) + (0,) * 3].copy() * self.volume

        axes = tuple(range(1, self.field.ndim))
        return (
            np.sum(self.field, axis=axes)
            * self.cell_volume
        )

    def mean_concentration(self):
        return self.matrix_inventory() / self.volume

    def conservation_check_for_deposit(self, particle_inventory):
        """Diagnostic error for CIC + Gaussian inventory preservation."""
        values = np.asarray(particle_inventory, dtype=float)
        rho = self.deposit_inventory(values)
        smooth = self.gaussian_smooth(rho)

        if values.ndim == 1:
            deposited = np.sum(smooth) * self.cell_volume
            expected = np.sum(values)
        else:
            axes = tuple(range(1, smooth.ndim))
            deposited = np.sum(smooth, axis=axes) * self.cell_volume
            expected = np.sum(values, axis=1)
        return deposited - expected
