import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import numpy as np
from scipy.signal import resample as fourier_resample

from ..greensfunction import (
    periodic_green_ewald_points,
    prepare_periodic_green_kernel,
)
from ..utilities import as_box_lengths, minimum_image_displacement
from ..hybrid_short_range import HybridShortRangeOperator


# ------------------------------------------------------------------
# Process-worker state
#
# These arrays are initialized once in each worker process instead of
# being transmitted again for every point chunk.
# ------------------------------------------------------------------
_WORKER_FIELD_STATE = {}


def _initialize_field_worker(
    positions,
    Q,
    box,
    alpha,
    real_lattice,
    kvecs,
    k2,
    recip_weights,
    const_term,
    kappa,
):
    """Initialize fixed Green-function data in one worker process."""
    global _WORKER_FIELD_STATE

    _WORKER_FIELD_STATE = {
        "positions": np.asarray(positions, dtype=float),
        "Q": np.asarray(Q, dtype=float),
        "box": as_box_lengths(box),
        "alpha": float(alpha),
        "real_lattice": np.asarray(real_lattice, dtype=float),
        "kvecs": np.asarray(kvecs, dtype=float),
        "k2": np.asarray(k2, dtype=float),
        "recip_weights": np.asarray(recip_weights, dtype=float),
        "const_term": float(const_term),
        "kappa": float(kappa),
    }


def _evaluate_field_chunk_worker(task):
    """Evaluate one point chunk in a worker process."""
    chunk_number, point_indices, chunk_points = task
    state = _WORKER_FIELD_STATE

    delta_c = evaluate_green_field_chunk_species(
        points=chunk_points,
        positions=state["positions"],
        Q=state["Q"],
        box=state["box"],
        alpha=state["alpha"],
        real_lattice=state["real_lattice"],
        kvecs=state["kvecs"],
        k2=state["k2"],
        recip_weights=state["recip_weights"],
        const_term=state["const_term"],
        kappa=state["kappa"],
    )

    return chunk_number, point_indices, delta_c


def validate_field_inputs_species(
    points,
    positions,
    radii,
    Q,
    cbar,
    solute_names,
):
    """Validate and convert the species-indexed composition-field inputs."""
    points = np.asarray(points, dtype=float)
    positions = np.asarray(positions, dtype=float)
    radii = np.asarray(radii, dtype=float)
    Q = np.asarray(Q, dtype=float)
    cbar = np.asarray(cbar, dtype=float)
    solute_names = tuple(str(name) for name in solute_names)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (Npoints, 3).")

    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (Nparticles, 3).")

    n_particles = len(radii)
    n_solutes = len(solute_names)

    if n_solutes < 1:
        raise ValueError("solute_names must contain at least one species.")

    if len(set(solute_names)) != n_solutes:
        raise ValueError("solute_names must contain unique species names.")

    if any(not name for name in solute_names):
        raise ValueError("solute_names must not contain empty names.")

    if positions.shape[0] != n_particles:
        raise ValueError("positions and radii must contain the same number of particles.")

    if Q.shape != (n_solutes, n_particles):
        raise ValueError(
            "Q must have shape (Nsolutes, Nparticles); "
            f"expected {(n_solutes, n_particles)}, got {Q.shape}."
        )

    if cbar.shape != (n_solutes,):
        raise ValueError(
            "cbar must have shape (Nsolutes,); "
            f"expected {(n_solutes,)}, got {cbar.shape}."
        )

    if np.any(~np.isfinite(points)):
        raise ValueError("points contains non-finite values.")

    if np.any(~np.isfinite(positions)):
        raise ValueError("positions contains non-finite values.")

    if np.any(~np.isfinite(radii)) or np.any(radii < 0.0):
        raise ValueError("radii must contain finite non-negative values.")

    if np.any(~np.isfinite(Q)):
        raise ValueError("Q contains non-finite values.")

    if np.any(~np.isfinite(cbar)):
        raise ValueError("cbar contains non-finite values.")

    return points, positions, radii, Q, cbar, solute_names


def validate_field_inputs(points, positions, radii, Q_mg, Q_si):
    """Legacy Mg/Si input validator retained for compatibility."""
    points, positions, radii, Q, _, _ = validate_field_inputs_species(
        points=points,
        positions=positions,
        radii=radii,
        Q=np.vstack(
            [
                np.asarray(Q_mg, dtype=float),
                np.asarray(Q_si, dtype=float),
            ]
        ),
        cbar=np.zeros(2, dtype=float),
        solute_names=("Mg", "Si"),
    )
    return points, positions, radii, Q[0], Q[1]


def particle_mask_on_points(
    points,
    positions,
    radii,
    box,
    *,
    return_particle_id=False,
):
    """
    Determine which evaluation points lie inside a precipitate.

    Periodic minimum-image distances are used.
    """
    points = np.asarray(points, dtype=float)
    positions = np.asarray(positions, dtype=float)
    radii = np.asarray(radii, dtype=float)
    box = as_box_lengths(box)

    inside = np.zeros(len(points), dtype=bool)

    if return_particle_id:
        particle_id = np.full(len(points), -1, dtype=np.int64)

    active_idx = np.flatnonzero(radii > 0.0)

    for particle_index in active_idx:
        dr = points - positions[particle_index]
        dr = minimum_image_displacement(dr, box)

        distance_squared = np.einsum("ij,ij->i", dr, dr)
        inside_this_particle = distance_squared <= radii[particle_index] ** 2

        if return_particle_id:
            newly_inside = inside_this_particle & ~inside
            particle_id[newly_inside] = particle_index

        inside |= inside_this_particle

    if return_particle_id:
        return inside, particle_id

    return inside


def evaluate_green_field_chunk_species(
    points,
    positions,
    Q,
    box,
    alpha,
    real_lattice,
    kvecs,
    k2,
    recip_weights,
    const_term,
    kappa=0.0,
):
    """
    Evaluate Green-function concentration perturbations for arbitrary solutes.

    Parameters
    ----------
    Q : ndarray, shape (Nsolutes, Nparticles)
        Species-indexed monopole strengths.

    Returns
    -------
    delta_c : ndarray, shape (Nsolutes, Npoints)
    """
    points = np.asarray(points, dtype=float)
    positions = np.asarray(positions, dtype=float)
    Q = np.asarray(Q, dtype=float)
    box = as_box_lengths(box)
    kappa = float(kappa)

    if Q.ndim != 2:
        raise ValueError("Q must have shape (Nsolutes, Nparticles).")

    if Q.shape[1] != len(positions):
        raise ValueError("Q.shape[1] must equal the number of particles.")

    if not np.isfinite(kappa) or kappa < 0.0:
        raise ValueError("kappa must be finite and non-negative.")

    delta_c = np.zeros((Q.shape[0], len(points)), dtype=np.float64)

    if len(points) == 0 or len(positions) == 0:
        return delta_c

    active_source_idx = np.flatnonzero(np.any(Q != 0.0, axis=0))

    for particle_index in active_source_idx:
        dr = points - positions[particle_index]
        dr = minimum_image_displacement(dr, box)

        G = periodic_green_ewald_points(
            dr=dr,
            box=box,
            alpha=alpha,
            real_lattice=real_lattice,
            kvecs=kvecs,
            k2=k2,
            recip_weights=recip_weights,
            const_term=const_term,
            kappa=kappa,
        )

        # Keep the same particle accumulation ordering used by the legacy
        # two-solute implementation; each species is updated independently.
        for species_index in range(Q.shape[0]):
            delta_c[species_index] += Q[species_index, particle_index] * G

    return delta_c


def evaluate_green_field_chunk(
    points,
    positions,
    Q_mg,
    Q_si,
    box,
    alpha,
    real_lattice,
    kvecs,
    k2,
    recip_weights,
    const_term,
    kappa=0.0,
):
    """Legacy Mg/Si wrapper for evaluate_green_field_chunk_species()."""
    delta_c = evaluate_green_field_chunk_species(
        points=points,
        positions=positions,
        Q=np.vstack(
            [
                np.asarray(Q_mg, dtype=float),
                np.asarray(Q_si, dtype=float),
            ]
        ),
        box=box,
        alpha=alpha,
        real_lattice=real_lattice,
        kvecs=kvecs,
        k2=k2,
        recip_weights=recip_weights,
        const_term=const_term,
        kappa=kappa,
    )
    return delta_c[0], delta_c[1]


def evaluate_composition_field(
    points,
    positions,
    radii,
    Q,
    cbar,
    solute_names,
    box,
    alpha,
    real_lattice,
    kvecs,
    k2,
    *,
    screening_length=None,
    cp=None,
    chunk_size=10_000,
    output_dtype=np.float32,
    return_particle_id=False,
    show_progress=False,
    n_workers=None,
):
    """
    Reconstruct arbitrary species-indexed concentration fields.

    Matrix fields are

        c_s(x) = cbar_s + sum_i Q_s,i G(x - x_i).

    Parameters
    ----------
    Q : ndarray, shape (Nsolutes, Nparticles)
    cbar : ndarray, shape (Nsolutes,)
    solute_names : sequence of str, length Nsolutes
    cp : ndarray, shape (Nsolutes,), or None
        Precipitate concentrations used for points inside particles.

    Returns
    -------
    result : dict
        ``c`` has shape (Nsolutes, Npoints). ``solute_names`` preserves the
        species ordering. Other geometric/kernel fields match the legacy API.
    """
    points, positions, radii, Q, cbar, solute_names = validate_field_inputs_species(
        points=points,
        positions=positions,
        radii=radii,
        Q=Q,
        cbar=cbar,
        solute_names=solute_names,
    )

    box = as_box_lengths(box)
    alpha = float(alpha)
    real_lattice = np.asarray(real_lattice, dtype=float)
    kvecs = np.asarray(kvecs, dtype=float)
    k2 = np.asarray(k2, dtype=float)
    n_solutes = len(solute_names)

    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("alpha must be positive and finite.")

    if real_lattice.ndim != 2 or real_lattice.shape[1] != 3:
        raise ValueError("real_lattice must have shape (Nreal, 3).")

    if kvecs.ndim != 2 or kvecs.shape[1] != 3:
        raise ValueError("kvecs must have shape (Nk, 3).")

    if k2.shape != (len(kvecs),):
        raise ValueError("k2 must have shape (len(kvecs),).")

    if np.any(~np.isfinite(real_lattice)):
        raise ValueError("real_lattice contains non-finite values.")

    if np.any(~np.isfinite(kvecs)):
        raise ValueError("kvecs contains non-finite values.")

    if np.any(~np.isfinite(k2)) or np.any(k2 <= 0.0):
        raise ValueError("k2 must contain finite positive values.")

    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")

    try:
        output_dtype = np.dtype(output_dtype)
    except (TypeError, ValueError) as exc:
        raise ValueError("output_dtype must be a valid NumPy dtype.") from exc

    if not np.issubdtype(output_dtype, np.floating):
        raise ValueError("output_dtype must be a floating-point dtype.")

    if cp is not None:
        cp = np.asarray(cp, dtype=float)
        if cp.shape != (n_solutes,):
            raise ValueError(
                "cp must have shape (Nsolutes,); "
                f"expected {(n_solutes,)}, got {cp.shape}."
            )
        if np.any(~np.isfinite(cp)):
            raise ValueError("cp contains non-finite values.")

    if n_workers is None:
        slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
        if slurm_cpus is not None:
            try:
                n_workers = int(slurm_cpus)
            except ValueError as exc:
                raise ValueError("SLURM_CPUS_PER_TASK must contain an integer.") from exc
        else:
            n_workers = os.cpu_count() or 1

    n_workers = int(n_workers)
    if n_workers <= 0:
        raise ValueError("n_workers must be a positive integer or None.")

    if return_particle_id:
        inside_precipitate, particle_id = particle_mask_on_points(
            points=points,
            positions=positions,
            radii=radii,
            box=box,
            return_particle_id=True,
        )
    else:
        inside_precipitate = particle_mask_on_points(
            points=points,
            positions=positions,
            radii=radii,
            box=box,
            return_particle_id=False,
        )
        particle_id = None

    if np.any(inside_precipitate) and cp is None:
        raise ValueError(
            "cp must be supplied when evaluation points lie inside precipitates."
        )

    matrix_idx = np.flatnonzero(~inside_precipitate)
    n_points = len(points)
    n_matrix_points = len(matrix_idx)

    c = np.full((n_solutes, n_points), np.nan, dtype=np.float64)
    c[:, matrix_idx] = cbar[:, None]

    if np.any(inside_precipitate):
        # The monopole concentration field describes the matrix.  At evaluation
        # points inside a precipitate, report the precipitate composition rather
        # than continuing the matrix field through the particle.
        c[:, inside_precipitate] = cp[:, None]


    green_kernel = prepare_periodic_green_kernel(
        box=box,
        alpha=alpha,
        k2=k2,
        screening_length=screening_length,
    )
    recip_weights = green_kernel["reciprocal_weights"]
    const_term = float(green_kernel["constant_term"])
    kappa = float(green_kernel["kappa"])

    n_chunks = (n_matrix_points + chunk_size - 1) // chunk_size

    if show_progress:
        print(
            f"Evaluating {n_matrix_points} matrix points in "
            f"{n_chunks} chunks using {n_workers} worker(s)."
        )

    def field_task_iterator():
        for chunk_number, start in enumerate(
            range(0, n_matrix_points, chunk_size),
            start=1,
        ):
            stop = min(start + chunk_size, n_matrix_points)
            chunk_idx = matrix_idx[start:stop]
            yield chunk_number, chunk_idx, points[chunk_idx]

    if n_workers == 1 or n_chunks <= 1:
        for completed, task in enumerate(field_task_iterator(), start=1):
            chunk_number, chunk_idx, chunk_points = task

            delta_c = evaluate_green_field_chunk_species(
                points=chunk_points,
                positions=positions,
                Q=Q,
                box=box,
                alpha=alpha,
                real_lattice=real_lattice,
                kvecs=kvecs,
                k2=k2,
                recip_weights=recip_weights,
                const_term=const_term,
                kappa=kappa,
            )
            c[:, chunk_idx] += delta_c

            if show_progress:
                print(
                    f"Completed field chunk {completed}/{n_chunks} "
                    f"(chunk {chunk_number})."
                )
    else:
        n_workers_use = min(n_workers, n_chunks)
        max_pending_tasks = 2 * n_workers_use
        task_iterator = iter(field_task_iterator())
        completed = 0

        with ProcessPoolExecutor(
            max_workers=n_workers_use,
            initializer=_initialize_field_worker,
            initargs=(
                positions,
                Q,
                box,
                alpha,
                real_lattice,
                kvecs,
                k2,
                recip_weights,
                const_term,
                kappa,
            ),
        ) as executor:
            pending_futures = set()

            for _ in range(min(max_pending_tasks, n_chunks)):
                try:
                    task = next(task_iterator)
                except StopIteration:
                    break
                pending_futures.add(executor.submit(_evaluate_field_chunk_worker, task))

            while pending_futures:
                done_futures, pending_futures = wait(
                    pending_futures,
                    return_when=FIRST_COMPLETED,
                )

                for future in done_futures:
                    chunk_number, chunk_idx, delta_c = future.result()
                    c[:, chunk_idx] += delta_c
                    completed += 1

                    if show_progress:
                        print(
                            f"Completed field chunk {completed}/{n_chunks} "
                            f"(chunk {chunk_number})."
                        )

                    try:
                        task = next(task_iterator)
                    except StopIteration:
                        continue
                    pending_futures.add(executor.submit(_evaluate_field_chunk_worker, task))

    if np.any(~np.isfinite(c)):
        raise RuntimeError("The evaluated composition field contains non-finite values.")

    result = {
        "points": points.copy(),
        "solute_names": solute_names,
        "c": c.astype(output_dtype, copy=False),
        "inside_precipitate": inside_precipitate,
        "green_kernel_type": green_kernel["kernel_type"],
        "screening_length": green_kernel["screening_length"],
        "screening_kappa": green_kernel["kappa"],
    }

    if return_particle_id:
        result["particle_id"] = particle_id

    return result



def sample_periodic_long_range_grid(points, field, box):
    """CIC-interpolate a saved long-range grid at arbitrary points.

    This is the solver-consistent sampler: it reproduces
    ``PeriodicLongRangeField.sample_at_points()`` used during the
    scale-separated precipitation calculation.
    """
    points = np.asarray(points, dtype=float)
    field = np.asarray(field, dtype=float)
    box = as_box_lengths(box).astype(float)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (M,3), got {points.shape}.")
    if field.ndim != 4:
        raise ValueError(
            "field must have shape (Nsolutes, nx, ny, nz); "
            f"got {field.shape}."
        )
    if np.any(~np.isfinite(points)) or np.any(~np.isfinite(field)):
        raise ValueError("points and field must contain only finite values.")

    grid_shape = np.asarray(field.shape[1:], dtype=int)
    spacing = box / grid_shape
    pts = np.mod(points, box[None, :])
    u = pts / spacing[None, :]
    base_floor = np.floor(u)
    i0 = base_floor.astype(int)
    frac = u - base_floor

    out = np.zeros((field.shape[0], len(pts)), dtype=float)
    nx, ny, nz = map(int, grid_shape)

    for ox in (0, 1):
        wx = (1.0 - frac[:, 0]) if ox == 0 else frac[:, 0]
        ix = (i0[:, 0] + ox) % nx
        for oy in (0, 1):
            wy = (1.0 - frac[:, 1]) if oy == 0 else frac[:, 1]
            iy = (i0[:, 1] + oy) % ny
            for oz in (0, 1):
                wz = (1.0 - frac[:, 2]) if oz == 0 else frac[:, 2]
                iz = (i0[:, 2] + oz) % nz
                out += field[:, ix, iy, iz] * (wx * wy * wz)[None, :]

    return out


def spectrally_upsample_periodic_long_range_grid(field, factor=4):
    """Fourier-resample a periodic LR grid onto a finer uniform grid.

    The operation retains only Fourier modes represented by the original FFT
    grid; it does not create new physical information.  ``scipy.signal.resample``
    performs periodic Fourier-domain zero padding/truncation and handles the
    even-grid Nyquist bookkeeping.

    Parameters
    ----------
    field : ndarray, shape (S,nx,ny,nz)
        Saved long-range concentration grid.
    factor : int
        Integer refinement factor along each spatial direction.

    Returns
    -------
    ndarray, shape (S,factor*nx,factor*ny,factor*nz)
        Smooth trigonometric interpolation sampled on the finer grid.
    """
    field = np.asarray(field, dtype=float)
    if field.ndim != 4:
        raise ValueError(
            "field must have shape (Nsolutes,nx,ny,nz); "
            f"got {field.shape}."
        )
    if np.any(~np.isfinite(field)):
        raise ValueError("field contains non-finite values.")

    factor = int(factor)
    if factor < 1:
        raise ValueError("spectral upsample factor must be >= 1.")
    if factor == 1:
        return field.copy()

    out = field
    for axis in (1, 2, 3):
        out = fourier_resample(out, factor * field.shape[axis], axis=axis)

    # A real input has a real trigonometric interpolant; remove only numerical
    # roundoff imaginary parts should SciPy ever return a complex array.
    out = np.real_if_close(out, tol=1000)
    if np.iscomplexobj(out):
        imag_max = float(np.max(np.abs(np.imag(out))))
        real_scale = max(1.0, float(np.max(np.abs(np.real(out)))))
        if imag_max > 1.0e-12 * real_scale:
            raise RuntimeError(
                "Fourier LR interpolation produced a non-negligible imaginary "
                f"component ({imag_max:.3e})."
            )
        out = np.real(out)

    return np.asarray(out, dtype=float)


def sample_periodic_long_range_grid_rectilinear(
    points,
    field,
    box,
    coordinates,
):
    """Trilinearly sample a periodic rectilinear LR grid.

    Sparse/coarse diagnostic LR snapshots are direct subsamples of the
    production FFT grid. Their stride need not divide the production grid
    dimensions, so the final periodic interval can differ from the interior
    spacing and they must not be treated as a uniform FFT grid spanning box.
    """
    points = np.asarray(points, dtype=float)
    field = np.asarray(field, dtype=float)
    box = as_box_lengths(box).astype(float)
    coordinates = tuple(np.asarray(v, dtype=float).ravel() for v in coordinates)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (M,3), got {points.shape}.")
    if field.ndim != 4:
        raise ValueError("field must have shape (Nsolutes,nx,ny,nz).")
    if len(coordinates) != 3:
        raise ValueError("coordinates must contain x, y and z arrays.")

    shape = field.shape[1:]
    for axis, (coord, n, length) in enumerate(zip(coordinates, shape, box)):
        if len(coord) != n:
            raise ValueError(
                f"Coordinate axis {axis} has length {len(coord)}; expected {n}."
            )
        if len(coord) == 0:
            raise ValueError("Sparse LR coordinate arrays may not be empty.")
        if np.any(~np.isfinite(coord)):
            raise ValueError("Sparse LR coordinates contain non-finite values.")
        if len(coord) > 1 and np.any(np.diff(coord) <= 0.0):
            raise ValueError("Sparse LR coordinates must be strictly increasing.")
        if coord[0] < -1.0e-12 or coord[-1] >= length + 1.0e-12:
            raise ValueError("Sparse LR coordinates lie outside the periodic box.")

    p = np.mod(points, box[None, :])
    lower = []
    upper = []
    frac = []

    for axis in range(3):
        coord = coordinates[axis]
        length = float(box[axis])
        x = p[:, axis]
        n = len(coord)

        if n == 1:
            i0 = np.zeros(len(x), dtype=np.intp)
            i1 = i0.copy()
            w = np.zeros(len(x), dtype=float)
        else:
            i0 = np.searchsorted(coord, x, side="right") - 1
            before = i0 < 0
            i0[before] = n - 1
            i1 = (i0 + 1) % n

            x0 = coord[i0].astype(float, copy=True)
            x1 = coord[i1].astype(float, copy=True)
            wrapped = i1 == 0
            x1[wrapped] += length
            x_eval = x.copy()
            x_eval[before] += length

            denom = x1 - x0
            if np.any(denom <= 0.0):
                raise ValueError("Invalid periodic sparse LR coordinate interval.")
            w = (x_eval - x0) / denom
            w = np.clip(w, 0.0, 1.0)

        lower.append(i0)
        upper.append(i1)
        frac.append(w)

    ix0, iy0, iz0 = lower
    ix1, iy1, iz1 = upper
    wx, wy, wz = frac

    out = np.zeros((field.shape[0], len(points)), dtype=float)
    for bx, ix in ((0, ix0), (1, ix1)):
        fx = (1.0 - wx) if bx == 0 else wx
        for by, iy in ((0, iy0), (1, iy1)):
            fy = (1.0 - wy) if by == 0 else wy
            for bz, iz in ((0, iz0), (1, iz1)):
                fz = (1.0 - wz) if bz == 0 else wz
                out += field[:, ix, iy, iz] * (fx * fy * fz)[None, :]

    return out


def sample_periodic_long_range_grid_spectral(
    points,
    field,
    box,
    *,
    upsample_factor=4,
):
    """Sample the periodic trigonometric LR field at arbitrary points.

    The saved FFT-grid samples are first Fourier-resampled to a finer periodic
    grid and then sampled by CIC on that fine grid.  The final CIC step is only
    a cheap sub-grid evaluator; the smooth field itself comes from the retained
    Fourier modes rather than from spatial smoothing of the coarse image.

    ``upsample_factor=1`` reduces exactly to the production CIC sampler.
    """
    fine_field = spectrally_upsample_periodic_long_range_grid(
        field, factor=upsample_factor
    )
    return sample_periodic_long_range_grid(points, fine_field, box)

def evaluate_hybrid_composition_field(
    points,
    positions,
    radii,
    Q,
    solute_names,
    box,
    alpha,
    long_range_field,
    *,
    long_range_coordinates=None,
    short_cutoff=None,
    cp=None,
    chunk_size=10_000,
    output_dtype=np.float32,
    return_particle_id=False,
    show_progress=False,
    long_range_interpolation="spectral",
    spectral_upsample_factor=4,
):
    """Reconstruct a scale-separated concentration field.

    For matrix points the physical field is

        C(x) = C_LR(x) + C_short(x),

    where C_short is evaluated with the production truncated erfc kernel.

    By default C_LR is reconstructed from the Fourier modes represented on the
    saved transient FFT grid (spectral interpolation).  Set
    ``long_range_interpolation="cic"`` to reproduce the exact LR sampler used
    by the production particle-coupling solve.
    """
    points = np.asarray(points, dtype=float)
    positions = np.asarray(positions, dtype=float)
    radii = np.asarray(radii, dtype=float)
    Q = np.asarray(Q, dtype=float)
    solute_names = tuple(str(name) for name in solute_names)
    box = as_box_lengths(box).astype(float)
    alpha = float(alpha)
    long_range_field = np.asarray(long_range_field, dtype=float)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (M,3), got {points.shape}.")
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(
            f"positions must have shape (N,3), got {positions.shape}."
        )
    if radii.shape != (len(positions),):
        raise ValueError("radii must have shape (Nparticles,).")
    if Q.shape != (len(solute_names), len(positions)):
        raise ValueError(
            "Q must have shape (Nsolutes, Nparticles); "
            f"got {Q.shape}."
        )
    expected_lr_shape = (len(solute_names),) + tuple(long_range_field.shape[1:])
    if long_range_field.ndim != 4 or long_range_field.shape[0] != len(solute_names):
        raise ValueError(
            "long_range_field must have shape (Nsolutes,nx,ny,nz); "
            f"got {long_range_field.shape}."
        )
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("alpha must be finite and positive.")

    long_range_interpolation = str(long_range_interpolation).strip().lower()
    if long_range_interpolation not in {"spectral", "cic"}:
        raise ValueError(
            "long_range_interpolation must be 'spectral' or 'cic'."
        )
    spectral_upsample_factor = int(spectral_upsample_factor)
    if spectral_upsample_factor < 1:
        raise ValueError("spectral_upsample_factor must be >= 1.")

    if short_cutoff is None:
        short_cutoff = 3.0 / alpha
    short_cutoff = float(short_cutoff)
    if not np.isfinite(short_cutoff) or short_cutoff <= 0.0:
        raise ValueError("short_cutoff must be finite and positive.")

    if cp is not None:
        cp = np.asarray(cp, dtype=float)
        if cp.shape != (len(solute_names),):
            raise ValueError("cp must have shape (Nsolutes,).")

    if return_particle_id:
        inside_precipitate, particle_id = particle_mask_on_points(
            points=points,
            positions=positions,
            radii=radii,
            box=box,
            return_particle_id=True,
        )
    else:
        inside_precipitate = particle_mask_on_points(
            points=points,
            positions=positions,
            radii=radii,
            box=box,
            return_particle_id=False,
        )
        particle_id = None

    if np.any(inside_precipitate) and cp is None:
        raise ValueError(
            "cp must be supplied when evaluation points lie inside precipitates."
        )

    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")

    n_solutes = len(solute_names)
    n_points = len(points)
    matrix_idx = np.flatnonzero(~inside_precipitate)

    c = np.full((n_solutes, n_points), np.nan, dtype=float)
    c_lr = np.full_like(c, np.nan)
    c_short = np.full_like(c, np.nan)

    # Production calculations solve only surviving particles. Using the same
    # active subset also avoids a removable singularity at centres of extinct
    # particles whose saved Q is identically zero.
    active_idx = np.flatnonzero(radii > 0.0)
    short_operator = None
    if len(active_idx):
        short_operator = HybridShortRangeOperator(
            positions=positions[active_idx],
            box=box,
            alpha=alpha,
            cutoff=short_cutoff,
        )

    # Full-resolution restart fields live on a uniform periodic FFT grid and
    # may be spectrally resampled. Sparse diagnostic histories carry their own
    # rectilinear coordinates; these must be sampled in physical coordinate
    # space because the decimation stride need not divide the production grid.
    use_rectilinear_lr = long_range_coordinates is not None
    if use_rectilinear_lr:
        lr_evaluation_grid = long_range_field
        lr_interpolation_used = "periodic_rectilinear"
    elif long_range_interpolation == "spectral":
        lr_evaluation_grid = spectrally_upsample_periodic_long_range_grid(
            long_range_field,
            factor=spectral_upsample_factor,
        )
        lr_interpolation_used = "spectral"
    else:
        lr_evaluation_grid = long_range_field
        lr_interpolation_used = "cic"

    n_chunks = (len(matrix_idx) + chunk_size - 1) // chunk_size
    for chunk_number, start in enumerate(
        range(0, len(matrix_idx), chunk_size),
        start=1,
    ):
        stop = min(start + chunk_size, len(matrix_idx))
        chunk_idx = matrix_idx[start:stop]
        chunk_points = points[chunk_idx]

        if use_rectilinear_lr:
            c_lr_chunk = sample_periodic_long_range_grid_rectilinear(
                chunk_points,
                lr_evaluation_grid,
                box,
                long_range_coordinates,
            )
        else:
            c_lr_chunk = sample_periodic_long_range_grid(
                chunk_points,
                lr_evaluation_grid,
                box,
            )
        if short_operator is None:
            c_short_chunk = np.zeros_like(c_lr_chunk)
        else:
            c_short_chunk = short_operator.evaluate_at_points(
                chunk_points,
                Q[:, active_idx],
            )

        c_lr[:, chunk_idx] = c_lr_chunk
        c_short[:, chunk_idx] = c_short_chunk
        c[:, chunk_idx] = c_lr_chunk + c_short_chunk

        if show_progress:
            print(
                f"Completed hybrid field chunk {chunk_number}/{n_chunks}."
            )

    #if np.any(inside_precipitate):
    #    c[:, inside_precipitate] = cp[:, None]
    if np.any(inside_precipitate):
        # LR/SR decomposition is only meaningful in the matrix.
        # Use a finite convention inside precipitates while preserving
        # c_total = c_long_range + c_short_range = cp.
        c[:, inside_precipitate] = cp[:, None]
        c_lr[:, inside_precipitate] = cp[:, None]
        c_short[:, inside_precipitate] = 0.0


    result = {
        "points": points.copy(),
        "solute_names": solute_names,
        "c": c.astype(output_dtype, copy=False),
        "c_long_range": c_lr.astype(output_dtype, copy=False),
        "c_short_range": c_short.astype(output_dtype, copy=False),
        "inside_precipitate": inside_precipitate,
        "field_model": "v11_hybrid_transient",
        "hybrid_alpha": alpha,
        "hybrid_matching_length": 1.0 / alpha,
        "hybrid_short_cutoff": short_cutoff,
        "hybrid_grid_shape": np.asarray(long_range_field.shape[1:], dtype=int),
        "hybrid_lr_interpolation": lr_interpolation_used,
        "hybrid_lr_spectral_upsample_factor": (
            spectral_upsample_factor
            if lr_interpolation_used == "spectral"
            else 1
        ),
        "hybrid_lr_evaluation_grid_shape": np.asarray(
            lr_evaluation_grid.shape[1:], dtype=int
        ),
    }

    if return_particle_id:
        result["particle_id"] = particle_id

    return result

def evaluate_ternary_composition_field(
    points,
    positions,
    radii,
    Q_mg,
    Q_si,
    cbar_mg,
    cbar_si,
    box,
    alpha,
    real_lattice,
    kvecs,
    k2,
    *,
    screening_length=None,
    cp_mg=None,
    cp_si=None,
    chunk_size=10_000,
    output_dtype=np.float32,
    return_particle_id=False,
    show_progress=False,
    n_workers=None,
):
    """
    Legacy Mg/Si wrapper for :func:`evaluate_composition_field`.

    The returned dictionary retains the historical ``c_mg`` and ``c_si``
    fields so existing post-processing scripts continue to work unchanged.
    """
    if (cp_mg is None) != (cp_si is None):
        raise ValueError("cp_mg and cp_si must either both be supplied or both be None.")

    cp = None if cp_mg is None else np.asarray([cp_mg, cp_si], dtype=float)

    generic = evaluate_composition_field(
        points=points,
        positions=positions,
        radii=radii,
        Q=np.vstack(
            [
                np.asarray(Q_mg, dtype=float),
                np.asarray(Q_si, dtype=float),
            ]
        ),
        cbar=np.asarray([cbar_mg, cbar_si], dtype=float),
        solute_names=("Mg", "Si"),
        box=box,
        alpha=alpha,
        real_lattice=real_lattice,
        kvecs=kvecs,
        k2=k2,
        screening_length=screening_length,
        cp=cp,
        chunk_size=chunk_size,
        output_dtype=output_dtype,
        return_particle_id=return_particle_id,
        show_progress=show_progress,
        n_workers=n_workers,
    )

    # Preserve the old public result structure. The generic ``c`` and
    # ``solute_names`` entries are intentionally retained as well so callers
    # can migrate incrementally.
    result = dict(generic)
    result["c_mg"] = generic["c"][0]
    result["c_si"] = generic["c"][1]
    return result


def reshape_field_result(result, grid_shape):
    """Reshape flat field arrays using the shape supplied by field_grids.py."""
    grid_shape = tuple(int(value) for value in grid_shape)
    expected_size = int(np.prod(grid_shape))

    if len(result["points"]) != expected_size:
        raise ValueError(
            "The number of points does not match grid_shape: "
            f"{len(result['points'])} points versus {expected_size} expected."
        )

    reshaped = dict(result)
    reshaped["points"] = result["points"].reshape(grid_shape + (3,))

    for concentration_key in ("c", "c_long_range", "c_short_range"):
        if concentration_key in result:
            values = np.asarray(result[concentration_key])
            if values.ndim != 2 or values.shape[1] != expected_size:
                raise ValueError(
                    f"result[{concentration_key!r}] must have shape "
                    "(Nsolutes, Npoints) before reshaping."
                )
            reshaped[concentration_key] = values.reshape(
                (values.shape[0],) + grid_shape
            )

    # Backward-compatible named fields, if present.
    if "c_mg" in result:
        reshaped["c_mg"] = result["c_mg"].reshape(grid_shape)

    if "c_si" in result:
        reshaped["c_si"] = result["c_si"].reshape(grid_shape)

    reshaped["inside_precipitate"] = result["inside_precipitate"].reshape(grid_shape)

    if "particle_id" in result:
        reshaped["particle_id"] = result["particle_id"].reshape(grid_shape)

    return reshaped
