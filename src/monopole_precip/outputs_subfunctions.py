import numpy as np

from pathlib import Path
import os
import pickle

from .utilities import (
    as_box_lengths,
    minimum_image_displacement,
    make_particle_masks_periodic,
)

from .greensfunction import periodic_green_ewald

def dump_monopole_concentration_history_precomputed(
    filename_prefix,
    dump_steps,
    X, Y, Z,
    positions,
    R_hist, Q_hist, cbar_hist,
    G_fields,
    box, c_p,
    show_progress=True
):
    """
    Write selected concentration fields to LAMMPS-style text dumps using precomputed G fields.
    """
    iterator = dump_steps

    for step in iterator:
        R = R_hist[step]
        Q = Q_hist[step]
        cbar = cbar_hist[step]
        positions_step = positions[step]

        cgrid, masks, union_mask = concentration_field_on_grid_precomputed(
            Q, cbar, G_fields,
            X, Y, Z,
            positions_step, R, box,
            c_inside=c_p
        )

        write_concentration_dump_lammps(
            filename=filename_prefix,
            step=step,
            X=X, Y=Y, Z=Z,
            c=cgrid,
            mask_union=union_mask,
            mode="w",
            box=box
        )

def make_uniform_xy_slice(box, Nx, Ny=None, z0=None):
    """
    Make a uniform XY slice at constant z = z0.

    Parameters
    ----------
    box : float or (3,)
        Box lengths
    Nx : int
        Number of points in x
    Ny : int or None
        Number of points in y; if None, uses Nx
    z0 : float or None
        Slice location. If None, uses the box midplane z = L/2.

    Returns
    -------
    X, Y, Z : (Nx, Ny) arrays
    dx, dy : float
    """
    box = as_box_lengths(box)
    Lx,Ly,Lz = box
    
    if Ny is None:
        Ny = Nx
    if z0 is None:
        z0 = 0.5 * Lz

    dx = Lx / Nx
    dy = Ly / Ny

    x = (np.arange(Nx) + 0.5) * dx
    y = (np.arange(Ny) + 0.5) * dy

    X, Y = np.meshgrid(x, y, indexing="ij")
    Z = np.full_like(X, z0, dtype=float)

    return X, Y, Z, dx, dy

def concentration_field_on_grid_from_Q(
    X, Y, Z,
    positions,
    R,
    Q,
    cbar,
    box,
    alpha,
    real_lattice,
    kvecs,
    k2,
    c_inside=None,
    dtype=np.float32
):
    """
    Reconstruct concentration field on a grid using the scalar
    periodic_green_ewald() function.

    Simpler, but slower than the chunked/vectorized version.
    Works for both 2D slices and 3D grids.
    """
    box = as_box_lengths(box)

    positions = np.asarray(positions, dtype=float)
    R = np.asarray(R, dtype=float)
    Q = np.asarray(Q, dtype=float)

    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    npts = pts.shape[0]

    cflat = np.full(npts, cbar, dtype=np.float64)

    for j in range(len(Q)):
        dr = pts - positions[j][None, :]
        dr = minimum_image_displacement(dr, box)

        Gj = np.empty(npts, dtype=np.float64)
        for m in range(npts):
            rnorm = np.linalg.norm(dr[m])
            if rnorm < 1e-14:
                Gj[m] = 0.0
            else:
                Gj[m] = periodic_green_ewald(
                    dr[m], box, alpha, real_lattice, kvecs, k2
                )

        cflat += Q[j] * Gj

    cgrid = cflat.reshape(X.shape).astype(dtype, copy=False)

    masks, union_mask = make_particle_masks_periodic(
        X, Y, Z, positions, R, box
    )

    if c_inside is not None:
        cgrid = np.array(cgrid, copy=True)
        cgrid[union_mask] = c_inside

    return cgrid, masks, union_mask

def write_particles_ovito_dump(
    filename,
    step,
    positions,
    radii,
    box,
    ids=None,
    types=None,
    phase_ids=None,
    gb_flags=None,
    dRdt=None,
    peclet=None,
    peclet_names=None,
    monopoles=None,
    monopole_names=None,
    mode="a",
):
    """
    Write particle data to a LAMMPS-style dump file for OVITO.

    Parameters
    ----------
    filename : str
    step : int
    positions : (N, 3) array
    radii : (N,) array
    box : float or (3,)
        Periodic box lengths
    ids : (N,) array, optional
    types : (N,) array, optional
        Legacy OVITO/LAMMPS particle type.  Existing callers use
        nuc_kind + 1, so this remains unchanged for backward compatibility.
    phase_ids : (N,) array, optional
        Zero-based precipitate phase ID written as the custom ``phase`` field.
    gb_flags : (N,) array, optional
        Grain-boundary-origin flag written as the custom ``gb`` field
        (0 = bulk, 1 = GB).
    dRdt : (N,) array, optional
        Instantaneous accepted particle radial velocity [nm/s].
    peclet : (n_species, N) or (N, n_species) array, optional
        Particle Péclet numbers from the accepted coupled solve.  The current
        The integrator convention is Pe_s = R*dRdt/(2*D_s). The dump writes
        absolute values so the diagnostic measures quasi-static validity for
        both growing and dissolving particles.
    peclet_names : sequence of str, optional
        Species names used to create columns ``Pe_<name>``.
    monopoles : (n_species, N) or (N, n_species) array, optional
        Instantaneous monopole strengths Q for each particle.  These are
        written as ``Q_<species>`` and, together with an LR-grid snapshot,
        are sufficient to reconstruct the scale-separated concentration field later.
    monopole_names : sequence of str, optional
        Species names used to create columns ``Q_<name>``.
    mode : "w" or "a"
    """

    box = as_box_lengths(box)
    Lx, Ly, Lz = box

    positions = np.asarray(positions)
    radii = np.asarray(radii)

    N = positions.shape[0]

    if ids is None:
        ids = np.arange(1, N + 1)

    if types is None:
        types = np.ones(N, dtype=int)

    if phase_ids is None:
        phase_ids = np.zeros(N, dtype=int)

    if gb_flags is None:
        gb_flags = np.zeros(N, dtype=int)

    ids = np.asarray(ids)
    types = np.asarray(types)
    phase_ids = np.asarray(phase_ids)
    gb_flags = np.asarray(gb_flags)

    if dRdt is not None:
        dRdt = np.asarray(dRdt, dtype=float)
        if dRdt.shape != (N,):
            raise ValueError(
                f"dRdt must have shape ({N},); got {dRdt.shape}."
            )

    pe_values = None
    pe_names = None
    if peclet is not None:
        pe_values = np.asarray(peclet, dtype=float)
        if pe_values.ndim == 1:
            pe_values = pe_values.reshape(1, -1)
        if pe_values.ndim != 2:
            raise ValueError("peclet must be one- or two-dimensional.")
        if pe_values.shape[1] == N:
            pass
        elif pe_values.shape[0] == N:
            pe_values = pe_values.T
        else:
            raise ValueError(
                "peclet must have particle dimension N; "
                f"got shape {pe_values.shape} for N={N}."
            )

        if peclet_names is None:
            pe_names = [f"species{i}" for i in range(pe_values.shape[0])]
        else:
            pe_names = [str(name) for name in peclet_names]
            if len(pe_names) != pe_values.shape[0]:
                raise ValueError(
                    "peclet_names length must match the number of species in peclet."
                )
        pe_values = np.abs(pe_values)

    q_values = None
    q_names = None
    if monopoles is not None:
        q_values = np.asarray(monopoles, dtype=float)
        if q_values.ndim == 1:
            q_values = q_values.reshape(1, -1)
        if q_values.ndim != 2:
            raise ValueError("monopoles must be one- or two-dimensional.")
        if q_values.shape[1] == N:
            pass
        elif q_values.shape[0] == N:
            q_values = q_values.T
        else:
            raise ValueError(
                "monopoles must have particle dimension N; "
                f"got shape {q_values.shape} for N={N}."
            )
        if monopole_names is None:
            q_names = [f"species{i}" for i in range(q_values.shape[0])]
        else:
            q_names = [str(name) for name in monopole_names]
            if len(q_names) != q_values.shape[0]:
                raise ValueError(
                    "monopole_names length must match the number of species in monopoles."
                )

    for name, values in (
        ("ids", ids),
        ("types", types),
        ("phase_ids", phase_ids),
        ("gb_flags", gb_flags),
    ):
        if values.shape != (N,):
            raise ValueError(
                f"{name} must have shape ({N},); got {values.shape}."
            )

    xlo, xhi = 0.0, Lx
    ylo, yhi = 0.0, Ly
    zlo, zhi = 0.0, Lz

    with open(filename, mode) as f:
        f.write("ITEM: TIMESTEP\n")
        f.write(f"{step}\n")

        f.write("ITEM: NUMBER OF ATOMS\n")
        f.write(f"{N}\n")

        f.write("ITEM: BOX BOUNDS pp pp pp\n")
        f.write(f"{xlo} {xhi}\n")
        f.write(f"{ylo} {yhi}\n")
        f.write(f"{zlo} {zhi}\n")

        columns = ["id", "type", "phase", "gb", "x", "y", "z", "Radius"]
        if dRdt is not None:
            columns.append("dRdt")
        if q_values is not None:
            columns.extend([f"Q_{name}" for name in q_names])
        if pe_values is not None:
            columns.extend([f"Pe_{name}" for name in pe_names])
            columns.append("Pe_max")
        f.write("ITEM: ATOMS " + " ".join(columns) + "\n")

        for i in range(N):
            x, y, z = positions[i]
            values = [
                f"{ids[i]}",
                f"{types[i]}",
                f"{phase_ids[i]}",
                f"{gb_flags[i]}",
                f"{x:.6e}",
                f"{y:.6e}",
                f"{z:.6e}",
                f"{radii[i]:.6e}",
            ]
            if dRdt is not None:
                values.append(f"{dRdt[i]:.6e}")
            if q_values is not None:
                values.extend(f"{q_values[j, i]:.10e}" for j in range(q_values.shape[0]))
            if pe_values is not None:
                values.extend(f"{pe_values[j, i]:.6e}" for j in range(pe_values.shape[0]))
                values.append(f"{np.max(pe_values[:, i]):.6e}")
            f.write(" ".join(values) + "\n")

# For outputting solute field to dump file
def make_uniform_grid(box, N):
    """
    Make a uniform cell-centered 3D grid for an orthorhombic box.

    Parameters
    ----------
    box : float or (3,)
        Box lengths. Scalar -> cubic, length-3 -> [Lx, Ly, Lz]
    N : int or (3,)
        Number of grid points. Scalar -> same in all directions,
        length-3 -> [Nx, Ny, Nz]

    Returns
    -------
    X, Y, Z : arrays of shape (Nx, Ny, Nz)
    dx, dy, dz : float
    """
    box = as_box_lengths(box)
    Lx, Ly, Lz = box

    if np.isscalar(N):
        Nx = Ny = Nz = int(N)
    else:
        Nx, Ny, Nz = map(int, N)

    dx = Lx / Nx
    dy = Ly / Ny
    dz = Lz / Nz

    x = (np.arange(Nx) + 0.5) * dx
    y = (np.arange(Ny) + 0.5) * dy
    z = (np.arange(Nz) + 0.5) * dz

    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    return X, Y, Z, dx, dy, dz


def write_concentration_dump_lammps(
    filename,
    step,
    X,
    Y,
    Z,
    c,
    mask_union,
    mode="w",
    dz_slice=None,
    box=None
):
    """
    Write concentration field in a LAMMPS dump-style format readable by OVITO.

    Works for:
        - full 3D grids
        - 2D slices (e.g. X,Y at constant Z)

    type = 1 : matrix
    type = 2 : precipitate

    If box is provided, the dump bounds are taken directly from box.
    Otherwise they are inferred from the grid coordinates.
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    Z = np.asarray(Z)
    c = np.asarray(c)
    mask_union = np.asarray(mask_union, dtype=bool)

    if not (X.shape == Y.shape == Z.shape == c.shape == mask_union.shape):
        raise ValueError("X, Y, Z, c, and mask_union must all have the same shape")

    shape = c.shape
    natoms = c.size

    # Determine grid spacing
    if X.ndim == 3:
        nx, ny, nz = shape

        dx = X[1, 0, 0] - X[0, 0, 0] if nx > 1 else 1.0
        dy = Y[0, 1, 0] - Y[0, 0, 0] if ny > 1 else 1.0

        if nz > 1:
            dz = Z[0, 0, 1] - Z[0, 0, 0]
        else:
            dz = dz_slice if dz_slice is not None else min(dx, dy)

    elif X.ndim == 2:
        nx, ny = shape

        dx = X[1, 0] - X[0, 0] if nx > 1 else 1.0
        dy = Y[0, 1] - Y[0, 0] if ny > 1 else 1.0
        dz = dz_slice if dz_slice is not None else min(dx, dy)

    else:
        raise ValueError("X, Y, Z must be 2D or 3D arrays")

    if box is None:
        xlo = np.min(X) - 0.5 * dx
        xhi = np.max(X) + 0.5 * dx

        ylo = np.min(Y) - 0.5 * dy
        yhi = np.max(Y) + 0.5 * dy

        zlo = np.min(Z) - 0.5 * dz
        zhi = np.max(Z) + 0.5 * dz
    else:
        box = as_box_lengths(box)
        xlo, xhi = 0.0, box[0]
        ylo, yhi = 0.0, box[1]
        zlo, zhi = 0.0, box[2]

    ids = np.arange(1, natoms + 1, dtype=int)
    types = np.where(mask_union.ravel(), 2, 1)

    data = np.column_stack([
        ids,
        types,
        X.ravel(),
        Y.ravel(),
        Z.ravel(),
        c.ravel()
    ])

    outname = f"{filename}.dump"

    with open(outname, mode) as f:
        f.write("ITEM: TIMESTEP\n")
        f.write(f"{step}\n")

        f.write("ITEM: NUMBER OF ATOMS\n")
        f.write(f"{natoms}\n")

        f.write("ITEM: BOX BOUNDS pp pp pp\n")
        f.write(f"{xlo:.8e} {xhi:.8e}\n")
        f.write(f"{ylo:.8e} {yhi:.8e}\n")
        f.write(f"{zlo:.8e} {zhi:.8e}\n")

        f.write("ITEM: ATOMS id type x y z c\n")

        np.savetxt(
            f,
            data,
            fmt=["%d", "%d", "%.6e", "%.6e", "%.6e", "%.6e"]
        )

def concentration_field_on_grid_precomputed(
    Q, cbar, G_fields,
    X, Y, Z,
    positions, R, box,
    c_inside=None
):
    """
    Reconstruct concentration field using precomputed Green fields.

    c(x) = cbar + sum_p Q[p] * G_fields[p](x)

    Returns
    -------
    cgrid : ndarray
    masks : list of bool arrays
    union_mask : bool array
    """
    # Fast linear combination of precomputed unit fields
    cgrid = cbar + np.tensordot(Q, G_fields, axes=(0, 0))

    # Rebuild particle masks because radii change with time
    masks, union_mask = make_particle_masks_periodic(X, Y, Z, positions, R, box)

    if c_inside is not None:
        cgrid = np.array(cgrid, copy=True)
        cgrid[union_mask] = c_inside

    return cgrid, masks, union_mask


def write_hybrid_long_range_snapshot_npz(
    filename, *, time_s, temperature_K, long_range_field, solute_names,
    matching_length_nm, box=None, field_array=None, extra_metadata=None,
):
    """Write one native long-range concentration-grid snapshot.

    The file contains the *native solver grid*, not an upsampled visualization
    grid.  It is intentionally independent of the restart/checkpoint file so
    old snapshots never remain resident in memory or get rewritten.
    """
    if long_range_field is None:
        return None
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    field = np.asarray(
        long_range_field.copy_field() if field_array is None else field_array,
        dtype=float,
    ).copy()
    lr_box = np.asarray(long_range_field.box if box is None else box, dtype=float)
    shape = np.asarray(long_range_field.grid_shape, dtype=np.int32)
    spacing = np.asarray(long_range_field.spacing, dtype=float)
    payload = {
        "format": np.asarray("V11_native_long_range_snapshot_v1"),
        "time_s": np.asarray(float(time_s)),
        "temperature_K": np.asarray(float(temperature_K)),
        "temperature_C": np.asarray(float(temperature_K) - 273.15),
        "box": lr_box,
        "grid_shape": shape,
        "grid_spacing_nm": spacing,
        "solute_names": np.asarray([str(x) for x in solute_names]),
        "c_long_range": field,
        "hybrid_matching_length_nm": np.asarray(float(matching_length_nm)),
        "hybrid_alpha_nm_inv": np.asarray(1.0 / float(matching_length_nm)),
        "matrix_inventory": np.sum(field, axis=(1,2,3)) * float(long_range_field.cell_volume),
        "mean_concentration": np.mean(field, axis=(1,2,3)),
    }
    if extra_metadata:
        for key, value in extra_metadata.items():
            payload[str(key)] = np.asarray(value)
    np.savez_compressed(tmp, **payload)
    os.replace(tmp, path)
    return path
