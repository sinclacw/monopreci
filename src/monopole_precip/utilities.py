import numpy as np

#-----------------------------------------
# # vector / geometry basics

# normalize_vector
# make_plane_basis
# signed_distance_to_plane 
# project_point_to_plane
# plane_coordinates

# # box / periodic helpers

# as_box_lengths
# minimum_image_displacement
# wrap_point_into_box
# periodic_distance_points
# minimum_image_delta

# # plane geometry in periodic box

# point_plane_signed_distance_periodic
# project_point_to_plane_periodic
# plane_coordinates_periodic
# periodic_distance_2d
# gb_footprint_radius
# is_in_gb_slab

# # random point sampling

# sample_point_in_box
# sample_bulk_point
# sample_gb_point

# # masks / grids

# spherical_mask_periodic
# make_particle_masks_periodic

# # GB occupancy helpers

# estimate_gb_free_area_fraction
# gb_candidate_overlaps_existing
# plane_coordinates_xnormal
# gb_footprint_radius_xnormal

# # conservation helpers

# particle_volumes
# total_solute_inventory
# total_solute_from_state
# solute_conservation_error
# mean_matrix_concentration

#-----------------------------------------

def as_box_lengths(box):
    """
    Return box lengths as a numpy array [Lx, Ly, Lz].

    Accepts either:
      - scalar L  -> interpreted as cubic [L, L, L]
      - iterable of length 3 -> [Lx, Ly, Lz]
    """
    box = np.asarray(box, dtype=float)
    if box.ndim == 0:
        box = np.array([box, box, box], dtype=float)
    if box.shape != (3,):
        raise ValueError("box must be a scalar or an iterable of length 3")
    if np.any(box <= 0.0):
        raise ValueError("all box lengths must be > 0")
    return box

## Helper functions for heteogeneous nucleation
def normalize_vector(v, tol=1e-15):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < tol:
        raise ValueError("Cannot normalize near-zero vector")
    return v / n

def make_plane_basis(normal):
    """
    Construct two orthonormal in-plane vectors e1, e2 for a plane
    with unit normal 'normal'.
    """
    n = normalize_vector(normal)

    # choose a vector not parallel to n
    if abs(n[0]) < 0.9:
        a = np.array([1.0, 0.0, 0.0])
    else:
        a = np.array([0.0, 1.0, 0.0])

    e1 = a - np.dot(a, n) * n
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    return e1, e2

def minimum_image_displacement(dvec, box):
    """
    Apply minimum-image convention to displacement vector(s) in an orthorhombic
    periodic box.

    Works for shape (..., 3).
    """
    box = as_box_lengths(box)
    dvec = np.asarray(dvec, dtype=float)
    return dvec - box * np.round(dvec / box)

def wrap_point_into_box(point, box):
    """
    Wrap point into [0,Lx) x [0,Ly) x [0,Lz).
    """
    box = as_box_lengths(box)
    point = np.asarray(point, dtype=float)
    return np.mod(point, box)

def periodic_distance_points(p1, p2, box):
    """
    Minimum-image distance between two 3D points in a periodic orthorhombic box.
    """
    dvec = minimum_image_displacement(np.asarray(p2) - np.asarray(p1), box)
    return np.linalg.norm(dvec)


def point_plane_signed_distance_periodic(point, gb_point, gb_normal, box):
    """
    Signed distance from point to plane, using minimum-image displacement
    relative to gb_point.
    """
    n = normalize_vector(gb_normal)
    dvec = minimum_image_displacement(np.asarray(point) - np.asarray(gb_point), box)
    return np.dot(dvec, n)


def project_point_to_plane_periodic(point, gb_point, gb_normal, box, wrap=True):
    """
    Orthogonal projection of a point onto the plane, using minimum-image
    displacement relative to gb_point.

    If wrap=True, wraps the projected point back into the periodic box.
    """
    n = normalize_vector(gb_normal)
    point = np.asarray(point, dtype=float)
    dvec = minimum_image_displacement(point - np.asarray(gb_point), box)
    d = np.dot(dvec, n)
    proj = point - d * n
    return wrap_point_into_box(proj, box) if wrap else proj


def plane_coordinates_periodic(point, gb_point, e1, e2, box):
    """
    2D in-plane coordinates of a point relative to gb_point, using
    minimum-image displacement.

    e1 and e2 should be orthonormal vectors spanning the plane.
    """
    dvec = minimum_image_displacement(np.asarray(point) - np.asarray(gb_point), box)
    return np.array([np.dot(dvec, e1), np.dot(dvec, e2)])


def periodic_distance_2d(s1, s2, L1, L2):
    """
    Minimum-image distance in 2D periodic coordinates on a rectangular
    in-plane cell of side lengths L1 and L2.
    """
    ds = np.asarray(s2, dtype=float) - np.asarray(s1, dtype=float)
    ds[0] -= L1 * np.round(ds[0] / L1)
    ds[1] -= L2 * np.round(ds[1] / L2)
    return np.linalg.norm(ds)


def gb_footprint_radius(center, radius, gb_point, gb_normal, box):
    """
    Radius of the circular intersection of a spherical particle with the GB plane.
    Returns 0 if the sphere does not intersect the plane.
    """
    d = abs(point_plane_signed_distance_periodic(center, gb_point, gb_normal, box))
    if d >= radius:
        return 0.0
    return np.sqrt(max(0.0, radius**2 - d**2))


def is_in_gb_slab(point, gb_point, gb_normal, gb_thickness, box):
    """
    True if point lies within a slab of thickness gb_thickness centered on the GB plane.
    """
    d = point_plane_signed_distance_periodic(point, gb_point, gb_normal, box)
    return abs(d) <= 0.5 * gb_thickness


def sample_point_in_box(box, rng):
    """
    Uniform random point in a periodic orthorhombic box [0,Lx]x[0,Ly]x[0,Lz].
    """
    box = as_box_lengths(box)
    return rng.uniform(0.0, box, size=3)

def sample_bulk_point(box, rng, gb_point, gb_normal, gb_thickness, max_tries=200):
    """
    Sample a point outside the GB slab.
    """
    for _ in range(max_tries):
        pt = sample_point_in_box(box, rng)
        if not is_in_gb_slab(pt, gb_point, gb_normal, gb_thickness, box):
            return pt
    return None


def sample_gb_point(box, rng, gb_point, gb_normal, gb_thickness, max_tries=500):
    """
    Sample a point inside the GB slab.
    """
    for _ in range(max_tries):
        pt = sample_point_in_box(box, rng)
        if is_in_gb_slab(pt, gb_point, gb_normal, gb_thickness, box):
            return pt
    return None


def minimum_image_delta(x, x0, L):
    """
    Minimum-image displacement for one coordinate direction.
    """
    dx = x - x0
    dx -= L * np.round(dx / L)
    return dx


def spherical_mask_periodic(X, Y, Z, center, R, box):
    """
    Periodic spherical mask in an orthorhombic box.
    """
    box = as_box_lengths(box)
    center = np.asarray(center, dtype=float)

    dx = minimum_image_delta(X, center[0], box[0])
    dy = minimum_image_delta(Y, center[1], box[1])
    dz = minimum_image_delta(Z, center[2], box[2])

    return (dx*dx + dy*dy + dz*dz) <= R*R


def make_particle_masks_periodic(X, Y, Z, centers, radii, box):
    """
    Returns
    -------
    masks : list of boolean arrays
        One mask per particle.
    union_mask : boolean array
        True anywhere inside any particle.
    """
    box = as_box_lengths(box)
    centers = np.asarray(centers, dtype=float)
    radii = np.asarray(radii, dtype=float)

    masks = []
    union_mask = np.zeros_like(X, dtype=bool)

    for center, R in zip(centers, radii):
        m = spherical_mask_periodic(X, Y, Z, center, R, box)
        masks.append(m)
        union_mask |= m

    return masks, union_mask


def particle_volumes(R):
    return (4.0 / 3.0) * np.pi * R**3

def green_matrix_inventory_integrals(
    radii,
    G0,
    Gpair,
):
    """
    Approximate the integral of each periodic Green basis function
    over the matrix domain.

    For source particle i,

        I_i = integral_{Omega_m} G(x - x_i) dV

    is calculated using the zero-mean property over the complete
    periodic box:

        I_i = -integral_{Omega_p} G(x - x_i) dV.

    Each precipitate is treated as a small sphere. The self integral
    retains the exact singular contribution R_i**2 / 2 and the regular
    self contribution G0 * V_i. Cross-particle contributions are
    approximated by taking G to be constant over each other particle:

        I_i ~= -[
            R_i**2 / 2
            + G0 * V_i
            + sum_{j != i} G_{j,i} V_j
        ].

    Parameters
    ----------
    radii : ndarray, shape (N,)
        Positive particle radii.

    G0 : float
        Regular self constant of the periodic Green function.

    Gpair : ndarray, shape (N, N)
        Pair Green-function matrix. The diagonal must be zero because
        the self contribution is handled separately.

        Gpair[j, i] is the Green function at particle j due to source i.

    Returns
    -------
    matrix_integrals : ndarray, shape (N,)
        Approximate values of

            integral_{Omega_m} G(x - x_i) dV

        for each source particle i.

    Notes
    -----
    Units are length squared because G has units of inverse length and
    the integral is over volume.
    """
    radii = np.asarray(radii, dtype=float)
    Gpair = np.asarray(Gpair, dtype=float)
    G0 = float(G0)

    if radii.ndim != 1:
        raise ValueError(
            f"radii must be one-dimensional, got shape {radii.shape}."
        )

    n_particles = len(radii)

    if Gpair.shape != (n_particles, n_particles):
        raise ValueError(
            "Gpair must have shape "
            f"({n_particles}, {n_particles}), got {Gpair.shape}."
        )

    if np.any(~np.isfinite(radii)) or np.any(radii <= 0.0):
        raise ValueError(
            "radii must contain finite positive values."
        )

    if not np.isfinite(G0):
        raise ValueError("G0 must be finite.")

    if np.any(~np.isfinite(Gpair)):
        raise ValueError("Gpair contains non-finite values.")

    diagonal = np.diag(Gpair)

    if not np.allclose(
        diagonal,
        0.0,
        rtol=0.0,
        atol=1.0e-14,
    ):
        raise ValueError(
            "The diagonal of Gpair must be zero because the self "
            "contribution is handled separately."
        )

    volumes = particle_volumes(radii)

    # Integral over the source particle's own spherical volume:
    #
    # integral [1/(4*pi*r)] dV = R^2 / 2
    #
    # plus the regular periodic self contribution G0 * V.
    self_integrals = (
        0.5 * radii**2
        + G0 * volumes
    )

    # For source i, sum over all other excluded particle volumes j:
    #
    # sum_j Gpair[j, i] * V_j
    cross_integrals = Gpair.T @ volumes

    # Whole-box Green functions have zero volume average, so the
    # matrix-domain integral is the negative of the integral over all
    # excluded precipitate volumes.
    matrix_integrals = -(
        self_integrals
        + cross_integrals
    )

    if np.any(~np.isfinite(matrix_integrals)):
        raise RuntimeError(
            "Non-finite Green-function matrix integrals were obtained."
        )

    return matrix_integrals


def total_solute_inventory(box, R0, cbar0, c_p):
    box = as_box_lengths(box)
    V_box = np.prod(box)
    R0 = np.asarray(R0, dtype=float)
    
    Vp0 = particle_volumes(R0)
    Vm0 = V_box - np.sum(Vp0)
    if Vm0 <= 0.0:
        raise ValueError("Initial particles fill or exceed the box volume.")
    return Vm0 * cbar0 + np.sum(Vp0) * c_p

def total_solute_from_state(box, R, cbar, c_p):
    """
    Compute total solute from the current state.

    Parameters
    ----------
    box : float or (3,)
        Box lengths
    R : array_like
        Particle radii
    cbar : float
        Mean matrix concentration
    c_p : float
        Precipitate concentration

    Returns
    -------
    M_total : float
    """
    box = as_box_lengths(box)
    V_box = np.prod(box)

    R = np.asarray(R, dtype=float)

    Vp = particle_volumes(R)
    Vm = V_box - np.sum(Vp)

    return Vm * cbar + np.sum(Vp) * c_p


def solute_conservation_error(box, R, cbar, c_p, M_total):
    """
    Absolute and relative conservation error.

    Parameters
    ----------
    box : float or (3,)
        Box lengths
    R : array_like
        Particle radii
    cbar : float
        Mean matrix concentration
    c_p : float
        Precipitate concentration
    M_total : float
        Reference total solute

    Returns
    -------
    M_now : float
    abs_err : float
    rel_err : float
    """
    M_now = total_solute_from_state(box, R, cbar, c_p)
    abs_err = M_now - M_total
    rel_err = abs_err / M_total if M_total != 0 else abs_err
    return M_now, abs_err, rel_err


def estimate_gb_free_area_fraction(
    centers,
    radii,
    nuc_kind,
    gb_point,
    gb_normal,
    gb_area,
    box,
    exclusion_buffer=0.0,
):
    """
    Estimate free fraction of GB area using projected circular footprints of
    GB-origin particles only.

    Parameters
    ----------
    centers : (N,3) array
    radii : (N,) array
    nuc_kind : (N,) int array
        0 = bulk, 1 = GB
    gb_point : (3,) array
        A point on the GB plane.
    gb_normal : (3,) array
        Unit normal (or will be normalized internally).
    gb_area : float
        Total GB plane area in the periodic box.
        Example for GB normal to x: gb_area = Ly * Lz
    box : scalar or length-3 iterable
        Periodic box lengths [Lx, Ly, Lz]
    exclusion_buffer : float
        Optional additional radial buffer added to each footprint.

    Returns
    -------
    phi_free : float
        Estimated free area fraction in [0,1].

    Notes
    -----
    This is a simple area-sum estimate:
        A_occ = sum(pi * (r_foot + buffer)^2)
    so overlaps between footprints are ignored.
    Therefore it can underpredict the true free fraction if footprints overlap.
    """
    centers = np.asarray(centers, dtype=float)
    radii = np.asarray(radii, dtype=float)
    nuc_kind = np.asarray(nuc_kind, dtype=np.int32)

    if gb_area <= 0.0 or centers.shape[0] == 0:
        return 1.0

    mask = (nuc_kind == 1) & (radii > 0.0)
    if not np.any(mask):
        return 1.0

    rfoot = []
    for c, R in zip(centers[mask], radii[mask]):
        rf = gb_footprint_radius(c, R, gb_point, gb_normal, box)
        if rf > 0.0:
            rfoot.append(rf + exclusion_buffer)

    if len(rfoot) == 0:
        return 1.0

    A_occ = np.sum(np.pi * np.square(rfoot))
    phi_free = max(0.0, 1.0 - A_occ / gb_area)
    return phi_free

def gb_candidate_overlaps_existing(
    candidate_center,
    candidate_radius,
    centers,
    radii,
    nuc_kind,
    gb_point,
    box,
    exclusion_buffer=0.0,
):
    """
    Check whether a candidate GB nucleus overlaps the projected GB footprints
    of existing GB-origin particles, for a GB normal to x.
    """
    box = as_box_lengths(box)

    centers = np.asarray(centers, dtype=float)
    radii = np.asarray(radii, dtype=float)
    nuc_kind = np.asarray(nuc_kind, dtype=np.int32)
    gb_point = np.asarray(gb_point, dtype=float)

    gb_x = gb_point[0]
    gb_L1 = box[1]
    gb_L2 = box[2]

    rf_cand = gb_footprint_radius_xnormal(candidate_center, candidate_radius, gb_x, box)
    if rf_cand <= 0.0:
        return False

    mask = (nuc_kind == 1) & (radii > 0.0)
    if not np.any(mask):
        return False

    sc = plane_coordinates_xnormal(candidate_center, gb_point, box)

    for c, R in zip(centers[mask], radii[mask]):
        rf = gb_footprint_radius_xnormal(c, R, gb_x, box)
        if rf <= 0.0:
            continue

        si = plane_coordinates_xnormal(c, gb_point, box)
        d2d = periodic_distance_2d(sc, si, gb_L1, gb_L2)

        if d2d < (rf_cand + rf + exclusion_buffer):
            return True

    return False

def plane_coordinates_xnormal(point, gb_point, box):
    """
    In-plane GB coordinates for x-normal GB.
    Returns [y_rel, z_rel] relative to gb_point, with periodic
    minimum-image convention in y and z.
    """
    box = as_box_lengths(box)
    point = np.asarray(point, dtype=float)
    gb_point = np.asarray(gb_point, dtype=float)

    dy = point[1] - gb_point[1]
    dz = point[2] - gb_point[2]

    dy -= box[1] * np.round(dy / box[1])
    dz -= box[2] * np.round(dz / box[2])

    return np.array([dy, dz], dtype=float)

def gb_footprint_radius_xnormal(center, radius, gb_x, box):
    """
    Radius of the circular intersection of a spherical particle with
    the x-normal GB plane x = gb_x.
    Returns 0 if the sphere does not intersect the plane.
    """
    box = as_box_lengths(box)
    center = np.asarray(center, dtype=float)

    dx = center[0] - gb_x
    dx -= box[0] * np.round(dx / box[0])
    d = abs(dx)

    if d >= radius:
        return 0.0

    return np.sqrt(max(0.0, radius**2 - d**2))

def mean_matrix_concentration(box, R, M_total, c_p, tol=1e-14):
    """
    Compute mean matrix concentration from total solute inventory.

    Parameters
    ----------
    box : float or (3,)
        Box lengths
    R : array_like
        Particle radii
    M_total : float
        Total solute inventory
    c_p : float
        Precipitate concentration

    Returns
    -------
    cbar : float
        Mean matrix concentration
    """
    box = as_box_lengths(box)
    V_box = np.prod(box)

    R = np.asarray(R, dtype=float)

    Vp = particle_volumes(R)
    Vp_tot = np.sum(Vp)
    Vm = V_box - np.sum(Vp)

    if Vm <= 0.0:
        raise ValueError("Particles fill or exceed the box volume.")

    M_matrix = M_total - Vp_tot * c_p
    
    if M_matrix < -tol:
        raise ValueError(
            f"Negative matrix solute inventory detected: "
            f"M_matrix = {M_matrix:.6e}. "
            "Particles have consumed more solute than is available."
        )
    
    return (M_total - np.sum(Vp) * c_p) / Vm


# Surface concentration calculation
def evaluate_csurf(R, c_surf):
    """
    Evaluate interface concentration for current radii.

    Parameters
    ----------
    R : (N,) array
    c_surf : scalar, array_like, or callable

    Returns
    -------
    out : (N,) array
    """
    R = np.asarray(R, dtype=float)

    if callable(c_surf):
        out = np.asarray(c_surf(R), dtype=float)
    elif np.ndim(c_surf) == 0:
        out = np.full(len(R), float(c_surf))
    else:
        out = np.asarray(c_surf, dtype=float)

    if len(out) != len(R):
        raise ValueError(
            f"c_surf must be scalar, callable, or length {len(R)}; got length {len(out)}"
        )

    return out

#-------------------------------------
# For ternary model
#-------------------------------------

def ternary_total_solute_inventory(
    box,
    radii,
    cbar_mg,
    cbar_si,
    cp_mg,
    cp_si,
):
    """
    Return total Mg and Si inventories.

    Concentrations are number densities [atoms / volume].
    Radii and box dimensions must use consistent length units.
    """
    box = as_box_lengths(box)

    Vbox = np.prod(box)
    Vp = np.sum(particle_volumes(radii))
    Vm = Vbox - Vp

    if Vm < 0.0:
        raise ValueError(
            "Particle volume exceeds the simulation volume."
        )

    M_mg = cbar_mg * Vm + cp_mg * Vp
    M_si = cbar_si * Vm + cp_si * Vp

    return M_mg, M_si

def ternary_mean_matrix_concentrations(
    box,
    radii,
    M_total_mg,
    M_total_si,
    cp_mg,
    cp_si,
):
    """
    Calculate mean Mg and Si matrix concentrations from exact
    global conservation.
    """
    box = as_box_lengths(box)

    Vbox = np.prod(box)
    Vp = np.sum(particle_volumes(radii))
    Vm = Vbox - Vp

    if Vm <= 0.0:
        raise ValueError(
            "No positive matrix volume remains."
        )

    cbar_mg = (
        M_total_mg - cp_mg * Vp
    ) / Vm

    cbar_si = (
        M_total_si - cp_si * Vp
    ) / Vm

    return cbar_mg, cbar_si

def ternary_solute_conservation_error(
    box,
    radii,
    cbar_mg,
    cbar_si,
    cp_mg,
    cp_si,
    M_total_mg,
    M_total_si,
):
    """
    Return current inventories and relative conservation errors.
    """
    M_mg, M_si = ternary_total_solute_inventory(
        box=box,
        radii=radii,
        cbar_mg=cbar_mg,
        cbar_si=cbar_si,
        cp_mg=cp_mg,
        cp_si=cp_si,
    )

    abs_err_mg = M_mg - M_total_mg
    abs_err_si = M_si - M_total_si

    rel_err_mg = (
        abs_err_mg / M_total_mg
        if M_total_mg != 0.0
        else abs_err_mg
    )

    rel_err_si = (
        abs_err_si / M_total_si
        if M_total_si != 0.0
        else abs_err_si
    )

    return {
        "M_mg": M_mg,
        "M_si": M_si,
        "abs_err_mg": abs_err_mg,
        "abs_err_si": abs_err_si,
        "rel_err_mg": rel_err_mg,
        "rel_err_si": rel_err_si,
    }



def total_solute_inventory_with_green_field(
    box,
    radii,
    cbar,
    cp,
    Q,
    G0,
    Gpair,
):
    """
    Return species-indexed inventories represented by reconstructed
    Green-function concentration fields.

    Parameters
    ----------
    cbar : array_like, shape (S,)
        Mean matrix number concentrations.
    cp : array_like, shape (S,) or (S, N)
        Precipitate concentrations; two-dimensional input is particle-specific.
    Q : array_like, shape (S, N)
        Green-function monopole strengths, with species first.
    """
    box = as_box_lengths(box)
    radii = np.asarray(radii, dtype=float)
    cbar = np.asarray(cbar, dtype=float)
    cp = np.asarray(cp, dtype=float)
    Q = np.asarray(Q, dtype=float)

    if cbar.ndim != 1:
        raise ValueError("cbar must have shape (Nspecies,).")

    n_solutes = cbar.size
    n_particles = radii.size

    if cp.shape == (n_solutes,):
        cp_particle = np.repeat(cp[:, None], n_particles, axis=1)
    elif cp.shape == (n_solutes, n_particles):
        cp_particle = cp
    else:
        raise ValueError(
            f"cp must have shape ({n_solutes},) or "
            f"({n_solutes}, {n_particles}); got {cp.shape}."
        )

    if Q.shape != (n_solutes, n_particles):
        raise ValueError(
            "Q must have shape "
            f"({n_solutes}, {n_particles}); got {Q.shape}."
        )

    if (
        np.any(~np.isfinite(cbar))
        or np.any(~np.isfinite(cp))
        or np.any(~np.isfinite(Q))
    ):
        raise ValueError(
            "cbar, cp, and Q must contain only finite values."
        )

    Vbox = float(np.prod(box))
    volumes = particle_volumes(radii)
    Vp = float(np.sum(volumes))
    Vm = Vbox - Vp

    if Vm <= 0.0:
        raise ValueError("No positive matrix volume remains.")

    matrix_integrals = green_matrix_inventory_integrals(
        radii=radii,
        G0=G0,
        Gpair=Gpair,
    )

    matrix_uniform = cbar * Vm
    green_inventory = Q @ matrix_integrals
    particle_inventory = cp_particle @ volumes
    M = matrix_uniform + green_inventory + particle_inventory

    return {
        "M": M,
        "matrix_uniform": matrix_uniform,
        "green_inventory": green_inventory,
        "particle_inventory": particle_inventory,
        "Vbox": Vbox,
        "Vp": Vp,
        "Vm": Vm,
        "matrix_integrals": matrix_integrals,
    }

def ternary_total_solute_inventory_with_green_field(
    box,
    radii,
    cbar_mg,
    cbar_si,
    cp_mg,
    cp_si,
    Q_mg,
    Q_si,
    G0,
    Gpair,
):
    """
    Backward-compatible Mg/Si wrapper around the species-indexed
    total_solute_inventory_with_green_field().
    """
    result = total_solute_inventory_with_green_field(
        box=box,
        radii=radii,
        cbar=np.asarray([cbar_mg, cbar_si], dtype=float),
        cp=np.asarray([cp_mg, cp_si], dtype=float),
        Q=np.vstack(
            (
                np.asarray(Q_mg, dtype=float),
                np.asarray(Q_si, dtype=float),
            )
        ),
        G0=G0,
        Gpair=Gpair,
    )

    return {
        "M_mg": float(result["M"][0]),
        "M_si": float(result["M"][1]),
        "matrix_uniform_mg": float(result["matrix_uniform"][0]),
        "matrix_uniform_si": float(result["matrix_uniform"][1]),
        "green_inventory_mg": float(result["green_inventory"][0]),
        "green_inventory_si": float(result["green_inventory"][1]),
        "particle_inventory_mg": float(result["particle_inventory"][0]),
        "particle_inventory_si": float(result["particle_inventory"][1]),
        "Vbox": result["Vbox"],
        "Vp": result["Vp"],
        "Vm": result["Vm"],
        "matrix_integrals": result["matrix_integrals"],
    }
