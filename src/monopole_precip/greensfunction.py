import numpy as np
from scipy.special import erfc
from scipy.sparse.linalg import gmres, LinearOperator

from .utilities import (
    as_box_lengths,
    minimum_image_displacement,
    mean_matrix_concentration,
    evaluate_csurf,
)


# ============================================================
# Lattice / reciprocal vectors for Ewald sums
# ============================================================

def generate_real_lattice(box, n_real):
    """
    Generate real-space lattice translation vectors for an orthorhombic box.

    Parameters
    ----------
    box : float or (3,) array_like
        Box lengths. Scalar -> cubic, length-3 -> [Lx, Ly, Lz]
    n_real : int
        Include translations with nx, ny, nz in [-n_real, ..., n_real]

    Returns
    -------
    real_lattice : ((2*n_real+1)^3, 3) ndarray
        Real-space translation vectors.
    """
    box = as_box_lengths(box)
    vals = np.arange(-n_real, n_real + 1, dtype=int)
    nn = np.array(np.meshgrid(vals, vals, vals, indexing="ij")).reshape(3, -1).T
    return nn * box[None, :]


def generate_k_vectors(box, n_k):
    """
    Generate reciprocal lattice vectors for an orthorhombic box.

    Parameters
    ----------
    box : float or (3,) array_like
        Box lengths. Scalar -> cubic, length-3 -> [Lx, Ly, Lz]
    n_k : int
        Include reciprocal indices mx, my, mz in [-n_k, ..., n_k],
        excluding k = 0

    Returns
    -------
    kvecs : (Nk, 3) ndarray
        Reciprocal lattice vectors.
    k2 : (Nk,) ndarray
        Squared magnitudes of reciprocal vectors.
    """
    box = as_box_lengths(box)
    vals = np.arange(-n_k, n_k + 1, dtype=int)
    mm = np.array(np.meshgrid(vals, vals, vals, indexing="ij")).reshape(3, -1).T
    mm = mm[np.any(mm != 0, axis=1)]  # exclude k = 0

    scale = 2.0 * np.pi / box
    kvecs = mm * scale[None, :]
    k2 = np.sum(kvecs**2, axis=1)

    return kvecs, k2

# ============================================================
# Periodic Green-kernel configuration
# ============================================================

def validate_screening_length(screening_length):
    """
    Validate and normalize the Green-function screening length.

    Parameters
    ----------
    screening_length : float or None
        Screening length in the same length units as the simulation box.

        None or positive infinity selects the original unscreened
        periodic Laplace Green function.

        A positive finite value will select the screened periodic
        Yukawa Green function once that kernel is enabled.

    Returns
    -------
    normalized_length : float or None
        None for the unscreened kernel, otherwise a positive finite
        screening length.
    """
    if screening_length is None:
        return None

    screening_length = float(
        screening_length
    )

    if np.isposinf(screening_length):
        return None

    if (
        not np.isfinite(screening_length)
        or screening_length <= 0.0
    ):
        raise ValueError(
            "screening_length must be a positive finite value, "
            "positive infinity, or None."
        )

    return screening_length


def prepare_periodic_green_kernel(
    *,
    box,
    alpha,
    k2,
    screening_length=None,
):
    """
    Prepare coefficients shared by periodic Green-function evaluations.

    The current implementation supports the original unscreened
    periodic Laplace kernel. A finite screening length is validated but
    deliberately rejected until the screened real-space, reciprocal,
    and self terms have all been implemented consistently.

    Parameters
    ----------
    box : float or (3,) array_like
        Periodic box lengths.

    alpha : float
        Ewald splitting parameter, in inverse-length units.

    k2 : (Nk,) array_like
        Squared reciprocal-vector magnitudes.

    screening_length : float or None
        None or positive infinity selects the original unscreened
        periodic Laplace kernel.

    Returns
    -------
    kernel : dict
        Dictionary containing the coefficients and metadata required
        by the Green-function evaluator.
    """
    box = as_box_lengths(box)

    alpha = float(alpha)

    if (
        not np.isfinite(alpha)
        or alpha <= 0.0
    ):
        raise ValueError(
            "alpha must be positive and finite."
        )

    k2 = np.asarray(
        k2,
        dtype=float,
    )

    if k2.ndim != 1:
        raise ValueError(
            "k2 must be one-dimensional."
        )

    if np.any(~np.isfinite(k2)):
        raise ValueError(
            "k2 contains non-finite values."
        )

    if np.any(k2 <= 0.0):
        raise ValueError(
            "k2 must contain only positive values because the "
            "zero reciprocal mode is excluded."
        )

    screening_length = validate_screening_length(
        screening_length
    )

    volume = float(
        np.prod(box)
    )

    if (
        not np.isfinite(volume)
        or volume <= 0.0
    ):
        raise ValueError(
            "The periodic-box volume must be positive and finite."
        )

    # --------------------------------------------------------
    # Original unscreened periodic Laplace kernel
    # --------------------------------------------------------
    if screening_length is None:
        reciprocal_weights = (
            np.exp(
                -k2
                / (4.0 * alpha**2)
            )
            / k2
            / volume
        )

        constant_term = (
            -1.0
            / (
                4.0
                * alpha**2
                * volume
            )
        )

        return {
            "kernel_type": "periodic_laplace",
            "screened": False,
            "screening_length": None,
            "kappa": 0.0,
            "kappa_squared": 0.0,
            "volume": volume,
            "reciprocal_weights":
                reciprocal_weights,
            "constant_term":
                constant_term,
        }
#----
    # --------------------------------------------------------
    # Zero-mean screened periodic Yukawa kernel
    # --------------------------------------------------------
    kappa = 1.0 / screening_length
    kappa_squared = kappa**2

    reciprocal_weights = (
        np.exp(
            -(
                k2
                + kappa_squared
            )
            / (4.0 * alpha**2)
        )
        / (
            k2
            + kappa_squared
        )
        / volume
    )

    # Remove the uniform k=0 contribution so that the spatially
    # uniform concentration remains represented only by cbar.
    #
    # np.expm1 is used for accuracy when kappa is small.
    a_squared = (
        kappa_squared
        / (4.0 * alpha**2)
    )

    constant_term = (
        np.expm1(-a_squared)
        / (
            volume
            * kappa_squared
        )
    )

    return {
        "kernel_type":
            "periodic_yukawa_zero_mean",
        "screened":
            True,
        "screening_length":
            screening_length,
        "kappa":
            kappa,
        "kappa_squared":
            kappa_squared,
        "volume":
            volume,
        "reciprocal_weights":
            reciprocal_weights,
        "constant_term":
            constant_term,
    }


def periodic_green_real_term(
    distance,
    alpha,
    kappa=0.0,
):
    """
    Evaluate the real-space Ewald contribution for either the
    periodic Laplace or screened Yukawa Green kernel.

    Parameters
    ----------
    distance : array_like
        Positive source-observation distances.

    alpha : float
        Ewald splitting parameter.

    kappa : float, default=0
        Inverse screening length. kappa=0 gives the unscreened
        Laplace real-space term.

    Returns
    -------
    values : ndarray
        Real-space Green-function contributions.
    """
    distance = np.asarray(
        distance,
        dtype=float,
    )

    alpha = float(alpha)
    kappa = float(kappa)

    if np.any(~np.isfinite(distance)):
        raise ValueError(
            "distance contains non-finite values."
        )

    if np.any(distance <= 0.0):
        raise ValueError(
            "periodic_green_real_term requires positive distances."
        )

    if (
        not np.isfinite(alpha)
        or alpha <= 0.0
    ):
        raise ValueError(
            "alpha must be positive and finite."
        )

    if (
        not np.isfinite(kappa)
        or kappa < 0.0
    ):
        raise ValueError(
            "kappa must be finite and non-negative."
        )

    # Original unscreened Laplace term.
    if kappa == 0.0:
        return (
            erfc(alpha * distance)
            / (
                4.0
                * np.pi
                * distance
            )
        )

    shift = (
        kappa
        / (2.0 * alpha)
    )

    arg_minus = (
        alpha * distance
        - shift
    )

    arg_plus = (
        alpha * distance
        + shift
    )

    term_minus = (
        np.exp(
            -kappa * distance
        )
        * erfc(arg_minus)
    )

    term_plus = (
        np.exp(
            kappa * distance
        )
        * erfc(arg_plus)
    )

    return (
        term_minus
        + term_plus
    ) / (
        8.0
        * np.pi
        * distance
    )

# ============================================================
# Periodic Laplace Green function
# ============================================================

def periodic_green_ewald(
    rvec,
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
    Evaluate a zero-mean periodic Green kernel.

    kappa = 0
        Periodic Laplace Green function satisfying

            laplacian(G) = -delta + 1/V.

    kappa > 0
        Zero-mean screened periodic Yukawa kernel.

    Parameters
    ----------
    rvec : (3,) array_like
        Displacement vector.

    box : float or (3,) array_like
        Periodic box lengths.

    alpha : float
        Ewald splitting parameter.

    real_lattice : (Nreal, 3) ndarray
        Real-space lattice translations.

    kvecs : (Nk, 3) ndarray
        Nonzero reciprocal vectors.

    k2 : (Nk,) ndarray
        Squared reciprocal-vector magnitudes.

    recip_weights : (Nk,) ndarray
        Precomputed reciprocal-space coefficients.

    const_term : float
        Zero-mode correction.

    kappa : float, default=0
        Inverse screening length.
    """
    box = as_box_lengths(
        box
    )

    rvec = np.asarray(
        rvec,
        dtype=float,
    ).reshape(3)

    rvec = minimum_image_displacement(
        rvec,
        box,
    )

    # --------------------------------------------------------
    # Real-space sum
    # --------------------------------------------------------
    rr = (
        real_lattice
        + rvec[None, :]
    )

    distances = np.linalg.norm(
        rr,
        axis=1,
    )

    mask = (
        distances > 1.0e-14
    )

    if np.any(mask):
        real_sum = float(
            np.sum(
                periodic_green_real_term(
                    distance=distances[mask],
                    alpha=alpha,
                    kappa=kappa,
                )
            )
        )

    else:
        real_sum = 0.0

    # --------------------------------------------------------
    # Reciprocal-space sum
    # --------------------------------------------------------
    phase = (
        kvecs
        @ rvec
    )

    reciprocal_sum = float(
        np.sum(
            recip_weights
            * np.cos(phase)
        )
    )

    return float(
        real_sum
        + reciprocal_sum
        + const_term
    )


def periodic_green_self_constant(
    box,
    alpha,
    real_lattice,
    kvecs,
    k2,
    screening_length=None,
):
    """
    Return the regular self part of the selected periodic Green kernel:

        G(r) = 1/(4*pi*r) + G0 + O(r)

    as r approaches zero.

    screening_length=None gives the original periodic Laplace
    kernel exactly.
    """
    box = as_box_lengths(
        box
    )

    kernel = prepare_periodic_green_kernel(
        box=box,
        alpha=alpha,
        k2=k2,
        screening_length=screening_length,
    )

    kappa = float(
        kernel["kappa"]
    )

    reciprocal_weights = kernel[
        "reciprocal_weights"
    ]

    constant_term = float(
        kernel["constant_term"]
    )

    # --------------------------------------------------------
    # Nonzero real-space images
    # --------------------------------------------------------
    distances = np.linalg.norm(
        real_lattice,
        axis=1,
    )

    mask = (
        distances > 1.0e-14
    )

    if np.any(mask):
        real_sum = float(
            np.sum(
                periodic_green_real_term(
                    distance=distances[mask],
                    alpha=alpha,
                    kappa=kappa,
                )
            )
        )

    else:
        real_sum = 0.0

    # At r=0, every reciprocal phase is one.
    reciprocal_sum = float(
        np.sum(
            reciprocal_weights
        )
    )

    # --------------------------------------------------------
    # Regular part of the zero-image real-space contribution
    #
    # For kappa=0:
    #
    #     -alpha / (2*pi^(3/2))
    #
    # For finite kappa, this is the constant remaining after
    # subtracting 1/(4*pi*r) from the screened real-space term.
    # --------------------------------------------------------
    if kappa == 0.0:
        zero_image_regular = (
            -float(alpha)
            / (
                2.0
                * np.pi**1.5
            )
        )

    else:
        shift = (
            kappa
            / (2.0 * alpha)
        )

        zero_image_regular = (
            -float(alpha)
            * np.exp(
                -shift**2
            )
            / (
                2.0
                * np.pi**1.5
            )
            - kappa
            * (
                1.0
                - erfc(shift)
            )
            / (
                4.0
                * np.pi
            )
        )

    return float(
        real_sum
        + reciprocal_sum
        + zero_image_regular
        + constant_term
    )

def precompute_particle_green_fields(
    X, Y, Z,
    positions,
    box, alpha, real_lattice, kvecs, k2,
    show_progress=True,
    dtype=np.float32
):
    """
    Precompute Green field on the fixed grid for a unit monopole at each particle center.

    Returns
    -------
    G_fields : (Np, nx, ny, nz) array
        Field due to unit monopole from each particle.
    """
    box = as_box_lengths(box)
    
    positions = np.asarray(positions, dtype=float)
    Np = len(positions)

    grid_shape = X.shape
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    npts = pts.shape[0]

    G_fields = np.empty((Np, npts), dtype=dtype)

    iterator = range(Np)

    box = as_box_lengths(box)
    V = np.prod(box)
    recip_weights = np.exp(-k2 / (4.0 * alpha**2)) / k2 / V
    const_term = -1.0 / (4.0 * alpha**2 * V)
    
    for p in iterator:
        #dr = pts - positions[p][None, :]
        #dr = dr - box * np.round(dr / box)
        dr = pts - positions[p][None, :]
        dr = minimum_image_displacement(dr, box)

        Gp = periodic_green_ewald_points(dr, box, alpha, real_lattice, kvecs, k2,recip_weights, const_term)

        rnorm = np.linalg.norm(dr, axis=1)
        Gp[rnorm < 1e-14] = 0.0

        G_fields[p, :] = Gp.astype(dtype, copy=False)

    return G_fields.reshape((Np,) + grid_shape)

def periodic_green_ewald_points(
    dr,
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
    Vectorized zero-mean periodic Green evaluation.

    Parameters
    ----------
    dr : (M, 3) ndarray
        Displacement vectors.

    kappa : float, default=0
        Inverse screening length. Zero gives the original
        periodic Laplace Green function.

    Returns
    -------
    G : (M,) ndarray
        Green-function values.
    """
    box = as_box_lengths(
        box
    )

    dr = np.asarray(
        dr,
        dtype=float,
    )

    if (
        dr.ndim != 2
        or dr.shape[1] != 3
    ):
        raise ValueError(
            "dr must have shape (M, 3)."
        )

    if np.any(~np.isfinite(dr)):
        raise ValueError(
            "dr contains non-finite values."
        )

    dr = minimum_image_displacement(
        dr,
        box,
    )

    # --------------------------------------------------------
    # Real-space sum
    # --------------------------------------------------------
    rr = (
        real_lattice[:, None, :]
        + dr[None, :, :]
    )

    distances = np.linalg.norm(
        rr,
        axis=2,
    )

    real_terms = np.zeros_like(
        distances
    )

    mask = (
        distances > 1.0e-14
    )

    if np.any(mask):
        real_terms[mask] = (
            periodic_green_real_term(
                distance=distances[mask],
                alpha=alpha,
                kappa=kappa,
            )
        )

    real_sum = np.sum(
        real_terms,
        axis=0,
    )

    # --------------------------------------------------------
    # Reciprocal-space sum
    # --------------------------------------------------------
    phase = (
        kvecs
        @ dr.T
    )

    reciprocal_sum = np.sum(
        recip_weights[:, None]
        * np.cos(phase),
        axis=0,
    )

    return (
        real_sum
        + reciprocal_sum
        + const_term
    )

def build_pair_green_matrix(
    positions,
    box,
    alpha,
    real_lattice,
    kvecs,
    k2,
    show_progress=False,
    screening_length=None,
):
    """
    Build the off-diagonal periodic Green interaction matrix.

    Parameters
    ----------
    positions : (N, 3) array_like
        Particle centres.

    box : float or (3,) array_like
        Periodic box lengths.

    alpha : float
        Ewald splitting parameter.

    real_lattice : (Nreal, 3) ndarray
        Real-space periodic lattice translations.

    kvecs : (Nk, 3) ndarray
        Nonzero reciprocal lattice vectors.

    k2 : (Nk,) ndarray
        Squared reciprocal-vector magnitudes.

    show_progress : bool, default=False
        Display progress while constructing the matrix.

    screening_length : float or None, default=None
        None or positive infinity gives the original unscreened
        periodic Laplace kernel.

    Returns
    -------
    Gpair : (N, N) ndarray
        Symmetric off-diagonal Green interaction matrix.
    """
    positions = np.asarray(
        positions,
        dtype=float,
    )

    box = as_box_lengths(
        box
    )

    if (
        positions.ndim != 2
        or positions.shape[1] != 3
    ):
        raise ValueError(
            "positions must have shape (N, 3)."
        )

    if np.any(~np.isfinite(positions)):
        raise ValueError(
            "positions contains non-finite values."
        )

    kernel = prepare_periodic_green_kernel(
        box=box,
        alpha=alpha,
        k2=k2,
        screening_length=screening_length,
    )

    reciprocal_weights = kernel[
        "reciprocal_weights"
    ]

    constant_term = kernel[
        "constant_term"
    ]

    kappa = float(
        kernel["kappa"]
    )

    n_particles = len(
        positions
    )

    Gpair = np.zeros(
        (n_particles, n_particles),
        dtype=float,
    )

    iterator = range(
        n_particles
    )

    for i in iterator:
        for j in range(
            i + 1,
            n_particles,
        ):
            displacement = (
                positions[i]
                - positions[j]
            )

            displacement = (
                minimum_image_displacement(
                    displacement,
                    box,
                )
            )

            gij = periodic_green_ewald(
                displacement,
                box,
                alpha,
                real_lattice,
                kvecs,
                k2,
                reciprocal_weights,
                constant_term,
                kappa=kappa,
            )

            Gpair[i, j] = gij
            Gpair[j, i] = gij

    return Gpair

def make_diagonal_preconditioner(R, G0):
    """
    Simple diagonal preconditioner using only diag(A)^(-1).
    """
    diag = 1.0 / (4.0 * np.pi * R) + G0
    invdiag = 1.0 / diag
    N = len(R)

    def mv(x):
        return invdiag * x

    return LinearOperator((N, N), matvec=mv, dtype=float)

def solve_monopoles_iterative(
    R,
    c_surf,
    box,
    M_total,
    c_p,
    G0,
    Gpair,
    Q_init=None,
    rtol=1e-10,
    atol=0.0,
    maxiter=200,
    restart=50,
    use_preconditioner=True
):
    """
    Solve monopole system iteratively for a set of active particles.

    Parameters
    ----------
    R : (Na,) array
        Active particle radii
    c_surf : scalar/array/callable evaluated on active particles
    Gpair : (Na, Na) array
        Off-diagonal pair Green matrix for active particles

    Returns
    -------
    Q : (Na,) array
    cbar : float
    """
    R = np.asarray(R, dtype=float)
    Na = len(R)

    if Na == 0:
        return np.zeros(0, dtype=float), mean_matrix_concentration(box, R, M_total, c_p)

    cbar = mean_matrix_concentration(box, R, M_total, c_p)
    c_surf_now = evaluate_csurf(R, c_surf)
    rhs = c_surf_now - cbar

    diag = 1.0 / (4.0 * np.pi * R) + G0

    def A_mv(x):
        return diag * x + Gpair @ x

    Aop = LinearOperator((Na, Na), matvec=A_mv, dtype=float)

    Mop = make_diagonal_preconditioner(R, G0) if use_preconditioner else None

    Q, info = gmres(
        Aop,
        rhs,
        x0=Q_init,
        M=Mop,
        rtol=rtol,
        atol=atol,
        maxiter=maxiter,
        restart=restart
    )

    if info != 0:
        raise RuntimeError(f"GMRES failed to converge, info={info}")

    return Q, cbar

#------
def build_pair_green_matrix_vectorized(
    positions, box, alpha, real_lattice, kvecs, k2
):
    """
    Vectorized construction of pair Green matrix.

    Returns
    -------
    G : (N, N) array
        G[i, i] = 0
        G[i, j] = G_per(x_i - x_j)
    """
    positions = np.asarray(positions, dtype=float)
    box = as_box_lengths(box)

    N = len(positions)
    G = np.zeros((N, N), dtype=float)

    box = as_box_lengths(box)
    V = np.prod(box)
    recip_weights = np.exp(-k2 / (4.0 * alpha**2)) / k2 / V
    const_term = -1.0 / (4.0 * alpha**2 * V)
    
    # for i in range(N):
    #     dr = positions - positions[i]                  # (N, 3)
    #     dr = minimum_image_displacement(dr, box)       # periodic minimum image

    #     G[i, :] = periodic_green_ewald_points(
    #         dr, box, alpha, real_lattice, kvecs, k2, recip_weights, const_term
    #     )

    #np.fill_diagonal(G, 0.0)  # self-term handled separately via G0

    for i in range(N - 1):
        dr = positions[i+1:] - positions[i]          # (N-i-1, 3)
        dr = minimum_image_displacement(dr, box)

        gij = periodic_green_ewald_points(
            dr, box, alpha, real_lattice, kvecs, k2, recip_weights, const_term
        )

        G[i, i+1:] = gij
        G[i+1:, i] = gij

    return G

def extend_pair_green_matrix(
    G_old,
    old_positions,
    new_positions,
    box,
    alpha,
    real_lattice,
    kvecs,
    k2,
    recip_weights,
    const_term,
    kappa=0.0,
):
    old_positions = np.asarray(old_positions, dtype=float)
    new_positions = np.asarray(new_positions, dtype=float)
    box = as_box_lengths(box)

    N = len(old_positions)
    m = len(new_positions)

    if m == 0:
        return G_old

    box = as_box_lengths(box)

    # old-new block
    G_cross = np.zeros((N, m), dtype=float)
    for j in range(m):
        dr = old_positions - new_positions[j]
        dr = minimum_image_displacement(dr, box)
        G_cross[:, j] = periodic_green_ewald_points(
            dr, box, alpha, real_lattice, kvecs, k2, recip_weights, const_term
        )

    # new-new block
    # G_new = np.zeros((m, m), dtype=float)
    # for i in range(m):
    #     dr = new_positions - new_positions[i]
    #     dr = minimum_image_displacement(dr, box)
    #     G_new[i, :] = periodic_green_ewald_points(
    #         dr, box, alpha, real_lattice, kvecs, k2, recip_weights, const_term
    #     )
    # np.fill_diagonal(G_new, 0.0)
    G_new = np.zeros((m, m), dtype=float)
    for i in range(m - 1):
        dr = new_positions[i+1:] - new_positions[i]
        dr = minimum_image_displacement(dr, box)
        gij = periodic_green_ewald_points(
            dr, box, alpha, real_lattice, kvecs, k2, recip_weights, const_term
        )
        G_new[i, i+1:] = gij
        G_new[i+1:, i] = gij

    # assemble full matrix
    top = np.hstack([G_old, G_cross])
    bottom = np.hstack([G_cross.T, G_new])
    G_full = np.vstack([top, bottom])

    return G_full

def concentration_at_point(
    pt,
    centers,
    Q,
    cbar,
    box,
    alpha,
    real_lattice,
    kvecs,
    k2,
):
    """
    Evaluate the local matrix concentration at a single point pt using
    the monopole superposition:

        c(pt) = cbar + sum_j Q_j * G_per(pt - centers[j])

    Parameters
    ----------
    pt : (3,) array_like
        Single evaluation point.
    centers : (N, 3) array_like
        Particle centers.
    Q : (N,) array_like
        Monopole strengths.
    cbar : float
        Mean matrix concentration.
    L : float
        Cubic box size.
    alpha, real_lattice, kvecs, k2
        Ewald parameters used by periodic_green_ewald_points().

    Returns
    -------
    c_loc : float
        Local concentration at pt.
    """

    box = as_box_lengths(box)
    
    pt = np.asarray(pt, dtype=float).reshape(3,)
    centers = np.asarray(centers, dtype=float)
    Q = np.asarray(Q, dtype=float)

    if centers.shape[0] == 0:
        return float(cbar)

    c_loc = float(cbar)

    box = as_box_lengths(box)
    V = np.prod(box)
    recip_weights = np.exp(-k2 / (4.0 * alpha**2)) / k2 / V
    const_term = -1.0 / (4.0 * alpha**2 * V)
    
    # for j in range(centers.shape[0]):
    #     dr = pt - centers[j]                 # (3,)
    #     dr = minimum_image_displacement(dr,box)
    #     #dr = dr - L * np.round(dr / L)       # minimum-image displacement
    #     G = periodic_green_ewald_points(
    #         dr.reshape(1, 3), box, alpha, real_lattice, kvecs, k2, recip_weights, const_term
    #     )[0]
    #     c_loc += Q[j] * G

    #return c_loc

    pt = np.asarray(pt, dtype=float).reshape(1, 3)
    centers = np.asarray(centers, dtype=float)
    Q = np.asarray(Q, dtype=float)

    if centers.shape[0] == 0:
        return float(cbar)

    dr = pt - centers
    dr = minimum_image_displacement(dr, box)

    G = periodic_green_ewald_points(
        dr, box, alpha, real_lattice, kvecs, k2, recip_weights, const_term
    )

    return float(cbar + np.dot(Q, G))



def concentrations_at_points_species(
    points,
    centers,
    Q,
    cbar,
    box,
    alpha,
    real_lattice,
    kvecs,
    k2,
):
    """Evaluate all transported-species concentrations at many points.

    Parameters
    ----------
    points : (M, 3) array_like
    centers : (N, 3) array_like
    Q : (S, N) array_like
    cbar : (S,) array_like

    Returns
    -------
    c_loc : (M, S) ndarray
        Local concentrations.  Green-function geometry is evaluated once per
        source particle and reused for every transported species.
    """
    box = as_box_lengths(box)
    points = np.asarray(points, dtype=float)
    centers = np.asarray(centers, dtype=float)
    Q = np.asarray(Q, dtype=float)
    cbar = np.asarray(cbar, dtype=float)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (M, 3).")
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError("centers must have shape (N, 3).")
    if Q.ndim != 2 or Q.shape[1] != centers.shape[0]:
        raise ValueError("Q must have shape (S, N), matching centers.")
    if cbar.shape != (Q.shape[0],):
        raise ValueError("cbar must have shape (S,), matching Q.")

    m = points.shape[0]
    n = centers.shape[0]
    if m == 0:
        return np.empty((0, Q.shape[0]), dtype=float)
    if n == 0:
        return np.broadcast_to(cbar, (m, Q.shape[0])).copy()

    V = np.prod(box)
    recip_weights = np.exp(-k2 / (4.0 * alpha**2)) / k2 / V
    const_term = -1.0 / (4.0 * alpha**2 * V)

    # G[p, j] is the periodic Green function from particle j to point p.
    # Building this once is the key optimization: all species reuse it.
    Gmat = np.empty((m, n), dtype=float)
    for j in range(n):
        dr = points - centers[j]
        dr = minimum_image_displacement(dr, box)
        Gmat[:, j] = periodic_green_ewald_points(
            dr, box, alpha, real_lattice, kvecs, k2,
            recip_weights, const_term,
        )

    return cbar[None, :] + Gmat @ Q.T
