import math
import time
import numpy as np
from scipy.optimize import brentq

from .utilities import (
    as_box_lengths,
    normalize_vector,
    make_plane_basis,
    minimum_image_displacement,
    sample_point_in_box,
    sample_bulk_point,
    sample_gb_point,
    particle_volumes,
    estimate_gb_free_area_fraction,
    gb_candidate_overlaps_existing,
)

from .greensfunction import concentration_at_point, concentrations_at_points_species

# ----------- CONSTANTS -----------------
kB = 1.380649e-23 # J/K  Boltzmann constant

# -----------------------------------------------------
# Ideal dilute driving force for nucleation (J/m^3)
# Uses site fractions X = C * vatm (dimensionless)
# ------------------------------------------------------
def deltaGv_ideal_dilute_atoms(T, C0Sc, aa, bb,cc, vatm, vatp, x=3, y=1):
    """
    For AxBy (here Al3Sc): reaction per formula unit.
    Supersaturation based on solubility product K: S = (X_Al^x * X_Sc^y) / K, where log10 K = bb - aa/T.
    ΔGv = -(kB T)/(v_fu) * ln(S), where v_fu = (x+y) * vatp.

    Returns ΔGv (J/m^3), negative when S>1 else 0.
    """
    if C0Sc <= 0:   # Concentrations can't be less than zero!
        return 0.0

    log10K = bb - aa / T + cc/T**2 
    K = 10.0 ** log10K   # Solubility product

    XSc = C0Sc * vatm   # Convert composition from concentration (C0 = atoms/m^3) to atomic fraction (XMg) 
    
    # need because solubility product is defined in terms of atomic fraction
    XAl = 1-XSc
    
    if XSc <= 0 or K <= 0:
        return 0.0

    S = ((1-XSc)**x) * (XSc**y) / K
    #print("S=", S)
    if S <= 1.0:
        return 0.0

    v_fu = (x + y) * vatp     # v_fu is the atomic volume based on the functional unit (here x+y atoms) compared to vatp which is the atomic volume of the precipitate in m^3/atom
    return -(kB * T / v_fu) * math.log(S)


# -----------------------------
# Nucleation: 
# Z: Eq.(4); b: Eq.(5) + multicomponent; rate: Eq.(6); RkBT: Eq.(7)
# -----------------------------
def nucleation_step(
    dt,T,t,
    C0Sc,      # atoms/m^3 in matrix
    aa, bb, cc,         # log10K = bb - aa/T
    vatm, vatp,      # m^3/atom
    gamma,           # J/m^2
    N0,              # nucleation sites 1/m^3
    DSc,        # m^2/s
    a_latt,          # m (lattice parameter / jump distance scale)
    x=3, y=1,
    barrier_factor = 1.0,  # multiplier for nucleation barrier to account for heterogeneous nucleation
):
    """
    Returns: dN, RkBT, Rstar, J, 

    NOTE: DGstar_eff = barrier_factor * DGstar
    Use barrier_factor = 1 for bulk nucleation (homogeneous)
    Use barrier_facror < 1 for heterogeneous nucleation
    
    """

    if barrier_factor < 0.0:
        raise ValueError("barrier_factor must be > 0")
    
    # Driving force per unit volume (paper's Δg): use ideal dilute closure
    deltaGv = deltaGv_ideal_dilute_atoms(T, C0Sc, aa, bb,cc, vatm, vatp, x=x, y=y)
    
    if deltaGv >= 0.0:
        print('deltaGv > 0!')  # The driving force needs to always be less than zero here assuming that we are precipitating!
        return 0.0, float("inf"), float("inf"), 0.0

    # Apply barrier factor on surface energy
    #gamma = gamma*barrier_factor
    
    # Critical radius and barrier (CNT)
    Rstar = -2.0 * gamma / deltaGv
    DGstar = 16.0 * np.pi * gamma**3 / (3.0 * deltaGv**2)

    # APPLED BARRIER_FACTOR on dGstar
    DGstar_eff = barrier_factor*DGstar
    
    # Zeldovich factor based on number, Eq.(4)
    Zn = (vatp * (x+y) / (2.0 * np.pi * Rstar**2)) * math.sqrt(gamma / (kB * T))
    #Zn = (vatp / (2.0 * pi * Rstar**2)) * math.sqrt(gamma / (kB * T))

    # Matrix site fractions (for multicomponent b)
    XSc = max(1e-60, C0Sc * vatm)
    XpSc = y/(x+y)
    
    # Eq.(5) 
    #denom = (1.0*XpMg / (max(1e-60, DMg) * XMg)) + (1.0*XpSi / (max(1e-60, DSi) * XSi))
    b = (4.0 * np.pi * Rstar**2 / a_latt**4) * DSc*XSc/XpSc

    # Incubation time tau = 4/(2π b Z^2)
    tau = 4.0 / (2.0 * np.pi * b * Zn**2)

    # Nucleation rate Eq.(6): dN/dt = N0 Z b exp(-DG*/kBT) * (1-exp(-t/tau))
    
    inc = 0.0 if t <= 0.0 else 1-math.exp(-t/tau)
    J = N0 * Zn * b * math.exp(-DGstar_eff / (kB * T)) * inc

    dN = J * dt

    # Nucleation radius Eq.(7): RkBT = R* + 1/(sqrt(pi) ZR) - ZR is the Zeldovich factor based on Radius
    RkBT = Rstar + 1.0 / 2.0*np.sqrt(kB*T/(np.pi*gamma))

    return dN, RkBT, Rstar, J


def surface_composition(
    R, T,
    aa, bb, cc,      # log10K planar: log10K = bb - aa/T  (your convention)
    vatm, vatp,  # atomic volumes [m^3/atom] (matrix, precip)
    gamma,       # interfacial energy [J/m^2]
    x=3, y=1,
    use_xy_factor=True
):

    if R <= 0.0 or T <= 0.0:
        return 0.0

    #print(f'T={T}, aa={aa}')
    # Planar solubility product (dimensionless, in site-fractions)
    log10K = bb - aa / T + cc/T**2
    K_planar = 10.0 ** log10K

    # Capillarity factor: R0 = 2γ v_at^p / (kBT)
    R0 = 2.0 * gamma * vatp / (kB * T)

    stoich = (x + y) if use_xy_factor else 1.0
    K_R = K_planar * math.exp(stoich * R0 / R)

    # Solve for interfacial site fraction XiSc in (0,1):
    # XiSc * (1 - XiSc)^3 = K_R
    # Note: maximum of f(X)=X(1-X)^3 occurs at X=1/4 with fmax=27/256 ≈ 0.10547
    fmax = 27.0 / 256.0
    if K_R <= 0.0:
        XiSc = 0.0
    elif K_R >= fmax:
        # Physically this means the requested K_R is too large for this functional form.
        # You can clamp at the maximum (Xi=0.25), or treat as "interface saturated".
        XiSc = 0.25
    else:
        def f(X):
            return X * (1.0 - X)**3 - K_R   # Preciso uses XAl = X0Al --> check  ---> dr_dt	double	5.0251506313407427E-13
        # For K_R in (0, fmax), there are two roots; the dilute one is near 0.
        # We want the dilute root for a dilute matrix: bracket near 0..0.25
        XiSc = brentq(f, 0.0, 0.25)

    if XiSc > 0.01:  # Artificially cap XiSc for small particles so that the routine does not become numerically unstable during dissolution
        XiSc = 0.01
    
    return XiSc/vatm


def nucleate_particles_cnt_monopole(
    centers,
    radii,
    Q,
    box,
    dt,
    t,
    T,
    cbar,
    aa, bb, cc,
    vatm, vatp,
    gamma,
    N0,
    DSc,
    a_latt,
    alpha,
    real_lattice,
    kvecs,
    k2,
    *,
    event_rate_bulk,
    event_rate_gb,
    J_ref_bulk,
    J_ref_gb,
    RkBT_ref_bulk,
    RkBT_ref_gb,
    lam_target=1.0,
    matrix_volume=None,
    safety_buffer=0.0,
    rng=None,
    gb_point=None,
    gb_normal=None,
    gb_thickness=0.0,
    gb_mode = "slab",
    gb_barrier_factor = 1.0,
    nuc_kind = None,
    gb_exclusion_buffer=0.0,
    gb_use_area_fraction=True,
):
    """
    Perform one nucleation update for a monopole + classical nucleation theory (CNT) model.

    This routine generates stochastic nucleation events within a periodic cubic domain
    using a monopole-based reconstruction of the solute field and a global Poisson
    proposal process derived from CNT.

    Nucleation is handled in three stages:
        1. A global Poisson draw determines the number of nucleation trials based on
           the mean matrix concentration.
        2. Candidate sites are sampled randomly in the domain and filtered using a
           geometric pre-screen to ensure a nucleus of size R_kBT_mean can be placed.
        3. For each valid site, the local concentration is evaluated using the
           monopole field, and CNT is applied to determine the local nucleation rate
           and insertion radius. A stochastic accept/reject step is then performed.

    During a timestep:
        - The concentration field is evaluated using the *pre-existing* particles only
          (i.e., monopole strengths Q are not updated for newly accepted nuclei).
        - Newly accepted nuclei are included in geometric overlap checks for subsequent
          trials within the same timestep.
        - Monopole strengths and global mass balance must be updated outside this
          function after nucleation.

    Parameters
    ----------
    centers : (N, 3) ndarray
        Cartesian coordinates of existing particle centers.

    radii : (N,) ndarray
        Radii of existing particles.

    Q : (N,) ndarray
        Monopole strengths associated with existing particles.

    L : float
        Side length of the cubic periodic simulation domain.

    dt : float
        Current timestep size.

    t : float
        Current simulation time.

    T : float
        Temperature (K).

    cbar : float
        Mean matrix concentration (atoms/m^3), used for the global nucleation rate.

    aa, bb, cc : float
        Thermodynamic parameters for equilibrium solubility (e.g. log10 K relation).

    vatm, vatp : float
        Atomic volumes of matrix and precipitate phases (m^3/atom).

    gamma : float
        Interfacial energy (J/m^2).

    N0 : float
        Density of nucleation sites (1/m^3).

    DSc : float
        Diffusivity of solute (m^2/s).

    a_latt : float
        Atomic jump distance or lattice parameter (m).

    alpha : float
        Ewald splitting parameter.

    real_lattice : (Nr, 3) ndarray
        Real-space lattice vectors used in the Ewald sum.

    kvecs : (Nk, 3) ndarray
        Reciprocal lattice vectors.

    k2 : (Nk,) ndarray
        Squared magnitudes of reciprocal lattice vectors.

    lam_target : float, optional
        Target upper bound for the expected number of nucleation trials per timestep.
        Used to adaptively reduce dt if the nucleation rate is too high. Default is 1.0.

    matrix_volume : float, optional
        Volume of matrix phase (m^3). If None, computed as total box volume minus
        particle volumes.

    safety_buffer : float, optional
        Additional exclusion distance added to overlap checks. Default is 0.0.

    rng : np.random.Generator, optional
        Random number generator. If None, a default generator is created.

    Returns
    -------
    out : dict
        Dictionary containing:

        - "Ntrial" : int
            Number of nucleation trials drawn from Poisson distribution.

        - "n_accepted" : int
            Number of nuclei successfully inserted.

        - "new_centers" : (M, 3) ndarray
            Positions of newly nucleated particles.

        - "new_radii" : (M,) ndarray
            Radii of newly nucleated particles (R_kBT).

        - "c_local" : (M,) ndarray
            Local matrix concentration at accepted nucleation sites.

        - "J_local" : (M,) ndarray
            Local nucleation rates at accepted sites.

        - "accepted_mask" : (Ntrial,) ndarray of bool
            Boolean mask indicating which trials resulted in successful nucleation.

        - "lambda_mean" : float
            Expected number of nucleation trials (Poisson mean).

        - "dt_used" : float
            Possibly reduced timestep after enforcing lam_target.

    Notes
    -----
    - The concentration field is assumed to be:
          c(x) = cbar + sum_j Q_j G_per(x - x_j)

      where G_per is the periodic Green's function evaluated via Ewald summation.

    - Newly nucleated particles do NOT contribute to the concentration field within
      the same timestep (operator-splitting approximation).

    - Geometric rejection (points inside particles or too close for insertion) does
      not count as a nucleation trial; such points are resampled.

    - Kinetic rejection (based on local CNT rate and stochastic acceptance) does
      count as a nucleation trial.

    - The timestep is adaptively reduced to ensure:
          J_mean * V_matrix * dt ≲ lam_target

      which prevents unphysically large nucleation bursts.

    - This function does NOT update:
        * monopole strengths Q
        * mean concentration cbar
        * solute mass balance

      These must be handled externally after nucleation.

    - For stability, nuclei are inserted at a slightly supercritical size R_kBT,
      which avoids marginal stability at the critical radius R*.
    """

    if nuc_kind is None:
        nuc_kind = np.zeros(len(radii),dtype=np.int32)
    else:
        nuc_kind = np.asarray(nuc_kind, dtype=np.int32).copy()
        
    if rng is None:
        rng = np.random.default_rng()

    centers = np.asarray(centers, dtype=float)
    radii = np.asarray(radii, dtype=float)

    if centers.size == 0:
        centers = np.zeros((0, 3), dtype=float)
        radii = np.zeros((0,), dtype=float)

    box = as_box_lengths(box)
    Lx, Ly, Lz = box
    V_box = Lx * Ly * Lz
    
    if matrix_volume is None:
        Vp = particle_volumes(radii)
        V_matrix = V_box - np.sum(Vp)
    else:
        V_matrix = float(matrix_volume)

    gb_enabled = (gb_point is not None and gb_normal is not None and gb_thickness is not None and gb_thickness > 0.0)
    if gb_enabled:
        gb_normal = normalize_vector(gb_normal)
        V_gb_geom = Ly * Lz *gb_thickness
        V_gb = min(V_matrix, V_gb_geom)
        #V_gb = min(V_matrix, L**2*gb_thickness)
        V_bulk = max(0.0,V_matrix-V_gb)
        e1,e2 = make_plane_basis(gb_normal)
        gb_area = Ly*Lz
        gb_L1 = Ly  # !!!!!!!
        gb_L2 = Lz

    else:
        V_gb = 0.0
        V_bulk = V_matrix
        gb_point = None
        gb_L1 = None
        gb_L2 = None
        gb_area = 0.0

    

        

    #J_bulk_mean = J_mean
    #J_gb_mean = J_mean if gb_enabled else 0.0
    phi_gb_free = 1.0
    if gb_enabled and gb_use_area_fraction:
        phi_gb_free = estimate_gb_free_area_fraction(
            centers=centers,
            radii=radii,
            nuc_kind = nuc_kind,
            gb_point = gb_point,
            gb_normal = gb_normal,
            gb_area = gb_area,
            box=box,
            exclusion_buffer=gb_exclusion_buffer,
        )
        

    if (not np.isfinite(V_matrix) or V_matrix <= 0.0):
        lam_bulk = 0.0
        lam_gb = 0.0
    
    else:
        if (
            event_rate_bulk is not None
            and np.isfinite(event_rate_bulk)
            and event_rate_bulk > 0.0
        ):
            lam_bulk = event_rate_bulk * dt
        else:
            lam_bulk = 0.0
    
        if (
            gb_enabled
            and event_rate_gb is not None
            and np.isfinite(event_rate_gb)
            and event_rate_gb > 0.0
        ):
            lam_gb = event_rate_gb * dt
        else:
            lam_gb = 0.0
    
    lam_tot = lam_bulk + lam_gb

    
    Ntrial_bulk = int(rng.poisson(lam_bulk))
    Ntrial_gb   = int(rng.poisson(lam_gb))
    Ntrial      = Ntrial_bulk + Ntrial_gb

    if Ntrial == 0:
        return {
            "Ntrial": 0,
            "Ntrial_bulk": 0,
            "Ntrial_gb": 0,
            "n_accepted": 0,
            "new_centers": np.zeros((0, 3)),
            "new_radii": np.zeros((0,)),
            "c_local": np.zeros((0,)),
            "J_local": np.zeros((0,)),
            "accepted_mask": np.zeros((0,), dtype=bool),
            "lambda_bulk": lam_bulk,
            "lambda_gb": lam_gb,
            "lambda_mean": lam_bulk+lam_gb,
            "dt_used": dt,
            "new_kind": np.zeros((0,),dtype=np.int32),
        }

    accepted_centers = []
    accepted_radii = []
    accepted_c = []
    accepted_J = []
    accepted_mask = []
    accepted_kind = []
    
    work_centers = centers.copy()
    work_radii = radii.copy()
    work_kind = nuc_kind.copy()

    #------------------------------------------------------------------
    def process_one_trial(kind):
        nonlocal work_centers
        nonlocal work_radii
        nonlocal work_kind
    
        max_geo_reject = 100
        pt = None
    
        # Regional reference values calculated previously by
        # nucleation_dt_limit_ternary() and passed into this function.
        if kind == 0:
            J_ref = J_ref_bulk
            RkBT_ref = RkBT_ref_bulk
        else:
            J_ref = J_ref_gb
            RkBT_ref = RkBT_ref_gb
    
        # A trial should not have been generated for a region with a
        # non-positive reference rate, but guard against it explicitly.
        if (
            J_ref is None
            or not np.isfinite(J_ref)
            or J_ref <= 0.0
        ):
            accepted_mask.append(False)
            return
    
        if (
            RkBT_ref is None
            or not np.isfinite(RkBT_ref)
            or RkBT_ref <= 0.0
        ):
            accepted_mask.append(False)
            return
    
        # ----------------------------------------------------------
        # Find a candidate location that passes the preliminary
        # geometric test using the regional reference radius.
        # ----------------------------------------------------------
        for _ in range(max_geo_reject):
            if kind == 0:
                if gb_enabled:
                    trial_pt = sample_bulk_point(
                        box,
                        rng,
                        gb_point,
                        gb_normal,
                        gb_thickness,
                    )
                else:
                    trial_pt = sample_point_in_box(
                        box,
                        rng,
                    )
            else:
                trial_pt = sample_gb_point(
                    box,
                    rng,
                    gb_point,
                    gb_normal,
                    gb_thickness,
                )
    
            if trial_pt is None:
                break
    
            if work_centers.shape[0] == 0:
                pt = trial_pt
                break
    
            dvec = minimum_image_displacement(
                work_centers - trial_pt,
                box,
            )
    
            trial_dist = np.linalg.norm(
                dvec,
                axis=1,
            )
    
            overlaps = np.any(
                trial_dist
                < (
                    work_radii
                    + RkBT_ref
                    + safety_buffer
                )
            )
    
            if overlaps:
                continue
    
            # Optional projected-overlap check for GB nuclei.
            if (
                kind == 1
                and gb_enabled
                and gb_exclusion_buffer >= 0.0
            ):
                overlaps_gb = (
                    gb_candidate_overlaps_existing(
                        candidate_center=trial_pt,
                        candidate_radius=RkBT_ref,
                        centers=work_centers,
                        radii=work_radii,
                        nuc_kind=work_kind,
                        gb_point=gb_point,
                        box=box,
                        exclusion_buffer=
                            gb_exclusion_buffer,
                    )
                )
    
                if overlaps_gb:
                    continue
    
            pt = trial_pt
            break
    
        if pt is None:
            accepted_mask.append(False)
            return
    
        # ----------------------------------------------------------
        # Evaluate the actual local concentration.
        # ----------------------------------------------------------
        c_loc = concentration_at_point(
            pt,
            centers,
            Q,
            cbar,
            box,
            alpha,
            real_lattice,
            kvecs,
            k2,
        )
    
        if (
            not np.isfinite(c_loc)
            or c_loc <= 0.0
        ):
            accepted_mask.append(False)
            return
    
        barrier_factor_loc = (
            1.0
            if kind == 0
            else gb_barrier_factor
        )
    
        (
            _,
            RkBT_loc,
            Rstar_loc,
            J_loc,
        ) = nucleation_step(
            dt,
            T,
            t,
            c_loc,
            aa,
            bb,
            cc,
            vatm,
            vatp,
            gamma,
            N0,
            DSc,
            a_latt,
            x=3,
            y=1,
            barrier_factor=barrier_factor_loc,
        )
    
        if (
            not np.isfinite(J_loc)
            or J_loc <= 0.0
        ):
            accepted_mask.append(False)
            return
    
        if (
            not np.isfinite(RkBT_loc)
            or RkBT_loc <= 0.0
        ):
            accepted_mask.append(False)
            return
    
        # ----------------------------------------------------------
        # Re-check using the actual local critical radius.
        # ----------------------------------------------------------
        if work_centers.shape[0] > 0:
            dvec = minimum_image_displacement(
                work_centers - pt,
                box,
            )
    
            trial_dist = np.linalg.norm(
                dvec,
                axis=1,
            )
    
            if np.any(
                trial_dist
                < (
                    work_radii
                    + RkBT_loc
                    + safety_buffer
                )
            ):
                accepted_mask.append(False)
                return
    
            if (
                kind == 1
                and gb_enabled
                and gb_exclusion_buffer >= 0.0
            ):
                overlaps_gb = (
                    gb_candidate_overlaps_existing(
                        candidate_center=pt,
                        candidate_radius=RkBT_loc,
                        centers=work_centers,
                        radii=work_radii,
                        nuc_kind=work_kind,
                        gb_point=gb_point,
                        box=box,
                        exclusion_buffer=
                            gb_exclusion_buffer,
                    )
                )
    
                if overlaps_gb:
                    accepted_mask.append(False)
                    return
    
        # ----------------------------------------------------------
        # Local-rate thinning.
        #
        # J_ref is the regional CNT rate density, not the integrated
        # event rate used to construct the Poisson proposal count.
        # ----------------------------------------------------------
        p_acc = min(
            1.0,
            J_loc / J_ref,
        )
    
        if rng.random() >= p_acc:
            accepted_mask.append(False)
            return
    
        # ----------------------------------------------------------
        # Accept nucleus.
        # ----------------------------------------------------------
        accepted_centers.append(
            pt.copy()
        )
    
        accepted_radii.append(
            RkBT_loc
        )
    
        accepted_c.append(
            c_loc
        )
    
        accepted_J.append(
            J_loc
        )
    
        accepted_kind.append(
            kind
        )
    
        accepted_mask.append(
            True
        )
    
        work_centers = np.vstack([
            work_centers,
            pt[None, :],
        ])
    
        work_radii = np.append(
            work_radii,
            RkBT_loc,
        )
    
        work_kind = np.append(
            work_kind,
            np.int32(kind),
        )

def nucleation_dt_limit(cbar, V_matrix, dt_try, t, T, cnt_params, lam_target):
    dN_mean, RkBT_mean, Rstar_mean, J_mean = nucleation_step(
        dt_try, T, t,
        cbar,
        cnt_params["aa"], cnt_params["bb"], cnt_params["cc"],
        cnt_params["vatm"], cnt_params["vatp"],
        cnt_params["gamma"],
        cnt_params["N0"],
        cnt_params["DSc"],
        cnt_params["a_latt"],
        x=3, y=1
    )

    if (not np.isfinite(J_mean)) or (J_mean <= 0.0) or (V_matrix <= 0.0):
        return np.inf, J_mean, RkBT_mean, Rstar_mean

    dt_nuc = lam_target / (J_mean * V_matrix)
    return dt_nuc, J_mean, RkBT_mean, Rstar_mean


def c_surf_gibbs_thomson(T,aa,bb,cc,vatm,vatp,gamma,x=3,y=1):
    def c_surf(Rvec): 
        Rvec = np.asarray(Rvec, dtype=float)
        out = np.empty_like(Rvec)

        for i, Ri in enumerate(Rvec):
            out[i] = surface_composition(
                R=Ri,
                T=T,
                aa=aa, bb=bb, cc=cc,
                vatm=vatm, vatp=vatp,
                gamma=gamma,
                x=3, y=1,
                use_xy_factor=True
            )

        return out
    return c_surf

def growth_rate(R, Q, D, c_p, c_surf, denom_tol=1e-14):
    """
    dR_i/dt = -D Q_i / [4π (c_p - c_surf_i) R_i^2]
    """
    R = np.asarray(R, dtype=float)
    Q = np.asarray(Q, dtype=float)
    c_surf = np.asarray(c_surf, dtype=float)

    denom = 4.0 * np.pi * (c_p - c_surf) * R**2
    dRdt = np.zeros_like(R, dtype=float)

    safe = (np.abs(denom) > denom_tol) & (R > 0.0)
    dRdt[safe] = -D * Q[safe] / denom[safe]

    tiny = 1e-14
    bad = (R < tiny) & (dRdt < 0.0)
    dRdt[bad] = 0.0

    return dRdt

#-----------------------------------------
# Ternary code
#-----------------------------------------
def deltaGv_ideal_dilute_atoms_ternary(
    T,
    C0Mg,
    C0Si,
    aa,
    bb,
    cc,
    vatm,
    vatp,
    x=5,
    y=6,
):
    """
    Ideal-dilute chemical driving force for nucleation of Mg_x Si_y.

    Concentrations C0Mg and C0Si are number densities in atoms/m^3.
    The corresponding matrix site fractions are

        XMg = C0Mg * vatm
        XSi = C0Si * vatm

    The supersaturation is

        S = XMg**x * XSi**y / K(T),

    where

        log10 K(T) = bb - aa/T + cc/T**2.

    The volumetric driving force is

        deltaGv = -(kB*T / v_fu) * ln(S),

    with

        v_fu = (x+y) * vatp.

    Returns
    -------
    float
        Volumetric driving force in J/m^3. It is negative when
        the matrix is supersaturated and zero otherwise.
    """
    if T <= 0.0:
        raise ValueError("T must be positive.")

    if vatm <= 0.0 or vatp <= 0.0:
        raise ValueError("vatm and vatp must be positive.")

    if x <= 0 or y <= 0:
        raise ValueError(
            "Stoichiometric coefficients x and y must be positive."
        )

    if (
        not np.isfinite(C0Mg)
        or not np.isfinite(C0Si)
        or C0Mg <= 0.0
        or C0Si <= 0.0
    ):
        return 0.0

    XMg = C0Mg * vatm
    XSi = C0Si * vatm

    if (
        not np.isfinite(XMg)
        or not np.isfinite(XSi)
        or XMg <= 0.0
        or XSi <= 0.0
    ):
        return 0.0

    log_K = np.log(10.0) * (
        bb
        - aa / T
        + cc / T**2
    )

    log_S = (
        x * np.log(XMg)
        + y * np.log(XSi)
        - log_K
    )

    if not np.isfinite(log_S) or log_S <= 0.0:
        return 0.0

    v_fu = (x + y) * vatp

    return -(kB * T / v_fu) * log_S

def nucleation_step_ternary(
    dt,
    T,
    t,
    C0Mg,
    C0Si,
    aa,
    bb,
    cc,
    vatm,
    vatp,
    gamma,
    N0,
    DMg,
    DSi,
    a_latt,
    x=5,
    y=6,
    barrier_factor=1.0,
    apply_incubation_correction=True,
):
    """
    Evaluate the CNT nucleation rate for a ternary Mg-Si matrix.

    The thermodynamic driving force is calculated from the Mg-Si
    solubility product. The attachment rate uses a multicomponent
    diffusion-resistance expression:

        1 / b proportional to
            XpMg / (DMg*XMg)
            + XpSi / (DSi*XSi),

    where

        XpMg = x/(x+y)
        XpSi = y/(x+y).

    Returns
    -------
    dN : float
        Nucleation rate integrated over dt, per unit volume.

    RkBT : float
        Slightly supercritical insertion radius.

    Rstar : float
        CNT critical radius.

    J : float
        Nucleation rate in m^-3 s^-1.
    """
    if dt < 0.0:
        raise ValueError("dt must be non-negative.")

    if T <= 0.0:
        raise ValueError("T must be positive.")

    if gamma <= 0.0:
        raise ValueError("gamma must be positive.")

    if N0 < 0.0:
        raise ValueError("N0 must be non-negative.")

    if DMg <= 0.0 or DSi <= 0.0:
        raise ValueError(
            "DMg and DSi must be positive."
        )

    if a_latt <= 0.0:
        raise ValueError("a_latt must be positive.")

    if barrier_factor < 0.0:
        raise ValueError(
            "barrier_factor must be non-negative."
        )

    deltaGv = deltaGv_ideal_dilute_atoms_ternary(
        T=T,
        C0Mg=C0Mg,
        C0Si=C0Si,
        aa=aa,
        bb=bb,
        cc=cc,
        vatm=vatm,
        vatp=vatp,
        x=x,
        y=y,
    )

    if deltaGv >= 0.0:
        return (
            0.0,
            float("inf"),
            float("inf"),
            0.0,
        )

    Rstar = -2.0 * gamma / deltaGv

    DGstar = (
        16.0
        * np.pi
        * gamma**3
        / (3.0 * deltaGv**2)
    )

    DGstar_eff = barrier_factor * DGstar

    v_fu = (x + y) * vatp

    Zn = (
        v_fu
        / (2.0 * np.pi * Rstar**2)
        * math.sqrt(gamma / (kB * T))
    )

    XMg = max(
        np.finfo(float).tiny,
        C0Mg * vatm,
    )

    XSi = max(
        np.finfo(float).tiny,
        C0Si * vatm,
    )

    XpMg = x / (x + y)
    XpSi = y / (x + y)

    diffusion_resistance = (
        XpMg / (DMg * XMg)
        + XpSi / (DSi * XSi)
    )

    if (
        not np.isfinite(diffusion_resistance)
        or diffusion_resistance <= 0.0
    ):
        return (
            0.0,
            Rstar,
            Rstar,
            0.0,
        )

    
    b = (
        4.0
        * np.pi
        * Rstar**2
        / a_latt**4
        / diffusion_resistance
    )
    
    if not np.isfinite(b) or b <= 0.0:
        return (
            0.0,
            Rstar,
            Rstar,
            0.0,
        )
    
    tau = 4.0 / (
        2.0
        * np.pi
        * b
        * Zn**2
    )
    
    # Check to see if we should use the incubation time - make sure this is false for non-isothermal conditions
    apply_incubation_correction = bool(apply_incubation_correction)
    
    if not apply_incubation_correction:
        incubation_factor = 1.0
    
    elif (
        not np.isfinite(tau)
        or tau <= 0.0
    ):
        incubation_factor = 1.0
    
    elif t <= 0.0:
        incubation_factor = 0.0
    
    else:
        incubation_factor = -math.expm1(
            -t / tau
        )
    
    exponent = -DGstar_eff / (
        kB * T
    )

    # Avoid an unnecessary underflow warning. Values below this
    # threshold are numerically indistinguishable from zero.
    if exponent < -745.0:
        J = 0.0
    else:
        J = (
            N0
            * Zn
            * b
            * math.exp(exponent)
            * incubation_factor
        )

    if not np.isfinite(J) or J <= 0.0:
        J = 0.0

    dN = J * dt

    RkBT = (
        Rstar
        + 0.5
        * np.sqrt(
            kB * T
            / (np.pi * gamma)
        )
    )

    return dN, RkBT, Rstar, J

def nucleation_dt_limit_ternary(
    cbar_mg,
    cbar_si,
    V_matrix,
    dt_try,
    t,
    T,
    cnt_params_by_phase,
    lam_target,
    *,
    cbar_gb_mg=None,
    cbar_gb_si=None,
    centers=None,
    radii=None,
    nuc_kind=None,
    box=None,
    gb_point=None,
    gb_normal=None,
    gb_thickness=0.0,
    gb_barrier_factor=1.0,
    gb_exclusion_buffer=0.0,
    gb_use_area_fraction=True,
    diagnostics=False,
):
    """
    Return the ternary nucleation timestep limit for competing phases.

    The timestep limiter is based on the *sum* of the bulk and grain-boundary
    proposal rates over all precipitate phases.  Phase-resolved rates and CNT
    reference quantities are also returned so that the stochastic nucleation
    step can subsequently select/propose events by phase.

    The caller may choose to use only phase 0 for actual particle creation;
    the limiter itself is fully phase-aware.
    """
    required = (
        "aa",
        "bb",
        "cc",
        "vatm",
        "vatp",
        "gamma",
        "N0",
        "DMg",
        "DSi",
        "a_latt",
    )

    cnt_params_by_phase = list(cnt_params_by_phase)
    n_phases = len(cnt_params_by_phase)

    if n_phases == 0:
        raise ValueError(
            "cnt_params_by_phase must contain at least one phase."
        )

    for p, cnt_params in enumerate(cnt_params_by_phase):
        missing = [
            key
            for key in required
            if key not in cnt_params
        ]

        if missing:
            raise ValueError(
                f"Missing ternary CNT parameters for phase {p}: "
                + ", ".join(missing)
            )

    # Phase-resolved outputs.  These are deliberately arrays even for a
    # one-phase calculation so the caller has one consistent interface.
    J_bulk_phase = np.zeros(n_phases, dtype=float)
    J_gb_phase = np.zeros(n_phases, dtype=float)
    RkBT_bulk_phase = np.full(n_phases, np.inf, dtype=float)
    RkBT_gb_phase = np.full(n_phases, np.inf, dtype=float)
    event_rate_bulk_phase = np.zeros(n_phases, dtype=float)
    event_rate_gb_phase = np.zeros(n_phases, dtype=float)

    if (
        not np.isfinite(V_matrix)
        or V_matrix <= 0.0
    ):
        return (
            np.inf,
            0.0,
            0.0,
            J_bulk_phase,
            J_gb_phase,
            RkBT_bulk_phase,
            RkBT_gb_phase,
            event_rate_bulk_phase,
            event_rate_gb_phase,
        )

    # ----------------------------------------------------------
    # Grain-boundary geometry is common to all candidate phases.
    # ----------------------------------------------------------
    gb_enabled = (
        gb_point is not None
        and gb_normal is not None
        and gb_thickness is not None
        and gb_thickness > 0.0
    )

    if gb_enabled:
        if box is None:
            raise ValueError(
                "box must be supplied when grain-boundary "
                "nucleation is enabled."
            )

        if cbar_gb_mg is None or cbar_gb_si is None:
            raise ValueError(
                "cbar_gb_mg and cbar_gb_si must be supplied "
                "when grain-boundary nucleation is enabled."
            )

        if (
            not np.isfinite(cbar_gb_mg)
            or not np.isfinite(cbar_gb_si)
            or cbar_gb_mg < 0.0
            or cbar_gb_si < 0.0
        ):
            raise ValueError(
                "Grain-boundary mean compositions must be "
                "finite and non-negative."
            )

        box = as_box_lengths(box)
        _, Ly, Lz = box

        V_gb_geom = Ly * Lz * gb_thickness
        V_gb = min(V_matrix, V_gb_geom)
        V_bulk = max(0.0, V_matrix - V_gb)

        phi_gb_free = 1.0

        if gb_use_area_fraction:
            if centers is None:
                centers = np.zeros(
                    (0, 3),
                    dtype=float,
                )

            if radii is None:
                radii = np.zeros(
                    (0,),
                    dtype=float,
                )

            if nuc_kind is None:
                nuc_kind = np.zeros(
                    len(radii),
                    dtype=np.int32,
                )

            gb_normal_unit = normalize_vector(
                gb_normal
            )

            gb_area = Ly * Lz

            phi_gb_free = estimate_gb_free_area_fraction(
                centers=np.asarray(
                    centers,
                    dtype=float,
                ),
                radii=np.asarray(
                    radii,
                    dtype=float,
                ),
                nuc_kind=np.asarray(
                    nuc_kind,
                    dtype=np.int32,
                ),
                gb_point=gb_point,
                gb_normal=gb_normal_unit,
                gb_area=gb_area,
                box=box,
                exclusion_buffer=
                    gb_exclusion_buffer,
            )
    else:
        V_bulk = V_matrix
        V_gb = 0.0
        phi_gb_free = 1.0

    # ----------------------------------------------------------
    # Evaluate the CNT reference rate independently for each phase.
    # ----------------------------------------------------------
    for p, cnt_params in enumerate(cnt_params_by_phase):
        x = cnt_params.get("x", 5)
        y = cnt_params.get("y", 6)
        gb_barrier_factor_p = float(
            cnt_params.get("gb_barrier_factor", gb_barrier_factor)
        )
        if (
            not np.isfinite(gb_barrier_factor_p)
            or gb_barrier_factor_p <= 0.0
        ):
            raise ValueError(
                f"Phase {p} GB barrier factor must be positive and finite."
            )

        (
            _,
            RkBT_bulk,
            _Rstar_bulk,
            J_bulk,
        ) = nucleation_step_ternary(
            dt=dt_try,
            T=T,
            t=t,
            C0Mg=cbar_mg,
            C0Si=cbar_si,
            aa=cnt_params["aa"],
            bb=cnt_params["bb"],
            cc=cnt_params["cc"],
            vatm=cnt_params["vatm"],
            vatp=cnt_params["vatp"],
            gamma=cnt_params["gamma"],
            N0=cnt_params["N0"],
            DMg=cnt_params["DMg"],
            DSi=cnt_params["DSi"],
            a_latt=cnt_params["a_latt"],
            x=x,
            y=y,
            barrier_factor=1.0,
            apply_incubation_correction=cnt_params.get(
                "apply_incubation_correction",
                True,
            ),
        )

        J_bulk_phase[p] = J_bulk
        RkBT_bulk_phase[p] = RkBT_bulk

        if (
            np.isfinite(J_bulk)
            and J_bulk > 0.0
            and V_bulk > 0.0
        ):
            event_rate_bulk_phase[p] = (
                J_bulk * V_bulk
            )

        if gb_enabled:
            (
                _,
                RkBT_gb,
                _Rstar_gb,
                J_gb,
            ) = nucleation_step_ternary(
                dt=dt_try,
                T=T,
                t=t,
                C0Mg=cbar_gb_mg,
                C0Si=cbar_gb_si,
                aa=cnt_params["aa"],
                bb=cnt_params["bb"],
                cc=cnt_params["cc"],
                vatm=cnt_params["vatm"],
                vatp=cnt_params["vatp"],
                gamma=cnt_params["gamma"],
                N0=cnt_params["N0"],
                DMg=cnt_params["DMg"],
                DSi=cnt_params["DSi"],
                a_latt=cnt_params["a_latt"],
                x=x,
                y=y,
                barrier_factor=gb_barrier_factor_p,
                apply_incubation_correction=cnt_params.get(
                    "apply_incubation_correction",
                    True,
                ),
            )

            J_gb_phase[p] = J_gb
            RkBT_gb_phase[p] = RkBT_gb

            if (
                np.isfinite(J_gb)
                and J_gb > 0.0
                and V_gb > 0.0
            ):
                event_rate_gb_phase[p] = (
                    J_gb
                    * V_gb
                    * phi_gb_free
                )


    # ----------------------------------------------------------
    # Competing-phase proposal event rate [s^-1].
    # lam_target constrains the total number of proposed events, not the
    # proposal population of each phase independently.
    # ----------------------------------------------------------
    event_rate_bulk_total = float(
        np.sum(event_rate_bulk_phase)
    )
    event_rate_gb_total = float(
        np.sum(event_rate_gb_phase)
    )
    event_rate_total = (
        event_rate_bulk_total
        + event_rate_gb_total
    )

    if (
        not np.isfinite(event_rate_total)
        or event_rate_total <= 0.0
    ):
        dt_nuc = np.inf
    else:
        dt_nuc = (
            lam_target
            / event_rate_total
        )

    return (
        dt_nuc,
        event_rate_bulk_total,
        event_rate_gb_total,
        J_bulk_phase,
        J_gb_phase,
        RkBT_bulk_phase,
        RkBT_gb_phase,
        event_rate_bulk_phase,
        event_rate_gb_phase,
    )

def nucleation_step_species(
    dt,
    T,
    t,
    cbar,
    aa,
    bb,
    cc,
    vatm,
    vatp,
    gamma,
    N0,
    D,
    a_latt,
    stoichiometric_weights,
    solvent_stoichiometric_weight=0.0,
    barrier_factor=1.0,
    apply_incubation_correction=True,
):
    """Evaluate the CNT rate for an arbitrary transported-solute vector.

    ``cbar`` and ``D`` are ordered on the transported-species axis.  The
    corresponding phase stoichiometric coefficients are supplied through
    ``stoichiometric_weights``.  A non-diffusing balance solvent may be
    represented by ``solvent_stoichiometric_weight``.  Its matrix site fraction
    is taken as

        X_solvent = 1 - sum_s X_s,

    so the ideal solution supersaturation is

        ln S = nu_solvent ln(X_solvent)
             + sum_s nu_s ln(X_s)
             - ln K(T).

    The diffusion attachment resistance contains only transported species,
    while their precipitate site fractions use the *full* formula-unit atom
    count, including solvent atoms.  Thus S=1 with solvent coefficient 3 gives
    the same attachment factor as the existing Al3Sc binary implementation.

    For the established two-solute/no-solvent case this function dispatches
    directly to ``nucleation_step_ternary`` to preserve the exact regression
    path.
    """
    cbar = np.asarray(cbar, dtype=float)
    D = np.asarray(D, dtype=float)
    weights = np.asarray(stoichiometric_weights, dtype=float)

    if cbar.ndim != 1 or cbar.size == 0:
        raise ValueError("cbar must have shape (S,) with S >= 1.")
    n_solutes = cbar.size
    if D.shape != (n_solutes,):
        raise ValueError(
            f"D must have shape ({n_solutes},); got {D.shape}."
        )
    if weights.shape != (n_solutes,):
        raise ValueError(
            "stoichiometric_weights must match the transported-species axis; "
            f"expected ({n_solutes},), got {weights.shape}."
        )
    if np.any(~np.isfinite(cbar)) or np.any(cbar <= 0.0):
        return 0.0, float("inf"), float("inf"), 0.0
    if np.any(~np.isfinite(D)) or np.any(D <= 0.0):
        raise ValueError("D must contain positive finite diffusivities.")
    if np.any(~np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError(
            "stoichiometric_weights must contain positive finite values."
        )

    solvent_weight = float(solvent_stoichiometric_weight)
    if not np.isfinite(solvent_weight) or solvent_weight < 0.0:
        raise ValueError(
            "solvent_stoichiometric_weight must be finite and non-negative."
        )

    # Exact compatibility dispatch for the current Mg-Si-style kernel.
    if n_solutes == 2 and solvent_weight == 0.0:
        return nucleation_step_ternary(
            dt=dt,
            T=T,
            t=t,
            C0Mg=float(cbar[0]),
            C0Si=float(cbar[1]),
            aa=aa,
            bb=bb,
            cc=cc,
            vatm=vatm,
            vatp=vatp,
            gamma=gamma,
            N0=N0,
            DMg=float(D[0]),
            DSi=float(D[1]),
            a_latt=a_latt,
            x=float(weights[0]),
            y=float(weights[1]),
            barrier_factor=barrier_factor,
            apply_incubation_correction=apply_incubation_correction,
        )

    if dt < 0.0:
        raise ValueError("dt must be non-negative.")
    if T <= 0.0:
        raise ValueError("T must be positive.")
    if vatm <= 0.0 or vatp <= 0.0:
        raise ValueError("vatm and vatp must be positive.")
    if gamma <= 0.0:
        raise ValueError("gamma must be positive.")
    if N0 < 0.0:
        raise ValueError("N0 must be non-negative.")
    if a_latt <= 0.0:
        raise ValueError("a_latt must be positive.")
    if barrier_factor < 0.0:
        raise ValueError("barrier_factor must be non-negative.")

    X = cbar * float(vatm)
    if np.any(~np.isfinite(X)) or np.any(X <= 0.0):
        return 0.0, float("inf"), float("inf"), 0.0

    X_solvent = 1.0 - float(np.sum(X))
    if solvent_weight > 0.0 and (
        not np.isfinite(X_solvent) or X_solvent <= 0.0
    ):
        return 0.0, float("inf"), float("inf"), 0.0

    log_K = np.log(10.0) * (
        float(bb)
        - float(aa) / float(T)
        + float(cc) / float(T) ** 2
    )
    log_S = float(np.dot(weights, np.log(X))) - log_K
    if solvent_weight > 0.0:
        log_S += solvent_weight * math.log(X_solvent)

    if not np.isfinite(log_S) or log_S <= 0.0:
        return 0.0, float("inf"), float("inf"), 0.0

    nu_total = solvent_weight + float(np.sum(weights))
    if not np.isfinite(nu_total) or nu_total <= 0.0:
        raise ValueError("The total formula-unit stoichiometry must be positive.")

    v_fu = nu_total * float(vatp)
    deltaGv = -(kB * float(T) / v_fu) * log_S
    Rstar = -2.0 * float(gamma) / deltaGv
    DGstar = (
        16.0
        * np.pi
        * float(gamma) ** 3
        / (3.0 * deltaGv ** 2)
    )
    DGstar_eff = float(barrier_factor) * DGstar

    Zn = (
        v_fu
        / (2.0 * np.pi * Rstar ** 2)
        * math.sqrt(float(gamma) / (kB * float(T)))
    )

    X_safe = np.maximum(X, np.finfo(float).tiny)
    Xp = weights / nu_total
    diffusion_resistance = float(np.sum(Xp / (D * X_safe)))
    if (
        not np.isfinite(diffusion_resistance)
        or diffusion_resistance <= 0.0
    ):
        return 0.0, Rstar, Rstar, 0.0

    b = (
        4.0
        * np.pi
        * Rstar ** 2
        / float(a_latt) ** 4
        / diffusion_resistance
    )
    if not np.isfinite(b) or b <= 0.0:
        return 0.0, Rstar, Rstar, 0.0

    tau = 4.0 / (2.0 * np.pi * b * Zn ** 2)
    apply_incubation_correction = bool(apply_incubation_correction)
    if not apply_incubation_correction:
        incubation_factor = 1.0
    elif not np.isfinite(tau) or tau <= 0.0:
        incubation_factor = 1.0
    elif t <= 0.0:
        incubation_factor = 0.0
    else:
        incubation_factor = -math.expm1(-float(t) / tau)

    exponent = -DGstar_eff / (kB * float(T))
    if exponent < -745.0:
        J = 0.0
    else:
        J = (
            float(N0)
            * Zn
            * b
            * math.exp(exponent)
            * incubation_factor
        )
    if not np.isfinite(J) or J <= 0.0:
        J = 0.0

    dN = J * float(dt)
    RkBT = (
        Rstar
        + 0.5
        * np.sqrt(kB * float(T) / (np.pi * float(gamma)))
    )
    return dN, RkBT, Rstar, J


def nucleation_dt_limit_species(
    cbar,
    V_matrix,
    dt_try,
    t,
    T,
    cnt_params_by_phase,
    lam_target,
    *,
    cbar_gb=None,
    centers=None,
    radii=None,
    nuc_kind=None,
    box=None,
    gb_point=None,
    gb_normal=None,
    gb_thickness=0.0,
    gb_barrier_factor=1.0,
    gb_exclusion_buffer=0.0,
    gb_use_area_fraction=True,
):
    """Species-indexed competing-phase CNT timestep limiter.

    The established two-solute/no-solvent parameter records retain an exact
    dispatch to ``nucleation_dt_limit_ternary``.  Other transported-species
    counts use ``nucleation_step_species`` and require each phase record to
    contain ``D``, ``stoichiometric_weights``, and optionally
    ``solvent_stoichiometric_weight``.
    """
    cbar = np.asarray(cbar, dtype=float)
    if cbar.ndim != 1 or cbar.size == 0:
        raise ValueError("cbar must have shape (S,) with S >= 1.")
    if np.any(~np.isfinite(cbar)) or np.any(cbar < 0.0):
        raise ValueError("cbar must contain finite, non-negative values.")
    n_solutes = cbar.size

    gb_enabled = (
        gb_point is not None
        and gb_normal is not None
        and gb_thickness is not None
        and gb_thickness > 0.0
    )

    if cbar_gb is not None:
        cbar_gb = np.asarray(cbar_gb, dtype=float)
        if cbar_gb.shape != cbar.shape:
            raise ValueError(
                "cbar_gb must have the same shape as cbar; "
                f"got cbar_gb.shape={cbar_gb.shape}, cbar.shape={cbar.shape}."
            )
        if np.any(~np.isfinite(cbar_gb)) or np.any(cbar_gb < 0.0):
            raise ValueError(
                "cbar_gb must contain finite, non-negative values."
            )
    if gb_enabled and cbar_gb is None:
        raise ValueError(
            "cbar_gb must be supplied when grain-boundary nucleation is enabled."
        )

    cnt_params_by_phase = list(cnt_params_by_phase)
    n_phases = len(cnt_params_by_phase)
    if n_phases == 0:
        raise ValueError(
            "cnt_params_by_phase must contain at least one phase."
        )

    # Preserve the exact current two-solute path whenever the parameter records
    # describe phases with no explicit balance-solvent stoichiometry.
    legacy_two_solute = n_solutes == 2 and all(
        float(params.get("solvent_stoichiometric_weight", 0.0)) == 0.0
        for params in cnt_params_by_phase
    )
    if legacy_two_solute:
        cbar_gb0 = None if cbar_gb is None else float(cbar_gb[0])
        cbar_gb1 = None if cbar_gb is None else float(cbar_gb[1])
        return nucleation_dt_limit_ternary(
            cbar_mg=float(cbar[0]),
            cbar_si=float(cbar[1]),
            cbar_gb_mg=cbar_gb0,
            cbar_gb_si=cbar_gb1,
            V_matrix=V_matrix,
            dt_try=dt_try,
            t=t,
            T=T,
            cnt_params_by_phase=cnt_params_by_phase,
            lam_target=lam_target,
            centers=centers,
            radii=radii,
            nuc_kind=nuc_kind,
            box=box,
            gb_point=gb_point,
            gb_normal=gb_normal,
            gb_thickness=gb_thickness,
            gb_barrier_factor=gb_barrier_factor,
            gb_exclusion_buffer=gb_exclusion_buffer,
            gb_use_area_fraction=gb_use_area_fraction,
        )

    required = (
        "aa", "bb", "cc", "vatm", "vatp", "gamma", "N0", "a_latt",
        "D", "stoichiometric_weights",
    )
    for p, params in enumerate(cnt_params_by_phase):
        missing = [key for key in required if key not in params]
        if missing:
            raise ValueError(
                f"Missing species-indexed CNT parameters for phase {p}: "
                + ", ".join(missing)
            )
        D_p = np.asarray(params["D"], dtype=float)
        w_p = np.asarray(params["stoichiometric_weights"], dtype=float)
        if D_p.shape != (n_solutes,) or w_p.shape != (n_solutes,):
            raise ValueError(
                f"Phase {p} CNT D and stoichiometric_weights must have shape "
                f"({n_solutes},)."
            )

    J_bulk_phase = np.zeros(n_phases, dtype=float)
    J_gb_phase = np.zeros(n_phases, dtype=float)
    RkBT_bulk_phase = np.full(n_phases, np.inf, dtype=float)
    RkBT_gb_phase = np.full(n_phases, np.inf, dtype=float)
    event_rate_bulk_phase = np.zeros(n_phases, dtype=float)
    event_rate_gb_phase = np.zeros(n_phases, dtype=float)

    if not np.isfinite(V_matrix) or V_matrix <= 0.0:
        return (
            np.inf, 0.0, 0.0,
            J_bulk_phase, J_gb_phase,
            RkBT_bulk_phase, RkBT_gb_phase,
            event_rate_bulk_phase, event_rate_gb_phase,
        )

    if gb_enabled:
        if box is None:
            raise ValueError(
                "box must be supplied when grain-boundary nucleation is enabled."
            )
        box = as_box_lengths(box)
        _, Ly, Lz = box
        V_gb_geom = Ly * Lz * gb_thickness
        V_gb = min(V_matrix, V_gb_geom)
        V_bulk = max(0.0, V_matrix - V_gb)
        phi_gb_free = 1.0
        if gb_use_area_fraction:
            if centers is None:
                centers = np.zeros((0, 3), dtype=float)
            if radii is None:
                radii = np.zeros((0,), dtype=float)
            if nuc_kind is None:
                nuc_kind = np.zeros(len(radii), dtype=np.int32)
            gb_normal_unit = normalize_vector(gb_normal)
            gb_area = Ly * Lz
            phi_gb_free = estimate_gb_free_area_fraction(
                centers=np.asarray(centers, dtype=float),
                radii=np.asarray(radii, dtype=float),
                nuc_kind=np.asarray(nuc_kind, dtype=np.int32),
                gb_point=gb_point,
                gb_normal=gb_normal_unit,
                gb_area=gb_area,
                box=box,
                exclusion_buffer=gb_exclusion_buffer,
            )
    else:
        V_bulk = V_matrix
        V_gb = 0.0
        phi_gb_free = 1.0

    for p, params in enumerate(cnt_params_by_phase):
        barrier_p = float(
            params.get("gb_barrier_factor", gb_barrier_factor)
        )
        if not np.isfinite(barrier_p) or barrier_p <= 0.0:
            raise ValueError(
                f"Phase {p} GB barrier factor must be positive and finite."
            )

        _, RkBT_bulk, _, J_bulk = nucleation_step_species(
            dt=dt_try,
            T=T,
            t=t,
            cbar=cbar,
            aa=params["aa"],
            bb=params["bb"],
            cc=params["cc"],
            vatm=params["vatm"],
            vatp=params["vatp"],
            gamma=params["gamma"],
            N0=params["N0"],
            D=params["D"],
            a_latt=params["a_latt"],
            stoichiometric_weights=params["stoichiometric_weights"],
            solvent_stoichiometric_weight=params.get(
                "solvent_stoichiometric_weight", 0.0
            ),
            barrier_factor=1.0,
            apply_incubation_correction=params.get(
                "apply_incubation_correction", True
            ),
        )
        J_bulk_phase[p] = J_bulk
        RkBT_bulk_phase[p] = RkBT_bulk
        if np.isfinite(J_bulk) and J_bulk > 0.0 and V_bulk > 0.0:
            event_rate_bulk_phase[p] = J_bulk * V_bulk

        if gb_enabled:
            _, RkBT_gb, _, J_gb = nucleation_step_species(
                dt=dt_try,
                T=T,
                t=t,
                cbar=cbar_gb,
                aa=params["aa"],
                bb=params["bb"],
                cc=params["cc"],
                vatm=params["vatm"],
                vatp=params["vatp"],
                gamma=params["gamma"],
                N0=params["N0"],
                D=params["D"],
                a_latt=params["a_latt"],
                stoichiometric_weights=params["stoichiometric_weights"],
                solvent_stoichiometric_weight=params.get(
                    "solvent_stoichiometric_weight", 0.0
                ),
                barrier_factor=barrier_p,
                apply_incubation_correction=params.get(
                    "apply_incubation_correction", True
                ),
            )
            J_gb_phase[p] = J_gb
            RkBT_gb_phase[p] = RkBT_gb
            if np.isfinite(J_gb) and J_gb > 0.0 and V_gb > 0.0:
                event_rate_gb_phase[p] = J_gb * V_gb * phi_gb_free

    event_rate_bulk_total = float(np.sum(event_rate_bulk_phase))
    event_rate_gb_total = float(np.sum(event_rate_gb_phase))
    event_rate_total = event_rate_bulk_total + event_rate_gb_total
    if not np.isfinite(event_rate_total) or event_rate_total <= 0.0:
        dt_nuc = np.inf
    else:
        dt_nuc = float(lam_target) / event_rate_total

    return (
        dt_nuc,
        event_rate_bulk_total,
        event_rate_gb_total,
        J_bulk_phase,
        J_gb_phase,
        RkBT_bulk_phase,
        RkBT_gb_phase,
        event_rate_bulk_phase,
        event_rate_gb_phase,
    )


def estimate_gb_matrix_composition_species(
    *,
    n_samples,
    centers,
    radii,
    Q,
    cbar,
    box,
    alpha,
    real_lattice,
    kvecs,
    k2,
    gb_point,
    gb_normal,
    gb_thickness,
    rng,
    max_attempts_per_sample=100,
    concentration_evaluator=None,
):
    """Estimate mean transported-solute concentrations in the GB matrix slab.

    ``Q`` has shape ``(S, N)`` and ``cbar`` has shape ``(S,)``.  The same
    accepted GB sample points are used for every transported species.  Points
    lying inside existing precipitates are excluded.

    Returns a dictionary containing species-indexed ``c_gb_mean`` and
    ``c_gb_std`` arrays plus ``n_valid``.
    """
    centers = np.asarray(centers, dtype=float)
    radii = np.asarray(radii, dtype=float)
    Q = np.asarray(Q, dtype=float)
    cbar = np.asarray(cbar, dtype=float)

    if cbar.ndim != 1 or cbar.size == 0:
        raise ValueError("cbar must have shape (S,) with S >= 1.")
    n_solutes = cbar.size
    if Q.ndim != 2 or Q.shape[0] != n_solutes:
        raise ValueError(
            "Q must have shape (S, N) with the same species axis as cbar; "
            f"got Q.shape={Q.shape}, cbar.shape={cbar.shape}."
        )
    if Q.shape[1] != centers.shape[0] or radii.shape != (centers.shape[0],):
        raise ValueError(
            "centers, radii, and the particle axis of Q must have matching lengths."
        )

    samples = []

    for _ in range(n_samples):
        pt = None

        for _ in range(max_attempts_per_sample):
            trial_pt = sample_gb_point(
                box,
                rng,
                gb_point,
                gb_normal,
                gb_thickness,
            )

            if trial_pt is None:
                break

            if centers.shape[0] == 0:
                pt = trial_pt
                break

            dvec = minimum_image_displacement(
                centers - trial_pt,
                box,
            )
            dist = np.linalg.norm(dvec, axis=1)

            if np.any(dist < radii):
                continue

            pt = trial_pt
            break

        if pt is None:
            continue

        if concentration_evaluator is None:
            c_loc = np.empty(n_solutes, dtype=float)
            for s in range(n_solutes):
                c_loc[s] = concentration_at_point(
                    pt,
                    centers,
                    Q[s],
                    float(cbar[s]),
                    box,
                    alpha,
                    real_lattice,
                    kvecs,
                    k2,
                )
        else:
            c_eval = np.asarray(
                concentration_evaluator(
                    np.asarray(pt, dtype=float).reshape(1, 3)
                ),
                dtype=float,
            )
            if c_eval.shape != (1, n_solutes):
                raise ValueError(
                    "concentration_evaluator must return shape "
                    f"(M,{n_solutes}); got {c_eval.shape}."
                )
            c_loc = c_eval[0]

        if np.all(np.isfinite(c_loc)) and np.all(c_loc > 0.0):
            samples.append(c_loc)

    if len(samples) == 0:
        return {
            "c_gb_mean": cbar.copy(),
            "c_gb_std": np.full(n_solutes, np.nan, dtype=float),
            "n_valid": 0,
        }

    samples = np.asarray(samples, dtype=float)
    return {
        "c_gb_mean": np.mean(samples, axis=0),
        "c_gb_std": np.std(samples, axis=0),
        "n_valid": int(samples.shape[0]),
    }


def estimate_gb_matrix_composition(
    *,
    n_samples,
    centers,
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
    gb_point,
    gb_normal,
    gb_thickness,
    rng,
    max_attempts_per_sample=100,
    concentration_evaluator=None,
):
    """Backward-compatible two-solute wrapper around the generic sampler."""
    result = estimate_gb_matrix_composition_species(
        n_samples=n_samples,
        centers=centers,
        radii=radii,
        Q=np.vstack((Q_mg, Q_si)),
        cbar=np.asarray((cbar_mg, cbar_si), dtype=float),
        box=box,
        alpha=alpha,
        real_lattice=real_lattice,
        kvecs=kvecs,
        k2=k2,
        gb_point=gb_point,
        gb_normal=gb_normal,
        gb_thickness=gb_thickness,
        rng=rng,
        max_attempts_per_sample=max_attempts_per_sample,
        concentration_evaluator=concentration_evaluator,
    )
    return {
        "c_gb_mg_mean": float(result["c_gb_mean"][0]),
        "c_gb_si_mean": float(result["c_gb_mean"][1]),
        "c_gb_mg_std": float(result["c_gb_std"][0]),
        "c_gb_si_std": float(result["c_gb_std"][1]),
        "n_valid": int(result["n_valid"]),
    }



def _empty_nucleation_rate_diagnostic():
    """Return zeroed accumulators for temporary stochastic CNT diagnostics."""
    return {
        "bulk": {
            "n_rate_evaluated": 0,
            "sum_Jloc_over_Jref": 0.0,
            "sum_Jloc": 0.0,
            "sum_Jloc_sq": 0.0,
            "sum_clipped_Jloc_over_Jref": 0.0,
            "n_Jloc_gt_Jref": 0,
            "max_Jloc_over_Jref": 0.0,
        },
        "gb": {
            "n_rate_evaluated": 0,
            "sum_Jloc_over_Jref": 0.0,
            "sum_Jloc": 0.0,
            "sum_Jloc_sq": 0.0,
            "sum_clipped_Jloc_over_Jref": 0.0,
            "n_Jloc_gt_Jref": 0,
            "max_Jloc_over_Jref": 0.0,
        },
    }


def _accumulate_nucleation_rate_diagnostic(rate_diagnostic, kind, J_loc, J_ref):
    """Accumulate J_local/J_ref without changing stochastic decisions or RNG use."""
    if (
        J_ref is None
        or not np.isfinite(J_ref)
        or J_ref <= 0.0
        or not np.isfinite(J_loc)
        or J_loc < 0.0
    ):
        return

    ratio = float(J_loc) / float(J_ref)
    region = rate_diagnostic["bulk" if kind == 0 else "gb"]
    region["n_rate_evaluated"] += 1
    region["sum_Jloc_over_Jref"] += ratio
    region["sum_Jloc"] += float(J_loc)
    region["sum_Jloc_sq"] += float(J_loc) * float(J_loc)
    region["sum_clipped_Jloc_over_Jref"] += min(1.0, ratio)
    if ratio > 1.0:
        region["n_Jloc_gt_Jref"] += 1
    region["max_Jloc_over_Jref"] = max(
        region["max_Jloc_over_Jref"],
        ratio,
    )

def nucleate_particles_cnt_monopole_ternary(
    centers,
    radii,
    Q_mg,
    Q_si,
    box,
    dt,
    t,
    T,
    cbar_mg,
    cbar_si,
    aa,
    bb,
    cc,
    vatm,
    vatp,
    gamma,
    N0,
    D_mg,
    D_si,
    a_latt,
    alpha,
    real_lattice,
    kvecs,
    k2,
    x=5,
    y=6,
    lam_target=1.0,
    apply_incubation_correction=True,
    matrix_volume=None,
    safety_buffer=0.0,
    rng=None,
    gb_point=None,
    gb_normal=None,
    gb_thickness=0.0,
    gb_mode="slab",
    gb_barrier_factor=1.0,
    nuc_kind=None,
    gb_exclusion_buffer=0.0,
    gb_use_area_fraction=True,
    cbar_gb_mg=None,
    cbar_gb_si=None,
    event_rate_bulk=None,
    event_rate_gb=None,
    J_ref_bulk=None,
    J_ref_gb=None,
    RkBT_ref_bulk=None,
    RkBT_ref_gb=None,
    candidate_sampling_alpha=2.0,
    candidate_sampling_min_pool=20,
    candidate_sampling_ess_per_event=1.5,
    candidate_sampling_min_ess=10.0,
    candidate_sampling_max_alpha=8.0,
    candidate_sampling_max_pool=80,
    spatial_rate_correction=True,
    spatial_rate_correction_max_factor=5.0,
    spatial_rate_sampling_rse_target=0.20,
    spatial_rate_sampling_min_samples=20,
    spatial_rate_sampling_max_samples=320,
    spatial_rate_probe_hazard=0.10,
    spatial_rate_skipped_hazard_bulk=0.0,
    spatial_rate_skipped_hazard_gb=0.0,
    spatial_rate_last_factor_bulk=1.0,
    spatial_rate_last_factor_gb=1.0,
    candidate_concentration_evaluator=None,
    diagnostics=False,
):
    """
    Perform one ternary Mg-Si CNT nucleation update using two
    monopole concentration fields.

    Existing-particle concentration fields are reconstructed as

        C_Mg(x) = cbar_mg + sum_j Q_mg[j] G(x-x_j)

        C_Si(x) = cbar_si + sum_j Q_si[j] G(x-x_j)

    Newly accepted particles are included in geometric overlap
    checks during this timestep, but do not contribute to either
    concentration field until the integrator performs the next
    coupled solve.
    """
    if rng is None:
        rng = np.random.default_rng()

    centers = np.asarray(
        centers,
        dtype=float,
    )

    radii = np.asarray(
        radii,
        dtype=float,
    )

    Q_mg = np.asarray(
        Q_mg,
        dtype=float,
    )

    Q_si = np.asarray(
        Q_si,
        dtype=float,
    )

    if centers.size == 0:
        centers = np.zeros(
            (0, 3),
            dtype=float,
        )

    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError(
            "centers must have shape (N, 3)."
        )

    n_existing = len(radii)

    if centers.shape[0] != n_existing:
        raise ValueError(
            "centers and radii must have the same length."
        )

    if Q_mg.shape != (n_existing,):
        raise ValueError(
            "Q_mg must have the same length as radii."
        )

    if Q_si.shape != (n_existing,):
        raise ValueError(
            "Q_si must have the same length as radii."
        )

    if np.any(~np.isfinite(Q_mg)):
        raise ValueError(
            "Q_mg contains non-finite values."
        )

    if np.any(~np.isfinite(Q_si)):
        raise ValueError(
            "Q_si contains non-finite values."
        )

    if nuc_kind is None:
        nuc_kind = np.zeros(
            n_existing,
            dtype=np.int32,
        )
    else:
        nuc_kind = np.asarray(
            nuc_kind,
            dtype=np.int32,
        ).copy()

        if nuc_kind.shape != (n_existing,):
            raise ValueError(
                "nuc_kind must have the same length as radii."
            )

    box = as_box_lengths(box)
    Lx, Ly, Lz = box
    V_box = Lx * Ly * Lz

    if matrix_volume is None:
        V_matrix = (
            V_box
            - np.sum(
                particle_volumes(radii)
            )
        )
    else:
        V_matrix = float(matrix_volume)

    gb_enabled = (
        gb_point is not None
        and gb_normal is not None
        and gb_thickness is not None
        and gb_thickness > 0.0
    )

    if gb_enabled:
        gb_normal = normalize_vector(
            gb_normal
        )

        V_gb_geom = (
            Ly
            * Lz
            * gb_thickness
        )

        V_gb = min(
            V_matrix,
            V_gb_geom,
        )

        V_bulk = max(
            0.0,
            V_matrix - V_gb,
        )

        gb_area = Ly * Lz

    else:
        V_gb = 0.0
        V_bulk = V_matrix
        gb_point = None
        gb_area = 0.0

#     (
#         _,
#         RkBT_bulk_mean,
#         Rstar_bulk_mean,
#         J_bulk_mean,
#     ) = nucleation_step_ternary(
#         dt=dt,
#         T=T,
#         t=t,
#         C0Mg=cbar_mg,
#         C0Si=cbar_si,
#         aa=aa,
#         bb=bb,
#         cc=cc,
#         vatm=vatm,
#         vatp=vatp,
#         gamma=gamma,
#         N0=N0,
#         DMg=D_mg,
#         DSi=D_si,
#         a_latt=a_latt,
#         x=x,
#         y=y,
#         barrier_factor=1.0,
#         apply_incubation_correction=apply_incubation_correction,
#     )

#     if gb_enabled:
# #------------
#         gb_comp = estimate_gb_matrix_composition(
#             n_samples=50,
#             centers=centers,
#             radii=radii,
#             Q_mg=Q_mg,
#             Q_si=Q_si,
#             cbar_mg=cbar_mg,
#             cbar_si=cbar_si,
#             box=box,
#             alpha=alpha,
#             real_lattice=real_lattice,
#             kvecs=kvecs,
#             k2=k2,
#             gb_point=gb_point,
#             gb_normal=gb_normal,
#             gb_thickness=gb_thickness,
#             rng=rng,
#         )
    
#         c_gb_mg_mean = gb_comp[
#             "c_gb_mg_mean"
#         ]
    
#         c_gb_si_mean = gb_comp[
#             "c_gb_si_mean"
#         ]
    
#         (
#             _,
#             RkBT_gb_mean,
#             Rstar_gb_mean,
#             J_gb_mean,
#         ) = nucleation_step_ternary(
#             dt=dt,
#             T=T,
#             t=t,
#             C0Mg=c_gb_mg_mean,
#             C0Si=c_gb_si_mean,
#             aa=aa,
#             bb=bb,
#             cc=cc,
#             vatm=vatm,
#             vatp=vatp,
#             gamma=gamma,
#             N0=N0,
#             DMg=D_mg,
#             DSi=D_si,
#             a_latt=a_latt,
#             x=x,
#             y=y,
#             barrier_factor=gb_barrier_factor,
#             apply_incubation_correction=
#                 apply_incubation_correction,
#         )
# #------------      
#         # (
#         #     _,
#         #     RkBT_gb_mean,
#         #     Rstar_gb_mean,
#         #     J_gb_mean,
#         # ) = nucleation_step_ternary(
#         #     dt=dt,
#         #     T=T,
#         #     t=t,
#         #     C0Mg=cbar_mg,
#         #     C0Si=cbar_si,
#         #     aa=aa,
#         #     bb=bb,
#         #     cc=cc,
#         #     vatm=vatm,
#         #     vatp=vatp,
#         #     gamma=gamma,
#         #     N0=N0,
#         #     DMg=D_mg,
#         #     DSi=D_si,
#         #     a_latt=a_latt,
#         #     x=x,
#         #     y=y,
#         #     barrier_factor=gb_barrier_factor,
#         #     apply_incubation_correction=apply_incubation_correction,
#         # )

#     else:
#         RkBT_gb_mean = float("inf")
#         Rstar_gb_mean = float("inf")
#         J_gb_mean = 0.0

    phi_gb_free = 1.0

    if gb_enabled and gb_use_area_fraction:
        phi_gb_free = estimate_gb_free_area_fraction(
            centers=centers,
            radii=radii,
            nuc_kind=nuc_kind,
            gb_point=gb_point,
            gb_normal=gb_normal,
            gb_area=gb_area,
            box=box,
            exclusion_buffer=gb_exclusion_buffer,
        )

    lam_bulk = 0.0
    lam_gb = 0.0

    # if (
    #     np.isfinite(V_matrix)
    #     and V_matrix > 0.0
    # ):
    #     if (
    #         np.isfinite(J_bulk_mean)
    #         and J_bulk_mean > 0.0
    #         and V_bulk > 0.0
    #     ):
    #         lam_bulk = (
    #             J_bulk_mean
    #             * V_bulk
    #             * dt
    #         )

    #     if (
    #         np.isfinite(J_gb_mean)
    #         and J_gb_mean > 0.0
    #         and V_gb > 0.0
    #     ):
    #         lam_gb = (
    #             J_gb_mean
    #             * V_gb
    #             * dt
    #             * phi_gb_free
    #         )

    if event_rate_bulk is None:
        raise ValueError(
            "event_rate_bulk must be supplied."
        )
    
    if event_rate_gb is None:
        raise ValueError(
            "event_rate_gb must be supplied."
        )
    
    if (
        not np.isfinite(event_rate_bulk)
        or event_rate_bulk < 0.0
    ):
        raise ValueError(
            "event_rate_bulk must be finite and non-negative."
        )
    
    if (
        not np.isfinite(event_rate_gb)
        or event_rate_gb < 0.0
    ):
        raise ValueError(
            "event_rate_gb must be finite and non-negative."
        )
    
    if (
        not np.isfinite(lam_target)
        or lam_target <= 0.0
    ):
        raise ValueError(
            "lam_target must be finite and positive."
        )
    
    lam_bulk = (
        float(event_rate_bulk)
        * dt
    )
    
    lam_gb = (
        float(event_rate_gb)
        * dt
    )
    
    lam_tot = (
        lam_bulk
        + lam_gb
    )
    
    if lam_tot > lam_target * (1.0 + 1.0e-10):
        raise RuntimeError(
            "The supplied timestep exceeds the ternary "
            "nucleation limit: "
            f"lambda={lam_tot:.12e}, "
            f"lam_target={lam_target:.12e}, "
            f"dt={dt:.12e}."
        )
    
    # print(
    #     "NUCLEATION PARAMETERS: "
    #     f"lam_target={lam_target:.6e}, "
    #     f"event_rate_bulk={event_rate_bulk:.6e}, "
    #     f"event_rate_gb={event_rate_gb:.6e}, "
    #     f"lam_bulk={lam_bulk:.6e}, "
    #     f"lam_gb={lam_gb:.6e}, "
    #     f"lam_total={lam_tot:.6e}"
    # )

    # print(
    #     "NUC CONSISTENCY: "
    #     f"event_rate_bulk={event_rate_bulk:.6e}, "
    #     f"event_rate_gb={event_rate_gb:.6e}, "
    #     f"lambda_total={lam_tot:.6e}, "
    #     f"dt={dt:.6e}"
    # )
#----------------------------   
    candidate_sampling_alpha = float(candidate_sampling_alpha)
    if (
        not np.isfinite(candidate_sampling_alpha)
        or candidate_sampling_alpha < 1.0
    ):
        raise ValueError(
            "candidate_sampling_alpha must be finite and >= 1.0."
        )

    candidate_sampling_min_pool = int(candidate_sampling_min_pool)
    if candidate_sampling_min_pool < 1:
        raise ValueError(
            "candidate_sampling_min_pool must be >= 1."
        )

    candidate_sampling_ess_per_event = float(
        candidate_sampling_ess_per_event
    )
    if (
        not np.isfinite(candidate_sampling_ess_per_event)
        or candidate_sampling_ess_per_event < 1.0
    ):
        raise ValueError(
            "candidate_sampling_ess_per_event must be finite and >= 1.0."
        )

    candidate_sampling_min_ess = float(candidate_sampling_min_ess)
    if (
        not np.isfinite(candidate_sampling_min_ess)
        or candidate_sampling_min_ess < 1.0
    ):
        raise ValueError(
            "candidate_sampling_min_ess must be finite and >= 1.0."
        )

    candidate_sampling_max_alpha = float(candidate_sampling_max_alpha)
    if (
        not np.isfinite(candidate_sampling_max_alpha)
        or candidate_sampling_max_alpha < candidate_sampling_alpha
    ):
        raise ValueError(
            "candidate_sampling_max_alpha must be finite and >= "
            "candidate_sampling_alpha."
        )

    candidate_sampling_max_pool = int(candidate_sampling_max_pool)
    if candidate_sampling_max_pool < candidate_sampling_min_pool:
        raise ValueError(
            "candidate_sampling_max_pool must be >= "
            "candidate_sampling_min_pool."
        )

    spatial_rate_correction = bool(spatial_rate_correction)
    spatial_rate_correction_max_factor = float(
        spatial_rate_correction_max_factor
    )
    if (
        not np.isfinite(spatial_rate_correction_max_factor)
        or spatial_rate_correction_max_factor < 1.0
    ):
        raise ValueError(
            "spatial_rate_correction_max_factor must be finite and >= 1.0."
        )

    spatial_rate_sampling_rse_target = float(
        spatial_rate_sampling_rse_target
    )
    if (
        not np.isfinite(spatial_rate_sampling_rse_target)
        or spatial_rate_sampling_rse_target <= 0.0
    ):
        raise ValueError(
            "spatial_rate_sampling_rse_target must be finite and > 0.0."
        )

    spatial_rate_sampling_min_samples = int(
        spatial_rate_sampling_min_samples
    )
    spatial_rate_sampling_max_samples = int(
        spatial_rate_sampling_max_samples
    )
    if spatial_rate_sampling_min_samples < 2:
        raise ValueError(
            "spatial_rate_sampling_min_samples must be >= 2."
        )
    if spatial_rate_sampling_max_samples < spatial_rate_sampling_min_samples:
        raise ValueError(
            "spatial_rate_sampling_max_samples must be >= "
            "spatial_rate_sampling_min_samples."
        )

    spatial_rate_probe_hazard = float(spatial_rate_probe_hazard)
    if (
        not np.isfinite(spatial_rate_probe_hazard)
        or spatial_rate_probe_hazard <= 0.0
    ):
        raise ValueError(
            "spatial_rate_probe_hazard must be finite and > 0.0."
        )

    spatial_rate_skipped_hazard_bulk = float(
        spatial_rate_skipped_hazard_bulk
    )
    spatial_rate_skipped_hazard_gb = float(
        spatial_rate_skipped_hazard_gb
    )
    spatial_rate_last_factor_bulk = float(spatial_rate_last_factor_bulk)
    spatial_rate_last_factor_gb = float(spatial_rate_last_factor_gb)
    if (
        not np.isfinite(spatial_rate_skipped_hazard_bulk)
        or spatial_rate_skipped_hazard_bulk < 0.0
        or not np.isfinite(spatial_rate_skipped_hazard_gb)
        or spatial_rate_skipped_hazard_gb < 0.0
        or not np.isfinite(spatial_rate_last_factor_bulk)
        or spatial_rate_last_factor_bulk <= 0.0
        or not np.isfinite(spatial_rate_last_factor_gb)
        or spatial_rate_last_factor_gb <= 0.0
    ):
        raise ValueError(
            "spatial-rate skipped hazards must be finite and >= 0.0, "
            "and last spatial-rate factors must be finite and > 0.0."
        )

    # SPATIAL-RATE-CORRECTED CANDIDATE-POOL SAMPLING.
    #
    # First draw the reference Poisson event counts.  The adaptive candidate
    # pool then estimates r = <J_loc>/J_ref over geometrically admissible
    # locations.  The reference draw is preserved exactly when r=1.  For r>1,
    # add an independent Poisson increment with mean (r-1)*lambda_ref; for r<1,
    # thin the reference draw binomially with probability r.  This allows the
    # sampled spatial CNT rate to modify both the global event count and the
    # spatial distribution of nucleation events.
    rate_diagnostic = _empty_nucleation_rate_diagnostic()

    Ntrial_bulk_ref = int(rng.poisson(lam_bulk))
    Ntrial_gb_ref = int(rng.poisson(lam_gb))

    Ntrial_bulk = 0
    Ntrial_gb = 0
    rate_factor_bulk = 1.0
    rate_factor_gb = 1.0
    rate_factor_bulk_raw = 1.0
    rate_factor_gb_raw = 1.0
    lambda_bulk_corrected = lam_bulk
    lambda_gb_corrected = lam_gb

    accepted_centers = []
    accepted_radii = []
    accepted_c_mg = []
    accepted_c_si = []
    accepted_J = []
    accepted_mask = []
    accepted_kind = []

    work_centers = centers.copy()
    work_radii = radii.copy()
    work_kind = nuc_kind.copy()

    Ncandidate_bulk = 0
    Ncandidate_gb = 0

    def sample_one_geometric_candidate(kind, RkBT_ref):
        """Draw one position that passes the cheap reference-radius screen."""
        max_geo_reject = 100

        for _ in range(max_geo_reject):
            if kind == 0:
                if gb_enabled:
                    trial_pt = sample_bulk_point(
                        box,
                        rng,
                        gb_point,
                        gb_normal,
                        gb_thickness,
                    )
                else:
                    trial_pt = sample_point_in_box(
                        box,
                        rng,
                    )
            else:
                trial_pt = sample_gb_point(
                    box,
                    rng,
                    gb_point,
                    gb_normal,
                    gb_thickness,
                )

            if trial_pt is None:
                return None

            if work_centers.shape[0] > 0:
                dvec = minimum_image_displacement(
                    work_centers - trial_pt,
                    box,
                )
                trial_dist = np.linalg.norm(dvec, axis=1)
                if np.any(
                    trial_dist
                    < (work_radii + RkBT_ref + safety_buffer)
                ):
                    continue

                if (
                    kind == 1
                    and gb_enabled
                    and gb_exclusion_buffer >= 0.0
                    and gb_candidate_overlaps_existing(
                        candidate_center=trial_pt,
                        candidate_radius=RkBT_ref,
                        centers=work_centers,
                        radii=work_radii,
                        nuc_kind=work_kind,
                        gb_point=gb_point,
                        box=box,
                        exclusion_buffer=gb_exclusion_buffer,
                    )
                ):
                    continue

            return trial_pt

        return None

    def evaluate_candidate(kind, pt, J_ref, c_loc=None):
        """Evaluate CNT rate/radius using an optional precomputed composition."""
        if c_loc is None:
            pts_eval = np.asarray(pt, dtype=float).reshape(1, 3)
            if candidate_concentration_evaluator is None:
                c_loc = concentrations_at_points_species(
                    pts_eval,
                    centers,
                    np.vstack((Q_mg, Q_si)),
                    np.asarray((cbar_mg, cbar_si), dtype=float),
                    box,
                    alpha,
                    real_lattice,
                    kvecs,
                    k2,
                )[0]
            else:
                c_eval = np.asarray(
                    candidate_concentration_evaluator(pts_eval),
                    dtype=float,
                )
                if c_eval.shape != (1, 2):
                    raise ValueError(
                        "candidate_concentration_evaluator must return "
                        f"shape (M,2); got {c_eval.shape}."
                    )
                c_loc = c_eval[0]
        c_loc_mg = float(c_loc[0])
        c_loc_si = float(c_loc[1])

        if (
            not np.isfinite(c_loc_mg)
            or not np.isfinite(c_loc_si)
            or c_loc_mg <= 0.0
            or c_loc_si <= 0.0
        ):
            return None

        barrier_factor_loc = 1.0 if kind == 0 else gb_barrier_factor
        (
            _,
            RkBT_loc,
            _Rstar_loc,
            J_loc,
        ) = nucleation_step_ternary(
            dt=dt,
            T=T,
            t=t,
            C0Mg=c_loc_mg,
            C0Si=c_loc_si,
            aa=aa,
            bb=bb,
            cc=cc,
            vatm=vatm,
            vatp=vatp,
            gamma=gamma,
            N0=N0,
            DMg=D_mg,
            DSi=D_si,
            a_latt=a_latt,
            x=x,
            y=y,
            barrier_factor=barrier_factor_loc,
            apply_incubation_correction=apply_incubation_correction,
        )

        _accumulate_nucleation_rate_diagnostic(
            rate_diagnostic,
            kind,
            J_loc,
            J_ref,
        )

        if not np.isfinite(J_loc) or J_loc <= 0.0:
            return None
        if not np.isfinite(RkBT_loc) or RkBT_loc <= 0.0:
            return None

        return {
            "pt": pt.copy(),
            "R": float(RkBT_loc),
            "J": float(J_loc),
            "c_mg": float(c_loc_mg),
            "c_si": float(c_loc_si),
        }

    def candidate_still_fits(kind, candidate):
        pt = candidate["pt"]
        Rloc = candidate["R"]

        if work_centers.shape[0] == 0:
            return True

        dvec = minimum_image_displacement(
            work_centers - pt,
            box,
        )
        trial_dist = np.linalg.norm(dvec, axis=1)
        if np.any(
            trial_dist
            < (work_radii + Rloc + safety_buffer)
        ):
            return False

        if (
            kind == 1
            and gb_enabled
            and gb_exclusion_buffer >= 0.0
            and gb_candidate_overlaps_existing(
                candidate_center=pt,
                candidate_radius=Rloc,
                centers=work_centers,
                radii=work_radii,
                nuc_kind=work_kind,
                gb_point=gb_point,
                box=box,
                exclusion_buffer=gb_exclusion_buffer,
            )
        ):
            return False

        return True

    def process_region(kind, n_events_ref, lambda_ref, skipped_hazard_in, last_rate_factor_in):
        nonlocal work_centers
        nonlocal work_radii
        nonlocal work_kind
        nonlocal Ncandidate_bulk
        nonlocal Ncandidate_gb

        if kind == 0:
            J_ref = J_ref_bulk
            RkBT_ref = RkBT_ref_bulk
        else:
            J_ref = J_ref_gb
            RkBT_ref = RkBT_ref_gb

        if (
            lambda_ref <= 0.0
            or J_ref is None
            or not np.isfinite(J_ref)
            or J_ref <= 0.0
            or RkBT_ref is None
            or not np.isfinite(RkBT_ref)
            or RkBT_ref <= 0.0
        ):
            return 0, 1.0, 1.0, float(lambda_ref), 0.0, np.nan, True, float(last_rate_factor_in), 0, 0, np.nan

        # With the spatial-rate correction disabled, the mean-field Poisson
        # draw completely determines the event count.  If it drew zero events
        # there is nothing to place, so return without evaluating any local
        # concentration fields.
        if not spatial_rate_correction and int(n_events_ref) <= 0:
            return 0, 1.0, 1.0, float(lambda_ref), 0.0, np.nan, True, float(last_rate_factor_in), 0, 0, np.nan

        # Late in nucleation, avoid probing the spatial field on every tiny
        # timestep.  Accumulate an estimate of the omitted *spatially corrected*
        # Poisson hazard using the most recently resolved spatial-rate factor.
        # Probe again once that estimated omitted hazard reaches the requested
        # threshold.  Any actual reference event also forces a probe because a
        # candidate pool is required for placement.
        corrected_hazard_increment = float(last_rate_factor_in) * float(lambda_ref)
        skipped_hazard_trial = float(skipped_hazard_in) + corrected_hazard_increment
        do_rate_probe = bool(
            spatial_rate_correction
            and (
                skipped_hazard_trial >= spatial_rate_probe_hazard
                or int(n_events_ref) > 0
            )
        )
        if spatial_rate_correction and not do_rate_probe:
            return (
                0,
                1.0,
                1.0,
                float(lambda_ref),
                skipped_hazard_trial,
                np.nan,
                True,
                float(last_rate_factor_in),
                0,
                0,
                np.nan,
            )

        # A completed probe resets the skipped-hazard accumulator.  When the
        # correction is disabled the accumulator is intentionally unused.
        skipped_hazard_out = 0.0

        pilot_events = max(1, int(n_events_ref))
        diagnostic_region = rate_diagnostic["bulk" if kind == 0 else "gb"]
        diagnostic_n_start = int(diagnostic_region["n_rate_evaluated"])
        diagnostic_J_sum_start = float(diagnostic_region["sum_Jloc"])
        diagnostic_J_sq_sum_start = float(diagnostic_region["sum_Jloc_sq"])
        pool = []
        max_pool_rounds = 40
        pool_round = 0

        # TEMP profiling only: split candidate work into rate-estimator and
        # placement-pool stages.  This is diagnostic bookkeeping and does not
        # alter any RNG draws, candidate counts, or acceptance decisions.
        timing = {
            "rate_sample": 0.0,
            "placement_sample": 0.0,
            "geometry_sample": 0.0,
            "candidate_eval": 0.0,
            "selection_fit": 0.0,
            "rate_candidate_evals": 0,
            "placement_candidate_evals": 0,
        }
        sampling_stage = "rate" if do_rate_probe else "placement"

        def sample_candidates(n_to_sample):
            nonlocal pool_round
            nonlocal Ncandidate_bulk
            nonlocal Ncandidate_gb
            nonlocal sampling_stage
            t_batch0 = time.perf_counter()
            batch_points = []
            for _ in range(max(0, int(n_to_sample))):
                t0 = time.perf_counter()
                pt = sample_one_geometric_candidate(kind, RkBT_ref)
                timing["geometry_sample"] += time.perf_counter() - t0
                if pt is None:
                    continue
                if kind == 0:
                    Ncandidate_bulk += 1
                else:
                    Ncandidate_gb += 1
                batch_points.append(pt)

            if batch_points:
                batch_points_array = np.asarray(batch_points, dtype=float)
                t0 = time.perf_counter()
                if candidate_concentration_evaluator is None:
                    batch_c = concentrations_at_points_species(
                        batch_points_array,
                        centers,
                        np.vstack((Q_mg, Q_si)),
                        np.asarray((cbar_mg, cbar_si), dtype=float),
                        box,
                        alpha,
                        real_lattice,
                        kvecs,
                        k2,
                    )
                else:
                    batch_c = np.asarray(
                        candidate_concentration_evaluator(batch_points_array),
                        dtype=float,
                    )
                    if batch_c.shape != (len(batch_points_array), 2):
                        raise ValueError(
                            "candidate_concentration_evaluator must return "
                            f"shape (M,2); got {batch_c.shape}."
                        )
                for pt, c_loc in zip(batch_points, batch_c):
                    candidate = evaluate_candidate(kind, pt, J_ref, c_loc=c_loc)
                    if candidate is not None:
                        pool.append(candidate)
                timing["candidate_eval"] += time.perf_counter() - t0
                n_eval = len(batch_points)
                if sampling_stage == "rate":
                    timing["rate_candidate_evals"] += n_eval
                else:
                    timing["placement_candidate_evals"] += n_eval
            timing[
                "rate_sample" if sampling_stage == "rate" else "placement_sample"
            ] += time.perf_counter() - t_batch0
            pool_round += 1

        def rate_sample_statistics():
            n = int(diagnostic_region["n_rate_evaluated"]) - diagnostic_n_start
            sum_J = float(diagnostic_region["sum_Jloc"]) - diagnostic_J_sum_start
            sum_J2 = (
                float(diagnostic_region["sum_Jloc_sq"])
                - diagnostic_J_sq_sum_start
            )
            if n <= 0:
                return 0, float(J_ref), np.nan, False
            mean_J = sum_J / float(n)
            if n < 2:
                return n, mean_J, np.inf, False
            var_J = max(
                0.0,
                (sum_J2 - float(n) * mean_J * mean_J) / float(n - 1),
            )
            if mean_J > 0.0:
                rse = np.sqrt(var_J / float(n)) / mean_J
            elif var_J == 0.0:
                rse = 0.0
            else:
                rse = np.inf
            converged = bool(
                n >= spatial_rate_sampling_min_samples
                and np.isfinite(rse)
                and rse <= spatial_rate_sampling_rse_target
            )
            return n, mean_J, float(rse), converged

        # Initial pool.  For a rate probe, the minimum sample count is set by
        # the statistical estimator; for pure placement it is set by the ESS
        # candidate-pool controls.
        if do_rate_probe:
            initial_target = max(
                spatial_rate_sampling_min_samples,
                int(np.ceil(candidate_sampling_alpha * pilot_events)),
            )
        else:
            initial_target = max(
                pilot_events,
                int(np.ceil(candidate_sampling_alpha * pilot_events)),
                candidate_sampling_min_pool,
            )
        sample_candidates(initial_target)

        # Accuracy-controlled spatial-rate integration.  Keep drawing uniform
        # candidate points until the relative standard error of <J_local> is
        # below the requested tolerance, or until the internal safety cap is
        # reached.  These samples are retained for subsequent placement.
        rate_n, J_spatial_est, rate_rse, rate_converged = rate_sample_statistics()
        while (
            do_rate_probe
            and not rate_converged
            and rate_n < spatial_rate_sampling_max_samples
            and pool_round < max_pool_rounds
        ):
            n_to_sample = min(
                candidate_sampling_min_pool,
                spatial_rate_sampling_max_samples - rate_n,
            )
            if n_to_sample <= 0:
                break
            sample_candidates(n_to_sample)
            rate_n, J_spatial_est, rate_rse, rate_converged = (
                rate_sample_statistics()
            )

        if do_rate_probe:
            raw_factor = J_spatial_est / float(J_ref)
            if not np.isfinite(raw_factor) or raw_factor < 0.0:
                raw_factor = 1.0
            rate_factor = min(raw_factor, spatial_rate_correction_max_factor)
            last_rate_factor_out = float(rate_factor)
        else:
            raw_factor = 1.0
            rate_factor = 1.0
            rate_rse = np.nan
            rate_converged = True
            last_rate_factor_out = float(last_rate_factor_in)

        # Freeze the rate-estimator diagnostics here, before any additional
        # candidate draws that may be required solely to satisfy placement ESS.
        rate_sample_count = int(rate_n) if do_rate_probe else 0
        rate_ratio_estimate = float(raw_factor) if do_rate_probe else np.nan
        placement_ess_last = np.nan
        sampling_stage = "placement"

        lambda_corrected = float(lambda_ref) * rate_factor

        # Poisson thinning/superposition preserves the historical reference
        # event draw when rate_factor == 1 exactly.
        if rate_factor >= 1.0:
            extra_mean = float(lambda_ref) * (rate_factor - 1.0)
            n_events = int(n_events_ref)
            if extra_mean > 0.0:
                n_events += int(rng.poisson(extra_mean))
        else:
            n_events = int(rng.binomial(int(n_events_ref), rate_factor))

        if n_events <= 0:
            return (
                0,
                rate_factor,
                raw_factor,
                lambda_corrected,
                skipped_hazard_out,
                rate_rse,
                rate_converged,
                last_rate_factor_out,
                rate_sample_count,
                int(diagnostic_region["n_rate_evaluated"]) - diagnostic_n_start,
                placement_ess_last,
            )

        n_inserted = 0
        while n_inserted < n_events and pool_round < max_pool_rounds:
            remaining = n_events - n_inserted
            base_target = max(
                remaining,
                int(np.ceil(candidate_sampling_alpha * remaining)),
                candidate_sampling_min_pool,
            )
            max_target = max(
                candidate_sampling_max_pool,
                int(np.ceil(candidate_sampling_max_alpha * remaining)),
            )
            desired = max(
                candidate_sampling_ess_per_event * remaining,
                candidate_sampling_min_ess,
            )

            while len(pool) < base_target and pool_round < max_pool_rounds:
                sample_candidates(base_target - len(pool))

            while pool and pool_round < max_pool_rounds:
                weights = np.asarray([c["J"] for c in pool], dtype=float)
                weight_sum = float(np.sum(weights))
                weight_sq_sum = float(np.sum(weights * weights))
                if (
                    not np.isfinite(weight_sum)
                    or weight_sum <= 0.0
                    or not np.isfinite(weight_sq_sum)
                    or weight_sq_sum <= 0.0
                ):
                    pool = []
                    break
                ess = (weight_sum * weight_sum) / weight_sq_sum
                placement_ess_last = float(ess)
                if ess >= desired or len(pool) >= max_target:
                    break
                n_to_sample = min(
                    max_target - len(pool),
                    max(candidate_sampling_min_pool, len(pool)),
                )
                if n_to_sample <= 0:
                    break
                sample_candidates(n_to_sample)

            if not pool:
                break

            weights = np.asarray([c["J"] for c in pool], dtype=float)
            weight_sum = float(np.sum(weights))
            if not np.isfinite(weight_sum) or weight_sum <= 0.0:
                pool = []
                break

            t0 = time.perf_counter()
            selected_index = int(rng.choice(len(pool), p=weights / weight_sum))
            candidate = pool.pop(selected_index)
            fits = candidate_still_fits(kind, candidate)
            timing["selection_fit"] += time.perf_counter() - t0
            if not fits:
                continue

            accepted_centers.append(candidate["pt"])
            accepted_radii.append(candidate["R"])
            accepted_c_mg.append(candidate["c_mg"])
            accepted_c_si.append(candidate["c_si"])
            accepted_J.append(candidate["J"])
            accepted_kind.append(kind)
            accepted_mask.append(True)

            work_centers = np.vstack((work_centers, candidate["pt"][None, :]))
            work_radii = np.append(work_radii, candidate["R"])
            work_kind = np.append(work_kind, np.int32(kind))
            n_inserted += 1

        if n_inserted < n_events:
            accepted_mask.extend([False] * (n_events - n_inserted))

        diagnostic_region["timing_rate_sample"] = timing["rate_sample"]
        diagnostic_region["timing_placement_sample"] = timing["placement_sample"]
        diagnostic_region["timing_geometry_sample"] = timing["geometry_sample"]
        diagnostic_region["timing_candidate_eval"] = timing["candidate_eval"]
        diagnostic_region["timing_selection_fit"] = timing["selection_fit"]
        diagnostic_region["timing_rate_candidate_evals"] = timing["rate_candidate_evals"]
        diagnostic_region["timing_placement_candidate_evals"] = timing["placement_candidate_evals"]

        return (
            n_events,
            rate_factor,
            raw_factor,
            lambda_corrected,
            skipped_hazard_out,
            rate_rse,
            rate_converged,
            last_rate_factor_out,
            rate_sample_count,
            int(diagnostic_region["n_rate_evaluated"]) - diagnostic_n_start,
            placement_ess_last,
        )

    # Preserve the historical region ordering: bulk first, then GB.
    (
        Ntrial_bulk,
        rate_factor_bulk,
        rate_factor_bulk_raw,
        lambda_bulk_corrected,
        skipped_hazard_bulk_out,
        rate_rse_bulk,
        rate_converged_bulk,
        last_rate_factor_bulk_out,
        rate_n_bulk,
        placement_n_bulk,
        placement_ess_bulk,
    ) = process_region(
        kind=0,
        n_events_ref=Ntrial_bulk_ref,
        lambda_ref=lam_bulk,
        skipped_hazard_in=spatial_rate_skipped_hazard_bulk,
        last_rate_factor_in=spatial_rate_last_factor_bulk,
    )
    (
        Ntrial_gb,
        rate_factor_gb,
        rate_factor_gb_raw,
        lambda_gb_corrected,
        skipped_hazard_gb_out,
        rate_rse_gb,
        rate_converged_gb,
        last_rate_factor_gb_out,
        rate_n_gb,
        placement_n_gb,
        placement_ess_gb,
    ) = process_region(
        kind=1,
        n_events_ref=Ntrial_gb_ref,
        lambda_ref=lam_gb,
        skipped_hazard_in=spatial_rate_skipped_hazard_gb,
        last_rate_factor_in=spatial_rate_last_factor_gb,
    )
    Ntrial = Ntrial_bulk + Ntrial_gb

    return {
        "Ntrial": Ntrial,
        "Ntrial_bulk": Ntrial_bulk,
        "Ntrial_gb": Ntrial_gb,
        "Ncandidate": Ncandidate_bulk + Ncandidate_gb,
        "Ncandidate_bulk": Ncandidate_bulk,
        "Ncandidate_gb": Ncandidate_gb,
        "candidate_sampling_alpha": candidate_sampling_alpha,
        "candidate_sampling_min_pool": candidate_sampling_min_pool,
        "candidate_sampling_ess_per_event": candidate_sampling_ess_per_event,
        "candidate_sampling_min_ess": candidate_sampling_min_ess,
        "candidate_sampling_max_alpha": candidate_sampling_max_alpha,
        "candidate_sampling_max_pool": candidate_sampling_max_pool,
        "spatial_rate_correction": spatial_rate_correction,
        "spatial_rate_correction_max_factor": spatial_rate_correction_max_factor,
        "spatial_rate_sampling_rse_target": spatial_rate_sampling_rse_target,
        "spatial_rate_sampling_min_samples": spatial_rate_sampling_min_samples,
        "spatial_rate_sampling_max_samples": spatial_rate_sampling_max_samples,
        "spatial_rate_probe_hazard": spatial_rate_probe_hazard,
        "spatial_rate_skipped_hazard_bulk": skipped_hazard_bulk_out,
        "spatial_rate_skipped_hazard_gb": skipped_hazard_gb_out,
        "spatial_rate_last_factor_bulk": last_rate_factor_bulk_out,
        "spatial_rate_last_factor_gb": last_rate_factor_gb_out,
        "spatial_rate_sampling_n_bulk": rate_n_bulk,
        "spatial_rate_sampling_n_gb": rate_n_gb,
        "spatial_rate_sampling_ratio_bulk": rate_factor_bulk_raw,
        "spatial_rate_sampling_ratio_gb": rate_factor_gb_raw,
        "spatial_placement_n_bulk": placement_n_bulk,
        "spatial_placement_n_gb": placement_n_gb,
        "spatial_placement_ess_bulk": placement_ess_bulk,
        "spatial_placement_ess_gb": placement_ess_gb,
        "spatial_rate_sampling_rse_bulk": rate_rse_bulk,
        "spatial_rate_sampling_rse_gb": rate_rse_gb,
        "spatial_rate_sampling_converged_bulk": rate_converged_bulk,
        "spatial_rate_sampling_converged_gb": rate_converged_gb,
        "spatial_rate_factor_bulk": rate_factor_bulk,
        "spatial_rate_factor_gb": rate_factor_gb,
        "spatial_rate_factor_bulk_raw": rate_factor_bulk_raw,
        "spatial_rate_factor_gb_raw": rate_factor_gb_raw,
        "lambda_bulk_corrected": lambda_bulk_corrected,
        "lambda_gb_corrected": lambda_gb_corrected,
        "lambda_mean_corrected": (
            lambda_bulk_corrected + lambda_gb_corrected
        ),
        "n_accepted": len(
            accepted_centers
        ),
        "new_centers": np.asarray(
            accepted_centers,
            dtype=float,
        ).reshape(-1, 3),
        "new_radii": np.asarray(
            accepted_radii,
            dtype=float,
        ),
        "c_local_mg": np.asarray(
            accepted_c_mg,
            dtype=float,
        ),
        "c_local_si": np.asarray(
            accepted_c_si,
            dtype=float,
        ),
        "J_local": np.asarray(
            accepted_J,
            dtype=float,
        ),
        "new_kind": np.asarray(
            accepted_kind,
            dtype=np.int32,
        ),
        "accepted_mask": np.asarray(
            accepted_mask,
            dtype=bool,
        ),
        "lambda_bulk": lam_bulk,
        "lambda_gb": lam_gb,
        "lambda_mean": lam_tot,
        "dt_used": dt,
        "rate_diagnostic": rate_diagnostic,
    }


def nucleate_particles_cnt_monopole_species(
    centers,
    radii,
    Q,
    box,
    dt,
    t,
    T,
    cbar,
    aa,
    bb,
    cc,
    vatm,
    vatp,
    gamma,
    N0,
    D,
    a_latt,
    alpha,
    real_lattice,
    kvecs,
    k2,
    stoichiometric_weights,
    *,
    solvent_stoichiometric_weight=0.0,
    lam_target=1.0,
    apply_incubation_correction=True,
    matrix_volume=None,
    safety_buffer=0.0,
    rng=None,
    gb_point=None,
    gb_normal=None,
    gb_thickness=0.0,
    gb_mode="slab",
    gb_barrier_factor=1.0,
    nuc_kind=None,
    gb_exclusion_buffer=0.0,
    gb_use_area_fraction=True,
    cbar_gb=None,
    event_rate_bulk=None,
    event_rate_gb=None,
    J_ref_bulk=None,
    J_ref_gb=None,
    RkBT_ref_bulk=None,
    RkBT_ref_gb=None,
    candidate_sampling_alpha=2.0,
    candidate_sampling_min_pool=20,
    candidate_sampling_ess_per_event=1.5,
    candidate_sampling_min_ess=10.0,
    candidate_sampling_max_alpha=8.0,
    candidate_sampling_max_pool=80,
    spatial_rate_correction=True,
    spatial_rate_correction_max_factor=5.0,
    spatial_rate_sampling_rse_target=0.20,
    spatial_rate_sampling_min_samples=20,
    spatial_rate_sampling_max_samples=320,
    spatial_rate_probe_hazard=0.10,
    spatial_rate_skipped_hazard_bulk=0.0,
    spatial_rate_skipped_hazard_gb=0.0,
    spatial_rate_last_factor_bulk=1.0,
    spatial_rate_last_factor_gb=1.0,
    candidate_concentration_evaluator=None,
    diagnostics=False,
):
    """Species-indexed stochastic CNT nucleation interface.

    ``Q`` has shape ``(S, N)`` and ``cbar``, ``D``, and
    ``stoichiometric_weights`` have shape ``(S,)``.  The established S=2,
    solvent_stoichiometric_weight=0 compatibility branch is preserved, while
    the generic arbitrary-S path supports either the standard monopole field
    reconstruction or an externally supplied candidate concentration evaluator
    (for example the scale-separated SR/LR reconstruction).
    """
    centers = np.asarray(centers, dtype=float)
    radii = np.asarray(radii, dtype=float)
    Q = np.asarray(Q, dtype=float)
    cbar = np.asarray(cbar, dtype=float)
    D = np.asarray(D, dtype=float)
    weights = np.asarray(stoichiometric_weights, dtype=float)

    if cbar.ndim != 1 or cbar.size == 0:
        raise ValueError("cbar must have shape (S,) with S >= 1.")
    n_solutes = cbar.size

    if D.shape != (n_solutes,):
        raise ValueError(
            "D must have shape (S,) matching cbar; "
            f"got D.shape={D.shape}, cbar.shape={cbar.shape}."
        )
    if weights.shape != (n_solutes,):
        raise ValueError(
            "stoichiometric_weights must have shape (S,) matching cbar; "
            f"got {weights.shape}, expected {(n_solutes,)}."
        )
    if Q.ndim != 2 or Q.shape[0] != n_solutes:
        raise ValueError(
            "Q must have shape (S, N) with the same species axis as cbar; "
            f"got Q.shape={Q.shape}, cbar.shape={cbar.shape}."
        )
    if Q.shape[1] != len(radii):
        raise ValueError(
            "The particle axis of Q must have the same length as radii."
        )
    if np.any(~np.isfinite(Q)):
        raise ValueError("Q contains non-finite values.")
    if np.any(~np.isfinite(cbar)) or np.any(cbar < 0.0):
        raise ValueError("cbar must contain finite, non-negative values.")
    if np.any(~np.isfinite(D)) or np.any(D <= 0.0):
        raise ValueError("D must contain finite, positive values.")
    if np.any(~np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError(
            "stoichiometric_weights must contain finite, positive values."
        )

    solvent_weight = float(solvent_stoichiometric_weight)
    if not np.isfinite(solvent_weight) or solvent_weight < 0.0:
        raise ValueError(
            "solvent_stoichiometric_weight must be finite and non-negative."
        )

    if cbar_gb is None:
        cbar_gb_arr = None
    else:
        cbar_gb_arr = np.asarray(cbar_gb, dtype=float)
        if cbar_gb_arr.shape != (n_solutes,):
            raise ValueError(
                "cbar_gb must have shape (S,) matching cbar; "
                f"got {cbar_gb_arr.shape}, expected {(n_solutes,)}."
            )
        if np.any(~np.isfinite(cbar_gb_arr)) or np.any(cbar_gb_arr < 0.0):
            raise ValueError(
                "cbar_gb must contain finite, non-negative values."
            )

    # Exact regression branch.  Do not replace this call with a reimplementation:
    # preserving the original RNG consumption/order is required for regression
    # compatibility.
    if n_solutes == 2 and solvent_weight == 0.0:
        if not np.allclose(weights, np.rint(weights), rtol=0.0, atol=0.0):
            raise ValueError(
                "The legacy two-solute stochastic CNT kernel requires integer "
                "stoichiometric coefficients."
            )

        out = nucleate_particles_cnt_monopole_ternary(
            centers=centers,
            radii=radii,
            Q_mg=Q[0],
            Q_si=Q[1],
            box=box,
            dt=dt,
            t=t,
            T=T,
            cbar_mg=float(cbar[0]),
            cbar_si=float(cbar[1]),
            aa=aa,
            bb=bb,
            cc=cc,
            vatm=vatm,
            vatp=vatp,
            gamma=gamma,
            N0=N0,
            D_mg=float(D[0]),
            D_si=float(D[1]),
            a_latt=a_latt,
            alpha=alpha,
            real_lattice=real_lattice,
            kvecs=kvecs,
            k2=k2,
            x=int(weights[0]),
            y=int(weights[1]),
            lam_target=lam_target,
            apply_incubation_correction=apply_incubation_correction,
            matrix_volume=matrix_volume,
            safety_buffer=safety_buffer,
            rng=rng,
            gb_point=gb_point,
            gb_normal=gb_normal,
            gb_thickness=gb_thickness,
            gb_mode=gb_mode,
            gb_barrier_factor=gb_barrier_factor,
            nuc_kind=nuc_kind,
            gb_exclusion_buffer=gb_exclusion_buffer,
            gb_use_area_fraction=gb_use_area_fraction,
            cbar_gb_mg=(
                None if cbar_gb_arr is None else float(cbar_gb_arr[0])
            ),
            cbar_gb_si=(
                None if cbar_gb_arr is None else float(cbar_gb_arr[1])
            ),
            event_rate_bulk=event_rate_bulk,
            event_rate_gb=event_rate_gb,
            J_ref_bulk=J_ref_bulk,
            J_ref_gb=J_ref_gb,
            RkBT_ref_bulk=RkBT_ref_bulk,
            RkBT_ref_gb=RkBT_ref_gb,
            candidate_sampling_alpha=candidate_sampling_alpha,
            candidate_sampling_min_pool=candidate_sampling_min_pool,
            candidate_sampling_ess_per_event=candidate_sampling_ess_per_event,
            candidate_sampling_min_ess=candidate_sampling_min_ess,
            candidate_sampling_max_alpha=candidate_sampling_max_alpha,
            candidate_sampling_max_pool=candidate_sampling_max_pool,
            spatial_rate_correction=spatial_rate_correction,
            spatial_rate_correction_max_factor=(
                spatial_rate_correction_max_factor
            ),
            spatial_rate_sampling_rse_target=(
                spatial_rate_sampling_rse_target
            ),
            spatial_rate_sampling_min_samples=(
                spatial_rate_sampling_min_samples
            ),
            spatial_rate_sampling_max_samples=(
                spatial_rate_sampling_max_samples
            ),
            spatial_rate_probe_hazard=spatial_rate_probe_hazard,
            spatial_rate_skipped_hazard_bulk=(
                spatial_rate_skipped_hazard_bulk
            ),
            spatial_rate_skipped_hazard_gb=(
                spatial_rate_skipped_hazard_gb
            ),
            spatial_rate_last_factor_bulk=spatial_rate_last_factor_bulk,
            spatial_rate_last_factor_gb=spatial_rate_last_factor_gb,
            candidate_concentration_evaluator=candidate_concentration_evaluator,
            diagnostics=diagnostics,
        )

        out = dict(out)
        out["c_local"] = np.vstack(
            (
                np.asarray(out["c_local_mg"], dtype=float),
                np.asarray(out["c_local_si"], dtype=float),
            )
        )
        return out

    # ------------------------------------------------------------------
    # Generic arbitrary-S stochastic proposal/acceptance path.
    # ------------------------------------------------------------------
    if rng is None:
        rng = np.random.default_rng()

    if centers.size == 0:
        centers = np.zeros((0, 3), dtype=float)
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError("centers must have shape (N, 3).")

    n_existing = len(radii)
    if centers.shape[0] != n_existing:
        raise ValueError("centers and radii must have the same length.")

    if nuc_kind is None:
        nuc_kind = np.zeros(n_existing, dtype=np.int32)
    else:
        nuc_kind = np.asarray(nuc_kind, dtype=np.int32).copy()
        if nuc_kind.shape != (n_existing,):
            raise ValueError("nuc_kind must have the same length as radii.")

    box = as_box_lengths(box)
    Lx, Ly, Lz = box
    V_box = Lx * Ly * Lz

    if matrix_volume is None:
        V_matrix = V_box - np.sum(particle_volumes(radii))
    else:
        V_matrix = float(matrix_volume)

    gb_enabled = (
        gb_point is not None
        and gb_normal is not None
        and gb_thickness is not None
        and gb_thickness > 0.0
    )

    if gb_enabled:
        if cbar_gb_arr is None:
            raise ValueError(
                "cbar_gb must be supplied when grain-boundary nucleation is enabled."
            )
        gb_normal = normalize_vector(gb_normal)
        V_gb_geom = Ly * Lz * gb_thickness
        V_gb = min(V_matrix, V_gb_geom)
        V_bulk = max(0.0, V_matrix - V_gb)
        gb_area = Ly * Lz
    else:
        V_gb = 0.0
        V_bulk = V_matrix
        gb_point = None
        gb_area = 0.0

    # Keep the same validation/proposal-rate contract as the established
    # ternary implementation.  The rates and reference CNT quantities are
    # supplied by nucleation_dt_limit_species(), so this routine only draws
    # proposals and performs local-composition rejection sampling.
    if event_rate_bulk is None:
        raise ValueError("event_rate_bulk must be supplied.")
    if event_rate_gb is None:
        raise ValueError("event_rate_gb must be supplied.")
    if not np.isfinite(event_rate_bulk) or event_rate_bulk < 0.0:
        raise ValueError("event_rate_bulk must be finite and non-negative.")
    if not np.isfinite(event_rate_gb) or event_rate_gb < 0.0:
        raise ValueError("event_rate_gb must be finite and non-negative.")
    if not np.isfinite(lam_target) or lam_target <= 0.0:
        raise ValueError("lam_target must be finite and positive.")

    lam_bulk = float(event_rate_bulk) * dt
    lam_gb = float(event_rate_gb) * dt
    lam_tot = lam_bulk + lam_gb

    if lam_tot > lam_target * (1.0 + 1.0e-10):
        raise RuntimeError(
            "The supplied timestep exceeds the species-indexed nucleation limit: "
            f"lambda={lam_tot:.12e}, lam_target={lam_target:.12e}, "
            f"dt={dt:.12e}."
        )

    candidate_sampling_alpha = float(candidate_sampling_alpha)
    if (
        not np.isfinite(candidate_sampling_alpha)
        or candidate_sampling_alpha < 1.0
    ):
        raise ValueError(
            "candidate_sampling_alpha must be finite and >= 1.0."
        )

    candidate_sampling_min_pool = int(candidate_sampling_min_pool)
    if candidate_sampling_min_pool < 1:
        raise ValueError(
            "candidate_sampling_min_pool must be >= 1."
        )

    candidate_sampling_ess_per_event = float(
        candidate_sampling_ess_per_event
    )
    if (
        not np.isfinite(candidate_sampling_ess_per_event)
        or candidate_sampling_ess_per_event < 1.0
    ):
        raise ValueError(
            "candidate_sampling_ess_per_event must be finite and >= 1.0."
        )

    candidate_sampling_min_ess = float(candidate_sampling_min_ess)
    if (
        not np.isfinite(candidate_sampling_min_ess)
        or candidate_sampling_min_ess < 1.0
    ):
        raise ValueError(
            "candidate_sampling_min_ess must be finite and >= 1.0."
        )

    candidate_sampling_max_alpha = float(candidate_sampling_max_alpha)
    if (
        not np.isfinite(candidate_sampling_max_alpha)
        or candidate_sampling_max_alpha < candidate_sampling_alpha
    ):
        raise ValueError(
            "candidate_sampling_max_alpha must be finite and >= "
            "candidate_sampling_alpha."
        )

    candidate_sampling_max_pool = int(candidate_sampling_max_pool)
    if candidate_sampling_max_pool < candidate_sampling_min_pool:
        raise ValueError(
            "candidate_sampling_max_pool must be >= "
            "candidate_sampling_min_pool."
        )

    spatial_rate_correction = bool(spatial_rate_correction)
    spatial_rate_correction_max_factor = float(
        spatial_rate_correction_max_factor
    )
    if (
        not np.isfinite(spatial_rate_correction_max_factor)
        or spatial_rate_correction_max_factor < 1.0
    ):
        raise ValueError(
            "spatial_rate_correction_max_factor must be finite and >= 1.0."
        )

    spatial_rate_sampling_rse_target = float(
        spatial_rate_sampling_rse_target
    )
    if (
        not np.isfinite(spatial_rate_sampling_rse_target)
        or spatial_rate_sampling_rse_target <= 0.0
    ):
        raise ValueError(
            "spatial_rate_sampling_rse_target must be finite and > 0.0."
        )

    spatial_rate_sampling_min_samples = int(
        spatial_rate_sampling_min_samples
    )
    spatial_rate_sampling_max_samples = int(
        spatial_rate_sampling_max_samples
    )
    if spatial_rate_sampling_min_samples < 2:
        raise ValueError(
            "spatial_rate_sampling_min_samples must be >= 2."
        )
    if spatial_rate_sampling_max_samples < spatial_rate_sampling_min_samples:
        raise ValueError(
            "spatial_rate_sampling_max_samples must be >= "
            "spatial_rate_sampling_min_samples."
        )

    spatial_rate_probe_hazard = float(spatial_rate_probe_hazard)
    if (
        not np.isfinite(spatial_rate_probe_hazard)
        or spatial_rate_probe_hazard <= 0.0
    ):
        raise ValueError(
            "spatial_rate_probe_hazard must be finite and > 0.0."
        )

    spatial_rate_skipped_hazard_bulk = float(
        spatial_rate_skipped_hazard_bulk
    )
    spatial_rate_skipped_hazard_gb = float(
        spatial_rate_skipped_hazard_gb
    )
    spatial_rate_last_factor_bulk = float(spatial_rate_last_factor_bulk)
    spatial_rate_last_factor_gb = float(spatial_rate_last_factor_gb)
    if (
        not np.isfinite(spatial_rate_skipped_hazard_bulk)
        or spatial_rate_skipped_hazard_bulk < 0.0
        or not np.isfinite(spatial_rate_skipped_hazard_gb)
        or spatial_rate_skipped_hazard_gb < 0.0
        or not np.isfinite(spatial_rate_last_factor_bulk)
        or spatial_rate_last_factor_bulk <= 0.0
        or not np.isfinite(spatial_rate_last_factor_gb)
        or spatial_rate_last_factor_gb <= 0.0
    ):
        raise ValueError(
            "spatial-rate skipped hazards must be finite and >= 0.0, "
            "and last spatial-rate factors must be finite and > 0.0."
        )

    # EXPERIMENTAL SPATIAL-RATE-CORRECTED CANDIDATE-POOL SAMPLING.
    rate_diagnostic = _empty_nucleation_rate_diagnostic()

    Ntrial_bulk_ref = int(rng.poisson(lam_bulk))
    Ntrial_gb_ref = int(rng.poisson(lam_gb))

    Ntrial_bulk = 0
    Ntrial_gb = 0
    rate_factor_bulk = 1.0
    rate_factor_gb = 1.0
    rate_factor_bulk_raw = 1.0
    rate_factor_gb_raw = 1.0
    lambda_bulk_corrected = lam_bulk
    lambda_gb_corrected = lam_gb

    accepted_centers = []
    accepted_radii = []
    accepted_c = []
    accepted_J = []
    accepted_mask = []
    accepted_kind = []

    work_centers = centers.copy()
    work_radii = radii.copy()
    work_kind = nuc_kind.copy()

    Ncandidate_bulk = 0
    Ncandidate_gb = 0

    def sample_one_geometric_candidate(kind, RkBT_ref):
        max_geo_reject = 100

        for _ in range(max_geo_reject):
            if kind == 0:
                if gb_enabled:
                    trial_pt = sample_bulk_point(
                        box, rng, gb_point, gb_normal, gb_thickness
                    )
                else:
                    trial_pt = sample_point_in_box(box, rng)
            else:
                trial_pt = sample_gb_point(
                    box, rng, gb_point, gb_normal, gb_thickness
                )

            if trial_pt is None:
                return None

            if work_centers.shape[0] > 0:
                dvec = minimum_image_displacement(
                    work_centers - trial_pt,
                    box,
                )
                trial_dist = np.linalg.norm(dvec, axis=1)
                if np.any(
                    trial_dist
                    < (work_radii + RkBT_ref + safety_buffer)
                ):
                    continue

                if (
                    kind == 1
                    and gb_enabled
                    and gb_exclusion_buffer >= 0.0
                    and gb_candidate_overlaps_existing(
                        candidate_center=trial_pt,
                        candidate_radius=RkBT_ref,
                        centers=work_centers,
                        radii=work_radii,
                        nuc_kind=work_kind,
                        gb_point=gb_point,
                        box=box,
                        exclusion_buffer=gb_exclusion_buffer,
                    )
                ):
                    continue

            return trial_pt

        return None

    def evaluate_candidate(kind, pt, J_ref, c_loc=None):
        if c_loc is None:
            point_array = np.asarray(pt, dtype=float).reshape(1, 3)
            if candidate_concentration_evaluator is None:
                c_loc = concentrations_at_points_species(
                    point_array,
                    centers,
                    Q,
                    cbar,
                    box,
                    alpha,
                    real_lattice,
                    kvecs,
                    k2,
                )[0]
            else:
                c_eval = np.asarray(
                    candidate_concentration_evaluator(point_array),
                    dtype=float,
                )
                if c_eval.shape != (1, n_solutes):
                    raise ValueError(
                        "candidate_concentration_evaluator must return "
                        f"shape (M,{n_solutes}); got {c_eval.shape}."
                    )
                c_loc = c_eval[0]
        else:
            c_loc = np.asarray(c_loc, dtype=float)

        if c_loc.shape != (n_solutes,):
            raise ValueError(
                f"Local candidate composition must have shape ({n_solutes},); "
                f"got {c_loc.shape}."
            )
        if np.any(~np.isfinite(c_loc)) or np.any(c_loc <= 0.0):
            return None

        barrier_factor_loc = 1.0 if kind == 0 else gb_barrier_factor
        (
            _,
            RkBT_loc,
            _Rstar_loc,
            J_loc,
        ) = nucleation_step_species(
            dt=dt,
            T=T,
            t=t,
            cbar=c_loc,
            aa=aa,
            bb=bb,
            cc=cc,
            vatm=vatm,
            vatp=vatp,
            gamma=gamma,
            N0=N0,
            D=D,
            a_latt=a_latt,
            stoichiometric_weights=weights,
            solvent_stoichiometric_weight=solvent_weight,
            barrier_factor=barrier_factor_loc,
            apply_incubation_correction=apply_incubation_correction,
        )

        _accumulate_nucleation_rate_diagnostic(
            rate_diagnostic,
            kind,
            J_loc,
            J_ref,
        )

        if not np.isfinite(J_loc) or J_loc <= 0.0:
            return None
        if not np.isfinite(RkBT_loc) or RkBT_loc <= 0.0:
            return None

        return {
            "pt": pt.copy(),
            "R": float(RkBT_loc),
            "J": float(J_loc),
            "c": c_loc.copy(),
        }

    def candidate_still_fits(kind, candidate):
        pt = candidate["pt"]
        Rloc = candidate["R"]

        if work_centers.shape[0] == 0:
            return True

        dvec = minimum_image_displacement(
            work_centers - pt,
            box,
        )
        trial_dist = np.linalg.norm(dvec, axis=1)
        if np.any(
            trial_dist
            < (work_radii + Rloc + safety_buffer)
        ):
            return False

        if (
            kind == 1
            and gb_enabled
            and gb_exclusion_buffer >= 0.0
            and gb_candidate_overlaps_existing(
                candidate_center=pt,
                candidate_radius=Rloc,
                centers=work_centers,
                radii=work_radii,
                nuc_kind=work_kind,
                gb_point=gb_point,
                box=box,
                exclusion_buffer=gb_exclusion_buffer,
            )
        ):
            return False

        return True

    def process_region(kind, n_events_ref, lambda_ref, skipped_hazard_in, last_rate_factor_in):
        nonlocal work_centers
        nonlocal work_radii
        nonlocal work_kind
        nonlocal Ncandidate_bulk
        nonlocal Ncandidate_gb

        if kind == 0:
            J_ref = J_ref_bulk
            RkBT_ref = RkBT_ref_bulk
        else:
            J_ref = J_ref_gb
            RkBT_ref = RkBT_ref_gb

        if (
            lambda_ref <= 0.0
            or J_ref is None
            or not np.isfinite(J_ref)
            or J_ref <= 0.0
            or RkBT_ref is None
            or not np.isfinite(RkBT_ref)
            or RkBT_ref <= 0.0
        ):
            return 0, 1.0, 1.0, float(lambda_ref), 0.0, np.nan, True, float(last_rate_factor_in), 0, 0, np.nan

        # With the spatial-rate correction disabled, the mean-field Poisson
        # draw completely determines the event count.  If it drew zero events
        # there is nothing to place, so return without evaluating any local
        # concentration fields.
        if not spatial_rate_correction and int(n_events_ref) <= 0:
            return 0, 1.0, 1.0, float(lambda_ref), 0.0, np.nan, True, float(last_rate_factor_in), 0, 0, np.nan

        # Late in nucleation, avoid probing the spatial field on every tiny
        # timestep.  Accumulate an estimate of the omitted *spatially corrected*
        # Poisson hazard using the most recently resolved spatial-rate factor.
        # Probe again once that estimated omitted hazard reaches the requested
        # threshold.  Any actual reference event also forces a probe because a
        # candidate pool is required for placement.
        corrected_hazard_increment = float(last_rate_factor_in) * float(lambda_ref)
        skipped_hazard_trial = float(skipped_hazard_in) + corrected_hazard_increment
        do_rate_probe = bool(
            spatial_rate_correction
            and (
                skipped_hazard_trial >= spatial_rate_probe_hazard
                or int(n_events_ref) > 0
            )
        )
        if spatial_rate_correction and not do_rate_probe:
            return (
                0,
                1.0,
                1.0,
                float(lambda_ref),
                skipped_hazard_trial,
                np.nan,
                True,
                float(last_rate_factor_in),
                0,
                0,
                np.nan,
            )

        # A completed probe resets the skipped-hazard accumulator.  When the
        # correction is disabled the accumulator is intentionally unused.
        skipped_hazard_out = 0.0

        pilot_events = max(1, int(n_events_ref))
        diagnostic_region = rate_diagnostic["bulk" if kind == 0 else "gb"]
        diagnostic_n_start = int(diagnostic_region["n_rate_evaluated"])
        diagnostic_J_sum_start = float(diagnostic_region["sum_Jloc"])
        diagnostic_J_sq_sum_start = float(diagnostic_region["sum_Jloc_sq"])
        pool = []
        max_pool_rounds = 40
        pool_round = 0

        # TEMP profiling only: split candidate work into rate-estimator and
        # placement-pool stages.  This is diagnostic bookkeeping and does not
        # alter any RNG draws, candidate counts, or acceptance decisions.
        timing = {
            "rate_sample": 0.0,
            "placement_sample": 0.0,
            "geometry_sample": 0.0,
            "candidate_eval": 0.0,
            "selection_fit": 0.0,
            "rate_candidate_evals": 0,
            "placement_candidate_evals": 0,
        }
        sampling_stage = "rate" if do_rate_probe else "placement"

        def sample_candidates(n_to_sample):
            nonlocal pool_round
            nonlocal Ncandidate_bulk
            nonlocal Ncandidate_gb
            nonlocal sampling_stage
            t_batch0 = time.perf_counter()
            batch_points = []
            for _ in range(max(0, int(n_to_sample))):
                t0 = time.perf_counter()
                pt = sample_one_geometric_candidate(kind, RkBT_ref)
                timing["geometry_sample"] += time.perf_counter() - t0
                if pt is None:
                    continue
                if kind == 0:
                    Ncandidate_bulk += 1
                else:
                    Ncandidate_gb += 1
                batch_points.append(pt)

            if batch_points:
                batch_points_array = np.asarray(batch_points, dtype=float)
                t0 = time.perf_counter()
                if candidate_concentration_evaluator is None:
                    batch_c = concentrations_at_points_species(
                        batch_points_array,
                        centers,
                        Q,
                        cbar,
                        box,
                        alpha,
                        real_lattice,
                        kvecs,
                        k2,
                    )
                else:
                    batch_c = np.asarray(
                        candidate_concentration_evaluator(batch_points_array),
                        dtype=float,
                    )
                    expected_shape = (len(batch_points_array), n_solutes)
                    if batch_c.shape != expected_shape:
                        raise ValueError(
                            "candidate_concentration_evaluator must return "
                            f"shape (M,{n_solutes}); got {batch_c.shape}."
                        )
                for pt, c_loc in zip(batch_points, batch_c):
                    candidate = evaluate_candidate(kind, pt, J_ref, c_loc=c_loc)
                    if candidate is not None:
                        pool.append(candidate)
                timing["candidate_eval"] += time.perf_counter() - t0
                n_eval = len(batch_points)
                if sampling_stage == "rate":
                    timing["rate_candidate_evals"] += n_eval
                else:
                    timing["placement_candidate_evals"] += n_eval
            timing[
                "rate_sample" if sampling_stage == "rate" else "placement_sample"
            ] += time.perf_counter() - t_batch0
            pool_round += 1

        def rate_sample_statistics():
            n = int(diagnostic_region["n_rate_evaluated"]) - diagnostic_n_start
            sum_J = float(diagnostic_region["sum_Jloc"]) - diagnostic_J_sum_start
            sum_J2 = (
                float(diagnostic_region["sum_Jloc_sq"])
                - diagnostic_J_sq_sum_start
            )
            if n <= 0:
                return 0, float(J_ref), np.nan, False
            mean_J = sum_J / float(n)
            if n < 2:
                return n, mean_J, np.inf, False
            var_J = max(
                0.0,
                (sum_J2 - float(n) * mean_J * mean_J) / float(n - 1),
            )
            if mean_J > 0.0:
                rse = np.sqrt(var_J / float(n)) / mean_J
            elif var_J == 0.0:
                rse = 0.0
            else:
                rse = np.inf
            converged = bool(
                n >= spatial_rate_sampling_min_samples
                and np.isfinite(rse)
                and rse <= spatial_rate_sampling_rse_target
            )
            return n, mean_J, float(rse), converged

        # Initial pool.  For a rate probe, the minimum sample count is set by
        # the statistical estimator; for pure placement it is set by the ESS
        # candidate-pool controls.
        if do_rate_probe:
            initial_target = max(
                spatial_rate_sampling_min_samples,
                int(np.ceil(candidate_sampling_alpha * pilot_events)),
            )
        else:
            initial_target = max(
                pilot_events,
                int(np.ceil(candidate_sampling_alpha * pilot_events)),
                candidate_sampling_min_pool,
            )
        sample_candidates(initial_target)

        # Accuracy-controlled spatial-rate integration.  Keep drawing uniform
        # candidate points until the relative standard error of <J_local> is
        # below the requested tolerance, or until the internal safety cap is
        # reached.  These samples are retained for subsequent placement.
        rate_n, J_spatial_est, rate_rse, rate_converged = rate_sample_statistics()
        while (
            do_rate_probe
            and not rate_converged
            and rate_n < spatial_rate_sampling_max_samples
            and pool_round < max_pool_rounds
        ):
            n_to_sample = min(
                candidate_sampling_min_pool,
                spatial_rate_sampling_max_samples - rate_n,
            )
            if n_to_sample <= 0:
                break
            sample_candidates(n_to_sample)
            rate_n, J_spatial_est, rate_rse, rate_converged = (
                rate_sample_statistics()
            )

        if do_rate_probe:
            raw_factor = J_spatial_est / float(J_ref)
            if not np.isfinite(raw_factor) or raw_factor < 0.0:
                raw_factor = 1.0
            rate_factor = min(raw_factor, spatial_rate_correction_max_factor)
            last_rate_factor_out = float(rate_factor)
        else:
            raw_factor = 1.0
            rate_factor = 1.0
            rate_rse = np.nan
            rate_converged = True
            last_rate_factor_out = float(last_rate_factor_in)

        # Freeze the rate-estimator diagnostics here, before any additional
        # candidate draws that may be required solely to satisfy placement ESS.
        rate_sample_count = int(rate_n) if do_rate_probe else 0
        rate_ratio_estimate = float(raw_factor) if do_rate_probe else np.nan
        placement_ess_last = np.nan
        sampling_stage = "placement"

        lambda_corrected = float(lambda_ref) * rate_factor

        # Poisson thinning/superposition preserves the historical reference
        # event draw when rate_factor == 1 exactly.
        if rate_factor >= 1.0:
            extra_mean = float(lambda_ref) * (rate_factor - 1.0)
            n_events = int(n_events_ref)
            if extra_mean > 0.0:
                n_events += int(rng.poisson(extra_mean))
        else:
            n_events = int(rng.binomial(int(n_events_ref), rate_factor))

        if n_events <= 0:
            return (
                0,
                rate_factor,
                raw_factor,
                lambda_corrected,
                skipped_hazard_out,
                rate_rse,
                rate_converged,
                last_rate_factor_out,
                rate_sample_count,
                int(diagnostic_region["n_rate_evaluated"]) - diagnostic_n_start,
                placement_ess_last,
            )

        n_inserted = 0
        while n_inserted < n_events and pool_round < max_pool_rounds:
            remaining = n_events - n_inserted
            base_target = max(
                remaining,
                int(np.ceil(candidate_sampling_alpha * remaining)),
                candidate_sampling_min_pool,
            )
            max_target = max(
                candidate_sampling_max_pool,
                int(np.ceil(candidate_sampling_max_alpha * remaining)),
            )
            desired = max(
                candidate_sampling_ess_per_event * remaining,
                candidate_sampling_min_ess,
            )

            while len(pool) < base_target and pool_round < max_pool_rounds:
                sample_candidates(base_target - len(pool))

            while pool and pool_round < max_pool_rounds:
                weights = np.asarray([c["J"] for c in pool], dtype=float)
                weight_sum = float(np.sum(weights))
                weight_sq_sum = float(np.sum(weights * weights))
                if (
                    not np.isfinite(weight_sum)
                    or weight_sum <= 0.0
                    or not np.isfinite(weight_sq_sum)
                    or weight_sq_sum <= 0.0
                ):
                    pool = []
                    break
                ess = (weight_sum * weight_sum) / weight_sq_sum
                placement_ess_last = float(ess)
                if ess >= desired or len(pool) >= max_target:
                    break
                n_to_sample = min(
                    max_target - len(pool),
                    max(candidate_sampling_min_pool, len(pool)),
                )
                if n_to_sample <= 0:
                    break
                sample_candidates(n_to_sample)

            if not pool:
                break

            weights = np.asarray([c["J"] for c in pool], dtype=float)
            weight_sum = float(np.sum(weights))
            if not np.isfinite(weight_sum) or weight_sum <= 0.0:
                pool = []
                break

            t0 = time.perf_counter()
            selected_index = int(rng.choice(len(pool), p=weights / weight_sum))
            candidate = pool.pop(selected_index)
            fits = candidate_still_fits(kind, candidate)
            timing["selection_fit"] += time.perf_counter() - t0
            if not fits:
                continue

            accepted_centers.append(candidate["pt"])
            accepted_radii.append(candidate["R"])
            accepted_c.append(candidate["c"])
            accepted_J.append(candidate["J"])
            accepted_kind.append(kind)
            accepted_mask.append(True)

            work_centers = np.vstack((work_centers, candidate["pt"][None, :]))
            work_radii = np.append(work_radii, candidate["R"])
            work_kind = np.append(work_kind, np.int32(kind))
            n_inserted += 1

        if n_inserted < n_events:
            accepted_mask.extend([False] * (n_events - n_inserted))

        diagnostic_region["timing_rate_sample"] = timing["rate_sample"]
        diagnostic_region["timing_placement_sample"] = timing["placement_sample"]
        diagnostic_region["timing_geometry_sample"] = timing["geometry_sample"]
        diagnostic_region["timing_candidate_eval"] = timing["candidate_eval"]
        diagnostic_region["timing_selection_fit"] = timing["selection_fit"]
        diagnostic_region["timing_rate_candidate_evals"] = timing["rate_candidate_evals"]
        diagnostic_region["timing_placement_candidate_evals"] = timing["placement_candidate_evals"]

        return (
            n_events,
            rate_factor,
            raw_factor,
            lambda_corrected,
            skipped_hazard_out,
            rate_rse,
            rate_converged,
            last_rate_factor_out,
            rate_sample_count,
            int(diagnostic_region["n_rate_evaluated"]) - diagnostic_n_start,
            placement_ess_last,
        )

    # Preserve the historical region ordering: bulk first, then GB.
    (
        Ntrial_bulk,
        rate_factor_bulk,
        rate_factor_bulk_raw,
        lambda_bulk_corrected,
        skipped_hazard_bulk_out,
        rate_rse_bulk,
        rate_converged_bulk,
        last_rate_factor_bulk_out,
        rate_n_bulk,
        placement_n_bulk,
        placement_ess_bulk,
    ) = process_region(
        kind=0,
        n_events_ref=Ntrial_bulk_ref,
        lambda_ref=lam_bulk,
        skipped_hazard_in=spatial_rate_skipped_hazard_bulk,
        last_rate_factor_in=spatial_rate_last_factor_bulk,
    )
    (
        Ntrial_gb,
        rate_factor_gb,
        rate_factor_gb_raw,
        lambda_gb_corrected,
        skipped_hazard_gb_out,
        rate_rse_gb,
        rate_converged_gb,
        last_rate_factor_gb_out,
        rate_n_gb,
        placement_n_gb,
        placement_ess_gb,
    ) = process_region(
        kind=1,
        n_events_ref=Ntrial_gb_ref,
        lambda_ref=lam_gb,
        skipped_hazard_in=spatial_rate_skipped_hazard_gb,
        last_rate_factor_in=spatial_rate_last_factor_gb,
    )
    Ntrial = Ntrial_bulk + Ntrial_gb

    if accepted_c:
        c_local = np.asarray(accepted_c, dtype=float).T
    else:
        c_local = np.zeros((n_solutes, 0), dtype=float)

    result = {
        "Ntrial": Ntrial,
        "Ntrial_bulk": Ntrial_bulk,
        "Ntrial_gb": Ntrial_gb,
        "Ncandidate": Ncandidate_bulk + Ncandidate_gb,
        "Ncandidate_bulk": Ncandidate_bulk,
        "Ncandidate_gb": Ncandidate_gb,
        "candidate_sampling_alpha": candidate_sampling_alpha,
        "candidate_sampling_min_pool": candidate_sampling_min_pool,
        "candidate_sampling_ess_per_event": candidate_sampling_ess_per_event,
        "candidate_sampling_min_ess": candidate_sampling_min_ess,
        "candidate_sampling_max_alpha": candidate_sampling_max_alpha,
        "candidate_sampling_max_pool": candidate_sampling_max_pool,
        "spatial_rate_correction": spatial_rate_correction,
        "spatial_rate_correction_max_factor": spatial_rate_correction_max_factor,
        "spatial_rate_sampling_rse_target": spatial_rate_sampling_rse_target,
        "spatial_rate_sampling_min_samples": spatial_rate_sampling_min_samples,
        "spatial_rate_sampling_max_samples": spatial_rate_sampling_max_samples,
        "spatial_rate_probe_hazard": spatial_rate_probe_hazard,
        "spatial_rate_skipped_hazard_bulk": skipped_hazard_bulk_out,
        "spatial_rate_skipped_hazard_gb": skipped_hazard_gb_out,
        "spatial_rate_last_factor_bulk": last_rate_factor_bulk_out,
        "spatial_rate_last_factor_gb": last_rate_factor_gb_out,
        "spatial_rate_sampling_n_bulk": rate_n_bulk,
        "spatial_rate_sampling_n_gb": rate_n_gb,
        "spatial_rate_sampling_ratio_bulk": rate_factor_bulk_raw,
        "spatial_rate_sampling_ratio_gb": rate_factor_gb_raw,
        "spatial_placement_n_bulk": placement_n_bulk,
        "spatial_placement_n_gb": placement_n_gb,
        "spatial_placement_ess_bulk": placement_ess_bulk,
        "spatial_placement_ess_gb": placement_ess_gb,
        "spatial_rate_sampling_rse_bulk": rate_rse_bulk,
        "spatial_rate_sampling_rse_gb": rate_rse_gb,
        "spatial_rate_sampling_converged_bulk": rate_converged_bulk,
        "spatial_rate_sampling_converged_gb": rate_converged_gb,
        "spatial_rate_factor_bulk": rate_factor_bulk,
        "spatial_rate_factor_gb": rate_factor_gb,
        "spatial_rate_factor_bulk_raw": rate_factor_bulk_raw,
        "spatial_rate_factor_gb_raw": rate_factor_gb_raw,
        "lambda_bulk_corrected": lambda_bulk_corrected,
        "lambda_gb_corrected": lambda_gb_corrected,
        "lambda_mean_corrected": (
            lambda_bulk_corrected + lambda_gb_corrected
        ),
        "n_accepted": len(accepted_centers),
        "new_centers": np.asarray(accepted_centers, dtype=float).reshape(-1, 3),
        "new_radii": np.asarray(accepted_radii, dtype=float),
        "c_local": c_local,
        "J_local": np.asarray(accepted_J, dtype=float),
        "new_kind": np.asarray(accepted_kind, dtype=np.int32),
        "accepted_mask": np.asarray(accepted_mask, dtype=bool),
        "lambda_bulk": lam_bulk,
        "lambda_gb": lam_gb,
        "lambda_mean": lam_tot,
        "dt_used": dt,
        "rate_diagnostic": rate_diagnostic,
    }

    # Temporary two-slot aliases for downstream compatibility.  The generic
    # c_local array is the source of truth.
    if n_solutes == 2:
        result["c_local_mg"] = c_local[0]
        result["c_local_si"] = c_local[1]

    return result

