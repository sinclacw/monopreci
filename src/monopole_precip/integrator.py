import csv
from contextlib import redirect_stdout
import numpy as np
import time
from .utilities import (
    as_box_lengths,
    total_solute_inventory,
    solute_conservation_error,
    particle_volumes,
    mean_matrix_concentration,
    evaluate_csurf,
    ternary_total_solute_inventory,
    ternary_mean_matrix_concentrations,
    green_matrix_inventory_integrals,
    total_solute_inventory_with_green_field,
)

from .greensfunction import (
    generate_real_lattice,
    generate_k_vectors,
    prepare_periodic_green_kernel,
    periodic_green_self_constant,
    build_pair_green_matrix,
    solve_monopoles_iterative,
    extend_pair_green_matrix,
)

from .nucleation_growth_subfunctions import (
    growth_rate,
    nucleation_dt_limit,
    nucleate_particles_cnt_monopole,
    c_surf_gibbs_thomson,
    nucleation_dt_limit_ternary,
    nucleation_dt_limit_species,
    nucleate_particles_cnt_monopole_ternary,
    nucleate_particles_cnt_monopole_species,
    estimate_gb_matrix_composition_species,
    estimate_gb_matrix_composition,
)

from .ternary_growth_coupled import (
    ParticlePhaseThermodynamics,
    solve_interface_coupled,
    solve_interface_with_continuation,
    initialize_ternary_interfaces_kwn,
    initialize_interfaces_kwn,
)

from .outputs_subfunctions import write_particles_ovito_dump
from .phases import SolubilityProductThermodynamics

from .hybrid_short_range import HybridShortRangeOperator
from .hybrid_long_range import PeriodicLongRangeField


# --------------------------------------------------------------------------
# Spatial nucleation-rate / candidate-pool safeguards.  The three accuracy
# controls below are supplied by run_configuration; these constants are
# implementation caps retained from the validated spatial-nucleation branch.
# --------------------------------------------------------------------------
NUCLEATION_CANDIDATE_SAMPLING_ALPHA = 2.0
NUCLEATION_CANDIDATE_MIN_POOL = 20
NUCLEATION_CANDIDATE_MIN_ESS = 10.0
NUCLEATION_CANDIDATE_MAX_ALPHA = 8.0
NUCLEATION_CANDIDATE_MAX_POOL = 80
NUCLEATION_SPATIAL_RATE_CORRECTION_MAX_FACTOR = 5.0
NUCLEATION_SPATIAL_RATE_MIN_SAMPLES = 20
NUCLEATION_SPATIAL_RATE_MAX_SAMPLES = 320
NUCLEATION_SPATIAL_RATE_PROBE_HAZARD = 0.10


#---- TERNARY CODE


def integrate_growth_ternary_iterative(
    R0,
    positions,
    box,
    solute_names,
    cp,
    cbar0,
    D,
    phases,
    t_end,
    dt,
    T,
    aa,
    bb,
    vatm,
    cnt_params,
    cc=0.0,
    use_xy_factor=True,
    alpha=None,
    screening_length=None,
    n_real=3,
    n_k=6,
    show_progress=True,
    enable_nucleation=True,
    check_conservation=True,
    conservation_tol=1e-10,
    gmres_rtol=1e-10,
    gmres_atol=0.0,
    gmres_maxiter=200,
    gmres_restart=50,
    use_preconditioner=True,
    nonlinear_xtol=1e-10,
    nonlinear_ftol=1e-10,
    nonlinear_gtol=1e-10,
    nonlinear_max_nfev=200,
    thermodynamic_tolerance=1e-8,
    velocity_mismatch_tolerance=1e-4,
    concentration_margin=1e-10,
    velocity_floor=0.1,
    stationary_radius_tolerance=1e-8,
    velocity_consistency_radius_tolerance=1e-6,
    remove_extinct=True,
    R_remove=1e-6,
    R_remove_buffer=0.0,
    adaptive_dt=False,
    dt_max=None,
    dt_min=1e-4,
    dt_safety=0.05,
    rng=None,
    print_diagnostics=False,
    print_every=100,
    save_every=500,
    dump_filename="particles_ternary.dump",
    dump_target_times=None,
    dump_time_offset=0.0,
    diagnostic_time_offset=0.0,
    dump_target_append=False,
    dump_target_print=False,
    dump_field_snapshot_callback=None,
    gb_point=None,
    gb_normal=None,
    gb_thickness=None,
    barrier_factor=1.0,
    cp_phase=None,
    Q0=None,
    c_surf0=None,
    dRdt0=None,
    dRdt_species0=None,
    phase_id0=None,
    nuc_kind0=None,
    t0=0.0,
    lam_target=1.0,
    lam_target_min=None,
    initial_lambda_step=0.20,
    minimum_lambda_step=1e-4,
    maximum_lambda_step=0.50,
    lambda_growth_factor=1.5,
    lambda_shrink_factor=0.5,
    max_continuation_attempts=100,
    continuation_verbose=False,
    M_total_input=None,
    hybrid_transport=False,
    hybrid_long_range_field=None,
    hybrid_matching_length=None,
    hybrid_grid_shape=(64, 64, 64),
    hybrid_short_cutoff=None,
    hybrid_forced_nucleus=None,
    profile_v11=False,
    solver_diagnostics_file=None,
    max_accepted_dt_growth_factor=2.0,
    post_retry_dt_growth_factor=1.0,
    effort_hold_nfev=18,
    effort_reduce_nfev=36,
    effort_reduce_dt_factor=0.5,
    enable_growth_surface_predictor=True,
    growth_surface_predictor_max_ratio=2.0,
    store_hybrid_long_range_history=False,
    return_hybrid_long_range_field_final=True,
    nucleation_spatial_rate_correction=True,
    nucleation_spatial_rate_rse_target=0.20,
    nucleation_placement_ess_per_event=1.5,
    nucleation_diagnostics_file=None,
    dt_last_accepted0=None,
    first_step_dt_max=None,
    previous_step_had_growth_retry0=False,
    previous_growth_nfev0=None,
):
    """
    Fully coupled ternary Mg-Si monopole growth integrator.

    At every particle configuration:

      1. Exact Mg and Si conservation determines the mean matrix
         concentrations.
      2. The coupled interfacial problem determines C_Mg^s, C_Si^s,
         Q_Mg, Q_Si, and the common particle growth rate.
      3. Particle radii are advanced explicitly.
      4. Extinct particles are removed from the active problem.
      5. Optional bulk and grain-boundary CNT nucleation is applied.
      6. The coupled state is re-solved after accepted nucleation events.

    The first coupled solve uses interaction continuation. Later solves
    first attempt a direct warm-started solve and fall back to continuation
    if the direct solve fails.

    The existing particle positions remain fixed. Newly nucleated particles
    are appended to the system and the pair Green matrix is extended.

    Returns
    -------
    t_hist : ndarray
        Saved times.

    R_hist : list of ndarray
        Full particle-radius vectors. Length may change after nucleation.

    Q_mg_hist, Q_si_hist : list of ndarray
        Full Mg and Si monopole vectors.

    cbar_mg_hist, cbar_si_hist : ndarray
        Mean matrix concentrations determined by exact conservation.

    G0 : float
        Periodic Green self constant.

    aux : dict
        Additional solver, conservation, nucleation, and geometry histories.

    """
    # Optional detailed coupled-solver diagnostics log.  When None, verbose
    # nonlinear/preconditioner diagnostics are suppressed.  When a path is
    # supplied they are appended there instead of printed to the terminal.
    if solver_diagnostics_file is not None:
        from pathlib import Path as _Path
        solver_diagnostics_file = _Path(solver_diagnostics_file)
        solver_diagnostics_file.parent.mkdir(parents=True, exist_ok=True)

    def _open_solver_diagnostics():
        if solver_diagnostics_file is None:
            return None
        return solver_diagnostics_file.open("a", buffering=1)

    # Optional write-only CSV diagnostics for spatial nucleation sampling.
    if nucleation_diagnostics_file is not None:
        from pathlib import Path as _Path
        nucleation_diagnostics_file = _Path(nucleation_diagnostics_file)
        nucleation_diagnostics_file.parent.mkdir(parents=True, exist_ok=True)

    def append_nucleation_diagnostic_rows(
        phase_output, J_bulk_ref, J_gb_ref, *,
        phase_index,
        c_bulk,
        c_gb=None,
        rate_bulk_ref=np.nan,
        rate_gb_ref=np.nan,
        gb_barrier_factor_used=np.nan,
    ):
        """Append species-generic spatial nucleation diagnostics."""
        if nucleation_diagnostics_file is None:
            return

        c_bulk = np.asarray(c_bulk, dtype=float)
        if c_bulk.shape != (n_solutes,):
            raise ValueError(
                f"c_bulk must have shape ({n_solutes},); got {c_bulk.shape}."
            )

        if c_gb is None:
            c_gb = np.full(n_solutes, np.nan, dtype=float)
        else:
            c_gb = np.asarray(c_gb, dtype=float)
            if c_gb.shape != (n_solutes,):
                raise ValueError(
                    f"c_gb must have shape ({n_solutes},); got {c_gb.shape}."
                )

        concentration_fields = [f"c_ref_{name}" for name in solute_names]
        fieldnames = [
            "time_s", "time_local_s", "time_global_s", "time_cnt_s",
            "step", "phase_id", "phase_name", "region",
            *concentration_fields,
            "J_reference",
            "event_rate_reference", "gb_barrier_factor_used",
            "lambda_reference", "lambda_corrected",
            "rate_factor", "rate_factor_raw", "rate_samples", "rate_rse",
            "rate_converged", "placement_samples", "placement_ess",
            "mean_Jloc_over_Jref_all", "fraction_Jloc_above_Jref_all",
            "max_Jloc_over_Jref_all", "old_clipped_rate_over_Jref_all",
            "old_clipped_over_spatial_all",
        ]
        diag = phase_output.get("rate_diagnostic", {})
        phase_name = getattr(phases[phase_index], "name", str(phase_index))
        rows = []
        for region, suffix, J_ref in (
            ("bulk", "bulk", J_bulk_ref),
            ("gb", "gb", J_gb_ref),
        ):
            rate_n = int(
                phase_output.get(f"spatial_rate_sampling_n_{suffix}", 0)
            )
            if rate_n <= 0:
                continue
            d = diag.get(region, {})
            n_all = int(d.get("n_rate_evaluated", 0))
            if n_all > 0:
                mean_ratio = (
                    float(d.get("sum_Jloc_over_Jref", 0.0)) / n_all
                )
                mean_clip = (
                    float(d.get("sum_clipped_Jloc_over_Jref", 0.0)) / n_all
                )
                frac_above = float(d.get("n_Jloc_gt_Jref", 0)) / n_all
                max_ratio = float(
                    d.get("max_Jloc_over_Jref", float("nan"))
                )
                clip_over_spatial = (
                    mean_clip / mean_ratio
                    if mean_ratio > 0.0
                    else float("nan")
                )
            else:
                mean_ratio = mean_clip = frac_above = max_ratio = (
                    clip_over_spatial
                ) = float("nan")

            cref = c_bulk if region == "bulk" else c_gb
            # ``time_cnt_s`` is the physical pre-step time actually used by
            # the CNT incubation correction.  This deliberately matches KWN,
            # which evaluates nucleation at the beginning of an accepted step.
            # ``time_local_s`` and ``time_global_s`` describe the accepted
            # end-of-step state represented by this diagnostic row.
            time_local_end = float(t + dt_use)
            time_global_end = float(diagnostic_time_offset + t + dt_use)
            time_cnt = float(diagnostic_time_offset + t)
            row = {
                "time_s": time_global_end,
                "time_local_s": time_local_end,
                "time_global_s": time_global_end,
                "time_cnt_s": time_cnt,
                "step": int(step_number),
                "phase_id": int(phase_index),
                "phase_name": phase_name,
                "region": region,
                "J_reference": float(J_ref),
                "event_rate_reference": float(
                    rate_bulk_ref if region == "bulk" else rate_gb_ref
                ),
                "gb_barrier_factor_used": float(
                    1.0 if region == "bulk"
                    else gb_barrier_factor_used
                ),
                "lambda_reference": float(
                    phase_output.get(
                        f"lambda_{suffix}", float("nan")
                    )
                ),
                "lambda_corrected": float(
                    phase_output.get(
                        f"lambda_{suffix}_corrected", float("nan")
                    )
                ),
                "rate_factor": float(
                    phase_output.get(
                        f"spatial_rate_factor_{suffix}", float("nan")
                    )
                ),
                "rate_factor_raw": float(
                    phase_output.get(
                        f"spatial_rate_factor_{suffix}_raw", float("nan")
                    )
                ),
                "rate_samples": rate_n,
                "rate_rse": float(
                    phase_output.get(
                        f"spatial_rate_sampling_rse_{suffix}",
                        float("nan"),
                    )
                ),
                "rate_converged": int(
                    bool(
                        phase_output.get(
                            f"spatial_rate_sampling_converged_{suffix}",
                            False,
                        )
                    )
                ),
                "placement_samples": int(
                    phase_output.get(
                        f"spatial_placement_n_{suffix}", 0
                    )
                ),
                "placement_ess": float(
                    phase_output.get(
                        f"spatial_placement_ess_{suffix}", float("nan")
                    )
                ),
                "mean_Jloc_over_Jref_all": mean_ratio,
                "fraction_Jloc_above_Jref_all": frac_above,
                "max_Jloc_over_Jref_all": max_ratio,
                "old_clipped_rate_over_Jref_all": mean_clip,
                "old_clipped_over_spatial_all": clip_over_spatial,
            }
            for s, species_name in enumerate(solute_names):
                row[f"c_ref_{species_name}"] = float(cref[s])
            rows.append(row)

        if not rows:
            return
        write_header = (
            not nucleation_diagnostics_file.exists()
            or nucleation_diagnostics_file.stat().st_size == 0
        )
        with nucleation_diagnostics_file.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerows(rows)

    # ------------------------------------------------------------------
    # Basic validation and setup
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Generic solute state and temporary Mg-Si compatibility bridge
    # ------------------------------------------------------------------
    solute_names = tuple(solute_names)
    n_solutes = len(solute_names)

    if n_solutes == 0:
        raise ValueError("At least one solute species is required.")

    if len(set(solute_names)) != n_solutes:
        raise ValueError("solute_names must contain unique species names.")

    cp = np.asarray(cp, dtype=float)
    cbar0 = np.asarray(cbar0, dtype=float)
    D = np.asarray(D, dtype=float)

    for name, values in (("cp", cp), ("cbar0", cbar0), ("D", D)):
        if values.shape != (n_solutes,):
            raise ValueError(
                f"{name} must have shape ({n_solutes},); got {values.shape}."
            )
        if np.any(~np.isfinite(values)):
            raise ValueError(f"{name} contains non-finite values.")

    if np.any(cp <= 0.0):
        raise ValueError("All precipitate solute concentrations in cp must be positive.")

    if cp_phase is None:
        cp_phase = cp[None, :].copy()
    else:
        cp_phase = np.asarray(cp_phase, dtype=float)
        if (
            cp_phase.ndim != 2
            or cp_phase.shape[0] == 0
            or cp_phase.shape[1] != n_solutes
        ):
            raise ValueError(
                "cp_phase must have shape (Nphases, Nspecies); "
                f"got {cp_phase.shape}."
            )
        if np.any(~np.isfinite(cp_phase)) or np.any(cp_phase <= 0.0):
            raise ValueError("cp_phase must contain finite positive concentrations.")
        if not np.array_equal(cp_phase[0], cp):
            raise ValueError(
                "For the current physical kernels, cp_phase[0] must exactly equal cp."
            )

    if np.any(D <= 0.0):
        raise ValueError("All solute diffusivities in D must be positive.")

    # The canonical integrator state is species-indexed.  Mg/Si names are
    # resolved only for the exact two-solute regression path; arbitrary-S
    # calculations must not manufacture a dummy second species.
    is_legacy_mg_si = (n_solutes == 2 and set(solute_names) == {"Mg", "Si"})
    if is_legacy_mg_si:
        mg_index = solute_names.index("Mg")
        si_index = solute_names.index("Si")
        cp_mg = float(cp[mg_index])
        cp_si = float(cp[si_index])
        cbar_mg0 = float(cbar0[mg_index])
        cbar_si0 = float(cbar0[si_index])
        D_mg = float(D[mg_index])
        D_si = float(D[si_index])
    else:
        mg_index = si_index = None
        cp_mg = cp_si = None
        cbar_mg0 = cbar_si0 = None
        D_mg = D_si = None

    phases = tuple(phases)
    if len(phases) == 0:
        raise ValueError("At least one precipitate phase is required.")

    # phase_id already identifies the phase of each particle.  The current
    # growth kernel remains restricted to phase 0, but its thermodynamics
    # now come from that phase object rather than from hard-coded equations.
    phase0 = phases[0]

    # Temporary compatibility scalars for the still Mg-Si-specific
    # isolated-interface and CNT kernels.  These are derived locally from
    # the authoritative phase definition rather than supplied separately
    # through the integrator API.
    phase0_vatp = float(phase0.atomic_volume)
    phase0_gamma = float(phase0.gamma)
    # Resolve the heterogeneous-nucleation barrier factor from the
    # authoritative phase definition.  The integrator-level barrier_factor
    # remains only the legacy fallback for phases that leave it unspecified.
    phase0_gb_barrier_factor = float(
        phase0.resolved_gb_barrier_factor(barrier_factor)
    )
    phase0_weights_species_order = (
        phase0.diffusing_stoichiometric_weights(solute_names)
    )
    # Legacy x/y and kernel-order weights exist only for the exact Mg-Si
    # regression path.  Generic CNT uses stoichiometric_weights plus the
    # separately computed solvent stoichiometric weight.
    if is_legacy_mg_si:
        phase0_x = int(phase0.stoichiometry["Mg"])
        phase0_y = int(phase0.stoichiometry["Si"])
        phase0_weights_kernel_order = np.asarray(
            (phase0_weights_species_order[mg_index],
             phase0_weights_species_order[si_index]), dtype=float
        )
    else:
        phase0_x = phase0_y = None
        phase0_weights_kernel_order = None
    phase0_weights_solver_order = np.asarray(
        phase0_weights_species_order,
        dtype=float,
    )

    # Numerical phase-property tables.  The coupled solver dispatches
    # thermodynamic evaluations by integer phase identity; phase names are
    # metadata only.
    phase_thermodynamic_models = tuple(
        phase.thermodynamics for phase in phases
    )
    phase_atomic_volumes = np.asarray(
        [phase.atomic_volume for phase in phases],
        dtype=float,
    )
    phase_gammas = np.asarray(
        [phase.gamma for phase in phases],
        dtype=float,
    )
    phase_weights_solver_order = np.vstack(
        [
            phase.diffusing_stoichiometric_weights(solute_names)
            for phase in phases
        ]
    ).astype(float, copy=False)

    def _coupled_result_in_species_order(result, values_name):
        """Return a generic coupled-result array reordered to solute_names."""
        if not hasattr(result, "solute_names"):
            raise RuntimeError(
                "Coupled result is missing solute_names metadata. "
                "All generic coupled/restart states must carry their species "
                "ordering explicitly."
            )
        result_names = tuple(result.solute_names)

        if len(set(result_names)) != len(result_names):
            raise RuntimeError(
                "Coupled solver returned duplicate species names."
            )

        try:
            row_order = [result_names.index(name) for name in solute_names]
        except ValueError as exc:
            raise RuntimeError(
                "Coupled solver species do not match integrator solute_names: "
                f"solver={result_names}, integrator={solute_names}."
            ) from exc

        values = np.asarray(getattr(result, values_name), dtype=float)

        if values_name == "cbar":
            if values.shape != (len(result_names),):
                raise RuntimeError(
                    f"Coupled solver returned cbar with shape {values.shape}."
                )
            return values[row_order]

        if values.ndim != 2 or values.shape[0] != len(result_names):
            raise RuntimeError(
                f"Coupled solver returned {values_name} with shape "
                f"{values.shape}."
            )

        return values[row_order, :]

    def _validate_species_particle_state(value, name, n_particles):
        if value is None:
            return None
        value = np.asarray(value, dtype=float)
        expected = (n_solutes, n_particles)
        if value.shape != expected:
            raise ValueError(
                f"{name} must have shape {expected}; got {value.shape}."
            )
        if np.any(~np.isfinite(value)):
            raise ValueError(f"{name} contains non-finite values.")
        return value.copy()

    box = as_box_lengths(box)
    box_volume = float(np.prod(box))

    cnt_params_ternary = dict(cnt_params)

    # Legacy ternary CNT requires the fcc matrix lattice parameter [nm].
    # Prefer an explicitly supplied value.  Current production/test callers
    # do not always carry a_latt in cnt_params, so for the Al fcc matrix use
    # the exact atomic-volume relation vatm = a^3 / 4 as the compatibility
    # fallback.  This gives ~0.405 nm for the present Al material definition.
    if "a_latt" in cnt_params_ternary:
        a_latt_cnt = float(cnt_params_ternary["a_latt"])
    else:
        a_latt_cnt = float((4.0 * float(vatm)) ** (1.0 / 3.0))

    if not np.isfinite(a_latt_cnt) or a_latt_cnt <= 0.0:
        raise ValueError(
            "CNT lattice parameter a_latt must be positive and finite."
        )

    # Keep the legacy CNT dictionary synchronized with the authoritative
    # phase-0 material definition.  Callers no longer provide independent
    # vatp/gamma/x/y values.
    cnt_params_ternary.update({
        "aa": float(aa), "bb": float(bb), "cc": float(cc),
        "vatm": float(vatm), "vatp": phase0_vatp,
        "gamma": phase0_gamma,
        "gb_barrier_factor": phase0_gb_barrier_factor,
        "D": D.copy(), "a_latt": a_latt_cnt,
        "stoichiometric_weights": np.asarray(phase0_weights_species_order, dtype=float).copy(),
    })
    if is_legacy_mg_si:
        cnt_params_ternary.update({
            "DMg": float(D_mg), "DSi": float(D_si),
            "x": phase0_x, "y": phase0_y,
        })

    # ------------------------------------------------------------------
    # Multiphase CNT parameters: one authoritative parameter record
    # per configured precipitate phase. Growth is phase-dispatched, and
    # nucleation uses the same phase definitions rather than phase 0 only.
    # ------------------------------------------------------------------
    phase_cnt_params = []
    for p, phase in enumerate(phases):
        thermo = phase.thermodynamics
        if not isinstance(thermo, SolubilityProductThermodynamics):
            raise TypeError(
                "Current two-solute CNT nucleation requires "
                "SolubilityProductThermodynamics for every precipitate phase; "
                f"phase {p} ({phase.name!r}) uses {type(thermo).__name__}."
            )

        weights_p = np.asarray(
            phase.diffusing_stoichiometric_weights(solute_names),
            dtype=float,
        )
        if weights_p.shape != (n_solutes,):
            raise ValueError(
                f"Phase {p} ({phase.name!r}) returned invalid diffusing "
                f"stoichiometric weights shape {weights_p.shape}."
            )

        solvent_weight_p = float(
            sum(
                coefficient
                for species_name, coefficient in phase.stoichiometry.items()
                if species_name not in solute_names
            )
        )

        params_p = dict(cnt_params)
        params_p.update({
            "aa": float(thermo.aa),
            "bb": float(thermo.bb),
            "cc": float(thermo.cc),
            "vatm": float(vatm),
            "vatp": float(phase.atomic_volume),
            "gamma": float(phase.gamma),
            "gb_barrier_factor": float(
                phase.resolved_gb_barrier_factor(barrier_factor)
            ),
            "D": np.asarray(D, dtype=float).copy(),
            "a_latt": a_latt_cnt,
            "stoichiometric_weights": weights_p.copy(),
            "solvent_stoichiometric_weight": solvent_weight_p,
        })
        if is_legacy_mg_si and solvent_weight_p == 0.0:
            params_p.update({
                "DMg": float(D[mg_index]),
                "DSi": float(D[si_index]),
                "x": int(round(float(weights_p[mg_index]))),
                "y": int(round(float(weights_p[si_index]))),
            })
        phase_cnt_params.append(params_p)

    
    positions = np.asarray(positions, dtype=float)
    R0 = np.asarray(R0, dtype=float)

    if R0.ndim != 1:
        raise ValueError("R0 must be one-dimensional.")

    if positions.shape != (len(R0), 3):
        raise ValueError(
            "positions must have shape (len(R0), 3)."
        )

    if np.any(~np.isfinite(R0)) or np.any(R0 < 0.0):
        raise ValueError(
            "R0 must contain finite non-negative radii."
        )

    if t_end < t0:
        raise ValueError("t_end must be no smaller than t0.")

    if dt <= 0.0:
        raise ValueError("dt must be positive.")

    if dt_min <= 0.0:
        raise ValueError("dt_min must be positive.")

    if dt_safety <= 0.0:
        raise ValueError("dt_safety must be positive.")

    # cp and D were validated species-wise above; no Mg/Si-only validation
    # belongs in the generic integrator core.

    if rng is None:
        rng = np.random.default_rng()

    # Persistent phase-local RNG streams. Phase 0 deliberately retains the
    # original stream so adding another phase does not perturb its trajectory.
    phase_nucleation_rng = [rng]
    for p in range(1, len(phases)):
        bitgen = rng.bit_generator
        if hasattr(bitgen, "jumped"):
            phase_nucleation_rng.append(np.random.Generator(bitgen.jumped(p)))
        else:
            import hashlib
            import pickle
            state_bytes = pickle.dumps(
                bitgen.state, protocol=pickle.HIGHEST_PROTOCOL
            )
            digest = hashlib.sha256(
                state_bytes + int(p).to_bytes(8, byteorder="little", signed=False)
            ).digest()
            seed_words = np.frombuffer(digest, dtype=np.uint32).copy()
            phase_nucleation_rng.append(
                np.random.default_rng(np.random.SeedSequence(seed_words))
            )

    if not isinstance(nucleation_spatial_rate_correction, (bool, np.bool_)):
        raise TypeError("nucleation_spatial_rate_correction must be boolean.")
    nucleation_spatial_rate_rse_target = float(nucleation_spatial_rate_rse_target)
    if (not np.isfinite(nucleation_spatial_rate_rse_target) or nucleation_spatial_rate_rse_target <= 0.0):
        raise ValueError("nucleation_spatial_rate_rse_target must be positive and finite.")
    nucleation_placement_ess_per_event = float(nucleation_placement_ess_per_event)
    if (not np.isfinite(nucleation_placement_ess_per_event) or nucleation_placement_ess_per_event <= 0.0):
        raise ValueError("nucleation_placement_ess_per_event must be positive and finite.")

    Ntot = len(R0)

    if phase_id0 is None:
        phase_id = np.zeros(Ntot, dtype=np.int32)
    else:
        phase_id = np.asarray(phase_id0, dtype=np.int32).copy()
        if phase_id.shape != (Ntot,):
            raise ValueError("phase_id0 and R0 must have the same length.")
        if np.any(phase_id < 0):
            raise ValueError("phase_id0 must contain non-negative integer phase IDs.")

    if np.any(phase_id >= len(phases)):
        raise ValueError(
            "phase_id0 contains an index outside the supplied phases tuple."
        )

    if nuc_kind0 is None:
        # 0 = bulk-origin particle
        # 1 = grain-boundary-origin particle
        nuc_kind = np.zeros(
            Ntot,
            dtype=np.int32,
        )
    else:
        nuc_kind = np.asarray(
            nuc_kind0,
            dtype=np.int32,
        ).copy()

        if nuc_kind.shape != (Ntot,):
            raise ValueError(
                "nuc_kind0 and R0 must have the same length."
            )

    if alpha is None:
        Lchar = box_volume ** (1.0 / 3.0)
        alpha = 6.0 / Lchar

    # -------------------------------------------------------------
    # Scale-separated transport
    #
    # Ordinary production extinction/removal is supported:
    # removed particles become zero-radius slots and their complete remaining
    # precipitate inventory is returned conservatively to the transient LR
    # matrix field.
    # -------------------------------------------------------------
    # One-entry cache for the hybrid short-range geometry.
    #
    # HybridShortRangeOperator depends only on particle positions, box,
    # alpha and cutoff.  Radii enter later through operator.factor(R), so
    # rebuilding the geometric neighbor structure on every coupled solve is
    # unnecessary.  The cache is keyed by the exact master-array indices
    # represented by the operator and is invalidated automatically whenever
    # topology/active membership changes.
    hybrid_short_operator = None
    hybrid_short_operator_idx = None
    hybrid_short_cache_hits = 0
    hybrid_short_cache_misses = 0

    forced_nucleus_inserted = False
    forced_nucleus_time = None
    forced_nucleus_position = None
    forced_nucleus_radius = None
    forced_nucleus_phase_id = 0

    if hybrid_transport:
        # Particle extinction is supported in the hybrid path.  Extinct
        # particles remain as zero-radius slots in the production arrays,
        # while the short-range operator is rebuilt on the surviving active
        # subset.  Their residual precipitate inventory is returned to the LR
        # field by the ordinary accepted-step inventory transfer.
        if np.any(R0 < 0.0):
            raise ValueError(
                "Initial particle radii must be non-negative."
            )
        if hybrid_matching_length is None:
            raise ValueError(
                "hybrid_matching_length [nm] must be supplied explicitly "
                "when hybrid_transport=True."
            )

        hybrid_alpha = 1.0 / float(hybrid_matching_length)
        if not np.isfinite(hybrid_alpha) or hybrid_alpha <= 0.0:
            raise ValueError("hybrid_matching_length must be finite and positive.")

        if hybrid_short_cutoff is None:
            hybrid_short_cutoff = 3.0 / hybrid_alpha

        # The present sparse SR operator uses periodic minimum-image
        # neighbour finding but does not explicitly sum second and higher
        # periodic images of a pair. Warn when the nominal SR support extends
        # beyond the smallest periodic box dimension. This is deliberately a
        # warning rather than a hard stop because no sharp convergence
        # boundary has yet been established for this approximation.
        hybrid_box_min = float(np.min(box))
        hybrid_cutoff = float(hybrid_short_cutoff)
        if hybrid_cutoff > hybrid_box_min:
            print(
                "[hybrid-split-warning] "
                f"short-range cutoff/box={hybrid_cutoff / hybrid_box_min:.6f} "
                f"(r_cut={hybrid_cutoff:.6e} nm, "
                f"L_min={hybrid_box_min:.6e} nm, "
                f"ell/L_min={float(hybrid_matching_length) / hybrid_box_min:.6f}). "
                "The current minimum-image SR operator does not explicitly "
                "sum non-nearest periodic images; this regime has not been "
                "validated. Prefer approximately r_cut <= L_min "
                "(for the default cutoff, ell <= L_min/3)."
            )

        # Optional deterministic N -> N+1 regression hook.
        if hybrid_forced_nucleus is not None:
            if enable_nucleation:
                raise ValueError(
                    "hybrid_forced_nucleus cannot be combined with enable_nucleation=True."
                )
            spec = dict(hybrid_forced_nucleus)
            forced_nucleus_time = float(spec["time"])
            forced_nucleus_position = np.asarray(spec["position"], dtype=float)
            forced_nucleus_radius = float(spec["radius"])
            forced_nucleus_phase_id = int(spec.get("phase_id", 0))
            if forced_nucleus_position.shape != (3,):
                raise ValueError("hybrid_forced_nucleus['position'] must have shape (3,).")
            if (not np.isfinite(forced_nucleus_time)) or forced_nucleus_time < t0:
                raise ValueError("hybrid_forced_nucleus['time'] must be finite and >= t0.")
            if (not np.isfinite(forced_nucleus_radius)) or forced_nucleus_radius <= 0.0:
                raise ValueError("hybrid_forced_nucleus['radius'] must be finite and positive.")
            if forced_nucleus_phase_id < 0 or forced_nucleus_phase_id >= len(phases):
                raise ValueError("hybrid_forced_nucleus phase_id is outside supplied phases.")

        if hybrid_long_range_field is None:
            hybrid_long_range_field = PeriodicLongRangeField(
                box=box,
                grid_shape=hybrid_grid_shape,
                alpha=hybrid_alpha,
                diffusivities=D,
                initial_concentrations=cbar0,
                positions=positions,
            )
        else:
            if not isinstance(hybrid_long_range_field, PeriodicLongRangeField):
                raise TypeError(
                    "hybrid_long_range_field must be a PeriodicLongRangeField."
                )
            if not np.allclose(
                hybrid_long_range_field.box, box, rtol=0.0, atol=1.0e-12
            ):
                raise ValueError(
                    "hybrid_long_range_field box does not match integrator box."
                )
            if not np.isclose(
                hybrid_long_range_field.alpha,
                hybrid_alpha,
                rtol=1.0e-12,
                atol=0.0,
            ):
                raise ValueError(
                    "hybrid_long_range_field alpha does not match "
                    "1/hybrid_matching_length."
                )
            # The same physical LR field is carried continuously through the
            # thermal history.  D changes with temperature at a segment
            # boundary, but C_LR must NOT be reinitialized.  Update only the
            # diffusivities used by subsequent spectral diffusion steps.
            hybrid_long_range_field.set_diffusivities(D)
            hybrid_long_range_field.update_positions(positions)




    # -------------------------------------------------------------
    # Adaptive nucleation batching
    #
    # lam_target retains its historical meaning and acts as the
    # maximum allowed nucleation target.
    # -------------------------------------------------------------

    lam_target_max = float(lam_target)

    if lam_target_min is None:
        # Backward-compatible behaviour:
        # no adaptive batching unless a minimum is explicitly given.
        lam_target_min = lam_target_max
    else:
        lam_target_min = float(lam_target_min)

    if lam_target_min <= 0.0:
        raise ValueError(
            "lam_target_min must be positive."
        )

    if lam_target_max < lam_target_min:
        raise ValueError(
            "lam_target must be >= lam_target_min."
        )


    real_lattice = generate_real_lattice(
        box,
        n_real,
    )

    kvecs, k2 = generate_k_vectors(
        box,
        n_k,
    )

    green_kernel = prepare_periodic_green_kernel(
        box=box,
        alpha=alpha,
        k2=k2,
        screening_length=screening_length,
    )

    screening_length = green_kernel[
        "screening_length"
    ]

    screening_kappa = float(
        green_kernel["kappa"]
    )

    G0 = periodic_green_self_constant(
        box,
        alpha,
        real_lattice,
        kvecs,
        k2,
        screening_length=screening_length,
    )

    if hybrid_transport:
        # Scale-separated transport uses the short-range operator plus the
        # transient long-range field. The dense all-pairs periodic Green matrix
        # used by the direct formulation is not required on this path, avoiding
        # its O(N^2) construction at each thermal-segment restart.
        Gpair_full = None
    else:
        Gpair_full = build_pair_green_matrix(
            positions,
            box,
            alpha,
            real_lattice,
            kvecs,
            k2,
            show_progress=False,
            screening_length=screening_length,
        )

    def particle_solute_inventory(R_full, phase_id_full):
        """Return precipitate solute inventory in species order."""
        R_full = np.asarray(R_full, dtype=float)
        phase_id_full = np.asarray(phase_id_full, dtype=np.int32)
        if phase_id_full.shape != R_full.shape:
            raise ValueError("phase_id and particle radii must have the same shape.")
        if np.any(phase_id_full < 0) or np.any(phase_id_full >= cp_phase.shape[0]):
            raise ValueError("Particle phase ID is outside cp_phase.")
        volumes = (4.0 / 3.0) * np.pi * R_full**3
        return cp_phase[phase_id_full].T @ volumes


    # Exact conserved inventories in generic species order.
    #
    # M_total[s] is the conserved number of atoms of solute species s in
    # the simulation cell.  For the current single precipitate phase,
    #
    #     M_total[s] = cbar0[s] * Vm0 + cp[s] * Vp0.
    #
    # The Mg/Si scalar aliases below are retained only as compatibility
    # views for lower-level kernels that have not yet been generalized.
    particle_volume0 = float(
        np.sum((4.0 / 3.0) * np.pi * np.asarray(R0, dtype=float) ** 3)
    )
    matrix_volume0 = box_volume - particle_volume0

    if matrix_volume0 <= 0.0:
        raise ValueError(
            "Initial precipitate volume leaves no positive matrix volume."
        )

    if M_total_input is None:
        if hybrid_transport:
            # The hybrid LR field is an auxiliary absolute concentration
            # field defined over the full periodic grid.  Its zero mode plus
            # the explicit precipitate inventory is the conserved quantity.
            M_total = (
                hybrid_long_range_field.matrix_inventory()
                + particle_solute_inventory(R0, phase_id)
            )
        else:
            M_total = (
                cbar0 * matrix_volume0
                + particle_solute_inventory(R0, phase_id)
            )
    else:
        M_total = np.asarray(M_total_input, dtype=float)
        if M_total.shape != (n_solutes,):
            raise ValueError(
                f"M_total_input must have shape ({n_solutes},); "
                f"got {M_total.shape}."
            )
        if np.any(~np.isfinite(M_total)):
            raise ValueError("M_total_input contains non-finite values.")
        M_total = M_total.copy()

    M_total_mg = (
        float(M_total[mg_index]) if is_legacy_mg_si else None
    )
    M_total_si = (
        float(M_total[si_index]) if is_legacy_mg_si else None
    )

    # Ensure that the CNT parameter dictionary contains the diffusivities
    # and stoichiometry expected by the ternary nucleation functions.
    #cnt_params_ternary = dict(cnt_params)

    # The diffusivities may change between thermal-history segments.
    # Always make the CNT calculation use the diffusivities supplied
    # to the current integrator call.
    #cnt_params_ternary["DMg"] = float(D_mg)    
    #cnt_params_ternary["DSi"] = float(D_si)

    #cnt_params_ternary["x"] = int(x)
    #cnt_params_ternary["y"] = int(y)


    # ------------------------------------------------------------------
    # Initial arrays
    # ------------------------------------------------------------------
    R = R0.copy()

    # Current split-invariance tests show increasing sensitivity as R/ell
    # approaches ~0.7 and pathological behaviour near the zero of the
    # present SR diagonal self coefficient, R/ell = sqrt(pi)/2. Until the
    # particle-mesh self response is made fully split-invariant, these are
    # numerical-validity guards rather than physical model parameters.
    hybrid_radius_ratio_warning = 0.65
    hybrid_radius_ratio_stop = float(np.sqrt(np.pi) / 2.0)

    def _hybrid_radius_ratio(R_values):
        if not hybrid_transport:
            return 0.0
        R_values = np.asarray(R_values, dtype=float)
        active_values = R_values[R_values > 0.0]
        if active_values.size == 0:
            return 0.0
        return float(np.max(active_values) / float(hybrid_matching_length))

    def _check_hybrid_radius_validity(R_values, *, where, time_value):
        if not hybrid_transport:
            return 0.0
        ratio = _hybrid_radius_ratio(R_values)
        if ratio >= hybrid_radius_ratio_stop:
            R_values = np.asarray(R_values, dtype=float)
            Rmax_local = float(np.max(R_values[R_values > 0.0]))
            raise RuntimeError(
                "Scale-separated transport validity limit reached: "
                f"Rmax/ell={ratio:.6f} >= sqrt(pi)/2="
                f"{hybrid_radius_ratio_stop:.6f} at t={time_value:.8e} s "
                f"({where}); Rmax={Rmax_local:.6e} nm, "
                f"ell={float(hybrid_matching_length):.6e} nm. "
                "The current SR/LR formulation is not validated beyond this "
                "point. Increase the matching length ell and rerun."
            )
        return ratio

    _check_hybrid_radius_validity(
        R, where="initial state", time_value=float(t0)
    )

    alive = R > 0.0
    coupled = alive.copy()

    Q_initial = _validate_species_particle_state(
        Q0,
        "Q0",
        Ntot,
    )

    c_surf_initial = _validate_species_particle_state(
        c_surf0,
        "c_surf0",
        Ntot,
    )

    # Legacy aliases are populated only for the exact Mg-Si regression path.
    Q_mg_initial = (
        None if (Q_initial is None or not is_legacy_mg_si)
        else Q_initial[mg_index].copy()
    )
    Q_si_initial = (
        None if (Q_initial is None or not is_legacy_mg_si)
        else Q_initial[si_index].copy()
    )
    c_surf_mg_initial = (
        None if (c_surf_initial is None or not is_legacy_mg_si)
        else c_surf_initial[mg_index].copy()
    )
    c_surf_si_initial = (
        None if (c_surf_initial is None or not is_legacy_mg_si)
        else c_surf_initial[si_index].copy()
    )

    # Optional accepted interface-velocity state passed across thermal
    # segment boundaries. These are restart diagnostics / timestep state,
    # not nonlinear unknowns. Carrying them avoids inventing zero velocity
    # at an otherwise continuous bookkeeping boundary.
    dRdt_initial = None
    if dRdt0 is not None:
        dRdt_initial = np.asarray(dRdt0, dtype=float)
        if dRdt_initial.shape != (Ntot,):
            raise ValueError(
                f"dRdt0 must have shape ({Ntot},); got "
                f"{dRdt_initial.shape}."
            )
        if np.any(~np.isfinite(dRdt_initial)):
            raise ValueError("dRdt0 contains non-finite values.")
        dRdt_initial = dRdt_initial.copy()

    dRdt_species_initial = None
    if dRdt_species0 is not None:
        dRdt_species_initial = np.asarray(dRdt_species0, dtype=float)
        if dRdt_species_initial.shape != (n_solutes, Ntot):
            raise ValueError(
                "dRdt_species0 must have shape "
                f"({n_solutes}, {Ntot}); got "
                f"{dRdt_species_initial.shape}."
            )
        if np.any(~np.isfinite(dRdt_species_initial)):
            raise ValueError("dRdt_species0 contains non-finite values.")
        dRdt_species_initial = dRdt_species_initial.copy()

    # ------------------------------------------------------------------
    # Histories
    # ------------------------------------------------------------------
    t_list = []
    R_list = []
    phase_id_list = []

    # Authoritative species-indexed particle-field histories.
    #
    # Each entry has shape (n_solutes, n_particles).  The existing
    # Mg/Si lists are retained below as compatibility histories for
    # legacy return values and diagnostics while the lower-level
    # numerical kernels are still ternary-specific.
    Q_list = []
    c_surf_list = []

    # Full 3-D LR history is intentionally NOT accumulated inside the
    # accepted-step integrator.  On fine FFT grids, even a few such snapshots
    # can dominate both RAM and NPZ size.  Sparse/coarsened diagnostic LR
    # history is owned by the thermal-history wrapper.  The argument is retained
    # only for API compatibility with older callers.
    if hybrid_transport and store_hybrid_long_range_history:
        raise ValueError(
            "Per-accepted-step long-range field history is disabled for "
            "scale-separated transport. Store sparse/coarsened LR history through "
            "integrate_ternary_thermal_history() instead."
        )
    hybrid_long_range_field_list = []

    Q_mg_list = []
    Q_si_list = []

    c_surf_mg_list = []
    c_surf_si_list = []

    # Authoritative species-indexed diagnostic histories.
    # Each entry is ordered according to solute_names.
    cbar_list = []
    dRdt_species_list = []
    Pe_list = []

    M_hist = []
    M_abs_err_hist = []
    M_rel_err_hist = []

    # Legacy Mg/Si compatibility histories retained for the existing
    # public return interface and downstream analysis.
    cbar_mg_list = []
    cbar_si_list = []

    dRdt_list = []
    dRdt_mg_list = []
    dRdt_si_list = []

    Pe_mg_list = []
    Pe_si_list = []

    M_mg_hist = []
    M_si_hist = []

    M_mg_abs_err_hist = []
    M_si_abs_err_hist = []

    M_mg_rel_err_hist = []
    M_si_rel_err_hist = []

    alive_count_hist = []
    coupled_count_hist = []
    
    solver_history = []

    number_ppt = []
    positions_out = []
    nuc_kind_out = []

    nuc_time = []
    Ntrial_hist = []
    Nacc_hist = []

    Jnuc = []
    Rnuc = []
    cnuc_mg = []
    cnuc_si = []

    # ------------------------------------------------------------------
    # Shared coupled-solver keyword arguments
    # ------------------------------------------------------------------
    coupled_solver_kwargs = {
        "solute_names": solute_names,
        "D": D,
        "T": T,
        "vatm": vatm,
        # Phase-local atomic volume, interfacial energy, and stoichiometry
        # are owned by ParticlePhaseThermodynamics rather than the generic
        # coupled-growth solver API.
        "gmres_rtol": gmres_rtol,
        "gmres_atol": gmres_atol,
        "gmres_maxiter": gmres_maxiter,
        "gmres_restart": gmres_restart,
        "use_preconditioner": use_preconditioner,
        "nonlinear_xtol": nonlinear_xtol,
        "nonlinear_ftol": nonlinear_ftol,
        "nonlinear_gtol": nonlinear_gtol,
        "nonlinear_max_nfev": nonlinear_max_nfev,
        "concentration_margin": concentration_margin,
        "velocity_floor": velocity_floor,
        "thermodynamic_tolerance":
            thermodynamic_tolerance,
        "velocity_mismatch_tolerance":
            velocity_mismatch_tolerance,
        "strict": True,
    }

    continuation_kwargs = {
        "initial_lambda_step":
            initial_lambda_step,
        "minimum_lambda_step":
            minimum_lambda_step,
        "maximum_lambda_step":
            maximum_lambda_step,
        "lambda_growth_factor":
            lambda_growth_factor,
        "lambda_shrink_factor":
            lambda_shrink_factor,
        "max_continuation_attempts":
            max_continuation_attempts,
        "verbose":
            continuation_verbose,
    }

    # ------------------------------------------------------------------
    # Local helpers
    # ------------------------------------------------------------------
    def isolated_surface_guess_species(
        R_selected,
        cbar_species,
    ):
        """Return the lambda=0 KWN surface state in solute_names order."""
        R_selected = np.asarray(R_selected, dtype=float)
        cbar_species = np.asarray(cbar_species, dtype=float)
        if cbar_species.shape != (n_solutes,):
            raise ValueError(
                f"cbar_species must have shape ({n_solutes},); "
                f"got {cbar_species.shape}."
            )
        if R_selected.size == 0:
            return np.zeros((n_solutes, 0), dtype=float)
        c_surf_selected, _ = initialize_interfaces_kwn(
            R=R_selected, cbar=cbar_species, cp=cp, D=D, T=T,
            vatm=vatm, vatp=phase0_vatp, gamma=phase0_gamma,
            thermodynamics=phase0.thermodynamics,
            stoichiometric_weights=phase0_weights_solver_order,
            concentration_margin=concentration_margin,
        )
        return np.asarray(c_surf_selected, dtype=float)


    def mean_concentrations(R_full, phase_id_full):
        """
        Return the conserved mean matrix concentrations in generic
        species order, shape (n_solutes,).
        """
        R_full = np.asarray(R_full, dtype=float)

        if hybrid_transport:
            cbar = np.asarray(
                hybrid_long_range_field.mean_concentration(),
                dtype=float,
            )
            if cbar.shape != (n_solutes,):
                raise RuntimeError(
                    "Hybrid long-range mean concentration has unexpected "
                    f"shape {cbar.shape}."
                )
            return cbar

        particle_volume = float(
            np.sum((4.0 / 3.0) * np.pi * R_full ** 3)
        )
        matrix_volume = box_volume - particle_volume

        if matrix_volume <= 0.0:
            raise RuntimeError(
                "Non-positive matrix volume encountered while "
                "enforcing solute conservation."
            )

        cbar = (
            M_total - particle_solute_inventory(R_full, phase_id_full)
        ) / matrix_volume

        if np.any(~np.isfinite(cbar)):
            raise RuntimeError(
                "Non-finite matrix concentration obtained "
                "from conservation."
            )

        if np.any(cbar <= 0.0):
            bad = np.where(cbar <= 0.0)[0]
            details = ", ".join(
                f"{solute_names[i]}={cbar[i]:.6e}"
                for i in bad
            )
            raise RuntimeError(
                "Particle volume requires more solute than is "
                "available in the conserved inventory: "
                + details
            )

        return cbar
#--------------------
    def inventory_errors(
        R_full,
        coupled_idx,
        result,
        cbar,
        phase_id_full,
    ):
        """
        Calculate species-indexed total-solute errors using the complete
        reconstructed Green-function field.
        """
        R_full = np.asarray(R_full, dtype=float)
        coupled_idx = np.asarray(coupled_idx, dtype=int)
        cbar = np.asarray(cbar, dtype=float)
        phase_id_full = np.asarray(phase_id_full, dtype=np.int32)
        if phase_id_full.shape != R_full.shape:
            raise ValueError("phase_id_full and R_full must have the same shape.")

        if cbar.shape != (n_solutes,):
            raise ValueError(
                f"cbar must have shape ({n_solutes},); got {cbar.shape}."
            )


        if hybrid_transport:
            M_now = (
                np.asarray(
                    hybrid_long_range_field.matrix_inventory(),
                    dtype=float,
                )
                + particle_solute_inventory(R_full, phase_id_full)
            )
            abs_err = M_now - M_total
            scale = np.maximum(
                np.abs(M_total),
                np.finfo(float).tiny,
            )
            rel_err = abs_err / scale
            return M_now, abs_err, rel_err

        if len(coupled_idx) == 0:
            particle_volume = float(
                np.sum(
                    (4.0 / 3.0)
                    * np.pi
                    * R_full**3
                )
            )
            matrix_volume = box_volume - particle_volume
            M_now = (
                cbar * matrix_volume
                + particle_solute_inventory(R_full, phase_id_full)
            )
        else:
            if result is None:
                raise RuntimeError(
                    "A coupled result is required when coupled particles "
                    "are present."
                )

            R_coupled = R_full[coupled_idx]
            Gpair_coupled = Gpair_full[
                np.ix_(coupled_idx, coupled_idx)
            ]

            Q_coupled = _coupled_result_in_species_order(
                result,
                "Q",
            )
            if Q_coupled.shape != (n_solutes, len(coupled_idx)):
                raise RuntimeError(
                    "Coupled solver returned Q with unexpected shape: "
                    f"{Q_coupled.shape}."
                )

            field_inventory = total_solute_inventory_with_green_field(
                box=box,
                radii=R_coupled,
                cbar=cbar,
                cp=cp_phase[phase_id_full[coupled_idx]].T,
                Q=Q_coupled,
                G0=G0,
                Gpair=Gpair_coupled,
            )
            M_now = np.asarray(
                field_inventory["M"],
                dtype=float,
            )

        abs_err = M_now - M_total
        scale = np.maximum(
            np.abs(M_total),
            np.finfo(float).tiny,
        )
        rel_err = abs_err / scale

        return (
            M_now,
            abs_err,
            rel_err,
        )

    def map_guess(
        old_active_idx,
        new_active_idx,
        old_values,
        default_value=0.0,
    ):
        if old_values is None:
            return None

        old_values = np.asarray(
            old_values,
            dtype=float,
        )

        if old_values.shape != (
            len(old_active_idx),
        ):
            return None

        mapped = np.full(
            len(new_active_idx),
            default_value,
            dtype=float,
        )

        old_lookup = {
            int(global_index): local_index
            for local_index, global_index
            in enumerate(old_active_idx)
        }

        for new_local, global_index in enumerate(
            new_active_idx
        ):
            old_local = old_lookup.get(
                int(global_index)
            )

            if old_local is not None:
                mapped[new_local] = (
                    old_values[old_local]
                )

        return mapped

    # --------------------------------------------------------------
    # Nonlinear-solver state
    #
    # LSMR is normally preferred.  Following repeated LSMR
    # continuation failures, a short sequence of accepted physical
    # timesteps may instead use dense continuation directly.
    #
    # These variables must be initialized before solve_active_state()
    # can be called, including during construction of the initial state.
    # --------------------------------------------------------------
    dense_steps_remaining = 0
    consecutive_lsmr_failures = 0
    dense_fallback_seen_this_step = False

    lsmr_failures_before_dense_window = 2
    dense_window_length = 3


    def get_hybrid_short_operator(master_idx):
        """Return/reuse the short-range operator for an exact active subset."""
        nonlocal hybrid_short_operator
        nonlocal hybrid_short_operator_idx
        nonlocal hybrid_short_cache_hits
        nonlocal hybrid_short_cache_misses

        master_idx = np.asarray(master_idx, dtype=int)

        cache_matches = (
            hybrid_short_operator is not None
            and hybrid_short_operator_idx is not None
            and hybrid_short_operator_idx.shape == master_idx.shape
            and np.array_equal(hybrid_short_operator_idx, master_idx)
        )

        if cache_matches:
            hybrid_short_cache_hits += 1
            return hybrid_short_operator

        hybrid_short_operator = HybridShortRangeOperator(
            positions=positions[master_idx],
            box=box,
            alpha=hybrid_alpha,
            cutoff=float(hybrid_short_cutoff),
        )
        hybrid_short_operator_idx = master_idx.copy()
        hybrid_short_cache_misses += 1
        return hybrid_short_operator


    # Diagnostic-only timing state for the most recent hybrid coupled solve.
    # These values do not participate in the physics.
    _last_hybrid_factor_wall = np.nan
    _last_hybrid_nonlinear_wall = np.nan
    _last_hybrid_nfev = None

    def solve_active_state(
        R_full,
        active_idx,
        old_active_idx=None,
        old_result=None,
        force_continuation=False,
        dt_for_velocity_tolerance=None,
        use_growth_surface_predictor=False,
        strict_solver=True,
    ):

        nonlocal dense_steps_remaining
        nonlocal consecutive_lsmr_failures
        nonlocal dense_fallback_seen_this_step
        nonlocal _last_hybrid_factor_wall
        nonlocal _last_hybrid_nonlinear_wall
        nonlocal _last_hybrid_nfev

        _last_hybrid_factor_wall = np.nan
        _last_hybrid_nonlinear_wall = np.nan
        _last_hybrid_nfev = None
        

        if len(active_idx) == 0:
            return None


        R_active = R_full[active_idx]

        if hybrid_transport:
            active_idx = np.asarray(active_idx, dtype=int)

            # The LR field retains the full fixed production position array so
            # inventory can be deposited/returned at extinct particle sites.
            # Only surviving particles participate in the coupled short-range
            # solve.
            Cbg = np.asarray(
                hybrid_long_range_field.sample_at_particles(),
                dtype=float,
            )[:, active_idx]

            active_short_operator = get_hybrid_short_operator(active_idx)
            _factor_t0 = time.perf_counter()
            factor = active_short_operator.factor(R_active)
            _last_hybrid_factor_wall = time.perf_counter() - _factor_t0

            particle_volumes_active = particle_volumes(R_active)
            cp_active = cp_phase[phase_id[active_idx]].T
            cbar_diag = np.mean(Cbg, axis=1)

            thermo_active = ParticlePhaseThermodynamics(
                phase_id=phase_id[active_idx],
                models=phase_thermodynamic_models,
                precipitate_atomic_volumes=phase_atomic_volumes,
                gammas=phase_gammas,
                stoichiometric_weights=phase_weights_solver_order,
            )

            # Species-consistency acceptance is intentionally separate from
            # the physical stationary-interface criterion.
            if (
                dt_for_velocity_tolerance is not None
                and dt_for_velocity_tolerance > 0.0
            ):
                velocity_absolute_tolerance = (
                    velocity_consistency_radius_tolerance
                    / dt_for_velocity_tolerance
                )
            else:
                velocity_absolute_tolerance = 0.0

            Q_guess = None
            c_surf_guess = None
            if old_result is not None and old_active_idx is not None:
                old_Q = _coupled_result_in_species_order(old_result, "Q")
                old_c_surf = _coupled_result_in_species_order(
                    old_result, "c_surf"
                )
                Q_guess = np.vstack([
                    map_guess(
                        old_active_idx,
                        active_idx,
                        old_Q[sidx],
                    )
                    for sidx in range(n_solutes)
                ])
                c_surf_guess = np.vstack([
                    map_guess(
                        old_active_idx,
                        active_idx,
                        old_c_surf[sidx],
                        default_value=np.nan,
                    )
                    for sidx in range(n_solutes)
                ])

                # Preserve converged surface concentrations for particles that
                # already existed.  Only newly introduced/unmapped particles
                # need an isolated-particle surface-state initializer.
                #
                # Previously any NaN from a newly nucleated particle caused the
                # entire warm start to be discarded, forcing all existing
                # particles back to isolated guesses.
                if np.all(np.isfinite(Q_guess)):
                    missing_surface = ~np.all(
                        np.isfinite(c_surf_guess),
                        axis=0,
                    )
                    if np.any(missing_surface):
                        # In hybrid prescribed-background mode the LR field is
                        # spatially heterogeneous.  Initialize each newly
                        # introduced/unmapped particle from its own sampled LR
                        # background rather than from the mean over all active
                        # particles.  Existing particles retain their converged
                        # surface concentrations unchanged.
                        missing_local_idx = np.where(missing_surface)[0]
                        for local_idx in missing_local_idx:
                            c_surf_guess[:, local_idx:local_idx + 1] = (
                                isolated_surface_guess_species(
                                    R_active[local_idx:local_idx + 1],
                                    Cbg[:, local_idx],
                                )
                            )
                else:
                    Q_guess = None
                    c_surf_guess = None

                if (
                    c_surf_guess is not None
                    and not np.all(np.isfinite(c_surf_guess))
                ):
                    Q_guess = None
                    c_surf_guess = None

                # ------------------------------------------------------
                # First-order accepted-state predictor for the initial
                # surface-concentration guess.
                #
                # This is deliberately limited to growth-candidate solves.
                # Post-nucleation re-solves at the same physical time do not
                # use it. Particle identity is the master-array index, so we
                # extrapolate only particles that existed and were alive in
                # both preceding accepted states.
                #
                #     c_s^(0) = c_s,n
                #               + (dt_target/dt_prev)
                #                 (c_s,n - c_s,n-1)
                #
                # The ratio is capped, and the predicted concentrations are
                # clipped strictly inside the solver's bounded domain.
                # ------------------------------------------------------
                if (
                    enable_growth_surface_predictor
                    and use_growth_surface_predictor
                    and c_surf_guess is not None
                    and dt_for_velocity_tolerance is not None
                    and dt_for_velocity_tolerance > 0.0
                    and len(c_surf_list) >= 2
                    and len(R_list) >= 2
                    and len(t_list) >= 2
                ):
                    dt_prev_predictor = float(
                        t_list[-1] - t_list[-2]
                    )

                    if (
                        np.isfinite(dt_prev_predictor)
                        and dt_prev_predictor > 0.0
                    ):
                        predictor_ratio_raw = (
                            float(dt_for_velocity_tolerance)
                            / dt_prev_predictor
                        )
                        predictor_ratio = min(
                            predictor_ratio_raw,
                            growth_surface_predictor_max_ratio,
                        )

                        c_hist_now = np.asarray(
                            c_surf_list[-1], dtype=float
                        )
                        c_hist_prev = np.asarray(
                            c_surf_list[-2], dtype=float
                        )
                        R_hist_now = np.asarray(
                            R_list[-1], dtype=float
                        )
                        R_hist_prev = np.asarray(
                            R_list[-2], dtype=float
                        )

                        n_common = min(
                            c_hist_now.shape[1],
                            c_hist_prev.shape[1],
                            R_hist_now.size,
                            R_hist_prev.size,
                        )

                        active_idx_array = np.asarray(
                            active_idx, dtype=int
                        )
                        predictor_local_mask = (
                            active_idx_array < n_common
                        )

                        if np.any(predictor_local_mask):
                            master_common = active_idx_array[
                                predictor_local_mask
                            ]
                            existed_both = (
                                (R_hist_now[master_common] > 0.0)
                                & (R_hist_prev[master_common] > 0.0)
                            )

                            local_to_predict = np.where(
                                predictor_local_mask
                            )[0][existed_both]

                            if local_to_predict.size > 0:
                                master_to_predict = active_idx_array[
                                    local_to_predict
                                ]

                                c_now_common = c_hist_now[
                                    :, master_to_predict
                                ]
                                c_prev_common = c_hist_prev[
                                    :, master_to_predict
                                ]

                                predicted = (
                                    c_now_common
                                    + predictor_ratio
                                    * (c_now_common - c_prev_common)
                                )

                                cp_predict = cp_active[
                                    :, local_to_predict
                                ]
                                predictor_lo = (
                                    10.0
                                    * concentration_margin
                                    * cp_predict
                                )
                                predictor_hi = (
                                    (1.0 - 10.0 * concentration_margin)
                                    * cp_predict
                                )
                                predicted = np.clip(
                                    predicted,
                                    predictor_lo,
                                    predictor_hi,
                                )

                                finite_predict = np.all(
                                    np.isfinite(predicted),
                                    axis=0,
                                )
                                if np.any(finite_predict):
                                    local_finite = local_to_predict[
                                        finite_predict
                                    ]
                                    c_surf_guess[:, local_finite] = (
                                        predicted[:, finite_predict]
                                    )

                                    if profile_v11:
                                        print(
                                            "[growth-surface-predictor] "
                                            f"Npredict={len(local_finite)} "
                                            f"Nactive={len(active_idx_array)} "
                                            f"ratio={predictor_ratio:.3f} "
                                            f"ratio_raw={predictor_ratio_raw:.3f} "
                                            f"dt_prev={dt_prev_predictor:.3e} "
                                            f"dt_target={float(dt_for_velocity_tolerance):.3e}"
                                        )

            if c_surf_guess is None:
                c_surf_guess = isolated_surface_guess_species(
                    R_active,
                    cbar_diag,
                )

            _nonlinear_t0 = time.perf_counter()
            try:
                if profile_v11 and solver_diagnostics_file is not None:
                    with _open_solver_diagnostics() as _solver_log:
                        with redirect_stdout(_solver_log):
                            result = solve_interface_coupled(
                                R=R_active,
                                G0=0.0,
                                # No dense pair matrix exists in the hybrid path.  The
                                # injected HybridShortRangeFactor is the complete monopole
                                # operator used by the coupled solve.
                                Gpair=None,
                                solute_names=solute_names,
                                cbar=cbar_diag,
                                matrix_integrals=np.zeros(len(R_active), dtype=float),
                                matrix_volume=box_volume,
                                particle_volume=particle_volumes_active,
                                M_total=M_total,
                                cp=cp_active,
                                D=D,
                                T=T,
                                vatm=vatm,
                                thermodynamics=thermo_active,
                                fixed_green_inventory=np.zeros(n_solutes, dtype=float),
                                c_surf_init=c_surf_guess,
                                Q_init=Q_guess,
                                gmres_rtol=gmres_rtol,
                                gmres_atol=gmres_atol,
                                gmres_maxiter=gmres_maxiter,
                                gmres_restart=gmres_restart,
                                use_preconditioner=use_preconditioner,
                                nonlinear_xtol=nonlinear_xtol,
                                nonlinear_ftol=nonlinear_ftol,
                                nonlinear_gtol=nonlinear_gtol,
                                nonlinear_max_nfev=nonlinear_max_nfev,
                                concentration_margin=concentration_margin,
                                velocity_floor=velocity_floor,
                                velocity_absolute_tolerance=velocity_absolute_tolerance,
                                thermodynamic_tolerance=thermodynamic_tolerance,
                                velocity_mismatch_tolerance=velocity_mismatch_tolerance,
                                nonlinear_solver="lsmr",
                                use_lsmr_block_preconditioner=True,
                                use_lsmr_neighbor_preconditioner=True,
                                lsmr_neighbor_preconditioner_k=12,
                                linear_solver=factor,
                                background_concentration=Cbg,
                                strict=strict_solver,
                                nonlinear_diagnostics=True,
                            )
                else:
                    result = solve_interface_coupled(
                        R=R_active,
                        G0=0.0,
                        # No dense pair matrix exists in the hybrid path.  The
                        # injected HybridShortRangeFactor is the complete monopole
                        # operator used by the coupled solve.
                        Gpair=None,
                        solute_names=solute_names,
                        cbar=cbar_diag,
                        matrix_integrals=np.zeros(len(R_active), dtype=float),
                        matrix_volume=box_volume,
                        particle_volume=particle_volumes_active,
                        M_total=M_total,
                        cp=cp_active,
                        D=D,
                        T=T,
                        vatm=vatm,
                        thermodynamics=thermo_active,
                        fixed_green_inventory=np.zeros(n_solutes, dtype=float),
                        c_surf_init=c_surf_guess,
                        Q_init=Q_guess,
                        gmres_rtol=gmres_rtol,
                        gmres_atol=gmres_atol,
                        gmres_maxiter=gmres_maxiter,
                        gmres_restart=gmres_restart,
                        use_preconditioner=use_preconditioner,
                        nonlinear_xtol=nonlinear_xtol,
                        nonlinear_ftol=nonlinear_ftol,
                        nonlinear_gtol=nonlinear_gtol,
                        nonlinear_max_nfev=nonlinear_max_nfev,
                        concentration_margin=concentration_margin,
                        velocity_floor=velocity_floor,
                        velocity_absolute_tolerance=velocity_absolute_tolerance,
                        thermodynamic_tolerance=thermodynamic_tolerance,
                        velocity_mismatch_tolerance=velocity_mismatch_tolerance,
                        nonlinear_solver="lsmr",
                        use_lsmr_block_preconditioner=True,
                        use_lsmr_neighbor_preconditioner=True,
                        lsmr_neighbor_preconditioner_k=12,
                        linear_solver=factor,
                        background_concentration=Cbg,
                        strict=strict_solver,
                        nonlinear_diagnostics=False,
                    )
            except Exception:
                _last_hybrid_nonlinear_wall = (
                    time.perf_counter() - _nonlinear_t0
                )
                _last_hybrid_nfev = None
                raise

            _last_hybrid_nonlinear_wall = (
                time.perf_counter() - _nonlinear_t0
            )
            _last_hybrid_nfev = getattr(result, "nfev", None)
            return result

        Gpair_active = Gpair_full[
            np.ix_(
                active_idx,
                active_idx,
            )
        ]

        # ----------------------------------------------------------
        # Current phase volumes
        #
        # active_idx contains every surviving
        # finite particle. Dead particles have zero radius and therefore
        # contribute zero volume.
        # ----------------------------------------------------------
        particle_volumes_full = particle_volumes(R_full)
        particle_volume = float(
            np.sum(particle_volumes_full)
        )

        matrix_volume = (
            box_volume
            - particle_volume
        )

        if matrix_volume <= 0.0:
            raise RuntimeError(
                "No positive matrix volume remains."
            )

        # ----------------------------------------------------------
        # Matrix-domain integral of each active Green basis function
        # ----------------------------------------------------------
        matrix_integrals = (
            green_matrix_inventory_integrals(
                radii=R_active,
                G0=G0,
                Gpair=Gpair_active,
            )
        )

        # These reduced-inventory concentrations are retained only as
        # initial guesses for the nonlinear interface solve.
        cbar_guess = np.asarray(
            mean_concentrations(R_full, phase_id),
            dtype=float,
        )

        if cbar_guess.shape != (n_solutes,):
            raise RuntimeError(
                "mean_concentrations returned an unexpected shape: "
                f"{cbar_guess.shape}."
            )

        local_kwargs = dict(
            coupled_solver_kwargs
        )

        # Thermodynamic constraints are dispatched particle-by-particle
        # according to structural phase identity while the concentration
        # fields remain part of one globally coupled nonlinear solve.
        local_kwargs["thermodynamics"] = ParticlePhaseThermodynamics(
            phase_id=phase_id[active_idx],
            models=phase_thermodynamic_models,
            precipitate_atomic_volumes=phase_atomic_volumes,
            gammas=phase_gammas,
            stoichiometric_weights=phase_weights_solver_order,
        )


        # ----------------------------------------------------------
        # Absolute velocity consistency tolerance
        #
        # This is deliberately distinct from stationary_radius_tolerance.
        # The former controls how accurately the species-resolved interface
        # velocities must agree over one timestep; the latter controls when
        # physical interface motion is negligible and may be stationary.
        # ----------------------------------------------------------
        if (
            dt_for_velocity_tolerance is not None
            and dt_for_velocity_tolerance > 0.0
        ):
            velocity_absolute_tolerance = (velocity_consistency_radius_tolerance/ dt_for_velocity_tolerance)
        else:
            velocity_absolute_tolerance = 0.0

        local_kwargs["velocity_absolute_tolerance"] = velocity_absolute_tolerance


        local_kwargs["cbar"] = cbar_guess

        # Full-field mass-conservation data in generic species order.
        local_kwargs["matrix_integrals"] = (
            matrix_integrals
        )

        local_kwargs["matrix_volume"] = (
            matrix_volume
        )

        # The coupled growth solver now receives the precipitate
        # composition and volume particle-by-particle.  This is identical
        # to the previous phase-0 calculation when every particle has the
        # same phase_id, but it removes the mass-balance assumption that
        # all precipitates have one common composition.
        local_kwargs["cp"] = (
            cp_phase[phase_id[active_idx]].T
        )
        local_kwargs["particle_volume"] = (
            particle_volumes_full[active_idx]
        )

        local_kwargs["M_total"] = M_total

        local_kwargs["fixed_green_inventory"] = np.zeros(
            n_solutes,
            dtype=float,
        )

        # ----------------------------------------------------------
        # Continuation nonlinear-solver preference
        #
        # Normally use automatic mode:
        #
        #     LSMR first -> dense fallback if required.
        #
        # Later, dense_steps_remaining will allow the integrator
        # to skip a repeatedly failing LSMR attempt for a short
        # sequence of difficult physical timesteps.
        # ----------------------------------------------------------

        if dense_steps_remaining > 0:
            continuation_solver_preference = "dense"
        else:
            continuation_solver_preference = "auto"

        # A continuation solve is used whenever the particle
        # population has changed through nucleation or when no
        # previous solution is available.
        def run_continuation():
            return solve_interface_with_continuation(
                R=R_active,
                G0=G0,
                Gpair=Gpair_active,
                c_surf_isolated_init=
                    isolated_surface_guess_species(
                        R_active,
                        cbar_guess,
                    ),
                solver_preference=
                    continuation_solver_preference,
                **continuation_kwargs,
                **local_kwargs,
            )

        if (
            force_continuation
            or old_result is None
            or old_active_idx is None
        ):
            result = run_continuation()
            solver_mode = "continuation"

        else:
            old_Q = _coupled_result_in_species_order(
                old_result,
                "Q",
            )
            old_c_surf = _coupled_result_in_species_order(
                old_result,
                "c_surf",
            )

            Q_guess = np.vstack(
                [
                    map_guess(
                        old_active_idx,
                        active_idx,
                        old_Q[sidx],
                    )
                    for sidx in range(n_solutes)
                ]
            )

            c_surf_guess = np.vstack(
                [
                    map_guess(
                        old_active_idx,
                        active_idx,
                        old_c_surf[sidx],
                        default_value=np.nan,
                    )
                    for sidx in range(n_solutes)
                ]
            )

            valid_surface_guess = (
                Q_guess.shape == (n_solutes, len(active_idx))
                and c_surf_guess.shape == (n_solutes, len(active_idx))
                and np.all(np.isfinite(Q_guess))
                and np.all(np.isfinite(c_surf_guess))
            )

            if not valid_surface_guess:
                Q_guess = None
                c_surf_guess = None

            try:
                result = solve_interface_coupled(
                    R=R_active,
                    G0=G0,
                    Gpair=Gpair_active,
                    c_surf_init=c_surf_guess,
                    Q_init=Q_guess,
                    **local_kwargs,
                )
                solver_mode = "direct"

            except (RuntimeError, ValueError):
                result = run_continuation()
                solver_mode = "continuation_fallback"

        # ----------------------------------------------------------
        # Track whether automatic continuation required the
        # dense fallback.
        #
        # dense_fallback_used=True means:
        #
        #     automatic continuation tried LSMR,
        #     LSMR failed,
        #     and the dense continuation rescued the solve.
        #
        # We only count consecutive failures here. We do NOT yet
        # use this counter to change solver behaviour.
        # ----------------------------------------------------------

        dense_fallback_used = bool(
            getattr(
                result,
                "dense_fallback_used",
                False,
            )
        )

        if dense_fallback_used:
            dense_fallback_seen_this_step = True
        #else:
        #    consecutive_lsmr_failures = 0

        solver_history.append({
            "mode": solver_mode,
            "n_active": len(active_idx),
            "nonlinear_solver_used":
                getattr(
                    result,
                    "nonlinear_solver_used",
                    None,
                ),

            "dense_fallback_used":
                dense_fallback_used,

            "consecutive_lsmr_failures":
                consecutive_lsmr_failures,

            "dense_steps_remaining":
                dense_steps_remaining,

            "success": result.success,
            "nfev": result.nfev,
            "cost": result.cost,
            "max_thermodynamic_residual":
                result.max_thermodynamic_residual,
            "max_relative_velocity_mismatch":
                result.max_relative_velocity_mismatch,
            "final_lambda":
                getattr(
                    result,
                    "final_lambda",
                    None,
                ),
            "continuation_history":
                getattr(
                    result,
                    "continuation_history",
                    None,
                ),
        })

        # ----------------------------------------------------------
        # Treat interfaces whose total predicted motion over this
        # timestep is below the radius resolution as stationary.
        #
        # Keep the species-specific interface velocities unchanged; only
        # the common radius-evolution rate is zeroed for stationary interfaces.
        # ----------------------------------------------------------
        if (dt_for_velocity_tolerance is not None and dt_for_velocity_tolerance > 0.0):
            velocity_scale = np.max(
                np.abs(
                    _coupled_result_in_species_order(
                        result, "dRdt_species"
                    )
                ),
                axis=0,
            )

            stationary_mask = (velocity_scale* dt_for_velocity_tolerance<= stationary_radius_tolerance)

            if np.any(stationary_mask):
                result.dRdt = result.dRdt.copy()
                result.dRdt[stationary_mask] = 0.0

        return result

    def build_full_state(
        n_full,
        active_idx,
        result,
    ):
        """Build full-particle state in canonical species order."""
        active_idx = np.asarray(active_idx, dtype=int)
        Q_full = np.zeros((n_solutes, n_full), dtype=float)
        c_surf_full = np.zeros((n_solutes, n_full), dtype=float)
        dRdt_full = np.zeros(n_full, dtype=float)
        dRdt_species_full = np.zeros((n_solutes, n_full), dtype=float)

        if result is not None:
            Q_result = _coupled_result_in_species_order(result, "Q")
            c_surf_result = _coupled_result_in_species_order(
                result, "c_surf"
            )
            dRdt_species_result = _coupled_result_in_species_order(
                result, "dRdt_species"
            )

            Q_full[:, active_idx] = Q_result
            c_surf_full[:, active_idx] = c_surf_result
            dRdt_full[active_idx] = np.asarray(result.dRdt, dtype=float)
            dRdt_species_full[:, active_idx] = dRdt_species_result

        return Q_full, c_surf_full, dRdt_full, dRdt_species_full

    def store_state(
        t_now,
        R_full,
        coupled_idx,
        result,
        cbar,
        phase_id_full,
        alive_mask,
        coupled_mask,
    ):
        """Store one accepted state using species-indexed arrays as authority."""
        cbar_now = np.asarray(cbar, dtype=float).copy()
        if cbar_now.shape != (n_solutes,):
            raise ValueError(
                f"cbar must have shape ({n_solutes},); got {cbar_now.shape}."
            )

        M_now, M_abs_err, M_rel_err = inventory_errors(
            R_full=R_full,
            coupled_idx=coupled_idx,
            result=result,
            cbar=cbar_now,
            phase_id_full=phase_id_full,
        )

        if check_conservation:
            bad_species = np.where(
                np.abs(M_rel_err) > conservation_tol
            )[0]
            if bad_species.size:
                i = int(bad_species[0])
                raise RuntimeError(
                    f"{solute_names[i]} conservation failed at "
                    f"t={t_now:.6g}: abs_err={M_abs_err[i]:.6e}, "
                    f"rel_err={M_rel_err[i]:.6e}."
                )

        n_full = len(R_full)
        phase_id_full = np.asarray(phase_id_full, dtype=np.int32)
        if phase_id_full.shape != (n_full,):
            raise ValueError(
                f"phase_id_full must have shape ({n_full},); "
                f"got {phase_id_full.shape}."
            )
        phase_id_list.append(phase_id_full.copy())

        Q_full, c_surf_full, dRdt_full, dRdt_species_full = (
            build_full_state(n_full, coupled_idx, result)
        )

        # Species-specific Peclet number based on the common interface speed.
        Pe_full = (
            R_full[None, :]
            * dRdt_full[None, :]
            / (2.0 * D[:, None])
        )

        t_list.append(float(t_now))
        R_list.append(R_full.copy())
        Q_list.append(Q_full.copy())
        c_surf_list.append(c_surf_full.copy())
        cbar_list.append(cbar_now.copy())
        dRdt_species_list.append(dRdt_species_full.copy())
        Pe_list.append(Pe_full.copy())
        M_hist.append(M_now.copy())
        M_abs_err_hist.append(M_abs_err.copy())
        M_rel_err_hist.append(M_rel_err.copy())
        dRdt_list.append(dRdt_full.copy())

        # Preserve the old Mg/Si histories only as a regression/output facade.
        if is_legacy_mg_si:
            Q_mg_list.append(Q_full[mg_index].copy())
            Q_si_list.append(Q_full[si_index].copy())
            c_surf_mg_list.append(c_surf_full[mg_index].copy())
            c_surf_si_list.append(c_surf_full[si_index].copy())
            cbar_mg_list.append(float(cbar_now[mg_index]))
            cbar_si_list.append(float(cbar_now[si_index]))
            dRdt_mg_list.append(dRdt_species_full[mg_index].copy())
            dRdt_si_list.append(dRdt_species_full[si_index].copy())
            Pe_mg_list.append(Pe_full[mg_index].copy())
            Pe_si_list.append(Pe_full[si_index].copy())
            M_mg_hist.append(float(M_now[mg_index]))
            M_si_hist.append(float(M_now[si_index]))
            M_mg_abs_err_hist.append(float(M_abs_err[mg_index]))
            M_si_abs_err_hist.append(float(M_abs_err[si_index]))
            M_mg_rel_err_hist.append(float(M_rel_err[mg_index]))
            M_si_rel_err_hist.append(float(M_rel_err[si_index]))

        alive_count_hist.append(int(np.count_nonzero(alive_mask)))
        coupled_count_hist.append(int(np.count_nonzero(coupled_mask)))
        nuc_kind_out.append(nuc_kind.copy())

    # ------------------------------------------------------------------
    # Initial state
    # ------------------------------------------------------------------
    t = float(t0)
    step_number = 0

    cbar_store = mean_concentrations(R, phase_id)

    # Canonical state is species-indexed. Mg/Si aliases are output-only.
    cbar_mg_store = (
        float(cbar_store[mg_index]) if is_legacy_mg_si else None
    )
    cbar_si_store = (
        float(cbar_store[si_index]) if is_legacy_mg_si else None
    )


    # Every surviving particle participates in the coupled solve.
    coupled = alive.copy()

    current_coupled_idx = np.where(
        coupled
    )[0]

    initial_result = None

    if len(current_coupled_idx) > 0:
        # A thermal-segment boundary is a bookkeeping boundary, not a new
        # physical state. When the preceding accepted Q and surface
        # concentrations are supplied, reuse that state exactly rather than
        # re-solving it and applying a fresh acceptance test before physical
        # time has advanced.
        restart_state_available = (
            Q_initial is not None
            and c_surf_initial is not None
            and Q_initial.shape == (n_solutes, Ntot)
            and c_surf_initial.shape == (n_solutes, Ntot)
            and np.all(np.isfinite(Q_initial[:, current_coupled_idx]))
            and np.all(np.isfinite(c_surf_initial[:, current_coupled_idx]))
        )

        if restart_state_available:
            class _RestartCoupledState:
                pass

            initial_result = _RestartCoupledState()

            # Preserve the complete species-indexed state across a thermal
            # segment boundary.  This synthetic result must satisfy the same
            # interface as a fresh coupled-solver result.
            initial_result.solute_names = tuple(solute_names)
            initial_result.cbar = np.asarray(cbar_store, dtype=float).copy()
            initial_result.Q = Q_initial[:, current_coupled_idx].copy()
            initial_result.c_surf = (
                c_surf_initial[:, current_coupled_idx].copy()
            )

            if dRdt_initial is not None:
                initial_result.dRdt = (
                    dRdt_initial[current_coupled_idx].copy()
                )
            else:
                initial_result.dRdt = np.zeros(
                    len(current_coupled_idx), dtype=float
                )

            if dRdt_species_initial is not None:
                initial_result.dRdt_species = (
                    dRdt_species_initial[:, current_coupled_idx].copy()
                )
            else:
                initial_result.dRdt_species = np.tile(
                    initial_result.dRdt[None, :], (n_solutes, 1)
                )

            initial_result.success = True
            initial_result.message = (
                "Reused accepted coupled state from preceding thermal "
                "segment without a same-state nonlinear re-solve."
            )
            initial_result.nfev = 0

            # A restart payload can be structurally complete yet still be
            # inconsistent with the Green-function inventory reconstructed
            # by this integrator call (for example after a thermal-history
            # boundary).  Verify conservation before accepting the reused
            # state.  If it is not inventory-consistent, use the carried
            # state as the nonlinear initial guess and re-solve at the same
            # physical particle state.
            (
                _restart_M_now,
                _restart_abs_err,
                _restart_rel_err,
            ) = inventory_errors(
                R_full=R,
                coupled_idx=current_coupled_idx,
                result=initial_result,
                cbar=cbar_store,
                phase_id_full=phase_id,
            )

            _restart_bad = np.where(
                np.abs(_restart_rel_err) > conservation_tol
            )[0]

            if _restart_bad.size:
                if profile_v11:
                    _restart_details = ", ".join(
                        f"{solute_names[s]} rel_err="
                        f"{_restart_rel_err[s]:.3e}"
                        for s in _restart_bad
                    )
                    print(
                        "[segment-restart] carried coupled state is not "
                        "inventory-consistent; re-solving same physical "
                        f"state ({_restart_details})"
                    )

                initial_result = solve_active_state(
                    R_full=R,
                    active_idx=current_coupled_idx,
                    old_active_idx=current_coupled_idx,
                    old_result=initial_result,
                    force_continuation=False,
                    dt_for_velocity_tolerance=float(dt),
                    use_growth_surface_predictor=False,
                    strict_solver=False,
                )
            elif profile_v11:
                print(
                    "[segment-restart] reused preceding accepted coupled "
                    f"state; N={len(current_coupled_idx)}"
                )

        else:
            # True cold start: no accepted coupled field was supplied.
            initial_result = solve_active_state(
                R_full=R,
                active_idx=current_coupled_idx,
                force_continuation=True,
                dt_for_velocity_tolerance=float(dt),
                strict_solver=False,
            )

            max_initial_state_refinements = 3
            if initial_result is not None and not initial_result.success:
                for refinement_attempt in range(
                    1, max_initial_state_refinements + 1
                ):
                    if profile_v11:
                        print(
                            "[initial-state-refinement] "
                            f"attempt={refinement_attempt}/"
                            f"{max_initial_state_refinements} "
                            f"N={len(current_coupled_idx)}"
                        )

                    initial_result = solve_active_state(
                        R_full=R,
                        active_idx=current_coupled_idx,
                        old_active_idx=current_coupled_idx,
                        old_result=initial_result,
                        force_continuation=False,
                        dt_for_velocity_tolerance=float(dt),
                        use_growth_surface_predictor=False,
                        strict_solver=False,
                    )

                    if initial_result.success:
                        if profile_v11:
                            print(
                                "[initial-state-refinement] "
                                f"accepted on attempt={refinement_attempt}"
                            )
                        break

            if initial_result is not None and not initial_result.success:
                raise RuntimeError(
                    "Initial coupled interface state failed acceptance after "
                    f"{max_initial_state_refinements} same-state refinement "
                    f"attempts. Last solver message: {initial_result.message}"
                )

    current_result = initial_result
    
    # The coupled solver now returns the mean concentrations that satisfy
    # the complete field inventory.
    if current_result is not None and not hybrid_transport:
        cbar_store[:] = _coupled_result_in_species_order(
            current_result,
            "cbar",
        )

        if is_legacy_mg_si:
            cbar_mg_store = float(cbar_store[mg_index])
            cbar_si_store = float(cbar_store[si_index])

    
    store_state(
        t_now=t,
        R_full=R,
        coupled_idx=
            current_coupled_idx,
        result=current_result,
        cbar=cbar_store,
        phase_id_full=phase_id,
        alive_mask=alive,
        coupled_mask=coupled,
    )

    # Optional target-time particle dump scheduler. Requested times are GLOBAL
    # simulation times. Dumps select the nearest accepted state without changing dt.
    dump_target_times_array = None
    dump_target_index = 0
    dump_target_written_count = 0
    dump_target_last_actual_time = None
    dump_time_offset = float(dump_time_offset)
    dump_target_append = bool(dump_target_append)
    dump_target_print = bool(dump_target_print)

    if dump_target_times is not None:
        dump_target_times_array = np.asarray(dump_target_times, dtype=float).reshape(-1)
        if np.any(~np.isfinite(dump_target_times_array)):
            raise ValueError("dump_target_times contains non-finite values.")
        if np.any(dump_target_times_array < 0.0):
            raise ValueError("dump_target_times must be non-negative global times.")
        dump_target_times_array = np.unique(dump_target_times_array)
        dump_target_times_array.sort()
        segment_global_start = dump_time_offset + float(t0)
        while dump_target_index < len(dump_target_times_array) and dump_target_times_array[dump_target_index] < segment_global_start - 1.0e-14:
            dump_target_index += 1

        def _write_target_dump_state(
            *, actual_global_time, R_state, positions_state, phase_id_state,
            nuc_kind_state, dRdt_state, Pe_state, Q_state, requested_global_time,
            lr_field_state=None
        ):
            nonlocal dump_target_written_count, dump_target_append, dump_target_last_actual_time
            actual_global_time = float(actual_global_time)
            if dump_target_last_actual_time is not None and np.isclose(actual_global_time, dump_target_last_actual_time, rtol=0.0, atol=max(1.0e-14,1.0e-12*max(1.0,abs(actual_global_time)))):
                return
            R_state=np.asarray(R_state,float); positions_state=np.asarray(positions_state,float)
            phase_id_state=np.asarray(phase_id_state,np.int32); nuc_kind_state=np.asarray(nuc_kind_state,np.int32)
            dRdt_state=np.asarray(dRdt_state,float)
            Pe_state=np.asarray(Pe_state,float)
            Q_state=np.asarray(Q_state,float)
            if not (len(R_state)==len(positions_state)==len(phase_id_state)==len(nuc_kind_state)==len(dRdt_state)):
                raise RuntimeError("Target-time dump state arrays have inconsistent lengths.")
            if Q_state.shape != (n_solutes, len(R_state)):
                raise RuntimeError(
                    "Target-time Q state has inconsistent shape: "
                    f"expected {(n_solutes, len(R_state))}, got {Q_state.shape}."
                )
            if Pe_state.shape != (n_solutes, len(R_state)):
                raise RuntimeError(
                    "Target-time Pe state has inconsistent shape: "
                    f"expected {(n_solutes, len(R_state))}, got {Pe_state.shape}."
                )
            active_idx_dump=np.flatnonzero(R_state>0.0)
            write_particles_ovito_dump(
                filename=dump_filename, step=actual_global_time,
                positions=positions_state[active_idx_dump],
                radii=R_state[active_idx_dump], box=box, ids=active_idx_dump+1,
                types=nuc_kind_state[active_idx_dump]+1,
                phase_ids=phase_id_state[active_idx_dump],
                gb_flags=nuc_kind_state[active_idx_dump],
                dRdt=dRdt_state[active_idx_dump],
                peclet=Pe_state[:, active_idx_dump],
                peclet_names=solute_names,
                monopoles=Q_state[:, active_idx_dump],
                monopole_names=solute_names,
                mode=("a" if dump_target_append else "w"),
            )
            if dump_field_snapshot_callback is not None and hybrid_transport:
                dump_field_snapshot_callback(
                    actual_global_time=actual_global_time,
                    requested_global_time=float(requested_global_time),
                    long_range_field=hybrid_long_range_field,
                    long_range_field_array=lr_field_state,
                    temperature_K=float(T),
                )
            dump_target_append=True; dump_target_written_count += 1; dump_target_last_actual_time=actual_global_time
            if dump_target_print:
                print(f"Scheduled OVITO dump: requested={requested_global_time:.8e} s, actual={actual_global_time:.8e} s, |dt|={abs(actual_global_time-requested_global_time):.3e} s")

        while dump_target_index < len(dump_target_times_array) and np.isclose(dump_target_times_array[dump_target_index], segment_global_start, rtol=0.0, atol=1.0e-14):
            requested_global_time=float(dump_target_times_array[dump_target_index])
            _write_target_dump_state(actual_global_time=segment_global_start,R_state=R,positions_state=positions,phase_id_state=phase_id,nuc_kind_state=nuc_kind,dRdt_state=dRdt_list[-1],Pe_state=Pe_list[-1],Q_state=Q_list[-1],requested_global_time=requested_global_time)
            dump_target_index += 1

# ------------------------------------------------------------------
# Main integration loop
# ------------------------------------------------------------------
    nucleation_dt_safety = 0.95
    max_nucleation_dt_retries = 25

    # Spatial-rate probes are expensive.  Accumulate reference hazard while
    # probes are skipped and retain the last resolved correction factor.
    # Initialize at the probe threshold so the first nucleation-capable step
    # of each thermal segment performs a conservative spatial probe.
    spatial_rate_skipped_hazard_bulk = np.full(
        len(phases), NUCLEATION_SPATIAL_RATE_PROBE_HAZARD, dtype=float
    )
    spatial_rate_skipped_hazard_gb = np.full(
        len(phases), NUCLEATION_SPATIAL_RATE_PROBE_HAZARD, dtype=float
    )
    spatial_rate_last_factor_bulk = np.ones(len(phases), dtype=float)
    spatial_rate_last_factor_gb = np.ones(len(phases), dtype=float)

    # Candidate-growth retry controls.  The radius update is explicit, so a
    # candidate step must also remain on the same resolved side of the local
    # kinetic fixed point.  Otherwise the start-of-step velocity has carried
    # the particle too far and the candidate is retried with a smaller dt.
    growth_dt_retry_safety = 0.5
    max_growth_dt_retries = 25
    velocity_reversal_fraction = 0.25 * dt_safety

    # Do not forget a timestep reduction immediately on the next accepted
    # physical step.  The ordinary growth/nucleation limiters remain free to
    # reduce dt further; this only limits upward rebound from the most recent
    # accepted timestep.
    max_accepted_dt_growth_factor = float(max_accepted_dt_growth_factor)
    if (
        not np.isfinite(max_accepted_dt_growth_factor)
        or max_accepted_dt_growth_factor < 1.0
    ):
        raise ValueError(
            "max_accepted_dt_growth_factor must be finite and >= 1."
        )

    post_retry_dt_growth_factor = float(post_retry_dt_growth_factor)
    if (
        not np.isfinite(post_retry_dt_growth_factor)
        or post_retry_dt_growth_factor < 1.0
    ):
        raise ValueError(
            "post_retry_dt_growth_factor must be finite and >= 1."
        )

    effort_hold_nfev = int(effort_hold_nfev)
    effort_reduce_nfev = int(effort_reduce_nfev)
    effort_reduce_dt_factor = float(effort_reduce_dt_factor)

    if effort_hold_nfev < 1:
        raise ValueError("effort_hold_nfev must be >= 1.")
    if effort_reduce_nfev < effort_hold_nfev:
        raise ValueError(
            "effort_reduce_nfev must be >= effort_hold_nfev."
        )
    if (
        not np.isfinite(effort_reduce_dt_factor)
        or effort_reduce_dt_factor <= 0.0
        or effort_reduce_dt_factor > 1.0
    ):
        raise ValueError(
            "effort_reduce_dt_factor must be finite and in (0, 1]."
        )

    enable_growth_surface_predictor = bool(
        enable_growth_surface_predictor
    )
    growth_surface_predictor_max_ratio = float(
        growth_surface_predictor_max_ratio
    )
    if (
        not np.isfinite(growth_surface_predictor_max_ratio)
        or growth_surface_predictor_max_ratio <= 0.0
    ):
        raise ValueError(
            "growth_surface_predictor_max_ratio must be finite and positive."
        )

    dt_last_accepted = (
        None
        if dt_last_accepted0 is None
        else float(dt_last_accepted0)
    )
    previous_step_had_growth_retry = bool(
        previous_step_had_growth_retry0
    )
    previous_growth_nfev = (
        None
        if previous_growth_nfev0 is None
        else int(previous_growth_nfev0)
    )


    # Wall-clock timing diagnostics
    wall_step_start = time.perf_counter()
    wall_window_start = wall_step_start
    wall_window_step_start = step_number
    wall_window_t_start = t
    wall_print_every = 10
    _prof = {"lr":0.0, "growth_solve":0.0, "gb":0.0,
             "cnt":0.0, "post_nuc_solve":0.0}

    while t < t_end:
        dump_prev_t=float(t); dump_prev_R=R; dump_prev_positions=positions; dump_prev_phase_id=phase_id; dump_prev_nuc_kind=nuc_kind
        dump_prev_dRdt=np.asarray(dRdt_list[-1], dtype=float).copy()
        dump_prev_Pe=np.asarray(Pe_list[-1], dtype=float).copy()
        dump_prev_Q=np.asarray(Q_list[-1], dtype=float).copy()
        dump_prev_lr_field=(
            hybrid_long_range_field.copy_field()
            if (dump_target_times_array is not None and dump_target_index < len(dump_target_times_array) and dump_field_snapshot_callback is not None and hybrid_transport)
            else None
        )
        step_number += 1

        # Wall-clock timing diagnostics
        wall_step_start = time.perf_counter()


        dense_fallback_seen_this_step = False
    
        old_coupled_idx = (
            current_coupled_idx.copy()
        )

        old_result = current_result

        dRdt_current = np.zeros(
            len(R),
            dtype=float,
        )

        if len(old_coupled_idx) > 0:
            if old_result is None:
                raise RuntimeError(
                    "old_result is None although coupled particles "
                    "are present."
                )

            dRdt_current[
                old_coupled_idx
            ] = old_result.dRdt


        # -------------------------------------------------------------
        # Adaptive nucleation target based on geometric crowding
        # -------------------------------------------------------------

        active_radii_for_lam = R[np.isfinite(R) & (R > 0.0)]

        N_active_for_lam = len(active_radii_for_lam)

        if (N_active_for_lam == 0 or lam_target_min == lam_target_max):

            lam_target_current = lam_target_max
            crowding_parameter = 0.0

        else:

            mean_radius_for_lam = float(np.mean(active_radii_for_lam))
            mean_spacing_for_lam = (box_volume/ N_active_for_lam)**(1.0 / 3.0)
            crowding_parameter = (2.0* mean_radius_for_lam/ mean_spacing_for_lam)

            crowding_low = 0.10
            crowding_high = 0.50

            crowding_fraction = np.clip((crowding_parameter- crowding_low)/ (crowding_high- crowding_low),0.0,1.0)

            lam_target_current = (lam_target_max- crowding_fraction* (lam_target_max- lam_target_min))

        #cbar_mg, cbar_si = (
        #    mean_concentrations(R, phase_id)
        #)

        # --------------------------------------------------------------
        # Initial candidate timestep
        # --------------------------------------------------------------
        dt_use = float(dt)

        # Cap only the first timestep of the complete thermal history.
        # dt_last_accepted is propagated across thermal segments, so this
        # condition is false after the first accepted step.
        if dt_last_accepted is None and first_step_dt_max is not None:
            first_step_dt_max_value = float(first_step_dt_max)
            if (
                not np.isfinite(first_step_dt_max_value)
                or first_step_dt_max_value <= 0.0
            ):
                raise ValueError(
                    "first_step_dt_max must be finite and > 0 when provided."
                )
            dt_use = min(dt_use, first_step_dt_max_value)

        if dt_last_accepted is not None:
            if previous_step_had_growth_retry:
                dt_history_growth_factor = post_retry_dt_growth_factor
                dt_history_reason = "retry_hold"
            elif (
                previous_growth_nfev is not None
                and previous_growth_nfev >= effort_reduce_nfev
            ):
                dt_history_growth_factor = effort_reduce_dt_factor
                dt_history_reason = "effort_reduce"
            elif (
                previous_growth_nfev is not None
                and previous_growth_nfev >= effort_hold_nfev
            ):
                dt_history_growth_factor = 1.0
                dt_history_reason = "effort_hold"
            else:
                dt_history_growth_factor = max_accepted_dt_growth_factor
                dt_history_reason = "normal_growth"

            dt_history_cap = (
                dt_history_growth_factor
                * dt_last_accepted
            )
            dt_use = min(
                dt_use,
                dt_history_cap,
            )
        else:
            dt_history_growth_factor = np.inf
            dt_history_reason = "initial"
            dt_history_cap = np.inf

        dt_lim_growth = np.inf
        dt_lim_nuc = np.inf

        # --------------------------------------------------------------
        # Minimum-time-resolution treatment of imminent dissolution.
        #
        # A particle that is predicted to cross the removal radius within
        # dt_min is not allowed to drive the global adaptive timestep below
        # dt_min.  Instead, it is marked as an extinction candidate.
        #
        # Importantly, dt_min is *not* used to override timestep limits from
        # particles that are expected to survive.  If the surviving
        # population still requires dt < dt_min, the usual RuntimeError is
        # retained below.
        # --------------------------------------------------------------
        extinction_candidate = np.zeros(
            len(R),
            dtype=bool,
        )
        dt_to_remove = np.full(
            len(R),
            np.inf,
            dtype=float,
        )

        if remove_extinct and np.any(alive):
            R_cut = R_remove + max(
                R_remove_buffer,
                0.0,
            )

            dissolving = (
                alive
                & (dRdt_current < 0.0)
            )

            if np.any(dissolving):
                dt_to_remove[dissolving] = (
                    (
                        R[dissolving]
                        - R_cut
                    )
                    / (
                        -dRdt_current[dissolving]
                    )
                )

                # A negative value means the particle is already inside the
                # buffered removal region and is still dissolving, so it is
                # also an extinction candidate.
                extinction_candidate = (
                    dissolving
                    & (dt_to_remove <= dt_min)
                )
#--------------
        # --------------------------------------------------------------
        # Adaptive growth/dissolution timestep limiter.
        #
        # Growing particles retain the ordinary relative-radius-change
        # limiter and are allowed to require dt < dt_min.
        #
        # Dissolving particles use the same limiter, but are not allowed
        # to drive the timestep below dt_min.  Their final approach to
        # the removal radius is therefore unresolved below dt_min.
        #
        # Nucleation is handled separately below and may also require
        # dt < dt_min.
        # --------------------------------------------------------------
        if adaptive_dt and np.any(alive):

            growing = (
                alive
                & (dRdt_current > 0.0)
            )

            dissolving_for_dt = (
                alive
                & (dRdt_current < 0.0)
            )

            dt_lim_grow = np.inf
            dt_lim_dissolve_raw = np.inf
            dt_lim_dissolve = np.inf

            # ----------------------------------------------------------
            # Growing particles: retain the full adaptive limiter.
            # ----------------------------------------------------------
            if np.any(growing):
                dt_lim_grow = (
                    dt_safety
                    * np.min(
                        R[growing]
                        / dRdt_current[growing]
                    )
                )

            # ----------------------------------------------------------
            # Dissolving particles: do not resolve their evolution on
            # timescales smaller than dt_min.
            # ----------------------------------------------------------
            if np.any(dissolving_for_dt):
                dt_lim_dissolve_raw = (
                    dt_safety
                    * np.min(
                        R[dissolving_for_dt]
                        / (
                            -dRdt_current[
                                dissolving_for_dt
                            ]
                        )
                    )
                )

                if (
                    np.isfinite(dt_lim_dissolve_raw)
                    and dt_lim_dissolve_raw > 0.0
                ):
                    dt_lim_dissolve = max(
                        dt_lim_dissolve_raw,
                        dt_min,
                    )

            # The growth-stage timestep is controlled by the stricter
            # of the growing and dissolution limits.
            dt_lim_growth = min(
                dt_lim_grow,
                dt_lim_dissolve,
            )

            if (
                np.isfinite(dt_lim_growth)
                and dt_lim_growth > 0.0
            ):
                dt_use = min(
                    dt_use,
                    dt_lim_growth,
                )

#---------------
        if dt_max is not None:
            dt_use = min(
                dt_use,
                dt_max,
            )

        # If one or more particles are being treated as unresolved
        # extinction events, do not step farther than dt_min.  This keeps the
        # timing of their disappearance localized to the minimum temporal
        # resolution rather than allowing an arbitrarily large step.
        if np.any(extinction_candidate):
            dt_use = min(
                dt_use,
                dt_min,
            )
    
        dt_use = min(
            dt_use,
            t_end - t,
        )

        if (
            hybrid_transport
            and hybrid_forced_nucleus is not None
            and not forced_nucleus_inserted
            and forced_nucleus_time is not None
            and t < forced_nucleus_time < t + dt_use
        ):
            dt_use = forced_nucleus_time - t
    
        if dt_use <= 0.0:
            raise RuntimeError(
                f"Non-positive timestep at t={t:.8e}."
            )
    
    
        # --------------------------------------------------------------
        # Candidate growth step with nucleation-timestep retries
        #
        # Growth is recomputed whenever the post-growth nucleation state
        # requires a smaller timestep. No random numbers are consumed
        # inside this retry loop.
        # --------------------------------------------------------------
        nucleation_dt_retry = 0
        growth_dt_retry = 0
        accepted_growth_nfev = None
        candidate_attempt = 0

        # Hybrid candidate growth mutates the transient LR field. A
        # nucleation-limiter retry rejects that candidate timestep, so
        # every retry must restart from the last accepted LR state.
        # A rollback snapshot is needed only when the candidate step can
        # actually mutate the LR field.  Before any particles exist, an exactly
        # uniform LR field is unchanged by diffusion and has no source term, so
        # copying a multi-GB grid here is unnecessary.
        need_hybrid_rollback_snapshot = bool(
            hybrid_transport
            and not (
                len(R) == 0
                and getattr(hybrid_long_range_field, "is_uniform", False)
            )
        )
        hybrid_lr_field_before_candidate = (
            hybrid_long_range_field.copy_field()
            if need_hybrid_rollback_snapshot
            else None
        )

        while True:
            candidate_attempt += 1
            if hybrid_lr_field_before_candidate is not None:
                hybrid_long_range_field.set_field(
                    hybrid_lr_field_before_candidate
                )
            dRdt_step = (
                dRdt_current.copy()
            )

            R_new = R.copy()

            R_new[alive] += (
                dt_use
                * dRdt_step[alive]
            )

            R_new = np.maximum(
                R_new,
                0.0,
            )

            if remove_extinct:

                # Standard extinction criterion.
                remove_mask = alive & (R_new < R_remove)

                # Optional buffered removal criterion.
                if R_remove_buffer > 0.0:
                    early_remove_mask = (
                        alive
                        & (dRdt_step < 0.0)
                        & (
                            R_new
                            < R_remove + R_remove_buffer
                        )
                    )
                    remove_mask |= early_remove_mask

                # Particles whose remaining lifetime was below dt_min were
                # excluded from the growth limiter.  Remove them on this
                # candidate step only if the step actually reaches their
                # predicted crossing time.  The second condition matters for
                # the final short step of a segment, where remaining < dt_min.
                unresolved_extinction_mask = (
                    extinction_candidate
                    & (
                        dt_to_remove
                        <= dt_use * (1.0 + 1.0e-12)
                    )
                )
                remove_mask |= unresolved_extinction_mask
            
                R_new[remove_mask] = 0.0


            # ----------------------------------------------------------
            # Check candidate particle volume
            # ----------------------------------------------------------
            particle_volume_new = np.sum(
                particle_volumes(R_new)
            )

            if particle_volume_new >= box_volume:
                raise RuntimeError(
                    "Particles fill the simulation box at "
                    f"t={t + dt_use:.6g}."
                )

            # ----------------------------------------------------------
            # Reclassify surviving particles
            # ----------------------------------------------------------
            alive_new = R_new > 0.0

            if hybrid_transport:
                # The accepted geometry change is converted into the actual
                # precipitate inventory change and communicated through the
                # transient smooth LR field before the new interface state is
                # solved.  Nucleation/retry is disabled in this bridge, so the
                # field is advanced exactly once per physical timestep.
                V_old = particle_volumes(R)
                V_new = particle_volumes(R_new)
                cp_particles = cp_phase[phase_id].T
                delta_particle_inventory = (
                    cp_particles
                    * (V_new - V_old)[None, :]
                )
                _pt0 = time.perf_counter() if profile_v11 else None
                hybrid_long_range_field.advance_with_particle_inventory_change(
                    dt=dt_use,
                    delta_particle_inventory=delta_particle_inventory,
                )
                if profile_v11:
                    _prof["lr"] += time.perf_counter() - _pt0

                # The long-range field is one component of the Ewald/hybrid
                # concentration decomposition, not a standalone physical
                # concentration.  It may therefore be locally zero or negative.
                # Physical positivity is enforced on the interfacial/total
                # concentrations by the coupled solve; do not reject a timestep
                # solely because C_LR sampled at a particle is non-positive.

                cbar_store = np.asarray(
                    hybrid_long_range_field.mean_concentration(),
                    dtype=float,
                )

                # ------------------------------------------------------
                # Global matrix-inventory admissibility
                #
                # C_LR is an auxiliary split component and may be signed
                # pointwise. Its zero mode, however, is the conserved
                # matrix inventory in the hybrid formulation. A candidate
                # growth step must therefore not consume more of any solute
                # than remains in the matrix. Reject and retry with a
                # smaller dt before attempting the post-growth interface
                # solve if the matrix inventory would become negative.
                # ------------------------------------------------------
                matrix_inventory_candidate = np.asarray(
                    hybrid_long_range_field.matrix_inventory(),
                    dtype=float,
                )

                inventory_scale = np.maximum(
                    np.abs(np.asarray(M_total, dtype=float)),
                    1.0,
                )
                inventory_negative_tolerance = (
                    100.0 * np.finfo(float).eps * inventory_scale
                )
                inventory_bad = (
                    ~np.isfinite(matrix_inventory_candidate)
                    | (
                        matrix_inventory_candidate
                        < -inventory_negative_tolerance
                    )
                )

                if np.any(inventory_bad):
                    bad_idx = np.where(inventory_bad)[0]

                    if profile_v11:
                        bad_desc = ", ".join(
                            f"{solute_names[i]}={matrix_inventory_candidate[i]:.6e}"
                            for i in bad_idx
                        )
                        print(
                            "[growth-attempt] "
                            f"step={step_number} "
                            f"candidate={candidate_attempt} "
                            f"growth_try={growth_dt_retry + 1} "
                            f"N={int(np.count_nonzero(alive_new))} "
                            f"dt={dt_use:.8e} "
                            "status=inventory_reject "
                            f"matrix_inventory=({bad_desc})"
                        )

                    growth_dt_retry += 1
                    if growth_dt_retry > max_growth_dt_retries:
                        raise RuntimeError(
                            "Unable to obtain a candidate growth step with "
                            "non-negative matrix solute inventory after "
                            f"{max_growth_dt_retries} timestep retries at "
                            f"t={t:.8e}; last candidate dt={dt_use:.8e}; "
                            f"matrix inventory={matrix_inventory_candidate}."
                        )

                    dt_reduced = growth_dt_retry_safety * dt_use
                    if (
                        not np.isfinite(dt_reduced)
                        or dt_reduced <= 0.0
                    ):
                        raise RuntimeError(
                            "Invalid reduced timestep after matrix-inventory "
                            f"rejection: dt={dt_reduced}."
                        )

                    dt_use = min(dt_reduced, t_end - t)
                    continue
            else:
                cbar_store = mean_concentrations(
                    R_new,
                    phase_id,
                )

            if is_legacy_mg_si:
                cbar_mg_store = float(cbar_store[mg_index])
                cbar_si_store = float(cbar_store[si_index])

            coupled_new = alive_new.copy()

            coupled_idx_new = np.where(
                coupled_new
            )[0]

            # ----------------------------------------------------------
            # Solve only the coupled-active post-growth population
            # ----------------------------------------------------------
            if len(coupled_idx_new) > 0:
                _pt0 = time.perf_counter() if profile_v11 else None
                _growth_attempt_t0 = time.perf_counter()
                _growth_retry_number = growth_dt_retry + 1
                try:
                    state_after_growth = (
                        solve_active_state(
                            R_full=R_new,
                            active_idx=coupled_idx_new,
                            old_active_idx=
                                old_coupled_idx,
                            old_result=
                                old_result,
                            force_continuation=True,
                            dt_for_velocity_tolerance=dt_use,
                            use_growth_surface_predictor=True,
                        )
                    )
                except RuntimeError as exc:
                    if profile_v11:
                        _prof["growth_solve"] += time.perf_counter() - _pt0

                    # A failed nonlinear acceptance check at the candidate
                    # post-growth state is treated as a rejected explicit
                    # timestep, not as a fatal simulation error.  Other
                    # RuntimeErrors still propagate so physical/inventory
                    # failures are never silently hidden.
                    if (
                        "Coupled interface solve failed acceptance checks"
                        not in str(exc)
                    ):
                        raise

                    _growth_attempt_elapsed = (
                        time.perf_counter() - _growth_attempt_t0
                    )
                    if profile_v11:
                        print(
                            "[growth-attempt] "
                            f"step={step_number} "
                            f"candidate={candidate_attempt} "
                            f"growth_try={_growth_retry_number} "
                            f"N={len(coupled_idx_new)} "
                            f"dt={dt_use:.8e} "
                            f"wall={_growth_attempt_elapsed:.3f}s "
                            f"factor={_last_hybrid_factor_wall:.3f}s "
                            f"nonlinear={_last_hybrid_nonlinear_wall:.3f}s "
                            f"nfev={_last_hybrid_nfev} "
                            "status=solve_reject"
                        )

                    growth_dt_retry += 1
                    if growth_dt_retry > max_growth_dt_retries:
                        raise RuntimeError(
                            "Unable to obtain a converged post-growth coupled "
                            "state after "
                            f"{max_growth_dt_retries} timestep retries at "
                            f"t={t:.8e}; last candidate dt={dt_use:.8e}."
                        ) from exc

                    dt_reduced = growth_dt_retry_safety * dt_use
                    if (
                        not np.isfinite(dt_reduced)
                        or dt_reduced <= 0.0
                    ):
                        raise RuntimeError(
                            "Invalid reduced timestep after coupled-solve "
                            f"failure: dt={dt_reduced}."
                        ) from exc

                    dt_use = min(dt_reduced, t_end - t)
                    continue

                _growth_attempt_elapsed = (
                    time.perf_counter() - _growth_attempt_t0
                )

                if profile_v11:
                    _prof["growth_solve"] += time.perf_counter() - _pt0

                # ------------------------------------------------------
                # Kinetic fixed-point crossing corrector
                # ------------------------------------------------------
                # The explicit radius predictor can cross a local kinetic
                # fixed point (dR/dt = 0).  Previously every resolved sign
                # reversal rejected the complete timestep and halved dt.  In
                # stiff cooling histories this can collapse the global
                # timestep even for a single otherwise well-behaved particle.
                #
                # For hybrid transport, correct the crossed radius inside the
                # *same* physical timestep.  A secant estimate brackets the
                # zero between the accepted start radius and the explicit
                # candidate radius.  After changing the radius we restore the
                # pre-candidate LR field, re-apply the particle inventory
                # change for the corrected radius, and re-solve the coupled
                # interface state.  Thus every corrected candidate remains
                # exactly consistent with the LR inventory transfer; no solute
                # is created/lost by the corrector.
                #
                # The corrector is intentionally restricted to the hybrid
                # path.  Legacy/non-hybrid calculations retain the established
                # reversal-retry behaviour.
                v_old_candidate = dRdt_step[coupled_idx_new]
                v_new_candidate = np.asarray(
                    state_after_growth.dRdt,
                    dtype=float,
                )

                max_fixed_point_corrections = 8
                fixed_point_corrections = 0
                restart_candidate_after_reversal = False

                while True:
                    sign_reversal = (
                        v_old_candidate * v_new_candidate < 0.0
                    )

                    if not np.any(sign_reversal):
                        break

                    idx_rev = coupled_idx_new[sign_reversal]
                    displacement = np.abs(
                        dt_use * dRdt_step[idx_rev]
                    )
                    reversal_allowance = np.maximum(
                        stationary_radius_tolerance,
                        velocity_reversal_fraction * R[idx_rev],
                    )
                    resolved_reversal = (
                        displacement > reversal_allowance
                    )

                    if not np.any(resolved_reversal):
                        break

                    # Keep only the reversals that are large enough to need a
                    # resolved correction.  Tiny near-stationary crossings are
                    # accepted exactly as before.
                    idx_fix = idx_rev[resolved_reversal]

                    if (
                        hybrid_transport
                        and hybrid_lr_field_before_candidate is not None
                        and fixed_point_corrections < max_fixed_point_corrections
                    ):
                        # Map full particle indices back to the coupled-state
                        # velocity vector.
                        coupled_lookup = {
                            int(full_idx): local_idx
                            for local_idx, full_idx in enumerate(coupled_idx_new)
                        }

                        # Outer-corrector diagnostics are deliberately based
                        # on radius/velocity changes, not on the inner nonlinear
                        # solver residual.  This lets cooling runs with different
                        # matching lengths be compared at the same global time,
                        # temperature, and particle position.
                        fp_diag = []
                        for full_idx in idx_fix:
                            local_idx = coupled_lookup[int(full_idx)]
                            v0 = float(dRdt_step[full_idx])
                            v1 = float(v_new_candidate[local_idx])
                            R_start = float(R[full_idx])
                            R_iter_before = float(R_new[full_idx])
                            denom = abs(v0) + abs(v1)
                            if not np.isfinite(denom) or denom <= 0.0:
                                theta = 0.5
                            else:
                                # Linear/secant estimate of the zero of v(R)
                                # between the accepted start state and the current
                                # corrected end-state estimate.
                                theta = abs(v0) / denom
                            theta = float(np.clip(theta, 0.0, 1.0))
                            R_iter_after = (
                                R_start
                                + theta * (R_iter_before - R_start)
                            )
                            R_new[full_idx] = R_iter_after
                            fp_diag.append({
                                "full_idx": int(full_idx),
                                "local_idx": int(local_idx),
                                "R_start": R_start,
                                "R_before": R_iter_before,
                                "R_after": float(R_iter_after),
                                "dR_outer": float(R_iter_after - R_iter_before),
                                "rel_dR_outer": float(
                                    abs(R_iter_after - R_iter_before)
                                    / max(abs(R_iter_before), stationary_radius_tolerance)
                                ),
                                "bracket_span": float(abs(R_iter_before - R_start)),
                                "v_start": v0,
                                "v_before": v1,
                                "theta": theta,
                            })

                        # Rebuild the LR candidate from the last accepted field
                        # using the *corrected* precipitate inventory change.
                        hybrid_long_range_field.set_field(
                            hybrid_lr_field_before_candidate
                        )
                        V_old = particle_volumes(R)
                        V_new = particle_volumes(R_new)
                        cp_particles = cp_phase[phase_id].T
                        delta_particle_inventory = (
                            cp_particles
                            * (V_new - V_old)[None, :]
                        )
                        _pt_fp = time.perf_counter() if profile_v11 else None
                        hybrid_long_range_field.advance_with_particle_inventory_change(
                            dt=dt_use,
                            delta_particle_inventory=delta_particle_inventory,
                        )
                        if profile_v11:
                            _prof["lr"] += time.perf_counter() - _pt_fp

                        cbar_store = np.asarray(
                            hybrid_long_range_field.mean_concentration(),
                            dtype=float,
                        )
                        if is_legacy_mg_si:
                            cbar_mg_store = float(cbar_store[mg_index])
                            cbar_si_store = float(cbar_store[si_index])

                        _pt_fp = time.perf_counter() if profile_v11 else None
                        state_after_growth = solve_active_state(
                            R_full=R_new,
                            active_idx=coupled_idx_new,
                            old_active_idx=old_coupled_idx,
                            old_result=old_result,
                            force_continuation=True,
                            dt_for_velocity_tolerance=dt_use,
                            use_growth_surface_predictor=True,
                        )
                        if profile_v11:
                            _prof["growth_solve"] += time.perf_counter() - _pt_fp

                        v_new_candidate = np.asarray(
                            state_after_growth.dRdt,
                            dtype=float,
                        )
                        fixed_point_corrections += 1

                        if profile_v11:
                            # Report the particle with the largest relative outer
                            # radius correction.  For the present single-particle
                            # cooling benchmark this is the particle itself.
                            worst_fp = max(
                                fp_diag,
                                key=lambda d: d["rel_dR_outer"],
                            )
                            wi = worst_fp["full_idx"]
                            wl = worst_fp["local_idx"]
                            pos_w = np.asarray(positions[wi], dtype=float)
                            v_after = float(v_new_candidate[wl])
                            global_t_end = float(
                                diagnostic_time_offset + t + dt_use
                            )
                            ell_value = (
                                float(hybrid_matching_length)
                                if hybrid_matching_length is not None
                                else float("nan")
                            )
                            r_over_ell = (
                                worst_fp["R_after"] / ell_value
                                if np.isfinite(ell_value) and ell_value > 0.0
                                else float("nan")
                            )
                            print(
                                "[growth-fixed-point] "
                                f"step={step_number} "
                                f"candidate={candidate_attempt} "
                                f"correction={fixed_point_corrections} "
                                f"Nfix={len(idx_fix)} "
                                f"dt={dt_use:.8e} "
                                f"t_global={global_t_end:.9e} "
                                f"T={float(T):.6f}K "
                                f"ell={ell_value:.6e}nm "
                                f"particle={wi} "
                                f"pos=({pos_w[0]:.6e},{pos_w[1]:.6e},{pos_w[2]:.6e})nm "
                                f"Rstart={worst_fp['R_start']:.9e} "
                                f"Rbefore={worst_fp['R_before']:.9e} "
                                f"Rafter={worst_fp['R_after']:.9e} "
                                f"dR_outer={worst_fp['dR_outer']:.9e} "
                                f"rel_dR_outer={worst_fp['rel_dR_outer']:.9e} "
                                f"bracket={worst_fp['bracket_span']:.9e} "
                                f"R_over_ell={r_over_ell:.9e} "
                                f"v_start={worst_fp['v_start']:.9e} "
                                f"v_before={worst_fp['v_before']:.9e} "
                                f"v_after={v_after:.9e} "
                                f"theta={worst_fp['theta']:.9e}"
                            )
                        continue

                    # Fallback: if the correction cannot be used (legacy path)
                    # or fails to remove the crossing after the bounded number
                    # of corrections, retain the proven timestep-retry path.
                    if profile_v11:
                        print(
                            "[growth-attempt] "
                            f"step={step_number} "
                            f"candidate={candidate_attempt} "
                            f"growth_try={_growth_retry_number} "
                            f"N={len(coupled_idx_new)} "
                            f"dt={dt_use:.8e} "
                            f"wall={_growth_attempt_elapsed:.3f}s "
                            f"factor={_last_hybrid_factor_wall:.3f}s "
                            f"nonlinear={_last_hybrid_nonlinear_wall:.3f}s "
                            f"nfev={_last_hybrid_nfev} "
                            "status=reversal_reject "
                            f"n_reversal={len(idx_fix)}"
                        )

                    growth_dt_retry += 1
                    if growth_dt_retry > max_growth_dt_retries:
                        raise RuntimeError(
                            "Unable to eliminate a resolved kinetic velocity "
                            "reversal after fixed-point correction and "
                            f"{max_growth_dt_retries} timestep retries at "
                            f"t={t:.8e}; candidate dt={dt_use:.8e}."
                        )

                    dt_reduced = growth_dt_retry_safety * dt_use
                    if not np.isfinite(dt_reduced) or dt_reduced <= 0.0:
                        raise RuntimeError(
                            "Invalid reduced timestep after kinetic reversal: "
                            f"dt={dt_reduced}."
                        )

                    dt_use = min(dt_reduced, t_end - t)
                    restart_candidate_after_reversal = True
                    break

                # A fallback reversal rejection changes dt and must restart the
                # complete candidate loop from the accepted pre-step state.
                if restart_candidate_after_reversal:
                    continue

                # The fixed-point corrector may have changed R_new after the
                # original candidate-volume calculation.  Refresh geometry-
                # dependent bookkeeping before nucleation limiting/output.
                if fixed_point_corrections > 0:
                    particle_volume_new = np.sum(particle_volumes(R_new))
                    if particle_volume_new >= box_volume:
                        raise RuntimeError(
                            "Particles fill the simulation box after kinetic "
                            f"fixed-point correction at t={t + dt_use:.6g}."
                        )
                    alive_new = R_new > 0.0
                    coupled_new = alive_new.copy()
                    coupled_idx_new = np.where(coupled_new)[0]

                # Check the fully corrected candidate before accepting growth
                # or proceeding to nucleation.
                _check_hybrid_radius_validity(
                    R_new,
                    where="post-growth candidate",
                    time_value=float(t + dt_use),
                )

                accepted_growth_nfev = (
                    None
                    if _last_hybrid_nfev is None
                    else int(_last_hybrid_nfev)
                )

                if profile_v11:
                    print(
                        "[growth-attempt] "
                        f"step={step_number} "
                        f"candidate={candidate_attempt} "
                        f"growth_try={_growth_retry_number} "
                        f"N={len(coupled_idx_new)} "
                        f"dt={dt_use:.8e} "
                        f"wall={_growth_attempt_elapsed:.3f}s "
                        f"factor={_last_hybrid_factor_wall:.3f}s "
                        f"nonlinear={_last_hybrid_nonlinear_wall:.3f}s "
                        f"nfev={_last_hybrid_nfev} "
                        "status=accept"
                    )

            else:
                state_after_growth = None

            if state_after_growth is not None and not hybrid_transport:
                cbar_store[:] = _coupled_result_in_species_order(
                    state_after_growth, "cbar"
                )
                if is_legacy_mg_si:
                    cbar_mg_store = float(cbar_store[mg_index])
                    cbar_si_store = float(cbar_store[si_index])

            (
                Q_full,
                c_surf_full,
                _,
                _,
            ) = build_full_state(
                len(R_new),
                coupled_idx_new,
                state_after_growth,
            )

            # ----------------------------------------------------------
            # Check the nucleation timestep using the candidate
            # post-growth state.
            #
            # This must use the same:
            #   - concentrations
            #   - radii
            #   - particle positions
            #   - grain-boundary geometry
            #   - physical time
            #
            # as the stochastic nucleation routine below.
            # ----------------------------------------------------------
            if enable_nucleation:
                V_matrix_candidate = (box_volume- particle_volume_new)
                alive_idx_new = np.where(alive_new)[0]
                
                if len(alive_idx_new) > 0:
                    centers_candidate = positions[alive_idx_new]
                    radii_candidate = R_new[alive_idx_new]
                    nuc_kind_candidate = nuc_kind[alive_idx_new]
                
                    Q_candidate = Q_full[:, alive_idx_new]
                
                else:
                    centers_candidate = np.zeros(
                        (0, 3),
                        dtype=float,
                    )
                
                    radii_candidate = np.zeros(
                        (0,),
                        dtype=float,
                    )
                
                    nuc_kind_candidate = np.zeros(
                        (0,),
                        dtype=np.int32,
                    )
                
                    Q_candidate = np.zeros(
                        (n_solutes, 0),
                        dtype=float,
                    )

                _pgb0 = time.perf_counter() if profile_v11 else None
                hybrid_gb_concentration_evaluator = None
                gb_field_diag = None
                if hybrid_transport:
                    gb_short_operator = get_hybrid_short_operator(
                        alive_idx_new
                    )
                    gb_Q = np.asarray(Q_candidate, dtype=float)

                    # Diagnostic accumulator for the actual GB sample points.
                    # All arrays use the canonical species ordering (S, M).
                    gb_field_diag = {
                        "lr": [],
                        "short": [],
                        "total": [],
                    }

                    def hybrid_gb_concentration_evaluator(points):
                        points = np.asarray(points, dtype=float)
                        c_lr = np.asarray(
                            hybrid_long_range_field.sample_at_points(points),
                            dtype=float,
                        )
                        c_short = np.asarray(
                            gb_short_operator.evaluate_at_points(
                                points,
                                gb_Q,
                            ),
                            dtype=float,
                        )

                        # LR sampler and SR operator both return (S, M).
                        if c_lr.shape[0] != n_solutes:
                            raise ValueError(
                                "Hybrid LR GB sampler returned an unexpected "
                                f"species axis: {c_lr.shape}; expected "
                                f"({n_solutes}, M)."
                            )
                        if c_short.shape != c_lr.shape:
                            raise ValueError(
                                "Hybrid SR/LR GB field shapes do not match: "
                                f"LR={c_lr.shape}, SR={c_short.shape}."
                            )

                        c_total = c_lr + c_short

                        gb_field_diag["lr"].append(c_lr.copy())
                        gb_field_diag["short"].append(c_short.copy())
                        gb_field_diag["total"].append(c_total.copy())

                        # Generic GB sampler expects (M, S).
                        return c_total.T

                gb_enabled = (
                    gb_point is not None
                    and gb_normal is not None
                    and gb_thickness is not None
                    and gb_thickness > 0.0
                )
                if gb_enabled:
                    gb_comp = estimate_gb_matrix_composition_species(
                        n_samples=50,
                        centers=centers_candidate,
                        radii=radii_candidate,
                        Q=Q_candidate,
                        cbar=cbar_store,
                        box=box,
                        alpha=alpha,
                        real_lattice=real_lattice,
                        kvecs=kvecs,
                        k2=k2,
                        gb_point=gb_point,
                        gb_normal=gb_normal,
                        gb_thickness=gb_thickness,
                        rng=rng,
                        concentration_evaluator=(
                            hybrid_gb_concentration_evaluator
                        ),
                    )
                    c_gb_store = np.asarray(
                        gb_comp["c_gb_mean"], dtype=float
                    ).copy()
                    if c_gb_store.shape != (n_solutes,):
                        raise ValueError(
                            "Generic GB sampler returned c_gb_mean with "
                            f"shape {c_gb_store.shape}; expected "
                            f"({n_solutes},)."
                        )

                    if (
                        profile_v11
                        and hybrid_transport
                        and gb_field_diag is not None
                        and gb_field_diag["total"]
                    ):
                        c_lr_diag = np.concatenate(
                            gb_field_diag["lr"], axis=1
                        )
                        c_short_diag = np.concatenate(
                            gb_field_diag["short"], axis=1
                        )
                        c_total_diag = np.concatenate(
                            gb_field_diag["total"], axis=1
                        )

                        n_eval_gb = int(c_total_diag.shape[1])
                        positive_mask = np.all(
                            np.isfinite(c_total_diag)
                            & (c_total_diag > 0.0),
                            axis=0,
                        )
                        n_positive_gb = int(
                            np.count_nonzero(positive_mask)
                        )
                        n_nonpositive_gb = int(
                            n_eval_gb - n_positive_gb
                        )

                        def _species_triplet(arr, species_index):
                            vals = np.asarray(
                                arr[species_index], dtype=float
                            )
                            finite = vals[np.isfinite(vals)]
                            if finite.size == 0:
                                return (np.nan, np.nan, np.nan)
                            return (
                                float(np.min(finite)),
                                float(np.mean(finite)),
                                float(np.max(finite)),
                            )

                        alive_radii_diag = np.asarray(
                            radii_candidate, dtype=float
                        )
                        if alive_radii_diag.size:
                            Rmin_diag = float(
                                np.min(alive_radii_diag)
                            )
                            Rmax_diag = float(
                                np.max(alive_radii_diag)
                            )
                        else:
                            Rmin_diag = np.nan
                            Rmax_diag = np.nan

                        ell_diag = (
                            float(hybrid_matching_length)
                            if hybrid_matching_length is not None
                            else np.nan
                        )
                        Roverell_min = (
                            Rmin_diag / ell_diag
                            if np.isfinite(ell_diag)
                            and ell_diag > 0.0
                            else np.nan
                        )
                        Roverell_max = (
                            Rmax_diag / ell_diag
                            if np.isfinite(ell_diag)
                            and ell_diag > 0.0
                            else np.nan
                        )

                        print(
                            "[hybrid-gb-field] "
                            f"t_global="
                            f"{diagnostic_time_offset + t + dt_use:.8e}s "
                            f"T={T:.3f}K "
                            f"N={len(radii_candidate)} "
                            f"ell={ell_diag:.6g}nm "
                            f"R/ell=({Roverell_min:.3f},"
                            f"{Roverell_max:.3f}) "
                            f"Neval={n_eval_gb} "
                            f"Npositive={n_positive_gb} "
                            f"Nnonpositive={n_nonpositive_gb} "
                            f"Nvalid_sampler="
                            f"{int(gb_comp.get('n_valid', -1))}"
                        )

                        for s, species_name in enumerate(solute_names):
                            lr_s = _species_triplet(c_lr_diag, s)
                            sr_s = _species_triplet(c_short_diag, s)
                            tot_s = _species_triplet(c_total_diag, s)
                            print(
                                f"[hybrid-gb-field-{species_name}] "
                                f"LR(min/mean/max)=({lr_s[0]:.6e},"
                                f"{lr_s[1]:.6e},{lr_s[2]:.6e}) "
                                f"SR=({sr_s[0]:.6e},{sr_s[1]:.6e},"
                                f"{sr_s[2]:.6e}) "
                                f"TOTAL=({tot_s[0]:.6e},"
                                f"{tot_s[1]:.6e},{tot_s[2]:.6e}) "
                                f"cbar={cbar_store[s]:.6e} "
                                f"cGB={c_gb_store[s]:.6e}"
                            )
                else:
                    c_gb_store = None

                # Legacy diagnostic facade only.  The nucleation calculation
                # itself consumes the canonical species-indexed c_gb_store.
                if is_legacy_mg_si and c_gb_store is not None:
                    c_gb_mg_mean = float(c_gb_store[mg_index])
                    c_gb_si_mean = float(c_gb_store[si_index])
                else:
                    c_gb_mg_mean = None
                    c_gb_si_mean = None
#--------------------------
                
                (
                    dt_lim_nuc,
                    rate_bulk,
                    rate_gb,
                    J_bulk_phase,
                    J_gb_phase,
                    RkBT_bulk_phase,
                    RkBT_gb_phase,
                    event_rate_bulk_phase,
                    event_rate_gb_phase,
                ) = nucleation_dt_limit_species(
                    cbar=np.asarray(cbar_store, dtype=float),
                    cbar_gb=c_gb_store,
                    V_matrix=V_matrix_candidate,
                    dt_try=dt_use,
                    # Use accumulated physical pre-step time for incubation.
                    # The outer thermal-history wrapper supplies
                    # diagnostic_time_offset = global segment start time.
                    # This prevents artificial incubation resets at thermal
                    # segment boundaries and matches the KWN pre-step convention.
                    t=diagnostic_time_offset + t,
                    T=T,
                    cnt_params_by_phase=phase_cnt_params,
                    lam_target=lam_target_current,
                    centers=centers_candidate,
                    radii=radii_candidate,
                    nuc_kind=nuc_kind_candidate,
                    box=box,
                    gb_point=gb_point,
                    gb_normal=gb_normal,
                    gb_thickness=gb_thickness,
                    gb_barrier_factor=barrier_factor,
                    gb_exclusion_buffer=0.0,
                    gb_use_area_fraction=True,
                )

                if profile_v11:
                    _prof["gb"] += time.perf_counter() - _pgb0

                # Multiphase CNT diagnostic: report the phase-resolved reference
                # rates used by the timestep limiter before stochastic spatial placement.
                # Print only on the accepted candidate timestep so retries do not flood
                # the console.
                _cnt_phase_diag = (
                    np.asarray(J_bulk_phase, dtype=float).copy(),
                    np.asarray(J_gb_phase, dtype=float).copy(),
                    np.asarray(RkBT_bulk_phase, dtype=float).copy(),
                    np.asarray(RkBT_gb_phase, dtype=float).copy(),
                    np.asarray(event_rate_bulk_phase, dtype=float).copy(),
                    np.asarray(event_rate_gb_phase, dtype=float).copy(),
                )

                nucleation_requires_retry = (
                    np.isfinite(dt_lim_nuc)
                    and dt_lim_nuc > 0.0
                    and dt_lim_nuc
                    < dt_use * (1.0 - 1.0e-12)
                )
    
                if nucleation_requires_retry:
                    nucleation_dt_retry += 1
    
                    if (
                        nucleation_dt_retry
                        > max_nucleation_dt_retries
                    ):
                        raise RuntimeError(
                            "Unable to obtain a timestep satisfying "
                            "the ternary nucleation limiter after "
                            f"{max_nucleation_dt_retries} retries. "
                            f"Current t={t:.8e}, "
                            f"candidate dt={dt_use:.8e}, "
                            f"nucleation limit="
                            f"{dt_lim_nuc:.8e}, "
                            f"bulk proposal rate="
                            f"{rate_bulk:.8e}, "
                            f"GB proposal rate="
                            f"{rate_gb:.8e}."
                        )
    
                    dt_reduced = (
                        nucleation_dt_safety
                        * dt_lim_nuc
                    )
    
                    # Ensure strict reduction even in the presence of
                    # floating-point roundoff.
                    dt_reduced = min(
                        dt_reduced,
                        np.nextafter(
                            dt_use,
                            0.0,
                        ),
                    )
    
                    dt_reduced = min(
                        dt_reduced,
                        t_end - t,
                    )
    
                    if (
                        not np.isfinite(dt_reduced)
                        or dt_reduced <= 0.0
                    ):
                        raise RuntimeError(
                            "The ternary nucleation limiter produced "
                            "an invalid timestep: "
                            f"dt_reduced={dt_reduced}, "
                            f"dt_lim_nuc={dt_lim_nuc:.8e}, "
                            f"t={t:.8e}."
                        )
    
                    remaining = t_end - t
    
                    if (
                        not np.isfinite(dt_reduced)
                        or dt_reduced <= 0.0
                    ):
                        raise RuntimeError(
                            "The ternary nucleation limiter produced "
                            "an invalid timestep: "
                            f"dt_reduced={dt_reduced:.8e}, "
                            f"dt_lim_nuc={dt_lim_nuc:.8e}, "
                            f"t={t:.8e}."
                        )
    
                    dt_use = dt_reduced
    
                    # Discard the candidate R_new/state_after_growth
                    # and recompute growth from the unchanged old state.
                    continue
    
            # Candidate growth state and nucleation timestep agree.
            break

    
        accepted_this_step = 0
        nuc_out = None
    
        # --------------------------------------------------------------
        # Ternary nucleation
        # --------------------------------------------------------------
        if enable_nucleation:
            alive_idx_new = np.where(
                alive_new
            )[0]

            if len(alive_idx_new) > 0:
                centers_for_nucleation = (
                    positions[alive_idx_new]
                )

                radii_for_nucleation = (
                    R_new[alive_idx_new]
                )

                Q_for_nucleation = (
                    Q_full[:, alive_idx_new]
                )

                nuc_kind_for_nucleation = (
                    nuc_kind[alive_idx_new]
                )
                
            else:
                centers_for_nucleation = np.zeros(
                    (0, 3),
                    dtype=float,
                )
    
                radii_for_nucleation = np.zeros(
                    (0,),
                    dtype=float,
                )
    
                Q_for_nucleation = np.zeros(
                    (n_solutes, 0),
                    dtype=float,
                )
    
                nuc_kind_for_nucleation = np.zeros(
                    (0,),
                    dtype=np.int32,
                )

            _pcnt0 = time.perf_counter() if profile_v11 else None

            hybrid_candidate_concentration_evaluator = None
            if hybrid_transport:
                candidate_short_operator = get_hybrid_short_operator(alive_idx_new)
                candidate_Q = np.asarray(Q_for_nucleation, dtype=float)

                def hybrid_candidate_concentration_evaluator(points):
                    points = np.asarray(points, dtype=float)
                    c_lr = np.asarray(
                        hybrid_long_range_field.sample_at_points(points), dtype=float
                    )
                    c_short = np.asarray(
                        candidate_short_operator.evaluate_at_points(points, candidate_Q),
                        dtype=float,
                    )
                    # Transport helpers use (S,M); stochastic CNT expects (M,S).
                    return (c_lr + c_short).T

            # Phase-competitive stochastic nucleation. Each phase has an
            # independent Poisson stream and phase-specific CNT physics.
            nuc_outputs_by_phase = []
            work_centers_nuc = centers_for_nucleation.copy()
            work_radii_nuc = radii_for_nucleation.copy()
            work_Q_nuc = Q_for_nucleation.copy()
            work_kind_nuc = nuc_kind_for_nucleation.copy()
            matrix_volume_for_nucleation = (
                box_volume - np.sum(particle_volumes(radii_for_nucleation))
            )

            for p, phase in enumerate(phases):
                cnt_p = phase_cnt_params[p]
                nuc_out_p = nucleate_particles_cnt_monopole_species(
                    centers=work_centers_nuc,
                    radii=work_radii_nuc,
                    Q=work_Q_nuc,
                    box=box,
                    dt=dt_use,
                    # Same physical pre-step CNT clock used by the limiter.
                    t=diagnostic_time_offset + t,
                    T=T,
                    cbar=np.asarray(cbar_store, dtype=float),
                    cbar_gb=c_gb_store,
                    event_rate_bulk=float(event_rate_bulk_phase[p]),
                    event_rate_gb=float(event_rate_gb_phase[p]),
                    J_ref_bulk=float(J_bulk_phase[p]),
                    J_ref_gb=float(J_gb_phase[p]),
                    RkBT_ref_bulk=float(RkBT_bulk_phase[p]),
                    RkBT_ref_gb=float(RkBT_gb_phase[p]),
                    aa=cnt_p["aa"],
                    bb=cnt_p["bb"],
                    cc=cnt_p["cc"],
                    vatm=cnt_p["vatm"],
                    vatp=cnt_p["vatp"],
                    gamma=cnt_p["gamma"],
                    N0=cnt_p["N0"],
                    D=cnt_p["D"],
                    a_latt=cnt_p["a_latt"],
                    alpha=alpha,
                    real_lattice=real_lattice,
                    kvecs=kvecs,
                    k2=k2,
                    stoichiometric_weights=cnt_p["stoichiometric_weights"],
                    solvent_stoichiometric_weight=cnt_p[
                        "solvent_stoichiometric_weight"
                    ],
                    apply_incubation_correction=cnt_p.get(
                        "apply_incubation_correction", True
                    ),
                    lam_target=lam_target_current,
                    matrix_volume=matrix_volume_for_nucleation,
                    safety_buffer=cnt_p.get("safety_buffer", 0.0),
                    rng=phase_nucleation_rng[p],
                    gb_point=gb_point,
                    gb_normal=gb_normal,
                    gb_thickness=gb_thickness,
                    gb_barrier_factor=cnt_p["gb_barrier_factor"],
                    nuc_kind=work_kind_nuc,
                    gb_exclusion_buffer=0.0,
                    gb_use_area_fraction=True,
                    candidate_sampling_alpha=NUCLEATION_CANDIDATE_SAMPLING_ALPHA,
                    candidate_sampling_min_pool=NUCLEATION_CANDIDATE_MIN_POOL,
                    candidate_sampling_ess_per_event=nucleation_placement_ess_per_event,
                    candidate_sampling_min_ess=NUCLEATION_CANDIDATE_MIN_ESS,
                    candidate_sampling_max_alpha=NUCLEATION_CANDIDATE_MAX_ALPHA,
                    candidate_sampling_max_pool=NUCLEATION_CANDIDATE_MAX_POOL,
                    spatial_rate_correction=nucleation_spatial_rate_correction,
                    spatial_rate_correction_max_factor=NUCLEATION_SPATIAL_RATE_CORRECTION_MAX_FACTOR,
                    spatial_rate_sampling_rse_target=nucleation_spatial_rate_rse_target,
                    spatial_rate_sampling_min_samples=NUCLEATION_SPATIAL_RATE_MIN_SAMPLES,
                    spatial_rate_sampling_max_samples=NUCLEATION_SPATIAL_RATE_MAX_SAMPLES,
                    spatial_rate_probe_hazard=NUCLEATION_SPATIAL_RATE_PROBE_HAZARD,
                    spatial_rate_skipped_hazard_bulk=float(
                        spatial_rate_skipped_hazard_bulk[p]
                    ),
                    spatial_rate_skipped_hazard_gb=float(
                        spatial_rate_skipped_hazard_gb[p]
                    ),
                    spatial_rate_last_factor_bulk=float(
                        spatial_rate_last_factor_bulk[p]
                    ),
                    spatial_rate_last_factor_gb=float(
                        spatial_rate_last_factor_gb[p]
                    ),
                    candidate_concentration_evaluator=(
                        hybrid_candidate_concentration_evaluator
                    ),
                    diagnostics=bool(profile_v11),
                )

                spatial_rate_skipped_hazard_bulk[p] = float(
                    nuc_out_p.get("spatial_rate_skipped_hazard_bulk", 0.0)
                )
                spatial_rate_skipped_hazard_gb[p] = float(
                    nuc_out_p.get("spatial_rate_skipped_hazard_gb", 0.0)
                )
                spatial_rate_last_factor_bulk[p] = float(
                    nuc_out_p.get(
                        "spatial_rate_last_factor_bulk",
                        spatial_rate_last_factor_bulk[p],
                    )
                )
                spatial_rate_last_factor_gb[p] = float(
                    nuc_out_p.get(
                        "spatial_rate_last_factor_gb",
                        spatial_rate_last_factor_gb[p],
                    )
                )

                append_nucleation_diagnostic_rows(
                    phase_output=nuc_out_p,
                    J_bulk_ref=float(J_bulk_phase[p]),
                    J_gb_ref=float(J_gb_phase[p]),
                    phase_index=p,
                    c_bulk=cbar_store,
                    c_gb=c_gb_store,
                    rate_bulk_ref=float(event_rate_bulk_phase[p]),
                    rate_gb_ref=float(event_rate_gb_phase[p]),
                    gb_barrier_factor_used=cnt_p["gb_barrier_factor"],
                )

                nuc_outputs_by_phase.append((p, nuc_out_p))

                nacc_p = int(nuc_out_p["n_accepted"])
                if nacc_p > 0:
                    accepted_centers_p = np.asarray(
                        nuc_out_p["new_centers"], dtype=float
                    )
                    accepted_radii_p = np.asarray(
                        nuc_out_p["new_radii"], dtype=float
                    )
                    accepted_kind_p = np.asarray(
                        nuc_out_p["new_kind"], dtype=np.int32
                    )
                    work_centers_nuc = np.vstack(
                        [work_centers_nuc, accepted_centers_p]
                    )
                    work_radii_nuc = np.concatenate(
                        [work_radii_nuc, accepted_radii_p]
                    )
                    work_kind_nuc = np.concatenate(
                        [work_kind_nuc, accepted_kind_p]
                    )
                    # New nuclei are overlap obstacles for later phases but do
                    # not acquire a field strength until the coupled post-solve.
                    work_Q_nuc = np.hstack(
                        [
                            work_Q_nuc,
                            np.zeros((n_solutes, nacc_p), dtype=float),
                        ]
                    )

            Ntrial_bulk = int(sum(int(o["Ntrial_bulk"]) for _, o in nuc_outputs_by_phase))
            Ntrial_gb = int(sum(int(o["Ntrial_gb"]) for _, o in nuc_outputs_by_phase))
            Ntrial = Ntrial_bulk + Ntrial_gb
            n_accepted_total = int(
                sum(int(o["n_accepted"]) for _, o in nuc_outputs_by_phase)
            )

            phase_parts = []
            center_parts = []
            radius_parts = []
            kind_parts = []
            c_parts = []
            J_parts = []
            mask_parts = []
            for p, o in nuc_outputs_by_phase:
                nacc_p = int(o["n_accepted"])
                if nacc_p > 0:
                    phase_parts.append(np.full(nacc_p, p, dtype=np.int32))
                    center_parts.append(np.asarray(o["new_centers"], dtype=float))
                    radius_parts.append(np.asarray(o["new_radii"], dtype=float))
                    kind_parts.append(np.asarray(o["new_kind"], dtype=np.int32))
                    c_p = np.asarray(o["c_local"], dtype=float)
                    if c_p.shape != (n_solutes, nacc_p):
                        raise RuntimeError(
                            "Species-indexed nucleation output has unexpected "
                            f"shape for phase {p}: {c_p.shape}."
                        )
                    c_parts.append(c_p)
                    J_parts.append(np.asarray(o["J_local"], dtype=float))
                mask_parts.append(np.asarray(o["accepted_mask"], dtype=bool))

            if n_accepted_total > 0:
                new_centers_all = np.vstack(center_parts)
                new_radii_all = np.concatenate(radius_parts)
                new_kind_all = np.concatenate(kind_parts)
                new_phase_id_all = np.concatenate(phase_parts)
                c_local_all = np.hstack(c_parts)
                J_local_all = np.concatenate(J_parts)
            else:
                new_centers_all = np.zeros((0, 3), dtype=float)
                new_radii_all = np.zeros(0, dtype=float)
                new_kind_all = np.zeros(0, dtype=np.int32)
                new_phase_id_all = np.zeros(0, dtype=np.int32)
                c_local_all = np.zeros((n_solutes, 0), dtype=float)
                J_local_all = np.zeros(0, dtype=float)

            nuc_out = {
                "Ntrial": Ntrial,
                "Ntrial_bulk": Ntrial_bulk,
                "Ntrial_gb": Ntrial_gb,
                "Ncandidate": int(sum(int(o.get("Ncandidate", 0)) for _, o in nuc_outputs_by_phase)),
                "Ncandidate_bulk": int(sum(int(o.get("Ncandidate_bulk", 0)) for _, o in nuc_outputs_by_phase)),
                "Ncandidate_gb": int(sum(int(o.get("Ncandidate_gb", 0)) for _, o in nuc_outputs_by_phase)),
                "n_accepted": n_accepted_total,
                "new_centers": new_centers_all,
                "new_radii": new_radii_all,
                "new_kind": new_kind_all,
                "new_phase_id": new_phase_id_all,
                "c_local": c_local_all,
                "c_local_mg": (
                    c_local_all[mg_index] if is_legacy_mg_si else None
                ),
                "c_local_si": (
                    c_local_all[si_index] if is_legacy_mg_si else None
                ),
                "J_local": J_local_all,
                "accepted_mask": (
                    np.concatenate(mask_parts) if mask_parts
                    else np.zeros(0, dtype=bool)
                ),
                "lambda_bulk": float(np.sum(event_rate_bulk_phase) * dt_use),
                "lambda_gb": float(np.sum(event_rate_gb_phase) * dt_use),
                "lambda_mean": float(
                    (np.sum(event_rate_bulk_phase) + np.sum(event_rate_gb_phase))
                    * dt_use
                ),
                "dt_used": dt_use,
            }

            if profile_v11:
                _prof["cnt"] += time.perf_counter() - _pcnt0

            # This should now be a genuine internal consistency check.
            if not np.isclose(
                nuc_out["dt_used"],
                dt_use,
                rtol=1.0e-10,
                atol=0.0,
            ):
                raise RuntimeError(
                    "The ternary nucleation routine still reduced "
                    "the timestep after the post-growth limiter was "
                    "applied. The limiter and nucleation routine are "
                    "not using identical proposal-rate calculations. "
                    f"growth dt={dt_use:.12e}, "
                    f"nucleation dt="
                    f"{nuc_out['dt_used']:.12e}, "
                    f"lambda_mean="
                    f"{nuc_out['lambda_mean']:.12e}, "
                    f"lam_target={lam_target_current:.12e}."
                    f"crowding={crowding_parameter:.6e}."
                )
                
            Ntrial = int(nuc_out["Ntrial"])
        
            acceptance_fraction = (
                nuc_out["n_accepted"] / Ntrial
                if Ntrial > 0
                else np.nan
            )
        
            if nuc_out["n_accepted"] > 0:
                print(
                    f"NUCLEATION: "
                    f"trials={Ntrial:4d}, "
                    f"bulk={int(nuc_out['Ntrial_bulk']):4d}, "
                    f"gb={int(nuc_out['Ntrial_gb']):4d}, "
                    f"accepted={nuc_out['n_accepted']:3d}, "
                    f"acceptance={acceptance_fraction:.3e}"
               )

                Jb_diag, Jg_diag, Rb_diag, Rg_diag, Eb_diag, Eg_diag = _cnt_phase_diag
                for p, phase in enumerate(phases):
                    phase_name = getattr(phase, "name", f"phase_{p}")
                    lambda_bulk_p = float(Eb_diag[p]) * float(dt_use)
                    lambda_gb_p = float(Eg_diag[p]) * float(dt_use)
                    print(
                        f"    CNT phase {p} ({phase_name}): "
                        f"Jbulk={Jb_diag[p]:.6e}, "
                        f"Jgb={Jg_diag[p]:.6e}, "
                        f"rate_bulk={Eb_diag[p]:.6e} s^-1, "
                        f"rate_gb={Eg_diag[p]:.6e} s^-1, "
                        f"lambda_bulk={lambda_bulk_p:.6e}, "
                        f"lambda_gb={lambda_gb_p:.6e}, "
                        f"RkBT_bulk={Rb_diag[p]:.6e} nm, "
                        f"RkBT_gb={Rg_diag[p]:.6e} nm"
                    )
            
            nuc_time.append(t + dt_use)
        
            Ntrial_hist.append([
                nuc_out["Ntrial"],
                nuc_out["Ntrial_bulk"],
                nuc_out["Ntrial_gb"],
            ])
        
            accepted_this_step = int(nuc_out["n_accepted"])
        
            Nacc_hist.append(accepted_this_step)
        
            Jnuc.append(nuc_out["J_local"])
        
            Rnuc.append(nuc_out["new_radii"])
        
            cnuc_mg.append(nuc_out["c_local_mg"])
        
            cnuc_si.append(nuc_out["c_local_si"])
    

            if accepted_this_step > 0:
                # Preserve the converged post-growth coupled solution
                # for warm-starting the post-nucleation solve.
                coupled_idx_before_nucleation = (
                    coupled_idx_new.copy()
                )

                result_before_nucleation = (
                    state_after_growth
                )

                new_centers = np.asarray(
                    nuc_out["new_centers"],
                    dtype=float,
                )

                new_radii = np.asarray(
                    nuc_out["new_radii"],
                    dtype=float,
                )

                new_kind = np.asarray(
                    nuc_out["new_kind"],
                    dtype=np.int32,
                )

                #n_new = len(new_radii)

                old_positions = positions.copy()

                # ------------------------------------------------------
                # Append the newly accepted nuclei
                # ------------------------------------------------------
                R_new = np.concatenate(
                    [
                        R_new,
                        new_radii,
                    ]
                )

                nuc_kind = np.concatenate(
                    [
                        nuc_kind,
                        new_kind,
                    ]
                )

                new_phase_id = np.asarray(
                    nuc_out["new_phase_id"], dtype=np.int32
                )
                if new_phase_id.shape != new_radii.shape:
                    raise RuntimeError(
                        "new_phase_id and new_radii must have the same shape "
                        "after multiphase nucleation."
                    )
                phase_id = np.concatenate([phase_id, new_phase_id])

                # ------------------------------------------------------
                # Extend dense Green data only on the direct transport path.
                #
                # Scale-separated growth uses the sparse short-range operator. The
                # existing CNT candidate/acceptance kernel is retained, but accepted
                # nuclei must not cause construction or extension of the dense N x N
                # growth matrix.
                # ------------------------------------------------------
                box = as_box_lengths(box)
                box_volume = float(np.prod(box))

                if not hybrid_transport:
                    recip_weights = green_kernel[
                        "reciprocal_weights"
                    ]
                    const_term = green_kernel[
                        "constant_term"
                    ]
                    Gpair_full = extend_pair_green_matrix(
                        Gpair_full,
                        old_positions=old_positions,
                        new_positions=new_centers,
                        box=box,
                        alpha=alpha,
                        kappa=screening_kappa,
                        real_lattice=real_lattice,
                        kvecs=kvecs,
                        k2=k2,
                        recip_weights=recip_weights,
                        const_term=const_term,
                    )

                positions = np.vstack(
                    [
                        old_positions,
                        new_centers,
                    ]
                )

                Ntot = len(R_new)

                # ------------------------------------------------------
                # Check the post-nucleation particle volume
                # ------------------------------------------------------
                particle_volume_new = np.sum(
                    particle_volumes(R_new)
                )

                if particle_volume_new >= box_volume:
                    raise RuntimeError(
                        "Particles fill the simulation box after "
                        f"nucleation at t={t + dt_use:.6g}."
                    )

                # ------------------------------------------------------
                # Recompute conserved matrix concentrations
                # ------------------------------------------------------
                alive_new = R_new > 0.0

                if hybrid_transport:
                    # Accepted nuclei enter with finite radius.  Withdraw their
                    # complete precipitate inventory from the transient LR
                    # matrix immediately, then update the interpolation geometry.
                    hybrid_long_range_field.update_positions(positions)

                    n_old = len(R_new) - len(new_radii)
                    delta_nucleus_inventory = np.zeros(
                        (n_solutes, len(R_new)),
                        dtype=float,
                    )
                    Vn = particle_volumes(new_radii)
                    new_phase_ids = phase_id[n_old:]
                    delta_nucleus_inventory[:, n_old:] = (
                        cp_phase[new_phase_ids].T
                        * Vn[None, :]
                    )

                    hybrid_long_range_field.apply_particle_inventory_change(
                        delta_nucleus_inventory,
                        sign=-1.0,
                    )
                    cbar_store = np.asarray(
                        hybrid_long_range_field.mean_concentration(),
                        dtype=float,
                    )
                else:
                    cbar_store = mean_concentrations(
                        R_new,
                        phase_id,
                    )

                if is_legacy_mg_si:
                    cbar_mg_store = float(cbar_store[mg_index])
                    cbar_si_store = float(cbar_store[si_index])

                # Every surviving particle participates in the coupled solve.
                coupled_new = alive_new.copy()
                

                coupled_idx_new = np.where(
                    coupled_new
                )[0]

                # ------------------------------------------------------
                # Solve only the coupled-active particles
                #
                # Preserve solutions for particles that were already
                # coupled-active before nucleation. Any newly coupled
                # particles receive newly initialized guesses.
                # ------------------------------------------------------
                if len(coupled_idx_new) > 0:
                    _pt0 = time.perf_counter() if profile_v11 else None

                    # --------------------------------------------------
                    # Post-nucleation coupled solve:
                    # one-shot first, population-continuation fallback.
                    #
                    # Most nucleation bursts converge cleanly when the full
                    # accepted population is solved in one warm-started step.
                    # Keep that fast path unchanged. Only if the full solve
                    # fails the strict coupled-interface acceptance checks do
                    # we retry by activating newly accepted nuclei in small
                    # deterministic batches.
                    #
                    # This fallback changes only the numerical continuation
                    # path. The accepted nuclei, their positions/radii/phase
                    # IDs, and the hybrid LR inventory withdrawal have already
                    # been committed above and are not resampled or altered.
                    # --------------------------------------------------
                    _old_idx = np.asarray(
                        coupled_idx_before_nucleation, dtype=int
                    )
                    _target_idx = np.asarray(
                        coupled_idx_new, dtype=int
                    )

                    try:
                        # Fast/default path: exactly the original one-shot
                        # post-nucleation solve.
                        state_after_growth = solve_active_state(
                            R_full=R_new,
                            active_idx=_target_idx,
                            old_active_idx=_old_idx,
                            old_result=result_before_nucleation,
                            force_continuation=True,
                            dt_for_velocity_tolerance=dt_use,
                        )

                    except RuntimeError as _post_nuc_exc:
                        # Do not hide unrelated physical/inventory/programming
                        # failures. The fallback is specifically for the
                        # marginal nonlinear acceptance failure seen after
                        # large stochastic nucleation bursts.
                        if (
                            "Coupled interface solve failed acceptance checks"
                            not in str(_post_nuc_exc)
                        ):
                            raise

                        post_nuc_batch_size = 25

                        _old_set = set(_old_idx.tolist())
                        _added_idx = np.asarray(
                            [
                                idx for idx in _target_idx.tolist()
                                if idx not in _old_set
                            ],
                            dtype=int,
                        )

                        if _added_idx.size == 0:
                            # Same population failed: there is no population
                            # continuation path available.
                            raise

                        _n_batches = int(
                            np.ceil(
                                _added_idx.size
                                / float(post_nuc_batch_size)
                            )
                        )

                        print(
                            "[post-nuc-fallback] "
                            "one-shot coupled solve failed acceptance; "
                            "retrying with population continuation: "
                            f"Nold={_old_idx.size} "
                            f"Nnew={_added_idx.size} "
                            f"Ntarget={_target_idx.size} "
                            f"batch_size={post_nuc_batch_size} "
                            f"Nbatches={_n_batches}"
                        )

                        _stage_idx = _old_idx.copy()
                        _stage_result = result_before_nucleation

                        for _batch_number, _start in enumerate(
                            range(
                                0,
                                _added_idx.size,
                                post_nuc_batch_size,
                            ),
                            start=1,
                        ):
                            _stop = min(
                                _start + post_nuc_batch_size,
                                _added_idx.size,
                            )
                            _batch_idx = _added_idx[_start:_stop]

                            _next_stage_idx = np.concatenate(
                                [_stage_idx, _batch_idx]
                            )

                            _next_stage_result = solve_active_state(
                                R_full=R_new,
                                active_idx=_next_stage_idx,
                                old_active_idx=_stage_idx,
                                old_result=_stage_result,
                                force_continuation=True,
                                dt_for_velocity_tolerance=dt_use,
                            )

                            print(
                                "[post-nuc-fallback] "
                                f"batch={_batch_number}/{_n_batches} "
                                f"Nsolve={_next_stage_idx.size} "
                                f"Nadded={_batch_idx.size} "
                                f"nfev={getattr(_next_stage_result, 'nfev', None)}"
                            )

                            _stage_idx = _next_stage_idx
                            _stage_result = _next_stage_result

                        # Normally the newly nucleated particles are appended
                        # and coupled_idx_new is already old+new in this exact
                        # order. Retain a defensive final mapped solve if a
                        # future refactor changes that ordering.
                        if (
                            _stage_idx.shape != _target_idx.shape
                            or not np.array_equal(
                                _stage_idx, _target_idx
                            )
                        ):
                            _stage_result = solve_active_state(
                                R_full=R_new,
                                active_idx=_target_idx,
                                old_active_idx=_stage_idx,
                                old_result=_stage_result,
                                force_continuation=True,
                                dt_for_velocity_tolerance=dt_use,
                            )

                        state_after_growth = _stage_result

                        print(
                            "[post-nuc-fallback] "
                            "population continuation converged; "
                            f"N={_target_idx.size}"
                        )

                    if profile_v11:
                        _prof["post_nuc_solve"] += time.perf_counter() - _pt0

                else:
                    state_after_growth = None
#----------------
                # Replace the reduced-inventory guesses with the complete
                # field-conserving mean concentrations returned by the solver.
                if state_after_growth is not None and not hybrid_transport:
                    cbar_store[:] = _coupled_result_in_species_order(
                        state_after_growth, "cbar"
                    )
                    if is_legacy_mg_si:
                        cbar_mg_store = float(cbar_store[mg_index])
                        cbar_si_store = float(cbar_store[si_index])
#----------------

                
                (
                    Q_full,
                    c_surf_full,
                    _,
                    _,
                ) = build_full_state(
                    len(R_new),
                    coupled_idx_new,
                    state_after_growth,
                )
                    
        # --------------------------------------------------------------
        # Deterministic hybrid N -> N+1 regression insertion
        # --------------------------------------------------------------
        if (
            hybrid_transport
            and hybrid_forced_nucleus is not None
            and not forced_nucleus_inserted
            and forced_nucleus_time is not None
            and np.isclose(
                t + dt_use,
                forced_nucleus_time,
                rtol=0.0,
                atol=max(1.0e-12, 1.0e-12 * max(1.0, abs(forced_nucleus_time))),
            )
        ):
            coupled_idx_before_insertion = coupled_idx_new.copy()
            result_before_insertion = state_after_growth

            new_position = np.asarray(forced_nucleus_position, dtype=float)[None, :]
            new_radius = float(forced_nucleus_radius)
            new_phase = int(forced_nucleus_phase_id)

            positions = np.vstack([positions, new_position])
            R_new = np.concatenate([R_new, np.array([new_radius], dtype=float)])
            phase_id = np.concatenate(
                [phase_id, np.array([new_phase], dtype=np.int32)]
            )
            nuc_kind = np.concatenate(
                [nuc_kind, np.array([0], dtype=np.int32)]
            )
            Ntot = len(R_new)

            # Extend LR geometry and withdraw the complete finite inventory
            # of the newly created precipitate from the matrix field.
            hybrid_long_range_field.update_positions(positions)

            Vn = 4.0 * np.pi * new_radius**3 / 3.0
            delta_nucleus_inventory = np.zeros((n_solutes, Ntot), dtype=float)
            delta_nucleus_inventory[:, -1] = cp_phase[new_phase] * Vn

            hybrid_long_range_field.apply_particle_inventory_change(
                delta_nucleus_inventory,
                sign=-1.0,
            )

            cbar_store = np.asarray(
                hybrid_long_range_field.mean_concentration(),
                dtype=float,
            )
            if is_legacy_mg_si:
                cbar_mg_store = float(cbar_store[mg_index])
                cbar_si_store = float(cbar_store[si_index])

            alive_new = R_new > 0.0
            coupled_new = alive_new.copy()

            coupled_idx_new = np.where(coupled_new)[0]

            if len(coupled_idx_new) > 0:
                state_after_growth = solve_active_state(
                    R_full=R_new,
                    active_idx=coupled_idx_new,
                    old_active_idx=coupled_idx_before_insertion,
                    old_result=result_before_insertion,
                    force_continuation=False,
                    dt_for_velocity_tolerance=dt_use,
                )
            else:
                state_after_growth = None

            if state_after_growth is not None and not hybrid_transport:
                cbar_store[:] = _coupled_result_in_species_order(
                    state_after_growth, "cbar"
                )
                if is_legacy_mg_si:
                    cbar_mg_store = float(cbar_store[mg_index])
                    cbar_si_store = float(cbar_store[si_index])

            Q_full, c_surf_full, _, _ = build_full_state(
                len(R_new),
                coupled_idx_new,
                state_after_growth,
            )

            forced_nucleus_inserted = True

        # --------------------------------------------------------------
        # Accept timestep
        # --------------------------------------------------------------
        
        # --------------------------------------------------------------
        # Update nonlinear-solver state using this ACCEPTED physical
        # timestep.
        # --------------------------------------------------------------

        if dense_steps_remaining > 0:

            # This timestep was inside a deliberately forced dense window.
            # Count down only when a physical timestep is actually accepted.
            dense_steps_remaining -= 1

            # A forced-dense timestep is not an LSMR failure, so do not
            # carry the old consecutive-failure count through the window.
            consecutive_lsmr_failures = 0

        else:

            # We were in normal automatic mode.
            if dense_fallback_seen_this_step:
                consecutive_lsmr_failures += 1
            else:
                consecutive_lsmr_failures = 0

            # Repeated LSMR failures indicate a locally difficult regime.
            if (consecutive_lsmr_failures >= lsmr_failures_before_dense_window):
                dense_steps_remaining = dense_window_length
                consecutive_lsmr_failures = 0

                print(
                    "Repeated LSMR continuation failures; "
                    f"using dense continuation for the next "
                    f"{dense_window_length} accepted timesteps."
                )


        # Persist the actual accepted physical timestep so the next outer
        # step cannot rebound arbitrarily far above a value that the coupled
        # solve has just demonstrated it can handle.
        dt_last_accepted = float(dt_use)
        previous_step_had_growth_retry = bool(growth_dt_retry > 0)
        previous_growth_nfev = accepted_growth_nfev

        t += dt_use

        R = R_new

        # --------------------------------------------------------------
        # Accept the particle classifications
        # --------------------------------------------------------------
        alive = alive_new
        coupled = coupled_new

        current_coupled_idx = np.where(
            coupled
        )[0]

        # Coupled solution corresponding to current_coupled_idx.
        current_result = (
            state_after_growth
        )


        store_state(
            t_now=t,
            R_full=R,
            coupled_idx=
                current_coupled_idx,
            result=current_result,
            cbar=cbar_store,
            phase_id_full=phase_id,
            alive_mask=alive,
            coupled_mask=coupled,
        )

        # Count every physically existing particle.
        number_ppt.append(
            int(np.count_nonzero(alive))
        )

        positions_out.append(
            positions.copy()
        )

        if dump_target_times_array is not None:
            prev_global_time=dump_time_offset+dump_prev_t
            current_global_time=dump_time_offset+float(t)
            while dump_target_index < len(dump_target_times_array):
                requested_global_time=float(dump_target_times_array[dump_target_index])
                if requested_global_time > current_global_time + 1.0e-14: break
                if requested_global_time < prev_global_time - 1.0e-14:
                    dump_target_index += 1; continue
                use_previous=abs(requested_global_time-prev_global_time) <= abs(current_global_time-requested_global_time)
                if use_previous:
                    _write_target_dump_state(actual_global_time=prev_global_time,R_state=dump_prev_R,positions_state=dump_prev_positions,phase_id_state=dump_prev_phase_id,nuc_kind_state=dump_prev_nuc_kind,dRdt_state=dump_prev_dRdt,Pe_state=dump_prev_Pe,Q_state=dump_prev_Q,requested_global_time=requested_global_time,lr_field_state=dump_prev_lr_field)
                else:
                    _write_target_dump_state(actual_global_time=current_global_time,R_state=R,positions_state=positions,phase_id_state=phase_id,nuc_kind_state=nuc_kind,dRdt_state=dRdt_list[-1],Pe_state=Pe_list[-1],Q_state=Q_list[-1],requested_global_time=requested_global_time,lr_field_state=None)
                dump_target_index += 1


        # --------------------------------------------------------------
        # Diagnostics and output
        # --------------------------------------------------------------
        if (
            print_diagnostics
            and (
                step_number % print_every == 0
                or step_number == 1
            )
        ):
            alive_idx_now = np.where(
                alive
            )[0]


            coupled_idx_now = np.where(
                coupled
            )[0]

            R_alive_now = R[
                alive_idx_now
            ]

            N_alive_now = len(
                alive_idx_now
            )


            N_coupled_now = len(
                coupled_idx_now
            )

            # Origin populations: 0 = bulk, 1 = grain boundary.
            # Count only physically surviving particles.
            kind_alive_now = np.asarray(
                nuc_kind, dtype=np.int32
            )[alive_idx_now]
            N_bulk_now = int(np.count_nonzero(kind_alive_now == 0))
            N_gb_now = int(np.count_nonzero(kind_alive_now == 1))

            if N_alive_now > 0:
                Rmin = float(
                    np.min(R_alive_now)
                )

                Rave = float(
                    np.mean(R_alive_now)
                )

                Rmax = float(
                    np.max(R_alive_now)
                )

            else:
                Rmin = 0.0
                Rave = 0.0
                Rmax = 0.0

            wall_now = time.perf_counter()
            wall_step = wall_now - wall_step_start

            hybrid_ratio_now = (
                _hybrid_radius_ratio(R) if hybrid_transport else 0.0
            )

            print(
                f"[step {step_number:6d}] "
                f"t={t:.4e}, "
                f"dt={dt_use:.2e}, "
                f"N={N_alive_now:5d}, "
                f"Nbulk={N_bulk_now:5d}, "
                f"NGB={N_gb_now:5d}, "
                f"Ncoupled={N_coupled_now:5d}, "
                f"dN={accepted_this_step:4d}, "
                f"R=({Rmin:.3e}, "
                f"{Rave:.3e}, "
                f"{Rmax:.3e}), "
                + (
                    f"Rmax/ell={hybrid_ratio_now:.3f}, "
                    if hybrid_transport else ""
                )
                + (
                    (
                        f"cbar_mg={cbar_mg_store:.4e}, "
                        f"cbar_si={cbar_si_store:.4e}, "
                    )
                    if is_legacy_mg_si
                    else "".join(
                        f"cbar_{name}={cbar_store[s]:.4e}, "
                        for s, name in enumerate(solute_names)
                    )
                )
                + f"lam={lam_target_current:.2f}, "
                f"chi={crowding_parameter:.3f}, "
                f"dt_hist_cap={dt_history_cap:.2e}, "
                f"dt_hist_factor={dt_history_growth_factor:.2f}, "
                f"dt_hist_reason={dt_history_reason}, "
                f"prev_retry={previous_step_had_growth_retry}, "
                f"growth_nfev={previous_growth_nfev}"
            )


            # Per-phase population/radius diagnostics. Keep this separate
            # from the main step line so the compact step diagnostic remains
            # readable while also reporting useful phase-resolved output.
            phase_id_alive_now = np.asarray(
                phase_id, dtype=np.int32
            )[alive_idx_now]
            for p, phase in enumerate(phases):
                phase_mask_now = phase_id_alive_now == p
                N_phase_now = int(np.count_nonzero(phase_mask_now))
                phase_name_now = getattr(phase, "name", str(p))

                if N_phase_now > 0:
                    R_phase_now = R_alive_now[phase_mask_now]
                    Rmin_phase_now = float(np.min(R_phase_now))
                    Rmean_phase_now = float(np.mean(R_phase_now))
                    Rmax_phase_now = float(np.max(R_phase_now))
                else:
                    Rmin_phase_now = 0.0
                    Rmean_phase_now = 0.0
                    Rmax_phase_now = 0.0

                print(
                    f"    phase {p} ({phase_name_now}): "
                    f"N={N_phase_now:5d}, "
                    f"R=({Rmin_phase_now:.3e}, "
                    f"{Rmean_phase_now:.3e}, "
                    f"{Rmax_phase_now:.3e}) nm"
                )

            if (
                hybrid_transport
                and hybrid_ratio_now >= hybrid_radius_ratio_warning
            ):
                print(
                    "[hybrid-split-warning] "
                    f"Rmax/ell={hybrid_ratio_now:.6f} >= "
                    f"{hybrid_radius_ratio_warning:.2f}; "
                    "hybrid growth is entering the empirically "
                    "sensitivity-prone regime. "
                    f"Hard stop at sqrt(pi)/2="
                    f"{hybrid_radius_ratio_stop:.6f}."
                )

            if step_number % wall_print_every == 0:

                wall_window = wall_now - wall_window_start
                n_window_steps = step_number - wall_window_step_start

                wall_per_step = (
                    wall_window / n_window_steps
                    if n_window_steps > 0
                    else np.nan
                )

                physical_time_advanced = (
                    t - wall_window_t_start
                )

                sim_time_per_wall_second = (
                    physical_time_advanced / wall_window
                    if wall_window > 0.0
                    else np.nan
                )

                if profile_v11:
                    _nprof = max(n_window_steps, 1)
                    _psum = sum(_prof.values()) / _nprof
                    print(
                        "[profile] "
                        + " ".join(f"{k}={v/_nprof:.4f}s/step"
                                   for k, v in _prof.items())
                        + f" measured={_psum:.4f}s/step"
                    )

                if profile_v11:
                    print(
                        f"[timing] wall_step={wall_step:.3f} s, "
                        f"wall_avg({n_window_steps} steps)="
                        f"{wall_per_step:.3f} s/step, "
                        f"sim_rate={sim_time_per_wall_second:.3e} "
                        f"sim-s/wall-s"
                    )

                wall_window_start = wall_now
                wall_window_step_start = step_number
                wall_window_t_start = t
                if profile_v11:
                    for _k in _prof:
                        _prof[_k] = 0.0

#----------------
        if (
            dump_target_times_array is None
            and save_every is not None
            and save_every > 0
            and step_number % save_every == 0
        ):
            print(
                "Saving ternary particle dump file..."
            )

            alive_idx_now = np.where(
                alive
            )[0]

            write_particles_ovito_dump(
                filename=dump_filename,
                step=t,
                positions=positions[
                    alive_idx_now
                ],
                radii=R[
                    alive_idx_now
                ],
                box=box,
                ids=alive_idx_now + 1,
                types=(
                    nuc_kind[
                        alive_idx_now
                    ]
                    + 1
                ),
                phase_ids=phase_id[alive_idx_now],
                gb_flags=nuc_kind[alive_idx_now],
                dRdt=np.asarray(dRdt_list[-1], dtype=float)[alive_idx_now],
                peclet=np.asarray(Pe_list[-1], dtype=float)[:, alive_idx_now],
                peclet_names=solute_names,
                mode=(
                    "a"
                    if step_number > 0
                    else "w"
                ),
            )

    # ------------------------------------------------------------------
    # Additional outputs
    # ------------------------------------------------------------------
    aux = {
        "alpha": alpha,
        "real_lattice": real_lattice,
        "kvecs": kvecs,
        "green_kernel_type": green_kernel["kernel_type"],
        "screening_length": screening_length,
        "screening_kappa": screening_kappa,
        "k2": k2,
        "M_total": M_total.copy(),
        "cbar": cbar_store.copy(),
        "solute_names": solute_names,
        "cp_phase": cp_phase.copy(),

        # Generic species-indexed final/restart state.  These are the
        # preferred fields for callers outside the legacy Mg-Si kernels.
        "Q_final": Q_list[-1].copy(),
        "c_surf_final": c_surf_list[-1].copy(),
        "dRdt_final": np.asarray(dRdt_list[-1], dtype=float).copy(),
        "dRdt_species_final": np.asarray(
            dRdt_species_list[-1], dtype=float
        ).copy(),
        "Pe_final": np.asarray(Pe_list[-1], dtype=float).copy(),
        "phase_id_final": phase_id.copy(),
        "phase_id_hist": phase_id_list,
        "nuc_kind_hist": nuc_kind_out,
        "cbar_final": cbar_store.copy(),
        "M_total": M_total.copy(),
        "M_total_mg": M_total_mg,
        "M_total_si": M_total_si,
        "G0": G0,
        "Gpair_full": Gpair_full,
        "target_dump_written_count": int(dump_target_written_count),
        "nucleation_spatial_rate_correction": bool(nucleation_spatial_rate_correction),
        "nucleation_spatial_rate_rse_target": float(nucleation_spatial_rate_rse_target),
        "nucleation_placement_ess_per_event": float(nucleation_placement_ess_per_event),
        "hybrid_transport": bool(hybrid_transport),
        "hybrid_matching_length": (
            None if not hybrid_transport
            else float(hybrid_matching_length)
        ),
        "hybrid_grid_shape": (
            None if not hybrid_transport
            else np.asarray(
                hybrid_long_range_field.grid_shape,
                dtype=int,
            ).copy()
        ),
        "hybrid_short_cutoff": (
            None if not hybrid_transport
            else float(hybrid_short_cutoff)
        ),
        "solver_diagnostics_file": (
            None if solver_diagnostics_file is None
            else str(solver_diagnostics_file)
        ),
        "hybrid_long_range_field_history_stored": False,
        "hybrid_long_range_field_hist": None,
        "hybrid_long_range_field_final": (
            None
            if not (hybrid_transport and return_hybrid_long_range_field_final)
            else hybrid_long_range_field.copy_field()
        ),
        "hybrid_long_range_matrix_inventory_final": (
            None
            if not (hybrid_transport and return_hybrid_long_range_field_final)
            else hybrid_long_range_field.matrix_inventory().copy()
        ),

        "M_mg_hist": np.asarray(
            M_mg_hist
        ),
        "M_si_hist": np.asarray(
            M_si_hist
        ),
        "M_mg_abs_err_hist": np.asarray(
            M_mg_abs_err_hist
        ),
        "M_si_abs_err_hist": np.asarray(
            M_si_abs_err_hist
        ),
        "M_mg_rel_err_hist": np.asarray(
            M_mg_rel_err_hist
        ),
        "M_si_rel_err_hist": np.asarray(
            M_si_rel_err_hist
        ),

        "alive_count_hist": np.asarray(
            alive_count_hist,
            dtype=int,
        ),
        "coupled_count_hist": np.asarray(
            coupled_count_hist,
            dtype=int,
        ),

        "R_remove": R_remove,
        "adaptive_dt": adaptive_dt,
        "dt_last_accepted_final": (
            None if dt_last_accepted is None else float(dt_last_accepted)
        ),
        "previous_step_had_growth_retry_final": bool(
            previous_step_had_growth_retry
        ),
        "previous_growth_nfev_final": (
            None if previous_growth_nfev is None
            else int(previous_growth_nfev)
        ),
        "solver_history": solver_history,

        "nuc_time": np.asarray(
            nuc_time,
            dtype=float,
        ),
        "Ntrial_hist": Ntrial_hist,
        "Nacc_hist": Nacc_hist,
        "cnuc_mg": cnuc_mg,
        "cnuc_si": cnuc_si,

        "Q_hist":
            Q_list,
        "c_surf_hist":
            c_surf_list,
        "cbar_hist":
            cbar_list,
        "dRdt_species_hist":
            dRdt_species_list,
        "Pe_hist":
            Pe_list,
        "M_hist":
            M_hist,
        "M_abs_err_hist":
            M_abs_err_hist,
        "M_rel_err_hist":
            M_rel_err_hist,

        "c_surf_mg_hist":
            c_surf_mg_list,
        "c_surf_si_hist":
            c_surf_si_list,
        "dRdt_mg_hist":
            dRdt_mg_list,
        "dRdt_si_hist":
            dRdt_si_list,
        "Pe_mg_hist":
            Pe_mg_list,
        "Pe_si_hist":
            Pe_si_list,

        "cnt_params_ternary":
            cnt_params_ternary,
        "cnt_params_by_phase":
            tuple(dict(p) for p in phase_cnt_params),

        # --------------------------------------------------------------
        # Final state for restarting another thermal segment
        # --------------------------------------------------------------
        "R_final": R.copy(),
        "positions_final": positions.copy(),
        "hybrid_forced_nucleus_inserted": bool(forced_nucleus_inserted),
        "nuc_kind_final": nuc_kind.copy(),

        "alive_final": alive.copy(),
        "coupled_final": coupled.copy(),

        "coupled_idx_final": (
            current_coupled_idx.copy()
        ),


        "Q_final": Q_list[-1].copy(),
        "c_surf_final": c_surf_list[-1].copy(),
        "cbar_final": np.asarray(cbar_list[-1], dtype=float).copy(),
        "dRdt_species_final": np.asarray(
            dRdt_species_list[-1], dtype=float
        ).copy(),
        "Pe_final": np.asarray(Pe_list[-1], dtype=float).copy(),

        "Q_mg_final": (
            Q_list[-1][mg_index].copy() if is_legacy_mg_si else None
        ),
        "Q_si_final": (
            Q_list[-1][si_index].copy() if is_legacy_mg_si else None
        ),
        "c_surf_mg_final": (
            c_surf_list[-1][mg_index].copy()
            if is_legacy_mg_si else None
        ),
        "c_surf_si_final": (
            c_surf_list[-1][si_index].copy()
            if is_legacy_mg_si else None
        ),
        "cbar_mg_final": (
            float(cbar_list[-1][mg_index]) if is_legacy_mg_si else None
        ),
        "cbar_si_final": (
            float(cbar_list[-1][si_index]) if is_legacy_mg_si else None
        ),
    }

    return (
        np.asarray(
            t_list,
            dtype=float,
        ),
        R_list,
        Q_mg_list if is_legacy_mg_si else [],
        Q_si_list if is_legacy_mg_si else [],
        np.asarray(cbar_mg_list, dtype=float)
            if is_legacy_mg_si else np.empty(0, dtype=float),
        np.asarray(cbar_si_list, dtype=float)
            if is_legacy_mg_si else np.empty(0, dtype=float),
        G0,
        aux,
        Jnuc,
        Rnuc,
        number_ppt,
        positions_out,
        nuc_kind_out,
        dRdt_list,
    )
