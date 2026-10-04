import numpy as np
import time
import csv
from pathlib import Path

from .integrator import integrate_growth_ternary_iterative
from .outputs_subfunctions import write_particles_ovito_dump


def arrhenius_diffusivity(T,D0,Q,R_gas=8.314462618):
    """
    Arrhenius diffusivity.

    Parameters
    ----------
    T : float
        Absolute temperature in K.
    D0 : float
        Diffusivity prefactor.
    Q : float
        Activation energy in J/mol.
    R_gas : float
        Gas constant in J/(mol K).
    """
    T = float(T)

    if not np.isfinite(T) or T <= 0.0:
        raise ValueError("T must be a positive finite temperature in K.")

    D = float(D0 * np.exp(-Q / (R_gas * T)))* 1.0e18  #nm^2/s
    return D


def _append_particle_statistics_history(
    filename, *, times, temperature_K, radii_hist, phase_hist, kind_hist,
    cbar_hist, box, solute_names, phase_names, append=True,
):
    """Append compact, species-general precipitation statistics to CSV.

    One row is written for the total population and for each phase split into
    all/bulk/GB origins.  Species-dependent matrix concentrations are emitted
    as cbar_<species>_nm3 columns.
    """
    if filename is None:
        return

    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0 and append

    fieldnames = [
        "time_s", "temperature_K", "temperature_C", "population",
        "phase_id", "origin", "N", "Nv_nm3", "M0", "M1_nm",
        "M2_nm2", "M3_nm3", "Rmean_nm", "Rrms_nm", "Rmax_nm",
        "volume_fraction",
    ] + [f"cbar_{name}_nm3" for name in solute_names]

    volume = float(np.prod(np.asarray(box, dtype=float)))
    rows = []

    for t, R, pid, kind, cbar in zip(
        times, radii_hist, phase_hist, kind_hist, cbar_hist
    ):
        R = np.asarray(R, dtype=float)
        pid = np.asarray(pid, dtype=np.int32)
        kind = np.asarray(kind, dtype=np.int32)
        cbar = np.asarray(cbar, dtype=float)

        alive = R > 0.0
        populations = [("all", -1, "all", alive)]
        for pidx, pname in enumerate(phase_names):
            pmask = alive & (pid == pidx)
            populations.extend([
                (str(pname), pidx, "all", pmask),
                (str(pname), pidx, "bulk", pmask & (kind == 0)),
                (str(pname), pidx, "gb", pmask & (kind == 1)),
            ])

        for label, pidx, origin, mask in populations:
            rr = R[mask]
            n = int(rr.size)
            if n:
                m1 = float(np.sum(rr))
                m2 = float(np.sum(rr**2))
                m3 = float(np.sum(rr**3))
                rmean = m1 / n
                rrms = float(np.sqrt(m2 / n))
                rmax = float(np.max(rr))
            else:
                m1 = m2 = m3 = rmean = rrms = rmax = 0.0

            row = {
                "time_s": float(t),
                "temperature_K": float(temperature_K),
                "temperature_C": float(temperature_K) - 273.15,
                "population": label,
                "phase_id": pidx,
                "origin": origin,
                "N": n,
                "Nv_nm3": n / volume,
                "M0": float(n),
                "M1_nm": m1,
                "M2_nm2": m2,
                "M3_nm3": m3,
                "Rmean_nm": rmean,
                "Rrms_nm": rrms,
                "Rmax_nm": rmax,
                "volume_fraction": (4.0 * np.pi / 3.0) * m3 / volume,
            }
            for i, name in enumerate(solute_names):
                row[f"cbar_{name}_nm3"] = float(cbar[i])
            rows.append(row)

    with path.open("a" if exists else "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)

def integrate_ternary_thermal_history(
    temperatures,
    segment_durations,
    *,
    R0,
    positions,
    box,
    solute_names,
    cp,
    cbar0,
    D0,
    Qdiff,
    phases,
    dt,
    aa,
    bb,
    cc,
    vatm,
    N0,
    a_latt,
    safety_buffer=0.0,
    apply_incubation_correction=True,
    cp_phase=None,
    nuc_kind0=None,
    phase_id0=None,
    Q0=None,
    c_surf0=None,
    dRdt0=None,
    dRdt_species0=None,
    M_total_input=None,
    # Legacy Mg/Si restart aliases retained for backward compatibility.
    Q_mg0=None,
    Q_si0=None,
    c_surf_mg0=None,
    c_surf_si0=None,
    M_total_mg_input=None,
    M_total_si_input=None,
    screening_length=None,
    print_segments=False,
    print_every_segments=1,
    rng=None,
    dump_seg_filename=None,
    dump_seg_every=1,
    dump_times=None,
    dump_include_final=True,
    history_csv_filename=None,
    field_snapshot_callback=None,
    periodic_save_callback=None,
    periodic_save_every_segments=None,
    store_hybrid_long_range_history=False,
    hybrid_long_range_history_every_segments=25,
    hybrid_long_range_history_spacing_nm=10.0,
    hybrid_long_range_history_dtype="float32",
    solver_diagnostics_file=None,
    **integrator_kwargs,
):
    """
    Integrate a multicomponent precipitation simulation over a piecewise
    isothermal thermal history.

    The public function name is retained for backward compatibility, but the
    thermal-history state is species-indexed and does not assume Mg/Si.

    Each temperature is held constant for the corresponding duration.
    """

    temperatures = np.asarray(
        temperatures,
        dtype=float,
    )

    segment_durations = np.asarray(
        segment_durations,
        dtype=float,
    )

    if temperatures.ndim != 1:
        raise ValueError(
            "temperatures must be one-dimensional."
        )

    if segment_durations.shape != temperatures.shape:
        raise ValueError(
            "temperatures and segment_durations must have "
            "the same shape."
        )

    if np.any(~np.isfinite(temperatures)):
        raise ValueError(
            "temperatures contains non-finite values."
        )

    if np.any(temperatures <= 0.0):
        raise ValueError(
            "All temperatures must be positive."
        )

    if np.any(~np.isfinite(segment_durations)):
        raise ValueError(
            "segment_durations contains non-finite values."
        )

    if np.any(segment_durations <= 0.0):
        raise ValueError(
            "All segment durations must be positive."
        )

    # ----------------------------------------------------------
    # Solute representation
    # ----------------------------------------------------------

    solute_names = tuple(solute_names)

    cp = np.asarray(cp, dtype=float)
    cp_phase = cp[None, :].copy() if cp_phase is None else np.asarray(cp_phase, dtype=float)
    cbar0 = np.asarray(cbar0, dtype=float)
    D0 = np.asarray(D0, dtype=float)
    Qdiff = np.asarray(Qdiff, dtype=float)

    n_solutes = len(solute_names)

    if n_solutes == 0:
        raise ValueError("At least one solute species is required.")

    for name, values in (
        ("cp", cp),
        ("cbar0", cbar0),
        ("D0", D0),
        ("Qdiff", Qdiff),
    ):
        if values.shape != (n_solutes,):
            raise ValueError(
                f"{name} must have shape ({n_solutes},); "
                f"got {values.shape}."
            )

        if np.any(~np.isfinite(values)):
            raise ValueError(
                f"{name} contains non-finite values."
            )

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
            "cp_phase[0] must exactly equal the compatibility vector cp."
        )

    phases = tuple(phases)
    if len(phases) == 0:
        raise ValueError("At least one precipitate phase is required.")
    if cp_phase.shape[0] != len(phases):
        raise ValueError(
            "cp_phase must contain one precipitate-composition row per phase; "
            f"got {cp_phase.shape[0]} rows for {len(phases)} phases."
        )

    if len(set(solute_names)) != n_solutes:
        raise ValueError(
            "solute_names must contain unique species names."
        )

    # ----------------------------------------------------------
    # Species-generic restart handling
    # ----------------------------------------------------------

    mg_index = solute_names.index("Mg") if "Mg" in solute_names else None
    si_index = solute_names.index("Si") if "Si" in solute_names else None

    def _validate_generic_restart(value, *, name, scalar=False):
        if value is None:
            return None

        values = np.asarray(value, dtype=float)
        expected = (n_solutes,) if scalar else None

        if scalar:
            if values.shape != expected:
                raise ValueError(
                    f"{name} must have shape {expected}; got {values.shape}."
                )
        else:
            if values.ndim < 1 or values.shape[0] != n_solutes:
                raise ValueError(
                    f"{name} must have species as its first axis with "
                    f"length {n_solutes}; got {values.shape}."
                )

        if np.any(~np.isfinite(values)):
            raise ValueError(f"{name} contains non-finite values.")

        return values.copy()

    def _pack_legacy_species_pair(mg_value, si_value, *, name, scalar=False):
        """Pack legacy Mg/Si restart data into species-indexed storage."""
        if mg_value is None and si_value is None:
            return None

        if mg_index is None or si_index is None:
            raise ValueError(
                f"{name}: legacy Mg/Si restart arguments can only be used "
                "when solute_names contains both 'Mg' and 'Si'."
            )

        if mg_value is None or si_value is None:
            raise ValueError(
                f"{name}: Mg and Si restart values must either both be "
                "provided or both be None."
            )

        if n_solutes != 2:
            raise ValueError(
                f"{name}: legacy Mg/Si restart arguments are only unambiguous "
                "for a two-solute Mg-Si system; use the generic restart "
                "argument instead."
            )

        if scalar:
            values = np.empty(n_solutes, dtype=float)
            values[mg_index] = float(mg_value)
            values[si_index] = float(si_value)
            return values

        mg_array = np.asarray(mg_value, dtype=float)
        si_array = np.asarray(si_value, dtype=float)

        if mg_array.shape != si_array.shape:
            raise ValueError(
                f"{name}: Mg and Si restart arrays must have the same "
                f"shape; got {mg_array.shape} and {si_array.shape}."
            )

        values = np.empty((n_solutes,) + mg_array.shape, dtype=float)
        values[mg_index] = mg_array
        values[si_index] = si_array
        return values

    if rng is None:
        rng = np.random.default_rng()

    if print_every_segments <= 0:
        raise ValueError(
            "print_every_segments must be a positive integer."
        )

    if periodic_save_every_segments is not None:
        periodic_save_every_segments = int(
            periodic_save_every_segments
        )
    
        if periodic_save_every_segments <= 0:
            raise ValueError(
                "periodic_save_every_segments must be a positive integer."
            )
    
        if periodic_save_callback is None:
            raise ValueError(
                "periodic_save_callback must be supplied when "
                "periodic_save_every_segments is enabled."
            )
    
    if (
        periodic_save_callback is not None
        and not callable(periodic_save_callback)
    ):
        raise TypeError(
            "periodic_save_callback must be callable."
        )

    # ----------------------------------------------------------
    # Particle-dump schedule
    #
    # When dump_times is supplied, OVITO dump output is decoupled from
    # thermal-segment boundaries.  Each requested global time is mapped to
    # the nearest accepted internal integration state.  This does not alter
    # the timestep or the physics.  dump_seg_every remains as the legacy
    # fallback when dump_times is None.
    # ----------------------------------------------------------
    total_duration = float(np.sum(segment_durations))

    if dump_times is None:
        dump_times_requested = None
    else:
        dump_times_requested = np.asarray(dump_times, dtype=float).reshape(-1)
        if np.any(~np.isfinite(dump_times_requested)):
            raise ValueError("dump_times contains non-finite values.")
        if np.any(dump_times_requested < 0.0):
            raise ValueError("dump_times must contain non-negative global times.")
        if np.any(dump_times_requested > total_duration):
            bad = dump_times_requested[dump_times_requested > total_duration]
            raise ValueError(
                "dump_times contains values beyond the thermal history: "
                f"max requested={np.max(bad):.8e} s, "
                f"total duration={total_duration:.8e} s."
            )

        if dump_include_final:
            dump_times_requested = np.concatenate(
                [dump_times_requested, np.asarray([total_duration], dtype=float)]
            )

        dump_times_requested = np.unique(dump_times_requested)
        dump_times_requested.sort()


    # These arguments are managed explicitly by the thermal-history wrapper
    # and must not also be forwarded through **integrator_kwargs.
    for key in (
        "t0",
        "t_end",
    
        # Generic restart arguments managed explicitly here.
        "Q0",
        "c_surf0",
        "dRdt0",
        "dRdt_species0",
        "M_total_input",

        # Legacy species-specific integrator arguments.
        "cp_mg",
        "cp_si",
        "cbar_mg0",
        "cbar_si0",
        "D_mg",
        "D_si",
        "Q_mg0",
        "Q_si0",
        "c_surf_mg0",
        "c_surf_si0",
        "M_total_mg_input",
        "M_total_si_input",
    ):
        integrator_kwargs.pop(key, None)


    # --------------------------------------------------------------
    # Temperature-independent CNT controls.  Phase-specific material
    # properties and stoichiometry are owned by the phase objects and are
    # assembled inside the integrator.  No chemical species names are
    # interpreted by the thermal-history wrapper.
    # --------------------------------------------------------------
    cnt_params_base = {
        "aa": float(aa),
        "bb": float(bb),
        "cc": float(cc),
        "vatm": float(vatm),
        "N0": float(N0),
        "a_latt": float(a_latt),
        "apply_incubation_correction":
            bool(apply_incubation_correction),
        "safety_buffer":
            float(safety_buffer),
    }


    # Restart state.
    R_current = np.asarray(
        R0,
        dtype=float,
    ).copy()

    positions_current = np.asarray(
        positions,
        dtype=float,
    ).copy()

    nuc_kind_current = (
        None
        if nuc_kind0 is None
        else np.asarray(
            nuc_kind0,
            dtype=np.int32,
        ).copy()
    )

    phase_id_current = (
        None
        if phase_id0 is None
        else np.asarray(
            phase_id0,
            dtype=np.int32,
        ).copy()
    )

    # Species-indexed restart state.
    cbar_current = cbar0.copy()

    Q_current = _validate_generic_restart(Q0, name="Q0")
    if Q_current is None:
        Q_current = _pack_legacy_species_pair(
            Q_mg0, Q_si0, name="Q0"
        )

    c_surf_current = _validate_generic_restart(
        c_surf0, name="c_surf0"
    )
    if c_surf_current is None:
        c_surf_current = _pack_legacy_species_pair(
            c_surf_mg0, c_surf_si0, name="c_surf0"
        )

    # Accepted velocity state is carried across artificial thermal-segment
    # boundaries so the restart state is reproduced exactly.
    dRdt_current = None
    if dRdt0 is not None:
        dRdt_current = np.asarray(dRdt0, dtype=float).copy()
        if dRdt_current.shape != R_current.shape:
            raise ValueError(
                "dRdt0 must have the same shape as R0; got "
                f"{dRdt_current.shape} and {R_current.shape}."
            )
        if np.any(~np.isfinite(dRdt_current)):
            raise ValueError("dRdt0 contains non-finite values.")

    dRdt_species_current = None
    if dRdt_species0 is not None:
        dRdt_species_current = np.asarray(
            dRdt_species0, dtype=float
        ).copy()
        expected_shape = (n_solutes, R_current.size)
        if dRdt_species_current.shape != expected_shape:
            raise ValueError(
                "dRdt_species0 must have shape "
                f"{expected_shape}; got {dRdt_species_current.shape}."
            )
        if np.any(~np.isfinite(dRdt_species_current)):
            raise ValueError("dRdt_species0 contains non-finite values.")


    # The thermal-history wrapper owns the compact kinetics CSV.  Consume
    # history_csv_filename here rather than forwarding it to the inner growth
    # integrator, whose clock restarts at zero for every thermal segment.
    if history_csv_filename is not None:
        history_path = Path(history_csv_filename)
        if history_path.exists():
            history_path.unlink()

    M_total = _validate_generic_restart(
        M_total_input, name="M_total_input", scalar=True
    )
    if M_total is None:
        M_total = _pack_legacy_species_pair(
            M_total_mg_input,
            M_total_si_input,
            name="M_total_input",
            scalar=True,
        )

    # Adaptive timestep/controller state is part of the numerical restart
    # state. Preserve it across thermal-segment boundaries so the segmented
    # thermal history behaves as one continuous integration.
    dt_last_accepted_current = None
    previous_step_had_growth_retry_current = False
    previous_growth_nfev_current = None

    global_time = 0.0

    segment_results = []

    # The transient long-range field is a single persistent object shared by
    # all thermal segments.  The thermal-history wrapper owns assembled-result
    # snapshots so individual segment results must not copy the full 3-D field.
    hybrid_long_range_field = integrator_kwargs.get(
        "hybrid_long_range_field", None
    )

    # Sparse diagnostic LR history.  The live solver field remains full
    # resolution/float64.  These snapshots are for visualization/diagnostics
    # only and are deliberately decoupled from accepted-step histories.
    hybrid_lr_hist = []
    hybrid_lr_hist_times = []
    hybrid_lr_hist_segments = []
    hybrid_lr_hist_indices = None
    hybrid_lr_hist_coordinates = None
    # Independent of len(hybrid_lr_hist): LR-only snapshots may precede the
    # first particle.  This preserves the historical first-particle snapshot.
    have_saved_particle_bearing_lr_snapshot = False
    hybrid_lr_hist_dtype = np.dtype(hybrid_long_range_history_dtype)
    if hybrid_lr_hist_dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError(
            "hybrid_long_range_history_dtype must be 'float32' or 'float64'."
        )

    if hybrid_long_range_history_every_segments is not None:
        hybrid_long_range_history_every_segments = int(
            hybrid_long_range_history_every_segments
        )
        if hybrid_long_range_history_every_segments < 1:
            raise ValueError(
                "hybrid_long_range_history_every_segments must be >= 1 or None."
            )

    if hybrid_long_range_history_spacing_nm is not None:
        hybrid_long_range_history_spacing_nm = float(
            hybrid_long_range_history_spacing_nm
        )
        if (
            not np.isfinite(hybrid_long_range_history_spacing_nm)
            or hybrid_long_range_history_spacing_nm <= 0.0
        ):
            raise ValueError(
                "hybrid_long_range_history_spacing_nm must be positive or None."
            )

    def _prepare_hybrid_lr_history_sampling():
        nonlocal hybrid_lr_hist_indices, hybrid_lr_hist_coordinates
        if hybrid_long_range_field is None:
            return

        spacing = np.asarray(hybrid_long_range_field.spacing, dtype=float)
        shape = np.asarray(hybrid_long_range_field.grid_shape, dtype=int)

        if hybrid_long_range_history_spacing_nm is None:
            strides = np.ones(3, dtype=int)
        else:
            # Diagnostic decimation only.  The LR field is smooth by
            # construction; choose the nearest integer stride that does not
            # make the saved grid finer than necessary.
            strides = np.maximum(
                1,
                np.rint(
                    hybrid_long_range_history_spacing_nm / spacing
                ).astype(int),
            )

        indices = tuple(
            np.arange(0, int(n), int(s), dtype=np.int32)
            for n, s in zip(shape, strides)
        )
        coordinates = tuple(
            idx.astype(float) * dx
            for idx, dx in zip(indices, spacing)
        )
        hybrid_lr_hist_indices = indices
        hybrid_lr_hist_coordinates = coordinates

    def _capture_hybrid_lr_snapshot(*, time_value, segment_index):
        if (
            not store_hybrid_long_range_history
            or hybrid_long_range_field is None
        ):
            return False

        if hybrid_lr_hist_indices is None:
            _prepare_hybrid_lr_history_sampling()

        ix, iy, iz = hybrid_lr_hist_indices
        # np.ix_ gives a direct coarse copy without first duplicating the full
        # solver field.  Cast only the much smaller diagnostic snapshot.
        field = hybrid_long_range_field.field
        coarse = field[
            :,
            ix[:, None, None],
            iy[None, :, None],
            iz[None, None, :],
        ].astype(hybrid_lr_hist_dtype, copy=False)

        hybrid_lr_hist.append(np.array(coarse, copy=True))
        hybrid_lr_hist_times.append(float(time_value))
        hybrid_lr_hist_segments.append(int(segment_index))
        return True

    global_t_hist = []
    global_R_hist = []
    global_phase_id_hist = []
    global_nuc_kind_hist = []
    global_Q_hist = []
    global_cbar_hist = []
    global_dRdt_hist = []

    n_segments = len(temperatures)

    dump_has_been_written = False

#--------
    def assemble_result():
        """
        Assemble the current simulation output using the same structure
        returned at the end of the complete thermal history.
        """

        if len(segment_results) == 0:
            final_aux = None
        else:
            # Keep historical segment records lightweight.  Only the assembled
            # assembled result receives one snapshot of the persistent LR field,
            # avoiding one multi-GB copy per thermal segment.
            final_aux = dict(segment_results[-1]["result"][7])
            if hybrid_long_range_field is not None:
                final_aux["hybrid_long_range_field_final"] = (
                    hybrid_long_range_field.copy_field()
                )
                final_aux["hybrid_long_range_matrix_inventory_final"] = (
                    hybrid_long_range_field.matrix_inventory().copy()
                )

        if final_aux is not None:
            final_aux["hybrid_long_range_field_hist"] = (
                None
                if not store_hybrid_long_range_history
                else list(hybrid_lr_hist)
            )
            final_aux["hybrid_long_range_field_hist_times"] = np.asarray(
                hybrid_lr_hist_times, dtype=float
            )
            final_aux["hybrid_long_range_field_hist_segments"] = np.asarray(
                hybrid_lr_hist_segments, dtype=np.int32
            )
            final_aux["hybrid_long_range_field_history_stored"] = bool(
                store_hybrid_long_range_history and len(hybrid_lr_hist) > 0
            )
            if hybrid_lr_hist_indices is not None:
                final_aux["hybrid_long_range_field_hist_indices"] = tuple(
                    np.asarray(v, dtype=np.int32)
                    for v in hybrid_lr_hist_indices
                )
                final_aux["hybrid_long_range_field_hist_coordinates_nm"] = tuple(
                    np.asarray(v, dtype=float)
                    for v in hybrid_lr_hist_coordinates
                )
                final_aux["hybrid_long_range_field_hist_dtype"] = str(
                    hybrid_lr_hist_dtype
                )

        result = {
            "t_hist": np.asarray(global_t_hist, dtype=float),
            "R_hist": list(global_R_hist),
            "phase_id_hist": list(global_phase_id_hist),
            "nuc_kind_hist": list(global_nuc_kind_hist),
            "Q_hist": list(global_Q_hist),
            "cbar_hist": np.asarray(global_cbar_hist, dtype=float),
            "dRdt_hist": list(global_dRdt_hist),
            "segments": list(segment_results),
            "final_aux": final_aux,
        }

        # Preserve historical Mg/Si result keys when this is an Mg-Si run.
        if mg_index is not None and si_index is not None:
            result.update({
                "Q_mg_hist": [q[mg_index] for q in global_Q_hist],
                "Q_si_hist": [q[si_index] for q in global_Q_hist],
                "cbar_mg_hist": np.asarray(
                    [c[mg_index] for c in global_cbar_hist], dtype=float
                ),
                "cbar_si_hist": np.asarray(
                    [c[si_index] for c in global_cbar_hist], dtype=float
                ),
            })

        return result
#--------
    
    # A prescribed LR field is a valid independent state even with N=0.
    # Save t=0 only for particle-free initial states.  Runs that already start
    # with particles retain their existing particle-bearing history semantics.
    if (
        store_hybrid_long_range_history
        and hybrid_long_range_field is not None
        and not bool(np.any(R_current > 0.0))
    ):
        _capture_hybrid_lr_snapshot(
            time_value=global_time,
            segment_index=0,
        )

    for segment_index, (T_segment,duration,) in enumerate(zip(temperatures,segment_durations,),start=1,):
    
        print_this_segment = (
            print_segments
            and (
                segment_index == 1
                or segment_index == n_segments
                or segment_index
                % print_every_segments == 0
            )
        )
    
        n_particles_before = int(np.count_nonzero(R_current > 0.0))

        segment_wall_start = (time.perf_counter())
        
        # Compute all segment diffusivities in the generic species order.
        D_segment = np.asarray(
            [
                arrhenius_diffusivity(
                    T=T_segment,
                    D0=D0_i,
                    Q=Qdiff_i,
                )
                for D0_i, Qdiff_i in zip(D0, Qdiff)
            ],
            dtype=float,
        )

        # The complete diffusivity vector is passed directly to the integrator.
        # Phase-specific CNT records are assembled there, so this wrapper does
        # not create species-specific diffusivity aliases.
        cnt_params_segment = dict(cnt_params_base)

        # Target-time dumps must be handled inside the accepted-step integrator
        # loop.  Waiting until a thermal segment returns would delay all dumps
        # until the segment boundary and defeats the purpose of decoupling dump
        # output from thermal segmentation.
        segment_integrator_kwargs = dict(integrator_kwargs)

        # The live LR object persists across segments.  Do not make another
        # full-field copy when each segment returns; assemble_result() creates
        # exactly one snapshot only for an assembled result.
        segment_integrator_kwargs[
            "return_hybrid_long_range_field_final"
        ] = False
        # Never accumulate full 3-D LR fields at accepted-step cadence.
        segment_integrator_kwargs["store_hybrid_long_range_history"] = False
        segment_integrator_kwargs["solver_diagnostics_file"] = (
            solver_diagnostics_file
        )
        if field_snapshot_callback is not None:
            segment_integrator_kwargs["dump_field_snapshot_callback"] = (
                field_snapshot_callback
            )

        # The inner integrator resets its local clock to zero for every thermal
        # segment.  Supply the accumulated thermal-history time separately so
        # fixed-point diagnostics from cooling runs can be compared at the same
        # physical time/temperature across matching-length sweeps.
        segment_integrator_kwargs["diagnostic_time_offset"] = float(global_time)

        if dump_times_requested is not None:
            segment_global_start = float(global_time)
            segment_global_end = segment_global_start + float(duration)
            if segment_index == 1:
                in_this_segment = (
                    dump_times_requested >= segment_global_start - 1.0e-14
                ) & (
                    dump_times_requested <= segment_global_end + 1.0e-14
                )
            else:
                # Shared thermal boundaries belong to the preceding segment,
                # preventing duplicate frames at exactly the same accepted state.
                in_this_segment = (
                    dump_times_requested > segment_global_start + 1.0e-14
                ) & (
                    dump_times_requested <= segment_global_end + 1.0e-14
                )

            segment_dump_times = dump_times_requested[in_this_segment]
            segment_integrator_kwargs["dump_filename"] = dump_seg_filename
            segment_integrator_kwargs["dump_target_times"] = segment_dump_times
            segment_integrator_kwargs["dump_time_offset"] = float(global_time)
            segment_integrator_kwargs["dump_target_append"] = bool(
                dump_has_been_written
            )
            segment_integrator_kwargs["dump_target_print"] = bool(print_segments)

            # Disable the legacy accepted-step cadence while an explicit target
            # schedule is active.
            segment_integrator_kwargs["save_every"] = None
        
        segment = integrate_growth_ternary_iterative(
            R0=R_current,
            positions=positions_current,
            box=box,

            # Generic species-indexed state.
            solute_names=solute_names,
            cp=cp,
            cbar0=cbar_current,
            D=D_segment,
            phases=phases,
            cp_phase=cp_phase,

            t0=0.0,
            t_end=float(duration),
            dt=dt,
            T=float(T_segment),

            aa=aa,
            bb=bb,
            cc=cc,
            vatm=vatm,

            cnt_params=cnt_params_segment,

            Q0=Q_current,
            c_surf0=c_surf_current,
            dRdt0=dRdt_current,
            dRdt_species0=dRdt_species_current,
            phase_id0=phase_id_current,

            nuc_kind0=nuc_kind_current,

            M_total_input=M_total,

            dt_last_accepted0=dt_last_accepted_current,
            previous_step_had_growth_retry0=(
                previous_step_had_growth_retry_current
            ),
            previous_growth_nfev0=previous_growth_nfev_current,

            screening_length=screening_length,


            rng=rng,

            **segment_integrator_kwargs,
        )

        segment_wall_time = (time.perf_counter()- segment_wall_start)
    

        
        # The integrator still returns its historical Mg/Si tuple for backward
        # compatibility.  Generic histories are obtained from aux_segment.
        t_segment = segment[0]
        R_segment = segment[1]
        aux_segment = segment[7]
        dRdt_segment = segment[13]

        if dump_times_requested is not None:
            dump_has_been_written = bool(
                dump_has_been_written
                or int(aux_segment.get("target_dump_written_count", 0)) > 0
            )
        Q_segment = aux_segment["Q_hist"]
        cbar_segment = aux_segment["cbar_hist"]

        # The first point of every segment after the first duplicates
        # the final point of the preceding segment.
        start_index = (
            0
            if segment_index == 1
            else 1
        )

        global_t_hist.extend(
            (
                global_time
                + t_segment[start_index:]
            ).tolist()
        )

        global_R_hist.extend(
            R_segment[start_index:]
        )

        phase_id_segment = aux_segment["phase_id_hist"]
        nuc_kind_segment = aux_segment["nuc_kind_hist"]

        _append_particle_statistics_history(
            history_csv_filename,
            times=(global_time + t_segment[start_index:]),
            temperature_K=float(T_segment),
            radii_hist=R_segment[start_index:],
            phase_hist=phase_id_segment[start_index:],
            kind_hist=nuc_kind_segment[start_index:],
            cbar_hist=cbar_segment[start_index:],
            box=box,
            solute_names=solute_names,
            phase_names=[
                getattr(phase, "name", f"phase_{i}")
                for i, phase in enumerate(phases)
            ],
            append=True,
        )

        global_phase_id_hist.extend(
            phase_id_segment[start_index:]
        )

        global_nuc_kind_hist.extend(
            nuc_kind_segment[start_index:]
        )

        global_Q_hist.extend(
            [np.asarray(q, dtype=float).copy() for q in Q_segment[start_index:]]
        )

        global_cbar_hist.extend(
            [np.asarray(c, dtype=float).copy() for c in cbar_segment[start_index:]]
        )

        global_dRdt_hist.extend(
            dRdt_segment[start_index:]
        )

        segment_record = {
            "segment_index": segment_index,
            "global_start_time": global_time,
            "global_end_time": global_time + duration,
            "temperature": float(T_segment),
            "duration": float(duration),
            "D": D_segment.copy(),
            "result": segment,
        }

        # Preserve historical segment metadata for Mg-Si runs.
        if mg_index is not None and si_index is not None:
            segment_record["D_mg"] = float(D_segment[mg_index])
            segment_record["D_si"] = float(D_segment[si_index])

        segment_results.append(segment_record)

        # Update the complete restart state.
        R_current = (
            aux_segment["R_final"]
        )

        positions_current = (
            aux_segment[
                "positions_final"
            ]
        )

        nuc_kind_current = (
            aux_segment[
                "nuc_kind_final"
            ]
        )


        # The integrator now exposes its restart state directly in the
        # generic species-indexed representation.
        Q_current = np.asarray(
            aux_segment["Q_final"],
            dtype=float,
        ).copy()

        phase_id_current = np.asarray(
            aux_segment["phase_id_final"],
            dtype=np.int32,
        ).copy()

        c_surf_current = np.asarray(
            aux_segment["c_surf_final"],
            dtype=float,
        ).copy()

        dRdt_current = np.asarray(
            aux_segment["dRdt_final"],
            dtype=float,
        ).copy()

        dRdt_species_current = np.asarray(
            aux_segment["dRdt_species_final"],
            dtype=float,
        ).copy()

        cbar_current = np.asarray(
            aux_segment["cbar_final"],
            dtype=float,
        ).copy()

        M_total = np.asarray(
            aux_segment["M_total"],
            dtype=float,
        ).copy()

        dt_last_accepted_current = aux_segment.get(
            "dt_last_accepted_final", None
        )
        previous_step_had_growth_retry_current = bool(
            aux_segment.get(
                "previous_step_had_growth_retry_final", False
            )
        )
        previous_growth_nfev_current = aux_segment.get(
            "previous_growth_nfev_final", None
        )

        # Sparse LR history is captured at thermal-segment boundaries, not at
        # accepted-step cadence.
        #
        # For particle-bearing states, preserve the existing semantics:
        # first particle-bearing field + requested cadence + final field.
        # LR-only snapshots use a separate path and do not change what a saved
        # LR field means for later SR+LR total-field reconstruction.
        if store_hybrid_long_range_history and hybrid_long_range_field is not None:
            has_particles = bool(np.any(R_current > 0.0))

            if has_particles:
                first_particle_snapshot = (
                    not have_saved_particle_bearing_lr_snapshot
                )
                cadence_snapshot = (
                    hybrid_long_range_history_every_segments is not None
                    and segment_index % hybrid_long_range_history_every_segments == 0
                )
                final_snapshot = segment_index == n_segments
                should_capture_lr = (
                    first_particle_snapshot
                    or cadence_snapshot
                    or final_snapshot
                )
            else:
                cadence_snapshot = (
                    hybrid_long_range_history_every_segments is not None
                    and segment_index % hybrid_long_range_history_every_segments == 0
                )
                final_snapshot = segment_index == n_segments
                should_capture_lr = cadence_snapshot or final_snapshot

            if should_capture_lr:
                # Avoid duplicate capture when multiple conditions coincide.
                if (
                    len(hybrid_lr_hist_segments) == 0
                    or hybrid_lr_hist_segments[-1] != segment_index
                ):
                    captured = _capture_hybrid_lr_snapshot(
                        time_value=global_time + float(duration),
                        segment_index=segment_index,
                    )
                    if captured and has_particles:
                        have_saved_particle_bearing_lr_snapshot = True
#--------------------------------------------
        if print_this_segment:
            active_radii = R_current[
                R_current > 0.0
            ]
        
            n_particles_after = len(
                active_radii
            )
        
            delta_particles = (
                n_particles_after
                - n_particles_before
            )
        
            if n_particles_after > 0:
                R_min = float(
                    np.min(active_radii)
                )
        
                R_mean = float(
                    np.mean(active_radii)
                )
        
                R_max = float(
                    np.max(active_radii)
                )
        
            else:
                R_min = 0.0
                R_mean = 0.0
                R_max = 0.0
        
            global_end_time = (
                global_time
                + float(duration)
            )
        
            n_internal_steps = max(
                len(t_segment) - 1,
                0,
            )
        
            print()
            print("=" * 72)
            print(
                f"Thermal segment "
                f"{segment_index}/{n_segments}"
            )
            print("=" * 72)
        
            print(
                f"Temperature     : "
                f"{T_segment:10.3f} K --> {T_segment-273:10.3f} C"
            )
        
            print(
                f"Global time     : "
                f"{global_time:10.4e} "
                f"-> "
                f"{global_end_time:10.4e} s"
            )
        
            print(
                f"Duration        : "
                f"{duration:10.4e} s"
            )
        
            print(
                f"Internal steps  : "
                f"{n_internal_steps:10d}"
            )
        
            print(
                f"Particles       : "
                f"{n_particles_after:10d} "
                f"(Prev. step {delta_particles:+d})"
            )

            # Bulk/GB populations are based on nucleation origin and include
            # only currently surviving particles.
            active_idx_summary = np.flatnonzero(R_current > 0.0)
            if active_idx_summary.size:
                kind_active_summary = np.asarray(
                    nuc_kind_current, dtype=np.int32
                )[active_idx_summary]
                n_bulk_summary = int(np.count_nonzero(kind_active_summary == 0))
                n_gb_summary = int(np.count_nonzero(kind_active_summary == 1))
            else:
                n_bulk_summary = 0
                n_gb_summary = 0

            print(
                f"Population bulk/GB: "
                f"{n_bulk_summary:10d} / {n_gb_summary:d}"
            )
        
            print(
                "Radius min/mean/max [nm]: "
                f"{R_min:.4e} / "
                f"{R_mean:.4e} / "
                f"{R_max:.4e}"
            )

            # Per-phase population/radius diagnostics, matching the useful
            # multiphase summaries.  Print every configured phase, even
            # when its current population is zero, so missing phase activity
            # is immediately visible.
            phase_id_arr_summary = np.asarray(phase_id_current, dtype=np.int32)
            R_arr_summary = np.asarray(R_current, dtype=float)
            for p, phase in enumerate(phases):
                phase_mask = (R_arr_summary > 0.0) & (phase_id_arr_summary == p)
                n_phase = int(np.count_nonzero(phase_mask))
                phase_name = getattr(phase, "name", str(p))
                if n_phase > 0:
                    R_phase = R_arr_summary[phase_mask]
                    rp_min = float(np.min(R_phase))
                    rp_mean = float(np.mean(R_phase))
                    rp_max = float(np.max(R_phase))
                else:
                    rp_min = rp_mean = rp_max = 0.0
                print(
                    f"Phase {p} ({phase_name}) : "
                    f"N={n_phase:6d}, "
                    f"R[min/mean/max]={rp_min:.4e} / "
                    f"{rp_mean:.4e} / {rp_max:.4e} nm"
                )
        
            for species_index, species_name in enumerate(solute_names):
                print(
                    f"Matrix {species_name} [nm^-3]: "
                    f"{cbar_current[species_index]:.6e}"
                )

            for species_index, species_name in enumerate(solute_names):
                print(
                    f"D_{species_name:<14}: "
                    f"{D_segment[species_index]:.6e} nm^2/s"
                )
        
            print(
                f"Wall time       : "
                f"{segment_wall_time:10.3f} s"
            )
#--------------------------------------------
        # Advance the global time to the end of the segment.
        global_time += float(duration)
 
        # --------------------------------------------------------------
        # Legacy segment-boundary OVITO particle dumps.  Explicit target-time
        # schedules are emitted inside the integrator as soon as the nearest
        # accepted state is known.
        # --------------------------------------------------------------
        if dump_seg_filename is not None and dump_times_requested is None:
            should_write_dump = (
                segment_index % dump_seg_every == 0
                or segment_index == n_segments
            )

            if should_write_dump:
                active_idx = np.flatnonzero(R_current > 0.0)

                write_particles_ovito_dump(
                    filename=dump_seg_filename,
                    step=global_time,
                    positions=positions_current[active_idx],
                    radii=R_current[active_idx],
                    box=box,
                    ids=active_idx + 1,
                    types=nuc_kind_current[active_idx] + 1,
                    phase_ids=phase_id_current[active_idx],
                    gb_flags=nuc_kind_current[active_idx],
                    dRdt=np.asarray(dRdt_segment[-1], dtype=float)[active_idx],
                    peclet=np.asarray(aux_segment["Pe_hist"][-1], dtype=float)[:, active_idx],
                    peclet_names=solute_names,
                    mode=("a" if dump_has_been_written else "w"),
                )

                if (
                    field_snapshot_callback is not None
                    and hybrid_long_range_field is not None
                ):
                    field_snapshot_callback(
                        actual_global_time=float(global_time),
                        requested_global_time=float(global_time),
                        long_range_field=hybrid_long_range_field,
                        long_range_field_array=None,
                        temperature_K=float(T_segment),
                    )

                dump_has_been_written = True

        #write output file
        should_save_periodically = (
            periodic_save_callback is not None
            and periodic_save_every_segments is not None
            and (
                segment_index
                % periodic_save_every_segments
                == 0
            )
        )
        
        if should_save_periodically:
            periodic_result = assemble_result()
        
            periodic_save_callback(
                periodic_result,
                completed_segments=segment_index,
                total_segments=n_segments,
                run_complete=False,
            )
        
            if print_segments:
                print(
                    f"Periodic results saved after "
                    f"segment {segment_index}/{n_segments}."
                )

    return assemble_result()
