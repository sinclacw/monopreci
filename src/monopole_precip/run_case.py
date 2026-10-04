import argparse
import os
import sys
from pathlib import Path


import numpy as np

from monopole_precip import integrate_ternary_thermal_history
from monopole_precip.hybrid_long_range import PeriodicLongRangeField
from monopole_precip.outputs_subfunctions import write_hybrid_long_range_snapshot_npz
from scipy.fft import next_fast_len

# Material and run configuration are supplied by the calculation directory.
# This allows the installed Mono'Preci executable to be launched from any
# directory containing material_definition.py and run_configuration.py.
sys.path.insert(0, str(Path.cwd()))

from material_definition import alloy, initial_composition

# Import the configuration module itself for settings that support
# backward-compatible names.
import run_configuration as run_config


from run_configuration import (
    box,
    positions,
    radii0,
    phase_id0,
    dt_initial,
    first_step_dt_max,
    dt_max,
    dt_min,
    dt_safety,
    isothermal_segment_time_s,
    enable_nucleation,
    safety_buffer,
    apply_incubation_correction,
    max_expected_nucleation_events,
    nucleation_spatial_rate_correction,
    nucleation_spatial_rate_rse_target,
    nucleation_placement_ess_per_event,
    save_nucleation_diagnostics,
    screening_length,
    R_remove,
    R_remove_buffer,
    save_every_segments,
    dump_times_s,
    dump_include_final,
    dump_seg_every,
    print_every,
    print_every_segments,
    save_every_internal_steps,
    store_hybrid_long_range_history,
    save_hybrid_long_range_field_snapshots,
    hybrid_long_range_history_every_segments,
    hybrid_long_range_history_spacing_nm,
    hybrid_long_range_history_dtype,
    gb_point,
    gb_normal,
    gb_thickness,
    hybrid_transport,
    hybrid_matching_length_nm,
    hybrid_short_cutoff_factor,
    hybrid_grid_spacing_nm,
    initial_lr_field,
    initial_lr_cosine_axis,
    initial_lr_cosine_mode,
    initial_lr_cosine_amplitude,
)

# Internal compatibility alias. ``profile_solver`` is the public configuration
# name; the integrator currently retains the historical ``profile_v11``
# keyword internally.
if hasattr(run_config, "profile_solver"):
    profile_v11 = bool(run_config.profile_solver)
elif hasattr(run_config, "profile_v11"):
    profile_v11 = bool(run_config.profile_v11)
else:
    profile_v11 = False


from monopole_precip.thermal_profile import linear_cooling

def _resolve_hybrid_grid(box_lengths, target_spacing_nm):
    """Return an FFT-friendly rectangular grid and its actual spacings."""
    box_lengths = np.asarray(box_lengths, dtype=float)
    target_spacing_nm = float(target_spacing_nm)
    if box_lengths.shape != (3,) or np.any(box_lengths <= 0.0):
        raise ValueError("box must contain three positive lengths.")
    if not np.isfinite(target_spacing_nm) or target_spacing_nm <= 0.0:
        raise ValueError("hybrid_grid_spacing_nm must be finite and positive.")

    raw = np.maximum(2, np.ceil(box_lengths / target_spacing_nm).astype(int))
    shape = np.asarray([next_fast_len(int(n)) for n in raw], dtype=int)
    spacing = box_lengths / shape
    return shape, spacing


def _diffusivities_nm2_s(T):
    """Return all configured solute diffusivities at T [K] in nm^2/s."""
    T = float(T)
    if not np.isfinite(T) or T <= 0.0:
        raise ValueError("Temperature must be finite and positive.")
    R_gas = 8.314462618
    return D0 * np.exp(-Qdiff / (R_gas * T)) * 1.0e18


def _get_solute_species(name):
    matches = [species for species in alloy.solutes if species.name == name]
    if len(matches) != 1:
        raise ValueError(
            f"Current ternary compatibility bridge requires exactly one "
            f"solute named {name!r}; found {len(matches)}."
        )
    return matches[0]


phase_default = alloy.phases[0]

# Generic species representation. Chemical names are lookup keys only; this
# upper-level configuration does not assume Mg/Si or a fixed number of solutes.
solute_names = tuple(species.name for species in alloy.solutes)
if len(solute_names) == 0:
    raise ValueError("alloy.solutes must contain at least one diffusing solute.")
if len(set(solute_names)) != len(solute_names):
    raise ValueError("alloy.solutes must contain unique species names.")

solute_species = tuple(_get_solute_species(name) for name in solute_names)

# Matrix composition and atomic volume. initial_composition is expressed as
# atomic/site fractions, while the solver state uses number densities.
vatm = float(alloy.matrix_atomic_volume)
cbar0 = np.asarray(
    [initial_composition[name] / vatm for name in solute_names],
    dtype=float,
)

# Generic diffusion parameters, ordered exactly like solute_names.
D0 = np.asarray([species.D0 for species in solute_species], dtype=float)
Qdiff = np.asarray([species.Qdiff for species in solute_species], dtype=float)

# Phase-0 compatibility thermodynamic scalars. Phase objects remain the source
# of truth; aa/bb/cc are retained only because the current thermal-history API
# still carries these legacy phase-0 arguments.
vatp = float(phase_default.atomic_volume)
aa = float(phase_default.solubility_aa)
bb = float(phase_default.solubility_bb)
cc = float(phase_default.solubility_cc)
gamma = float(phase_default.gamma)

cp = np.asarray(
    [phase_default.number_concentration(name) for name in solute_names],
    dtype=float,
)

# Phase-specific precipitate number concentrations.  The phase-0 row is
# retained as ``cp`` for compatibility with the current upper-level API.
cp_phase = np.asarray(
    [
        [phase.number_concentration(name) for name in solute_names]
        for phase in alloy.phases
    ],
    dtype=float,
)

# Matrix/CNT parameters derived directly from the material definition.
# These are material properties rather than per-job launch parameters.
N0 = 1.0 / vatm
a_latt = float(alloy.matrix_lattice_parameter)

# Legacy fallback used only if a phase leaves gb_barrier_factor=None.  Active
# Al-Mg-Si phases define their own factors in material_definition.py, so this
# value does not override phase-specific material physics.
gb_barrier_fallback = 1.0

# ============================================================================
# Command-line arguments
# ============================================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Run the monopole precipitation solver. Numerical/model controls "
            "are read from run_configuration.py; the command line identifies "
            "the particular thermal-history job and its output files."
        )
    )

    # Job-specific output names.
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("monopole_ternary_run.npz"),
        help="Output NPZ filename.",
    )
    parser.add_argument(
        "--dump-file",
        type=Path,
        default=Path("dump_segs.dump"),
        help="OVITO-compatible particle dump filename.",
    )

    # Thermal history for this particular calculation.
    parser.add_argument(
        "--T-start-C",
        dest="T_start_C",
        type=float,
        default=550.0,
        help="Starting temperature in degrees Celsius.",
    )
    parser.add_argument(
        "--T-end-C",
        dest="T_end_C",
        type=float,
        default=100.0,
        help="Final temperature in degrees Celsius.",
    )
    parser.add_argument(
        "--cooling-rate",
        type=float,
        default=1.0,
        help="Cooling rate in degrees Celsius per second.",
    )
    parser.add_argument(
        "--temperature-step",
        type=float,
        default=1.0,
        help="Temperature interval per thermal segment in degrees Celsius.",
    )
    parser.add_argument(
        "--isothermal-time",
        type=float,
        default=None,
        help=(
            "Isothermal hold time in seconds. Required when T-start-C equals "
            "T-end-C."
        ),
    )

    # Random realization for this particular calculation.
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random-number seed used for stochastic nucleation.",
    )

    return parser.parse_args()


# ============================================================================
# Validation
# ============================================================================

def validate_positive_finite(value, option_name):
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{option_name} must be positive and finite.")


def validate_nonnegative_finite(value, option_name):
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{option_name} must be non-negative and finite.")


def validate_positive_integer(value, option_name):
    if value <= 0:
        raise ValueError(f"{option_name} must be a positive integer.")


def validate_arguments(args):
    validate_positive_finite(args.cooling_rate, "--cooling-rate")
    validate_positive_finite(args.temperature_step, "--temperature-step")

    if args.T_start_C < args.T_end_C:
        raise ValueError(
            "--T-start-C must be greater than or equal to --T-end-C."
        )

    if args.T_start_C == args.T_end_C and args.isothermal_time is None:
        raise ValueError(
            "--isothermal-time must be specified when --T-start-C equals "
            "--T-end-C."
        )

    # Validate numerical/model controls imported from run_configuration.py.
    validate_positive_finite(dt_initial, "run_configuration.dt_initial")
    validate_positive_finite(dt_max, "run_configuration.dt_max")
    validate_positive_finite(dt_min, "run_configuration.dt_min")
    validate_positive_finite(dt_safety, "run_configuration.dt_safety")
    if dt_min > dt_max:
        raise ValueError(
            "run_configuration.dt_min cannot exceed run_configuration.dt_max."
        )

    validate_positive_finite(
        isothermal_segment_time_s,
        "run_configuration.isothermal_segment_time_s",
    )

    if not isinstance(enable_nucleation, (bool, np.bool_)):
        raise TypeError("run_configuration.enable_nucleation must be boolean.")
    validate_nonnegative_finite(
        safety_buffer, "run_configuration.safety_buffer"
    )
    if not isinstance(apply_incubation_correction, (bool, np.bool_)):
        raise TypeError(
            "run_configuration.apply_incubation_correction must be boolean."
        )
    validate_positive_finite(
        max_expected_nucleation_events,
        "run_configuration.max_expected_nucleation_events",
    )

    if screening_length is not None:
        validate_positive_finite(
            screening_length, "run_configuration.screening_length"
        )

    if not isinstance(hybrid_transport, (bool, np.bool_)):
        raise TypeError("run_configuration.hybrid_transport must be boolean.")
    validate_positive_finite(
        hybrid_matching_length_nm,
        "run_configuration.hybrid_matching_length_nm",
    )
    validate_positive_finite(
        hybrid_short_cutoff_factor,
        "run_configuration.hybrid_short_cutoff_factor",
    )
    validate_positive_finite(
        hybrid_grid_spacing_nm,
        "run_configuration.hybrid_grid_spacing_nm",
    )

    validate_positive_finite(gamma, "phase gamma")
    for p, phase in enumerate(alloy.phases):
        weights = phase.diffusing_stoichiometric_weights(solute_names)
        if np.any(~np.isfinite(weights)):
            raise ValueError(
                f"Phase {p} ({phase.name!r}) has non-finite diffusing-solute "
                "stoichiometry."
            )
        # Validate the phase-specific GB factor or the legacy fallback.
        phase.resolved_gb_barrier_factor(gb_barrier_fallback)

    validate_positive_finite(N0, "material-derived N0")
    validate_positive_finite(a_latt, "alloy.matrix_lattice_parameter")

    validate_nonnegative_finite(R_remove, "run_configuration.R_remove")
    validate_nonnegative_finite(
        R_remove_buffer, "run_configuration.R_remove_buffer"
    )

    validate_positive_integer(print_every, "run_configuration.print_every")
    validate_positive_integer(
        print_every_segments, "run_configuration.print_every_segments"
    )
    validate_positive_integer(
        dump_seg_every, "run_configuration.dump_seg_every"
    )
    validate_positive_integer(
        save_every_internal_steps,
        "run_configuration.save_every_internal_steps",
    )
    validate_positive_integer(
        save_every_segments, "run_configuration.save_every_segments"
    )


def validate_thermal_profile(temperatures, segment_durations):
    if temperatures.ndim != 1:
        raise ValueError("temperatures must be one-dimensional.")

    if segment_durations.shape != temperatures.shape:
        raise ValueError("temperatures and segment_durations must have the same shape.")

    if len(temperatures) == 0:
        raise ValueError("The generated thermal profile contains no segments.")

    if np.any(~np.isfinite(temperatures)):
        raise ValueError("The thermal profile contains non-finite temperatures.")

    if np.any(temperatures <= 0.0):
        raise ValueError("All thermal-profile temperatures must be positive.")

    if np.any(~np.isfinite(segment_durations)):
        raise ValueError("The thermal profile contains non-finite durations.")

    if np.any(segment_durations <= 0.0):
        raise ValueError("All thermal-segment durations must be positive.")


# ============================================================================
# Saving
# ============================================================================

def describe_value(value):
    value_type = type(value).__name__
    shape = getattr(value, "shape", None)

    if shape is not None:
        return f"{value_type}, shape={shape}"

    try:
        return f"{value_type}, length={len(value)}"
    except TypeError:
        return value_type


def to_saveable_array(value):
    """Convert result values to arrays that can be written safely by np.savez.

    Homogeneous sequences are stored as ordinary numeric arrays.  Ragged
    histories (for example particle arrays whose length changes after
    nucleation) are stored as a one-dimensional object array with one history
    entry per element.  Constructing that object array explicitly avoids
    NumPy attempting to broadcast nested arrays with incompatible shapes.
    """
    if isinstance(value, np.ndarray):
        return value

    if isinstance(value, dict):
        out = np.empty((), dtype=object)
        out[()] = value
        return out

    try:
        array = np.asarray(value)
        if array.dtype != object:
            return array
    except (TypeError, ValueError):
        pass

    if isinstance(value, (list, tuple)):
        out = np.empty(len(value), dtype=object)
        for i, item in enumerate(value):
            out[i] = item
        return out

    out = np.empty((), dtype=object)
    out[()] = value
    return out


def build_run_metadata(
    args,
    temperatures,
    segment_durations,
    *,
    completed_segments,
    total_segments,
    run_complete,
    apply_incubation_correction,
):
    metadata = {
        # Run status
        "completed_segments": np.asarray(completed_segments, dtype=int),
        "total_segments": np.asarray(total_segments, dtype=int),
        "run_complete": np.asarray(run_complete, dtype=bool),

        # Thermal history
        "temperatures": np.asarray(temperatures, dtype=float),
        "temperatures_C": np.asarray(temperatures, dtype=float) - 273.15,
        "segment_durations": np.asarray(segment_durations, dtype=float),

        # Initial state
        "box": np.asarray(box, dtype=float),
        "initial_positions": np.asarray(positions, dtype=float),
        "initial_radii": np.asarray(radii0, dtype=float),
        "initial_phase_id": np.asarray(phase_id0, dtype=np.int32),

        # Material definition
        "alloy_solvent": np.asarray(alloy.solvent),
        "phase_name": np.asarray(phase_default.name),
        "phase_names": np.asarray(
            [phase.name for phase in alloy.phases]
        ),
        "phase_atomic_volumes": np.asarray(
            [phase.atomic_volume for phase in alloy.phases],
            dtype=float,
        ),
        "phase_gammas": np.asarray(
            [phase.gamma for phase in alloy.phases],
            dtype=float,
        ),
        "phase_gb_barrier_factors": np.asarray(
            [
                phase.resolved_gb_barrier_factor(gb_barrier_fallback)
                for phase in alloy.phases
            ],
            dtype=float,
        ),
        "phase_stoichiometry": np.asarray(
            [dict(phase.stoichiometry) for phase in alloy.phases],
            dtype=object,
        ),
        "phase_solubility_aa": np.asarray(
            [phase.solubility_aa for phase in alloy.phases],
            dtype=float,
        ),
        "phase_solubility_bb": np.asarray(
            [phase.solubility_bb for phase in alloy.phases],
            dtype=float,
        ),
        "phase_solubility_cc": np.asarray(
            [phase.solubility_cc for phase in alloy.phases],
            dtype=float,
        ),
        "phase_cp": np.asarray(cp_phase, dtype=float),
        "solute_names": np.asarray([s.name for s in alloy.solutes]),
        "initial_composition": np.asarray(dict(initial_composition), dtype=object),

        # Generic species-indexed state/material data.
        "cbar0": np.asarray(cbar0, dtype=float),
        "cp": np.asarray(cp, dtype=float),
        "D0": np.asarray(D0, dtype=float),
        "Qdiff": np.asarray(Qdiff, dtype=float),

        # Thermodynamic parameters actually used
        "aa": np.asarray(aa, dtype=float),
        "bb": np.asarray(bb, dtype=float),
        "cc": np.asarray(cc, dtype=float),
        "vatm": np.asarray(vatm, dtype=float),
        "vatp": np.asarray(vatp, dtype=float),
        "gamma": np.asarray(gamma, dtype=float),

        # CNT parameters actually used
        "N0": np.asarray(N0, dtype=float),
        "a_latt": np.asarray(a_latt, dtype=float),
        "safety_buffer": np.asarray(safety_buffer, dtype=float),
        "apply_incubation_correction": np.asarray(apply_incubation_correction, dtype=bool),

        # Run controls
        "seed": np.asarray(args.seed, dtype=int),
        "T_start_C": np.asarray(args.T_start_C, dtype=float),
        "T_end_C": np.asarray(args.T_end_C, dtype=float),
        "cooling_rate_C_per_s": np.asarray(args.cooling_rate, dtype=float),
        "temperature_step_C": np.asarray(args.temperature_step, dtype=float),
        "dt_initial": np.asarray(dt_initial, dtype=float),
        "first_step_dt_max": np.asarray(first_step_dt_max, dtype=float),
        "dt_max": np.asarray(dt_max, dtype=float),
        "dt_min": np.asarray(dt_min, dtype=float),
        "dt_safety": np.asarray(dt_safety, dtype=float),
        "isothermal_segment_time_s": np.asarray(
            isothermal_segment_time_s, dtype=float
        ),
        "screening_length": np.asarray(screening_length, dtype=object),
        # Scale-separated transport controls. Store both requested controls
        # and the actual rectangular FFT discretization used.
        "hybrid_transport": np.asarray(hybrid_transport, dtype=bool),
        "hybrid_matching_length_nm": np.asarray(
            hybrid_matching_length_nm, dtype=float
        ),
        "hybrid_alpha_nm_inv": np.asarray(
            1.0 / hybrid_matching_length_nm, dtype=float
        ),
        "hybrid_short_cutoff_factor": np.asarray(
            hybrid_short_cutoff_factor, dtype=float
        ),
        "hybrid_short_cutoff_nm": np.asarray(
            hybrid_short_cutoff_factor * hybrid_matching_length_nm, dtype=float
        ),
        "hybrid_grid_spacing_target_nm": np.asarray(
            hybrid_grid_spacing_nm, dtype=float
        ),
        "hybrid_grid_shape": np.asarray(
            _resolve_hybrid_grid(box, hybrid_grid_spacing_nm)[0], dtype=int
        ),
        "hybrid_grid_spacing_actual_nm": np.asarray(
            _resolve_hybrid_grid(box, hybrid_grid_spacing_nm)[1], dtype=float
        ),
        "initial_lr_field": np.asarray(initial_lr_field),
        "initial_lr_cosine_axis": np.asarray(initial_lr_cosine_axis),
        "initial_lr_cosine_mode": np.asarray(initial_lr_cosine_mode, dtype=int),
        "initial_lr_cosine_amplitude": np.asarray(
            dict(initial_lr_cosine_amplitude), dtype=object
        ),
        "max_expected_nucleation_events": np.asarray(
            max_expected_nucleation_events, dtype=float
        ),
        # Legacy metadata alias retained for old analysis scripts.
        "lam_target": np.asarray(
            max_expected_nucleation_events, dtype=float
        ),
        "gb_barrier_fallback": np.asarray(gb_barrier_fallback, dtype=float),
        # Legacy metadata key retained for older analysis scripts.
        "barrier_factor": np.asarray(gb_barrier_fallback, dtype=float),
        "nucleation_spatial_rate_correction": np.asarray(
            nucleation_spatial_rate_correction, dtype=bool
        ),
        "nucleation_spatial_rate_rse_target": np.asarray(
            nucleation_spatial_rate_rse_target, dtype=float
        ),
        "nucleation_placement_ess_per_event": np.asarray(
            nucleation_placement_ess_per_event, dtype=float
        ),
        "save_nucleation_diagnostics": np.asarray(
            save_nucleation_diagnostics, dtype=bool
        ),
        "R_remove": np.asarray(R_remove, dtype=float),
        "R_remove_buffer": np.asarray(R_remove_buffer, dtype=float),
        "enable_nucleation": np.asarray(enable_nucleation, dtype=bool),

        # Output controls
        "save_every_internal_steps": np.asarray(save_every_internal_steps, dtype=int),
        "save_every_segments": np.asarray(save_every_segments, dtype=int),
        "dump_times_s": np.asarray(
            [] if dump_times_s is None else dump_times_s, dtype=float
        ),
        "dump_times_enabled": np.asarray(dump_times_s is not None, dtype=bool),
        "dump_include_final": np.asarray(dump_include_final, dtype=bool),
        "dump_seg_every": np.asarray(dump_seg_every, dtype=int),
        "print_every": np.asarray(print_every, dtype=int),
        "print_every_segments": np.asarray(print_every_segments, dtype=int),
        "profile_v11": np.asarray(profile_v11, dtype=bool),
        "dump_seg_filename": np.asarray(str(args.dump_file)),
    }

    # Preserve old Mg-Si output keys only when those species are present.
    # They are compatibility aliases derived from generic arrays and are not
    # inputs to the physics.
    if "Mg" in solute_names and "Si" in solute_names:
        mg_index = solute_names.index("Mg")
        si_index = solute_names.index("Si")
        metadata.update({
            "CtotMg": np.asarray(cbar0[mg_index], dtype=float),
            "CtotSi": np.asarray(cbar0[si_index], dtype=float),
            "CpMg": np.asarray(cp[mg_index], dtype=float),
            "CpSi": np.asarray(cp[si_index], dtype=float),
            "D0Mg": np.asarray(D0[mg_index], dtype=float),
            "QMg": np.asarray(Qdiff[mg_index], dtype=float),
            "D0Si": np.asarray(D0[si_index], dtype=float),
            "QSi": np.asarray(Qdiff[si_index], dtype=float),
            "x": np.asarray(phase_default.stoichiometry["Mg"], dtype=int),
            "y": np.asarray(phase_default.stoichiometry["Si"], dtype=int),
        })

    return metadata


def save_result(
    result,
    output_file,
    temperatures,
    segment_durations,
    args,
    *,
    completed_segments,
    total_segments,
    run_complete,
    apply_incubation_correction,
):
    if not isinstance(result, dict):
        raise TypeError("Expected integrate_ternary_thermal_history() to return a dictionary.")

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    temporary_file = output_file.with_name(output_file.name + ".tmp.npz")

    save_data = build_run_metadata(
        args=args,
        temperatures=temperatures,
        segment_durations=segment_durations,
        completed_segments=completed_segments,
        total_segments=total_segments,
        run_complete=run_complete,
        apply_incubation_correction=apply_incubation_correction,
    )

    for name, value in result.items():
        save_data[name] = to_saveable_array(value)

    np.savez_compressed(temporary_file, **save_data)
    os.replace(temporary_file, output_file)


# ============================================================================
# Reporting
# ============================================================================

def print_run_summary(
    args,
    output_file,
    dump_file,
    temperatures,
    segment_durations,
    apply_incubation_correction,
):
    characteristic_length = float(np.prod(box) ** (1.0 / 3.0))
    total_time = float(np.sum(segment_durations))

    print("=" * 72)
    print("Ternary monopole precipitation simulation")
    print("=" * 72)
    print(f"Output file               : {output_file}")
    print(f"OVITO dump file           : {dump_file}")
    print(f"Random seed               : {args.seed}")
    print()

    print("Thermal history")
    print(f"  Start temperature       : {args.T_start_C:.3f} C")
    print(f"  End temperature         : {args.T_end_C:.3f} C")
    print(f"  Cooling rate            : {args.cooling_rate:.6e} C/s")
    print(f"  Temperature step        : {args.temperature_step:.6e} C")
    print(f"  Thermal segments        : {len(temperatures)}")
    print(f"  Total duration          : {total_time:.6e} s")
    print()

    print("Initial configuration")
    print(f"  Initial particles       : {len(radii0)}")

    phase_id0_array = np.asarray(phase_id0, dtype=np.int32)
    if phase_id0_array.shape != np.asarray(radii0).shape:
        raise ValueError(
            "run_configuration.phase_id0 must have the same shape as radii0."
        )
    if np.any(phase_id0_array < 0) or np.any(phase_id0_array >= len(alloy.phases)):
        raise ValueError(
            "run_configuration.phase_id0 contains an invalid phase index."
        )

    for p, phase in enumerate(alloy.phases):
        n_p = int(np.count_nonzero(phase_id0_array == p))
        print(f"  Initial phase {p:<2d}       : {phase.name} (N={n_p})")

    print(f"  Box                     : {np.asarray(box)}")
    print(f"  Characteristic size     : {characteristic_length:.6e} nm")
    print()

    hybrid_grid_shape, hybrid_grid_spacing_actual = _resolve_hybrid_grid(
        box, hybrid_grid_spacing_nm
    )
    print("Transport formulation")
    if hybrid_transport:
        print("  Formulation             : scale-separated")
        print(f"  Matching length ell     : {hybrid_matching_length_nm:.6e} nm")
        print(
            f"  Short-range cutoff      : "
            f"{hybrid_short_cutoff_factor * hybrid_matching_length_nm:.6e} nm "
            f"({hybrid_short_cutoff_factor:g} ell)"
        )
        print(f"  LR target grid spacing  : {hybrid_grid_spacing_nm:.6e} nm")
        print(f"  LR FFT grid             : {tuple(hybrid_grid_shape)}")
        print(f"  LR actual spacing       : {hybrid_grid_spacing_actual} nm")
        print()
    else:
        print("  Formulation             : direct quasistatic")
        print()

    print("Periodic Green function")
    if screening_length is None:
        print("  Kernel                  : " 
              "unscreened periodic Laplace")
        print("  Screening length        : None")
        print("  Screening kappa         : 0.000000e+00 nm^-1"
        )
    else:
        print("  Kernel                  : "
              "zero-mean screened periodic Yukawa")
        print(f"  Screening length        : "
              f"{screening_length:.6e} nm")
        print(f"  Screening kappa         : "
              f"{1.0 / screening_length:.6e} nm^-1")
    print()
    
    print("Material system")
    print(f"  Solvent                 : {alloy.solvent}")
    print(f"  Solutes                 : {[s.name for s in alloy.solutes]}")
    print(f"  Precipitate phases      : {len(alloy.phases)}")
    for p, phase in enumerate(alloy.phases):
        stoich = ", ".join(
            f"{name}:{phase.stoichiometry.get(name, 0.0):g}"
            for name in solute_names
        )
        thermo = phase.thermodynamics
        aa_p = getattr(thermo, "aa", np.nan)
        bb_p = getattr(thermo, "bb", np.nan)
        cc_p = getattr(thermo, "cc", np.nan)
        print(
            f"  Phase {p:<2d}              : {phase.name}; "
            f"stoich=({stoich}); vatp={phase.atomic_volume:.6e}; "
            f"gamma={phase.gamma:.6e}; "
            f"gb_factor={phase.resolved_gb_barrier_factor(gb_barrier_fallback):.6e}; "
            f"aa={aa_p:.6e}; bb={bb_p:.6e}; cc={cc_p:.6e}"
        )
    print()

    print("Thermodynamics and CNT")
    print(f"  CNT phase              : 0 ({phase_default.name})")
    print(f"  Gamma                   : {gamma:.6e}")
    phase0_stoich = ", ".join(
        f"{name}:{phase_default.stoichiometry.get(name, 0.0):g}"
        for name in solute_names
    )
    print(f"  Diffusing stoichiometry : {phase0_stoich}")
    print(f"  N0                      : {N0:.6e}")
    print(f"  Lattice parameter       : {a_latt:.6e}")
    print(f"  Safety buffer           : {safety_buffer:.6e}")
    print(f"  Incubation correction   : {apply_incubation_correction}")
    print(f"  GB barrier fallback     : {gb_barrier_fallback:.6e}")
    print()

    print("Time stepping")
    print(f"  Initial dt              : {dt_initial:.6e} s")
    print(f"  First-step dt cap       : {first_step_dt_max:.6e} s")
    print(f"  Maximum dt              : {dt_max:.6e} s")
    print(f"  Minimum dt              : {dt_min:.6e} s")
    print(f"  Growth safety           : {dt_safety:.6e}")
    print(
        "  Max expected nuc events : "
        f"{max_expected_nucleation_events:.6e}"
    )
    print()

    print("Model controls")
    print(f"  Nucleation enabled      : {enable_nucleation}")
    print(f"  Particle removal radius : {R_remove:.6e}")
    print(f"  Removal radius buffer   : {R_remove_buffer:.6e}")
    print(
        f"  Early removal radius    : "
        f"{R_remove + R_remove_buffer:.6e}"
    )
    print("=" * 72)


# ============================================================================
# Main execution
# ============================================================================

def main():
    args = parse_arguments()
    validate_arguments(args)

    output_file = args.output.expanduser().resolve()
    dump_file = args.dump_file.expanduser().resolve()

    # Compact-output architecture:
    #   * output_file: current restart/checkpoint state only
    #   * history_csv_file: compact scalar/moment kinetics history
    #   * dump_file: particle snapshots including Q
    #   * field_snapshot_dir: native LR concentration-grid snapshots at the
    #     same particle-dump times.
    history_csv_file = output_file.with_name(output_file.stem + "_history.csv")
    field_snapshot_dir = output_file.with_name(output_file.stem + "_lr_fields")

    if save_nucleation_diagnostics:
        nucleation_diagnostics_file = output_file.with_name(
            f"{output_file.stem}_nucleation_diagnostics.csv"
        )
        # A new top-level run owns this diagnostics file.  Remove an older file
        # with the same name so repeated runs do not silently concatenate data.
        if nucleation_diagnostics_file.exists():
            nucleation_diagnostics_file.unlink()
    else:
        nucleation_diagnostics_file = None

    output_file.parent.mkdir(parents=True, exist_ok=True)
    dump_file.parent.mkdir(parents=True, exist_ok=True)

    # Save the resolved dump path in the metadata.
    args.dump_file = dump_file

    temperatures, segment_durations = linear_cooling(
        T_start_C=args.T_start_C,
        T_end_C=args.T_end_C,
        cooling_rate_C_per_s=args.cooling_rate,
        temperature_step_C=args.temperature_step,
        isothermal_time_s=args.isothermal_time,
        isothermal_segment_time_s=isothermal_segment_time_s,
    )

    temperatures = np.asarray(temperatures, dtype=float)
    segment_durations = np.asarray(segment_durations, dtype=float)

    validate_thermal_profile(temperatures, segment_durations)

    n_segments = len(temperatures)
    rng = np.random.default_rng(args.seed)
    characteristic_length = float(np.prod(box) ** (1.0 / 3.0))

    # Construct the long-range field once for the complete thermal history.
    # The same object is passed through every cooling/hold segment, so its
    # concentration field is continuous in time.  The integrator updates only
    # D(T) at segment boundaries.
    hybrid_grid_shape, hybrid_grid_spacing_actual = _resolve_hybrid_grid(
        box, hybrid_grid_spacing_nm
    )
    hybrid_long_range_field = None
    if hybrid_transport:
        D_initial = _diffusivities_nm2_s(float(temperatures[0]))
        hybrid_long_range_field = PeriodicLongRangeField(
            box=box,
            grid_shape=hybrid_grid_shape,
            alpha=1.0 / float(hybrid_matching_length_nm),
            diffusivities=D_initial,
            initial_concentrations=cbar0,
            positions=positions,
        )

        lr_kind = str(initial_lr_field).strip().lower()
        if lr_kind == "uniform":
            pass
        elif lr_kind == "cosine":
            amplitudes = np.asarray(
                [
                    float(initial_lr_cosine_amplitude.get(name, 0.0))
                    for name in solute_names
                ],
                dtype=float,
            )
            hybrid_long_range_field.set_cosine_initial_condition(
                amplitudes=amplitudes,
                axis=initial_lr_cosine_axis,
                mode=initial_lr_cosine_mode,
            )
            mean_after = hybrid_long_range_field.mean_concentration()
            if not np.allclose(mean_after, cbar0, rtol=5.0e-14, atol=1.0e-14):
                raise RuntimeError(
                    "Initial LR cosine field changed the mean composition: "
                    f"before={cbar0}, after={mean_after}."
                )
            print("  Initial LR field        : cosine")
            print(f"  Initial LR cosine axis  : {initial_lr_cosine_axis}")
            print(f"  Initial LR cosine mode  : {initial_lr_cosine_mode}")
            print(
                "  Initial LR amplitudes   : "
                + ", ".join(
                    f"{name}={amp:.6g}"
                    for name, amp in zip(solute_names, amplitudes)
                )
            )
            print(f"  Initial LR mean check   : {mean_after}")
        else:
            raise ValueError(
                "initial_lr_field must be 'uniform' or 'cosine'; "
                f"got {initial_lr_field!r}."
            )

    print_run_summary(
        args=args,
        output_file=output_file,
        dump_file=dump_file,
        temperatures=temperatures,
        segment_durations=segment_durations,
        apply_incubation_correction=apply_incubation_correction,
    )

    def field_snapshot_callback(
        *, actual_global_time, requested_global_time, long_range_field,
        long_range_field_array=None, temperature_K,
    ):
        if long_range_field is None:
            return
        field_snapshot_dir.mkdir(parents=True, exist_ok=True)
        # Actual accepted time is authoritative.  The requested time is stored
        # as metadata for target-time dump schedules.
        tag = f"t{float(actual_global_time):017.9f}s"
        filename = field_snapshot_dir / f"lr_{tag}.npz"
        write_hybrid_long_range_snapshot_npz(
            filename,
            time_s=float(actual_global_time),
            temperature_K=float(temperature_K),
            long_range_field=long_range_field,
            field_array=long_range_field_array,
            solute_names=solute_names,
            matching_length_nm=hybrid_matching_length_nm,
            box=box,
            extra_metadata={
                "requested_time_s": float(requested_global_time),
                "random_seed": int(args.seed),
            },
        )
        print(f"LR field snapshot saved: {filename}")

    def periodic_save_callback(
        partial_result,
        *,
        completed_segments,
        total_segments,
        run_complete,
    ):
        save_result(
            result=partial_result,
            output_file=output_file,
            temperatures=temperatures,
            segment_durations=segment_durations,
            args=args,
            completed_segments=completed_segments,
            total_segments=total_segments,
            run_complete=run_complete,
            apply_incubation_correction=apply_incubation_correction,
        )

        print(
            f"Partial results saved to {output_file} after segment "
            f"{completed_segments}/{total_segments}."
        )

    result = integrate_ternary_thermal_history(
        # Thermal profile
        temperatures=temperatures,
        segment_durations=segment_durations,

        # Initial state
        R0=radii0,
        positions=positions,
        phase_id0=phase_id0,
        box=box,

        # Solute properties
        solute_names=solute_names,
        cbar0=cbar0,
        cp=cp,
        cp_phase=cp_phase,
        D0=D0,
        Qdiff=Qdiff,
        phases=alloy.phases,

        # Thermodynamic parameters
        aa=aa,
        bb=bb,
        cc=cc,
        vatm=vatm,

        # CNT/model parameters. Phase-specific material properties
        # (atomic volume, gamma, stoichiometry) are owned by phases.
        N0=N0,
        a_latt=a_latt,
        safety_buffer=safety_buffer,
        apply_incubation_correction=apply_incubation_correction,

        # Time stepping
        dt=dt_initial,
        first_step_dt_max=first_step_dt_max,
        adaptive_dt=True,
        dt_max=dt_max,
        dt_min=dt_min,
        dt_safety=dt_safety,
        # Nucleation
        enable_nucleation=enable_nucleation,
        lam_target=max_expected_nucleation_events,
        # Internal adaptive floor defaults to 10% of the event budget.
        lam_target_min=None,
        barrier_factor=gb_barrier_fallback,
        nucleation_spatial_rate_correction=(
            nucleation_spatial_rate_correction
        ),
        nucleation_spatial_rate_rse_target=(
            nucleation_spatial_rate_rse_target
        ),
        nucleation_placement_ess_per_event=(
            nucleation_placement_ess_per_event
        ),
        nucleation_diagnostics_file=nucleation_diagnostics_file,

        # Particle removal
        remove_extinct=True,
        R_remove=R_remove,
        R_remove_buffer=R_remove_buffer,

        # Scale-separated transport. hybrid_long_range_field is a persistent
        # object shared by every thermal segment.
        hybrid_transport=hybrid_transport,
        hybrid_long_range_field=hybrid_long_range_field,
        hybrid_matching_length=hybrid_matching_length_nm,
        hybrid_grid_shape=tuple(int(n) for n in hybrid_grid_shape),
        hybrid_short_cutoff=(
            hybrid_short_cutoff_factor * hybrid_matching_length_nm
        ),

        # Ewald and Fourier-space parameters used by the direct quasistatic
        # formulation.
        alpha=6.0 / characteristic_length,
        n_real=3,
        n_k=6,
        screening_length=screening_length,

        # Solver tolerances
        check_conservation=True,
        conservation_tol=1.0e-6,
        gmres_rtol=1.0e-5,
        gmres_atol=0.0,
        gmres_maxiter=200,
        gmres_restart=50,
        nonlinear_xtol=1.0e-5,
        nonlinear_ftol=1.0e-5,
        nonlinear_gtol=1.0e-5,
        nonlinear_max_nfev=500,
        velocity_mismatch_tolerance=1.0e-4,
        thermodynamic_tolerance=1.0e-5,
        use_preconditioner=True,

        # Random-number generator
        rng=rng,

        # Diagnostics
        show_progress=False,
        print_segments=True,
        print_every_segments=print_every_segments,
        print_diagnostics=True,
        print_every=print_every,
        profile_v11=profile_v11,

        # Internal particle-history frequency plus sparse LR diagnostic history.
        # These options do not alter the live LR field used by the solver.
        save_every=save_every_internal_steps,
        store_hybrid_long_range_history=store_hybrid_long_range_history,
        hybrid_long_range_history_every_segments=(
            hybrid_long_range_history_every_segments
        ),
        hybrid_long_range_history_spacing_nm=(
            hybrid_long_range_history_spacing_nm
        ),
        hybrid_long_range_history_dtype=hybrid_long_range_history_dtype,

        # Grain-boundary geometry
        gb_point=gb_point,
        gb_normal=gb_normal,
        gb_thickness=gb_thickness,

        # Initial particle origin types
        nuc_kind0=None,

        # OVITO particle output.  If dump_times_s is supplied, frames are
        # selected at the nearest accepted internal states, independently of
        # thermal-segment boundaries.  dump_seg_every remains the legacy
        # fallback when dump_times_s is None.
        dump_seg_filename=str(dump_file),
        dump_seg_every=dump_seg_every,
        dump_times=dump_times_s,
        dump_include_final=dump_include_final,
        history_csv_filename=str(history_csv_file),
        field_snapshot_callback=(
            field_snapshot_callback
            if save_hybrid_long_range_field_snapshots
            else None
        ),

        # Periodic NPZ output
        periodic_save_callback=periodic_save_callback,
        periodic_save_every_segments=save_every_segments,
    )

    if not isinstance(result, dict):
        raise TypeError("Expected integrate_ternary_thermal_history() to return a dictionary.")

    print()
    print("=" * 72)
    print("Simulation completed")
    print("=" * 72)
    print(f"Integrator returned {len(result)} entries:")

    for name, value in result.items():
        print(f"  {name}: {describe_value(value)}")

    save_result(
        result=result,
        output_file=output_file,
        temperatures=temperatures,
        segment_durations=segment_durations,
        args=args,
        completed_segments=n_segments,
        total_segments=n_segments,
        run_complete=True,
        apply_incubation_correction=apply_incubation_correction,
    )

    print()
    print(f"Current-state checkpoint: {output_file}")
    print(f"Compact kinetics history: {history_csv_file}")
    if nucleation_diagnostics_file is not None:
        print(f"Nucleation diagnostics saved to: {nucleation_diagnostics_file}")
    print(f"Particle dump at         : {dump_file}")
    if save_hybrid_long_range_field_snapshots:
        print(f"LR field snapshots       : {field_snapshot_dir}")
    else:
        print("LR field snapshots       : disabled")
    print("=" * 72)


if __name__ == "__main__":
    main()
