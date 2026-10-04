"""
Load saved precipitation states from a thermal-history NPZ file.

The simulation output contains:

- top-level global histories;
- per-segment integrator results;
- a final_aux dictionary containing the final restart state.

This module hides the nested object-array structure and returns one
consistent state suitable for concentration-field reconstruction.
"""

from pathlib import Path

import numpy as np


def _as_object(value):
    """
    Unwrap a zero-dimensional NumPy object array.
    """
    if isinstance(value, np.ndarray) and value.shape == () and value.dtype == object:
        return value.item()

    return value


def _as_float(value, name):
    """
    Convert a scalar value to float and require it to be finite.
    """
    value = float(value)

    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite.")

    return value


def _as_array(value, name, *, dtype=None, ndim=None):
    """
    Convert a value to a NumPy array and perform basic validation.
    """
    array = np.asarray(value, dtype=dtype)

    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional; got shape {array.shape}.")

    if np.issubdtype(array.dtype, np.floating) and np.any(~np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values.")

    return array


def _segment_entries(data):
    """
    Return the saved segment dictionaries as a Python list.
    """
    segments = data["segments"]

    if segments.ndim != 1:
        raise ValueError(f"segments must be one-dimensional; got shape {segments.shape}.")

    entries = []

    for i, entry in enumerate(segments):
        entry = _as_object(entry)

        if not isinstance(entry, dict):
            raise TypeError(f"segments[{i}] must be a dictionary; got {type(entry).__name__}.")

        if "result" not in entry:
            raise KeyError(f"segments[{i}] does not contain a 'result' entry.")

        entries.append(entry)

    return entries


def _segment_result(entry, segment_array_index):
    """
    Validate and return one integrator result tuple.
    """
    result = _as_object(entry["result"])

    if not isinstance(result, tuple):
        raise TypeError(
            f"segments[{segment_array_index}]['result'] must be a tuple; "
            f"got {type(result).__name__}."
        )

    if len(result) < 14:
        raise ValueError(
            f"segments[{segment_array_index}]['result'] has length {len(result)}; "
            "at least 14 entries are required."
        )

    return result


def _extract_shared_data(data):
    """
    Extract data that do not depend on the requested simulation time.
    """
    required_top_level = [
        "box",
        "initial_positions",
        "initial_radii",
        "final_aux",
    ]

    for key in required_top_level:
        if key not in data.files:
            raise KeyError(f"Saved NPZ file does not contain required key {key!r}.")

    final_aux = _as_object(data["final_aux"])

    if not isinstance(final_aux, dict):
        raise TypeError("final_aux must contain a dictionary.")

    required_aux = [
        "alpha",
        "real_lattice",
        "kvecs",
        "k2",
    ]

    for key in required_aux:
        if key not in final_aux:
            raise KeyError(f"final_aux does not contain required key {key!r}.")

    # --------------------------------------------------------------
    # Green-kernel metadata
    #
    # Older simulation files do not contain these keys. Treat those
    # files as using the original unscreened periodic Laplace kernel.
    # --------------------------------------------------------------
    green_kernel_type = final_aux.get(
        "green_kernel_type",
        "periodic_laplace",
    )

    green_kernel_type = str(
        green_kernel_type
    )

    screening_length = final_aux.get(
        "screening_length",
        None,
    )

    if screening_length is not None:
        screening_length = float(
            screening_length
        )

        if (
            not np.isfinite(screening_length)
            or screening_length <= 0.0
        ):
            raise ValueError(
                "final_aux['screening_length'] must be positive "
                "and finite, or None."
            )

    screening_kappa = final_aux.get(
        "screening_kappa",
        (
            0.0
            if screening_length is None
            else 1.0 / screening_length
        ),
    )

    screening_kappa = float(
        screening_kappa
    )

    if (
        not np.isfinite(screening_kappa)
        or screening_kappa < 0.0
    ):
        raise ValueError(
            "final_aux['screening_kappa'] must be finite and "
            "non-negative."
        )

    # --------------------------------------------------------------
    # Scale-separated transport metadata. These keys are optional so
    # historical V8/V10 quasi-static files remain fully supported.
    # --------------------------------------------------------------
    hybrid_transport = bool(final_aux.get("hybrid_transport", False))
    hybrid_matching_length = final_aux.get("hybrid_matching_length", None)
    hybrid_grid_shape = final_aux.get("hybrid_grid_shape", None)
    hybrid_short_cutoff = final_aux.get("hybrid_short_cutoff", None)

    if hybrid_transport:
        if hybrid_matching_length is None:
            raise KeyError(
                "Scale-separated output is missing "
                "final_aux['hybrid_matching_length']."
            )
        hybrid_matching_length = float(hybrid_matching_length)
        if (
            not np.isfinite(hybrid_matching_length)
            or hybrid_matching_length <= 0.0
        ):
            raise ValueError(
                "hybrid_matching_length must be finite and positive."
            )

        if hybrid_grid_shape is None:
            raise KeyError(
                "Scale-separated output is missing "
                "final_aux['hybrid_grid_shape']."
            )
        hybrid_grid_shape = np.asarray(hybrid_grid_shape, dtype=int)
        if hybrid_grid_shape.shape != (3,) or np.any(hybrid_grid_shape < 2):
            raise ValueError(
                "hybrid_grid_shape must have shape (3,) with entries >= 2."
            )

        if hybrid_short_cutoff is None:
            # Backward-compatible convention used by earlier scale-separated output.
            hybrid_short_cutoff = 3.0 * hybrid_matching_length
        hybrid_short_cutoff = float(hybrid_short_cutoff)
        if not np.isfinite(hybrid_short_cutoff) or hybrid_short_cutoff <= 0.0:
            raise ValueError(
                "hybrid_short_cutoff must be finite and positive."
            )
    else:
        hybrid_matching_length = None
        hybrid_grid_shape = None
        hybrid_short_cutoff = None
    
    # --------------------------------------------------------------
    # Species metadata
    #
    # V10 stores the authoritative species-indexed metadata as
    # solute_names and cp.  Older V8/Mg-Si files contain only the
    # historical CpMg/CpSi scalars, so reconstruct the generic arrays
    # from those keys when needed.
    # --------------------------------------------------------------
    if "solute_names" in data.files and "cp" in data.files:
        solute_names = tuple(str(name) for name in np.asarray(data["solute_names"]).tolist())
        cp = _as_array(data["cp"], "cp", dtype=float, ndim=1).copy()

        if len(solute_names) == 0:
            raise ValueError("solute_names must contain at least one transported species.")

        if len(set(solute_names)) != len(solute_names):
            raise ValueError("solute_names contains duplicate species names.")

        if cp.shape != (len(solute_names),):
            raise ValueError(
                f"cp must have shape ({len(solute_names)},); got {cp.shape}."
            )
    else:
        for key in ("CpMg", "CpSi"):
            if key not in data.files:
                raise KeyError(
                    "Saved NPZ file contains neither generic solute_names/cp "
                    f"metadata nor legacy key {key!r}."
                )

        solute_names = ("Mg", "Si")
        cp = np.asarray(
            [
                _as_float(data["CpMg"], "CpMg"),
                _as_float(data["CpSi"], "CpSi"),
            ],
            dtype=float,
        )

    shared = {
        "box": _as_array(data["box"], "box", dtype=float, ndim=1),
        "solute_names": solute_names,
        "cp": cp,
        "alpha": _as_float(final_aux["alpha"], "alpha"),

        "green_kernel_type":
            green_kernel_type,

        "screening_length":
            screening_length,

        "screening_kappa":
            screening_kappa,

        "hybrid_transport": hybrid_transport,
        "hybrid_matching_length": hybrid_matching_length,
        "hybrid_grid_shape": (
            None if hybrid_grid_shape is None else hybrid_grid_shape.copy()
        ),
        "hybrid_short_cutoff": hybrid_short_cutoff,

        "real_lattice": _as_array(
            final_aux["real_lattice"],
            "real_lattice",
            dtype=float,
            ndim=2,
        ),
        "kvecs": _as_array(
            final_aux["kvecs"],
            "kvecs",
            dtype=float,
            ndim=2,
        ),
        "k2": _as_array(
            final_aux["k2"],
            "k2",
            dtype=float,
            ndim=1,
        ),
        "initial_positions": _as_array(
            data["initial_positions"],
            "initial_positions",
            dtype=float,
            ndim=2,
        ),
        "initial_radii": _as_array(
            data["initial_radii"],
            "initial_radii",
            dtype=float,
            ndim=1,
        ),
        "final_aux": final_aux,
    }

    # Historical Mg/Si aliases are exposed only when those species exist.
    if "Mg" in solute_names:
        shared["cp_mg"] = float(cp[solute_names.index("Mg")])

    if "Si" in solute_names:
        shared["cp_si"] = float(cp[solute_names.index("Si")])

    if shared["box"].shape != (3,):
        raise ValueError(f"box must have shape (3,); got {shared['box'].shape}.")

    if shared["initial_positions"].shape != (len(shared["initial_radii"]), 3):
        raise ValueError(
            "initial_positions must have shape "
            f"({len(shared['initial_radii'])}, 3); "
            f"got {shared['initial_positions'].shape}."
        )

    if shared["real_lattice"].shape[1] != 3:
        raise ValueError(
            "real_lattice must have shape (Nreal, 3); "
            f"got {shared['real_lattice'].shape}."
        )

    if shared["kvecs"].shape[1] != 3:
        raise ValueError(f"kvecs must have shape (Nk, 3); got {shared['kvecs'].shape}.")

    if shared["k2"].shape != (len(shared["kvecs"]),):
        raise ValueError(
            "k2 must have shape (len(kvecs),); "
            f"got {shared['k2'].shape}."
        )

    return shared


def _state_times_and_locations(data, segments):
    """
    Build the global saved-state times and their segment/local locations.

    The initial point of each segment after the first is omitted because
    it duplicates the final point of the preceding segment.
    """
    times = []
    locations = []

    for segment_array_index, entry in enumerate(segments):
        result = _segment_result(entry, segment_array_index)
        t_local = _as_array(result[0], "segment t_hist", dtype=float, ndim=1)

        global_start_time = _as_float(
            entry.get("global_start_time", 0.0),
            f"segments[{segment_array_index}].global_start_time",
        )

        start_local_index = 0 if segment_array_index == 0 else 1

        for local_index in range(start_local_index, len(t_local)):
            times.append(global_start_time + float(t_local[local_index]))
            locations.append((segment_array_index, local_index))

    times = np.asarray(times, dtype=float)

    if np.any(~np.isfinite(times)):
        raise ValueError("The reconstructed saved-state times contain non-finite values.")

    if len(times) > 1 and np.any(np.diff(times) < 0.0):
        raise ValueError("The reconstructed saved-state times are not monotonically increasing.")

    return times, locations


def _select_global_index(
    times,
    *,
    state,
    index,
    time,
    time_selection,
    exact_time_tolerance,
):
    """
    Select one global saved-state index.
    """
    supplied = sum(
        value is not None
        for value in (
            index,
            time,
        )
    )

    if supplied > 1:
        raise ValueError("Specify only one of index or time.")

    if index is not None:
        index = int(index)

        if index < 0:
            index += len(times)

        if index < 0 or index >= len(times):
            raise IndexError(
                f"State index {index} is outside the available range "
                f"0 to {len(times) - 1}."
            )

        return index

    if time is not None:
        requested_time = _as_float(time, "time")
        selection = str(time_selection).lower()

        if selection == "nearest":
            return int(np.argmin(np.abs(times - requested_time)))

        if selection == "before":
            candidates = np.flatnonzero(times <= requested_time)

            if len(candidates) == 0:
                raise ValueError(
                    f"No saved state occurs at or before t={requested_time:.16e}."
                )

            return int(candidates[-1])

        if selection == "after":
            candidates = np.flatnonzero(times >= requested_time)

            if len(candidates) == 0:
                raise ValueError(
                    f"No saved state occurs at or after t={requested_time:.16e}."
                )

            return int(candidates[0])

        if selection == "exact":
            matches = np.flatnonzero(
                np.isclose(
                    times,
                    requested_time,
                    rtol=0.0,
                    atol=float(exact_time_tolerance),
                )
            )

            if len(matches) == 0:
                raise ValueError(
                    f"No saved state matches t={requested_time:.16e} "
                    f"within absolute tolerance {exact_time_tolerance:.3e}."
                )

            return int(matches[0])

        raise ValueError(
            "time_selection must be one of "
            "'nearest', 'before', 'after', or 'exact'."
        )

    state = str(state).lower()

    if state == "final":
        return len(times) - 1

    if state == "initial":
        return 0

    raise ValueError(
        "state must be 'initial' or 'final' when index and time are not supplied."
    )


def _positions_for_local_state(
    data,
    segments,
    segment_array_index,
    local_index,
    result,
):
    """
    Recover particle positions for one segment-local state.

    positions_out contains post-timestep states, so local state zero uses
    the segment-start particle positions.
    """
    positions_out = result[11]

    if local_index == 0:
        if segment_array_index == 0:
            return np.asarray(data["initial_positions"], dtype=float).copy()

        previous_result = _segment_result(
            segments[segment_array_index - 1],
            segment_array_index - 1,
        )

        previous_aux = _as_object(previous_result[7])

        if not isinstance(previous_aux, dict):
            raise TypeError("The preceding segment auxiliary result must be a dictionary.")

        return np.asarray(previous_aux["positions_final"], dtype=float).copy()

    output_index = local_index - 1

    if output_index >= len(positions_out):
        raise IndexError(
            "The requested local state has no matching positions_out entry: "
            f"local_index={local_index}, len(positions_out)={len(positions_out)}."
        )

    return np.asarray(positions_out[output_index], dtype=float).copy()


def _particle_type_for_local_state(
    data,
    segments,
    segment_array_index,
    local_index,
    result,
    n_particles,
):
    """
    Recover particle-type data for one state.
    """
    nuc_kind_out = result[12]

    if len(nuc_kind_out) == len(result[0]):
        particle_type = np.asarray(nuc_kind_out[local_index], dtype=np.int32)

    elif local_index == 0:
        if segment_array_index == 0:
            particle_type = np.zeros(n_particles, dtype=np.int32)
        else:
            previous_result = _segment_result(
                segments[segment_array_index - 1],
                segment_array_index - 1,
            )

            previous_aux = _as_object(previous_result[7])
            particle_type = np.asarray(previous_aux["nuc_kind_final"], dtype=np.int32)

    else:
        output_index = local_index - 1

        if output_index >= len(nuc_kind_out):
            raise IndexError(
                "The requested state has no matching nuc_kind_out entry."
            )

        particle_type = np.asarray(nuc_kind_out[output_index], dtype=np.int32)

    if particle_type.shape != (n_particles,):
        raise ValueError(
            "Particle-type array has shape "
            f"{particle_type.shape}; expected ({n_particles},)."
        )

    return particle_type.copy()



def _phase_id_for_local_state(
    data,
    segments,
    segment_array_index,
    local_index,
    result,
    n_particles,
):
    """
    Recover precipitate phase identifiers for one saved state.

    V10 stores phase_id_hist/phase_id_final in the segment auxiliary
    dictionary and initial_phase_id at the top level. Legacy V8 files have
    no phase identity because only one precipitate phase exists; those are
    therefore assigned phase 0.
    """
    aux = _as_object(result[7])

    if (
        isinstance(aux, dict)
        and "phase_id_hist" in aux
        and len(aux["phase_id_hist"]) == len(result[0])
    ):
        phase_id = np.asarray(aux["phase_id_hist"][local_index], dtype=np.int32)

    elif local_index == 0:
        if segment_array_index == 0:
            if "initial_phase_id" in data.files:
                phase_id = np.asarray(data["initial_phase_id"], dtype=np.int32)
            else:
                phase_id = np.zeros(n_particles, dtype=np.int32)
        else:
            previous_result = _segment_result(
                segments[segment_array_index - 1],
                segment_array_index - 1,
            )
            previous_aux = _as_object(previous_result[7])

            if isinstance(previous_aux, dict) and "phase_id_final" in previous_aux:
                phase_id = np.asarray(
                    previous_aux["phase_id_final"], dtype=np.int32
                )
            else:
                phase_id = np.zeros(n_particles, dtype=np.int32)

    elif isinstance(aux, dict) and "phase_id_final" in aux and local_index == len(result[0]) - 1:
        phase_id = np.asarray(aux["phase_id_final"], dtype=np.int32)

    else:
        # Legacy single-phase files have no phase bookkeeping.
        phase_id = np.zeros(n_particles, dtype=np.int32)

    if phase_id.shape != (n_particles,):
        raise ValueError(
            "Phase-id array has shape "
            f"{phase_id.shape}; expected ({n_particles},)."
        )

    return phase_id.copy()

def _extract_state_from_location(
    data,
    shared,
    segments,
    times,
    global_index,
    location,
):
    """
    Extract one complete saved state using a segment/local location.
    """
    segment_array_index, local_index = location
    entry = segments[segment_array_index]
    result = _segment_result(entry, segment_array_index)

    t_local = np.asarray(result[0], dtype=float)
    R_hist = result[1]

    # V10 stores authoritative generic histories in the auxiliary state.
    # V8 and early V10 files retain only the historical two-slot tuple.
    aux = _as_object(result[7])
    use_generic_histories = (
        isinstance(aux, dict)
        and "Q_hist" in aux
        and "cbar_hist" in aux
    )

    if use_generic_histories:
        Q_hist = aux["Q_hist"]
        cbar_hist = aux["cbar_hist"]
    else:
        if shared["solute_names"] != ("Mg", "Si"):
            raise ValueError(
                "This saved segment contains only the legacy two-solute tuple "
                "but the file species are not exactly ('Mg', 'Si')."
            )

        Q_mg_hist = result[2]
        Q_si_hist = result[3]
        cbar_mg_hist = np.asarray(result[4], dtype=float)
        cbar_si_hist = np.asarray(result[5], dtype=float)

        Q_hist = [
            np.vstack(
                [
                    np.asarray(q_mg, dtype=float),
                    np.asarray(q_si, dtype=float),
                ]
            )
            for q_mg, q_si in zip(Q_mg_hist, Q_si_hist)
        ]
        cbar_hist = np.column_stack([cbar_mg_hist, cbar_si_hist])

    n_local_states = len(t_local)

    histories = {
        "R_hist": R_hist,
        "Q_hist": Q_hist,
        "cbar_hist": cbar_hist,
    }

    for name, history in histories.items():
        if len(history) != n_local_states:
            raise ValueError(
                f"{name} in segment {segment_array_index} has length "
                f"{len(history)}; expected {n_local_states}."
            )

    radii = np.asarray(R_hist[local_index], dtype=float).copy()
    Q = np.asarray(Q_hist[local_index], dtype=float).copy()
    cbar = np.asarray(cbar_hist[local_index], dtype=float).copy()

    # Long-range field for this exact saved scale-separated state.
    #
    # Current scale-separated output keeps:
    #   (1) one full-resolution final LR field for restart, and
    #   (2) sparse/coarse diagnostic LR snapshots at selected thermal-segment
    #       boundaries in final_aux.
    #
    # For the global final state always prefer the full-resolution restart
    # field.  For earlier states, use a sparse LR snapshot only when its saved
    # time matches the selected particle/Q state.
    hybrid_long_range_field = None
    hybrid_long_range_field_source = None
    hybrid_long_range_field_is_sparse = False
    hybrid_long_range_field_grid_shape = None
    hybrid_long_range_field_coordinates_nm = None

    if shared["hybrid_transport"]:
        final_aux = shared["final_aux"]

        # Exact final reconstruction: use the authoritative full solver grid.
        if global_index == len(times) - 1:
            final_field = final_aux.get(
                "hybrid_long_range_field_final", None
            )
            if final_field is not None:
                hybrid_long_range_field = np.asarray(
                    final_field, dtype=float
                ).copy()
                hybrid_long_range_field_source = "final_full"
                hybrid_long_range_field_is_sparse = False

        # Earlier reconstruction: current sparse/coarse diagnostic history.
        if hybrid_long_range_field is None:
            lr_hist = final_aux.get(
                "hybrid_long_range_field_hist", None
            )
            lr_times = final_aux.get(
                "hybrid_long_range_field_hist_times", None
            )

            if lr_hist is not None and lr_times is not None:
                lr_times = np.asarray(lr_times, dtype=float).ravel()
                lr_hist_list = list(lr_hist)

                if len(lr_hist_list) != len(lr_times):
                    raise ValueError(
                        "Sparse long-range history has inconsistent field/time "
                        f"lengths: {len(lr_hist_list)} fields versus "
                        f"{len(lr_times)} times."
                    )

                requested_state_time = float(times[global_index])
                atol = max(
                    1.0e-12,
                    1.0e-10 * max(1.0, abs(requested_state_time)),
                )
                matches = np.flatnonzero(
                    np.isclose(
                        lr_times,
                        requested_state_time,
                        rtol=0.0,
                        atol=atol,
                    )
                )

                if len(matches):
                    ih = int(matches[0])
                    hybrid_long_range_field = np.asarray(
                        lr_hist_list[ih], dtype=float
                    ).copy()
                    hybrid_long_range_field_source = "sparse_history"
                    hybrid_long_range_field_is_sparse = True

                    # Preserve the exact coordinates of the decimated LR
                    # snapshot.  The sparse grid need not divide the production
                    # FFT dimensions exactly; in that case it is a periodic
                    # rectilinear grid rather than a uniform FFT grid.  Field
                    # reconstruction handles that geometry explicitly.
                    hist_coordinates = final_aux.get(
                        "hybrid_long_range_field_hist_coordinates_nm", None
                    )
                    if hist_coordinates is not None:
                        coords = tuple(
                            np.asarray(v, dtype=float).ravel().copy()
                            for v in hist_coordinates
                        )
                        saved_shape = hybrid_long_range_field.shape[1:]
                        if len(coords) != 3:
                            raise ValueError(
                                "Sparse LR history coordinate metadata must "
                                "contain three coordinate arrays."
                            )
                        for axis, (coord, n_saved, length) in enumerate(
                            zip(coords, saved_shape, shared["box"])
                        ):
                            if len(coord) != int(n_saved):
                                raise ValueError(
                                    "Sparse LR history coordinate metadata does "
                                    "not match the saved field shape."
                                )
                            if np.any(~np.isfinite(coord)):
                                raise ValueError(
                                    "Sparse LR history coordinates contain "
                                    "non-finite values."
                                )
                            if len(coord) > 1 and np.any(np.diff(coord) <= 0.0):
                                raise ValueError(
                                    "Sparse LR history coordinates must be "
                                    "strictly increasing on each axis."
                                )
                            if len(coord) and (
                                coord[0] < -1.0e-12
                                or coord[-1] >= float(length) + 1.0e-12
                            ):
                                raise ValueError(
                                    "Sparse LR history coordinates lie outside "
                                    "the periodic simulation box."
                                )
                        hybrid_long_range_field_coordinates_nm = coords

        if hybrid_long_range_field is not None:
            if (
                hybrid_long_range_field.ndim != 4
                or hybrid_long_range_field.shape[0]
                != len(shared["solute_names"])
            ):
                raise ValueError(
                    "Saved hybrid LR field must have shape "
                    "(Nsolutes,nx,ny,nz); got "
                    f"{hybrid_long_range_field.shape}."
                )

            if np.any(~np.isfinite(hybrid_long_range_field)):
                raise ValueError(
                    "Saved hybrid LR field contains non-finite values."
                )

            hybrid_long_range_field_grid_shape = np.asarray(
                hybrid_long_range_field.shape[1:], dtype=int
            )

    n_particles = len(radii)
    n_solutes = len(shared["solute_names"])

    positions = _positions_for_local_state(
        data=data,
        segments=segments,
        segment_array_index=segment_array_index,
        local_index=local_index,
        result=result,
    )

    particle_type = _particle_type_for_local_state(
        data=data,
        segments=segments,
        segment_array_index=segment_array_index,
        local_index=local_index,
        result=result,
        n_particles=n_particles,
    )

    phase_id = _phase_id_for_local_state(
        data=data,
        segments=segments,
        segment_array_index=segment_array_index,
        local_index=local_index,
        result=result,
        n_particles=n_particles,
    )

    if positions.shape != (n_particles, 3):
        raise ValueError(
            "positions has shape "
            f"{positions.shape}; expected ({n_particles}, 3)."
        )

    if Q.shape != (n_solutes, n_particles):
        raise ValueError(
            f"Q has shape {Q.shape}; expected ({n_solutes}, {n_particles})."
        )

    if cbar.shape != (n_solutes,):
        raise ValueError(
            f"cbar has shape {cbar.shape}; expected ({n_solutes},)."
        )

    alive = radii > 0.0

    state = {
        "time": float(times[global_index]),
        "global_index": int(global_index),
        "segment_array_index": int(segment_array_index),
        "segment_index": int(entry.get("segment_index", segment_array_index + 1)),
        "local_index": int(local_index),
        "temperature": _as_float(
            entry.get("temperature", np.nan),
            f"segments[{segment_array_index}].temperature",
        ),
        "positions": positions,
        "radii": radii,
        "solute_names": shared["solute_names"],
        "Q": Q,
        "cbar": cbar,
        "particle_type": particle_type,
        "phase_id": phase_id,
        "alive": alive,
        "pending": None,
        "coupled": None,
        "box": shared["box"].copy(),
        "cp": shared["cp"].copy(),
        "alpha": shared["alpha"],
        "green_kernel_type":
            shared["green_kernel_type"],
        "screening_length":
            shared["screening_length"],
        "screening_kappa":
            shared["screening_kappa"],
        "hybrid_transport": shared["hybrid_transport"],
        "hybrid_matching_length": shared["hybrid_matching_length"],
        "hybrid_grid_shape": (
            None
            if shared["hybrid_grid_shape"] is None
            else shared["hybrid_grid_shape"].copy()
        ),
        "hybrid_short_cutoff": shared["hybrid_short_cutoff"],
        "hybrid_long_range_field": hybrid_long_range_field,
        "hybrid_long_range_field_available": (
            hybrid_long_range_field is not None
        ),
        "hybrid_long_range_field_source":
            hybrid_long_range_field_source,
        "hybrid_long_range_field_is_sparse":
            hybrid_long_range_field_is_sparse,
        "hybrid_long_range_field_grid_shape": (
            None
            if hybrid_long_range_field_grid_shape is None
            else hybrid_long_range_field_grid_shape.copy()
        ),
        "hybrid_long_range_field_coordinates_nm": (
            None
            if hybrid_long_range_field_coordinates_nm is None
            else tuple(v.copy() for v in hybrid_long_range_field_coordinates_nm)
        ),
        "real_lattice":
            shared["real_lattice"].copy(),
        "kvecs":
            shared["kvecs"].copy(),
        "k2":
            shared["k2"].copy(),
    }

    # Preserve historical Mg/Si state aliases when those species exist.
    solute_names = shared["solute_names"]

    if "Mg" in solute_names:
        mg_index = solute_names.index("Mg")
        state["Q_mg"] = Q[mg_index].copy()
        state["cbar_mg"] = float(cbar[mg_index])
        state["cp_mg"] = float(shared["cp"][mg_index])

    if "Si" in solute_names:
        si_index = solute_names.index("Si")
        state["Q_si"] = Q[si_index].copy()
        state["cbar_si"] = float(cbar[si_index])
        state["cp_si"] = float(shared["cp"][si_index])

    # Exact masks are currently saved only for the final state.
    if global_index == len(times) - 1:
        final_aux = shared["final_aux"]

        if "pending_final" in final_aux:
            state["pending"] = np.asarray(final_aux["pending_final"], dtype=bool).copy()

        if "coupled_final" in final_aux:
            state["coupled"] = np.asarray(final_aux["coupled_final"], dtype=bool).copy()

    return state


def load_simulation_state(
    filename,
    *,
    state="final",
    index=None,
    time=None,
    time_selection="nearest",
    exact_time_tolerance=1.0e-12,
):
    """
    Load one saved state from a precipitation simulation NPZ file.

    Parameters
    ----------
    filename : str or Path
        Saved simulation NPZ file.

    state : {"initial", "final"}, default="final"
        State selected when neither index nor time is supplied.

    index : int or None
        Global saved-state index. Negative indices follow normal Python
        indexing conventions.

    time : float or None
        Requested global simulation time.

    time_selection : {"nearest", "before", "after", "exact"}
        Rule used when selecting by time.

    exact_time_tolerance : float
        Absolute tolerance used when time_selection="exact".

    Returns
    -------
    result : dict
        Dictionary containing the particle state, monopole strengths,
        matrix concentrations, precipitate concentrations, and Ewald
        data required for composition-field reconstruction.

    Notes
    -----
    The file is opened with allow_pickle=True because the simulation
    stores nested segment dictionaries and ragged particle histories.
    Only load files from trusted sources.

    Exact pending/coupled masks are currently available only for the
    final state. For earlier states, those entries are returned as None.
    For legacy quasi-static runs the saved Q arrays are sufficient for field
    reconstruction. Scale-separated runs additionally require the LR field snapshot
    corresponding to the requested time.
    """
    filename = Path(filename).expanduser().resolve()

    if not filename.is_file():
        raise FileNotFoundError(f"Simulation file does not exist: {filename}")

    if exact_time_tolerance < 0.0:
        raise ValueError("exact_time_tolerance must be non-negative.")

    with np.load(filename, allow_pickle=True) as data:
        required_keys = [
            "segments",
            "final_aux",
            "box",
        ]

        for key in required_keys:
            if key not in data.files:
                raise KeyError(f"Saved NPZ file does not contain required key {key!r}.")

        shared = _extract_shared_data(data)
        segments = _segment_entries(data)

        if len(segments) == 0:
            raise ValueError("The saved simulation contains no thermal segments.")

        times, locations = _state_times_and_locations(data, segments)

        if len(times) == 0:
            raise ValueError("The saved simulation contains no saved states.")

        global_index = _select_global_index(
            times,
            state=state,
            index=index,
            time=time,
            time_selection=time_selection,
            exact_time_tolerance=exact_time_tolerance,
        )

        result = _extract_state_from_location(
            data=data,
            shared=shared,
            segments=segments,
            times=times,
            global_index=global_index,
            location=locations[global_index],
        )

    result["source_file"] = str(filename)
    result["number_of_saved_states"] = len(times)
    result["available_time_min"] = float(times[0])
    result["available_time_max"] = float(times[-1])

    return result


def list_saved_state_times(filename):
    """
    Return all unique global saved-state times in an NPZ file.
    """
    filename = Path(filename).expanduser().resolve()

    if not filename.is_file():
        raise FileNotFoundError(f"Simulation file does not exist: {filename}")

    with np.load(filename, allow_pickle=True) as data:
        segments = _segment_entries(data)
        times, _ = _state_times_and_locations(data, segments)

    return times