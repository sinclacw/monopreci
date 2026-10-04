#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np

from monopole_precip.concentrationfield.field_evaluation import (
    evaluate_composition_field,
    evaluate_hybrid_composition_field,
    reshape_field_result,
)
from monopole_precip.concentrationfield.field_grids import load_grid_request
from monopole_precip.concentrationfield.simulation_state import (
    load_simulation_state,
)
from monopole_precip.concentrationfield.vtk_output import (
    write_image_data_vti,
    write_particle_points_vtp,
    write_point_data_vtp,
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct species-indexed concentration fields from a saved "
            "monopole precipitation simulation."
        )
    )

    parser.add_argument(
        "simulation_file",
        type=Path,
        help="Input NPZ simulation file.",
    )

    state_group = parser.add_mutually_exclusive_group(required=True)

    state_group.add_argument(
        "--state",
        choices=("initial", "final"),
        help="Load the initial or final saved simulation state.",
    )

    state_group.add_argument(
        "--time",
        type=float,
        help="Load a saved simulation state at the requested global time.",
    )

    parser.add_argument(
        "--time-selection",
        choices=("nearest", "before", "after", "exact"),
        default="nearest",
        help="Rule used when selecting a saved state by time. Default: nearest.",
    )

    parser.add_argument(
        "--grid-file",
        type=Path,
        required=True,
        help="JSON file describing a line, plane, or volume grid.",
    )

    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help=(
            "Optional VTK output filename. Use .vtp for a line and .vti for "
            "a plane or volume. At least one of --output-file or --npz-output "
            "must be supplied."
        ),
    )

    parser.add_argument(
        "--npz-output",
        type=Path,
        default=None,
        help=(
            "Optional NumPy NPZ output containing the reconstructed "
            "field and grid metadata."
        ),
    )

    lr_group = parser.add_mutually_exclusive_group()

    lr_group.add_argument(
        "--lr-file",
        type=Path,
        default=None,
        help=(
            "Standalone long-range-field snapshot for a scale-separated run. "
            "The snapshot time must match the selected particle state."
        ),
    )

    lr_group.add_argument(
        "--lr-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing standalone lr_t*.npz long-range-field "
            "snapshots. The snapshot nearest the selected particle-state time "
            "is used, subject to --max-lr-time-mismatch."
        ),
    )

    parser.add_argument(
        "--max-lr-time-mismatch",
        type=float,
        default=1.0e-8,
        help=(
            "Maximum allowed absolute time mismatch [s] between the selected "
            "particle state and a standalone long-range-field snapshot. "
            "Default: 1e-8."
        ),
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "Number of worker processes used for field evaluation. "
            "Default: use SLURM_CPUS_PER_TASK when available."
        ),
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10_000,
        help="Number of matrix evaluation points assigned to each task.",
    )

    parser.add_argument(
        "--hybrid-lr-interpolation",
        choices=("spectral", "cic"),
        default="spectral",
        help=(
            "Long-range field interpolation for scale-separated simulations. "
            "'spectral' reconstructs "
            "the smooth periodic Fourier field for visualization; 'cic' "
            "reproduces the exact LR sampler used by the production solver. "
            "Default: spectral."
        ),
    )

    parser.add_argument(
        "--hybrid-lr-spectral-factor",
        type=int,
        default=4,
        help=(
            "Integer Fourier-resampling factor in each direction before "
            "sampling the long-range field. Used only with spectral interpolation. "
            "Default: 4."
        ),
    )

    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Print progress information during field reconstruction.",
    )

    return parser.parse_args()


def validate_arguments(args):
    if not args.simulation_file.is_file():
        raise FileNotFoundError(
            f"Simulation file does not exist: {args.simulation_file}"
        )

    if args.simulation_file.suffix.lower() != ".npz":
        raise ValueError("The simulation input file must have the .npz extension.")

    if not args.grid_file.is_file():
        raise FileNotFoundError(
            f"Grid request file does not exist: {args.grid_file}"
        )

    if args.grid_file.suffix.lower() != ".json":
        raise ValueError("The grid request file must have the .json extension.")

    if args.workers is not None and args.workers <= 0:
        raise ValueError("--workers must be a positive integer.")

    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be a positive integer.")

    if args.hybrid_lr_spectral_factor < 1:
        raise ValueError("--hybrid-lr-spectral-factor must be >= 1.")

    if args.time is not None and not np.isfinite(args.time):
        raise ValueError("--time must be finite.")

    if args.state is not None and args.time_selection != "nearest":
        raise ValueError(
            "--time-selection is only meaningful when --time is supplied."
        )

    if args.npz_output is not None and args.npz_output.suffix.lower() != ".npz":
        raise ValueError("--npz-output must have the .npz extension.")

    if args.output_file is None and args.npz_output is None:
        raise ValueError(
            "At least one of --output-file or --npz-output must be supplied."
        )

    if args.lr_file is not None:
        if not args.lr_file.is_file():
            raise FileNotFoundError(
                f"Long-range snapshot does not exist: {args.lr_file}"
            )
        if args.lr_file.suffix.lower() != ".npz":
            raise ValueError("--lr-file must have the .npz extension.")

    if args.lr_dir is not None and not args.lr_dir.is_dir():
        raise NotADirectoryError(
            f"Long-range snapshot directory does not exist: {args.lr_dir}"
        )

    if (
        not np.isfinite(args.max_lr_time_mismatch)
        or args.max_lr_time_mismatch < 0.0
    ):
        raise ValueError("--max-lr-time-mismatch must be finite and non-negative.")



def _snapshot_scalar(data, key, default=np.nan):
    """Read a scalar value from an NPZ snapshot."""
    if key not in data.files:
        return float(default)
    values = np.asarray(data[key]).reshape(-1)
    if values.size == 0:
        return float(default)
    return float(values[0])


def discover_long_range_snapshots(directory):
    """Return standalone LR snapshots as (time_s, path) records."""
    directory = Path(directory)
    files = sorted(directory.glob("lr_t*.npz"))
    if not files:
        files = sorted(directory.rglob("lr_t*.npz"))

    records = []
    for path in files:
        try:
            with np.load(path, allow_pickle=True) as data:
                if "time_s" not in data.files:
                    continue
                time_s = _snapshot_scalar(data, "time_s")
                if np.isfinite(time_s):
                    records.append((time_s, path))
        except Exception as exc:
            print(f"WARNING: skipping unreadable LR snapshot {path}: {exc}")

    records.sort(key=lambda item: item[0])

    if not records:
        raise RuntimeError(
            f"No readable lr_t*.npz snapshots with time_s metadata found under "
            f"{directory}."
        )

    return records


def select_long_range_snapshot(directory, target_time):
    """Select the standalone LR snapshot nearest target_time."""
    records = discover_long_range_snapshots(directory)
    times = np.asarray([record[0] for record in records], dtype=float)
    index = int(np.argmin(np.abs(times - float(target_time))))
    return records[index][1]


def load_long_range_snapshot(path, expected_box, expected_solute_names):
    """Load and validate one native long-range-field snapshot."""
    path = Path(path)

    with np.load(path, allow_pickle=True) as data:
        field_key = None
        for key in (
            "c_long_range",
            "c",
            "field",
            "C_LR",
            "c_lr",
            "concentration",
            "concentration_field",
        ):
            if key in data.files:
                field_key = key
                break

        if field_key is None:
            raise KeyError(
                f"Could not identify a long-range field in {path}. "
                f"Available keys: {list(data.files)}"
            )

        field = np.asarray(data[field_key], dtype=float)
        if field.ndim == 3:
            field = field[None, ...]
        if field.ndim != 4:
            raise ValueError(
                "The long-range field must have shape "
                "(n_solutes, nx, ny, nz) or (nx, ny, nz); "
                f"got {field.shape}."
            )

        time_s = _snapshot_scalar(data, "time_s")

        box = (
            np.asarray(data["box"], dtype=float).reshape(3)
            if "box" in data.files
            else np.asarray(expected_box, dtype=float).reshape(3)
        )

        snapshot_solute_names = (
            tuple(str(x) for x in np.asarray(data["solute_names"]).reshape(-1))
            if "solute_names" in data.files
            else None
        )

        matching_length = (
            _snapshot_scalar(data, "hybrid_matching_length_nm")
            if "hybrid_matching_length_nm" in data.files
            else np.nan
        )

    expected_box = np.asarray(expected_box, dtype=float).reshape(3)
    if not np.allclose(box, expected_box, rtol=0.0, atol=1.0e-10):
        raise ValueError(
            f"LR snapshot box {box} does not match simulation box {expected_box}."
        )

    expected_solute_names = tuple(str(x) for x in expected_solute_names)
    if field.shape[0] != len(expected_solute_names):
        raise ValueError(
            f"LR snapshot contains {field.shape[0]} solute fields but the "
            f"simulation state contains {len(expected_solute_names)} "
            f"({expected_solute_names})."
        )

    if (
        snapshot_solute_names is not None
        and snapshot_solute_names != expected_solute_names
    ):
        raise ValueError(
            f"LR snapshot solute ordering {snapshot_solute_names} does not "
            f"match simulation ordering {expected_solute_names}."
        )

    return {
        "path": path,
        "field": field,
        "time_s": float(time_s),
        "matching_length_nm": float(matching_length),
    }

def _positive_cartesian_axis(vector, name):
    """Return the Cartesian axis index for a +axis-aligned vector."""
    vector = np.asarray(vector, dtype=float)

    if vector.shape != (3,):
        raise ValueError(f"{name} must have shape (3,).")

    length = float(np.linalg.norm(vector))
    if not np.isfinite(length) or length <= 0.0:
        raise ValueError(f"{name} must have a finite positive length.")

    direction = vector / length
    axis = int(np.argmax(np.abs(direction)))
    expected = np.zeros(3, dtype=float)
    expected[axis] = 1.0

    if not np.allclose(direction, expected, rtol=0.0, atol=1.0e-12):
        raise ValueError(
            f"ImageData output requires {name} to align with one of the "
            "positive Cartesian axes (+x, +y, +z); got direction {direction}."
        )

    return axis, length


def get_image_data_geometry(grid_metadata):
    """
    Return VTK ImageData geometry for an axis-aligned plane or volume.

    Plane grids may span any two distinct positive Cartesian axes, e.g.
    xy, xz, or yz.  The returned VTK grid shape is always ordered as
    (Nx, Ny, Nz), with a singleton dimension normal to the plane.

    Returns
    -------
    image_origin : ndarray, shape (3,)
        Coordinate of the first sampled point.
    image_spacing : ndarray, shape (3,)
        Positive VTK spacing in x, y, z.
    vtk_grid_shape : tuple(int, int, int)
        VTK ImageData point dimensions (Nx, Ny, Nz).
    source_axes : tuple(int, ...)
        Cartesian axis occupied by each source grid dimension.  For a
        plane this is (axis_u, axis_v); for a volume it is (0, 1, 2).
    """
    geometry = grid_metadata["geometry"]
    source_shape = tuple(int(value) for value in grid_metadata["shape"])
    image_origin = np.asarray(grid_metadata["origin"], dtype=float).copy()
    image_spacing = np.ones(3, dtype=float)
    vtk_grid_shape = [1, 1, 1]

    if geometry == "plane":
        vectors = (
            ("vector_u", np.asarray(grid_metadata["vector_u"], dtype=float)),
            ("vector_v", np.asarray(grid_metadata["vector_v"], dtype=float)),
        )

        source_axes = []
        for (name, vector), n_points in zip(vectors, source_shape):
            axis, length = _positive_cartesian_axis(vector, name)
            if axis in source_axes:
                raise ValueError(
                    "Plane ImageData output requires vector_u and vector_v "
                    "to align with two different Cartesian axes."
                )

            spacing = length / n_points
            source_axes.append(axis)
            image_spacing[axis] = spacing
            vtk_grid_shape[axis] = n_points
            image_origin[axis] += 0.5 * spacing

        return (
            image_origin,
            image_spacing,
            tuple(vtk_grid_shape),
            tuple(source_axes),
        )

    if geometry == "volume":
        vectors = (
            ("vector_u", np.asarray(grid_metadata["vector_u"], dtype=float)),
            ("vector_v", np.asarray(grid_metadata["vector_v"], dtype=float)),
            ("vector_w", np.asarray(grid_metadata["vector_w"], dtype=float)),
        )

        expected_axes = (0, 1, 2)
        for (name, vector), n_points, expected_axis in zip(
            vectors, source_shape, expected_axes
        ):
            axis, length = _positive_cartesian_axis(vector, name)
            if axis != expected_axis:
                expected = np.zeros(3, dtype=float)
                expected[expected_axis] = 1.0
                raise ValueError(
                    f"volume ImageData output currently requires {name} "
                    f"to align with {expected}."
                )

            spacing = length / n_points
            image_spacing[axis] = spacing
            vtk_grid_shape[axis] = n_points
            image_origin[axis] += 0.5 * spacing

        return (
            image_origin,
            image_spacing,
            tuple(vtk_grid_shape),
            expected_axes,
        )

    raise ValueError(
        "ImageData geometry is defined only for plane and volume grids."
    )


def _plane_array_to_xyz(values, source_shape, source_axes, vtk_grid_shape):
    """
    Map a plane array indexed as (u, v[, components]) to VTK (x, y, z).
    """
    values = np.asarray(values)
    source_shape = tuple(int(v) for v in source_shape)
    source_axes = tuple(int(v) for v in source_axes)
    vtk_grid_shape = tuple(int(v) for v in vtk_grid_shape)

    if values.shape[:2] != source_shape:
        raise ValueError(
            f"Plane field array has shape {values.shape}; expected leading "
            f"shape {source_shape}."
        )

    # Reorder u,v so the non-singleton dimensions occur in x,y,z order.
    sorted_cartesian_axes = sorted(source_axes)
    permutation = [source_axes.index(axis) for axis in sorted_cartesian_axes]
    trailing_axes = list(range(2, values.ndim))
    values = np.transpose(values, axes=permutation + trailing_axes)

    trailing_shape = values.shape[2:]
    return values.reshape(vtk_grid_shape + trailing_shape)


def print_grid_description(grid_metadata, n_points):
    geometry = grid_metadata["geometry"]

    print()
    print("Calculation grid")
    print(f"  Geometry         : {geometry}")
    print(f"  Grid shape       : {grid_metadata['shape']}")
    print(f"  Number of points : {n_points}")

    if geometry == "line":
        print(f"  Start            : {grid_metadata['start']}")
        print(f"  End              : {grid_metadata['end']}")

    elif geometry == "plane":
        print(f"  Origin           : {grid_metadata['origin']}")
        print(f"  Vector u         : {grid_metadata['vector_u']}")
        print(f"  Vector v         : {grid_metadata['vector_v']}")

    elif geometry == "volume":
        print(f"  Origin           : {grid_metadata['origin']}")
        print(f"  Vector u         : {grid_metadata['vector_u']}")
        print(f"  Vector v         : {grid_metadata['vector_v']}")
        print(f"  Vector w         : {grid_metadata['vector_w']}")


def _species_field_data(field, solute_names):
    """Return total and, when available, scale-separated concentration components."""
    solute_names = tuple(str(name) for name in solute_names)
    output = {}

    for key, prefix in (
        ("c", "c"),
        ("c_long_range", "c_lr"),
        ("c_short_range", "c_short"),
    ):
        if key not in field:
            continue
        values = np.asarray(field[key])
        if values.ndim < 2 or values.shape[0] != len(solute_names):
            raise ValueError(
                f"field[{key!r}] must have one leading species axis "
                "matching solute_names."
            )
        output.update({
            f"{prefix}_{name}": values[species_index]
            for species_index, name in enumerate(solute_names)
        })

    return output


def write_reconstructed_field(
    output_file,
    geometry,
    field_result,
    reshaped_field,
    grid_metadata,
):
    output_file.parent.mkdir(parents=True, exist_ok=True)

    solute_names = tuple(field_result["solute_names"])
    if not solute_names:
        raise ValueError("At least one solute is required for field output.")

    active_scalars = f"c_{solute_names[0]}"

    if geometry == "line":
        if output_file.suffix.lower() != ".vtp":
            raise ValueError("Line grids require a .vtp output filename.")

        point_data = {
            "distance": grid_metadata["distance"],
            **_species_field_data(field_result, solute_names),
            "inside_precipitate": (
                field_result["inside_precipitate"].astype(np.uint8)
            ),
            "particle_id": field_result["particle_id"],
        }

        return write_point_data_vtp(
            filename=output_file,
            points=field_result["points"],
            point_data=point_data,
            active_scalars=active_scalars,
            binary=True,
        )

    if geometry in ("plane", "volume"):
        expected_suffix = ".vti"
        if output_file.suffix.lower() != expected_suffix:
            raise ValueError(
                f"{geometry.capitalize()} grids require a {expected_suffix} "
                "output filename."
            )

        (
            image_origin,
            image_spacing,
            vtk_grid_shape,
            source_axes,
        ) = get_image_data_geometry(grid_metadata)

        point_data = {
            **_species_field_data(reshaped_field, solute_names),
            "inside_precipitate": (
                reshaped_field["inside_precipitate"].astype(np.uint8)
            ),
            "particle_id": reshaped_field["particle_id"],
        }

        if geometry == "plane":
            point_data = {
                name: _plane_array_to_xyz(
                    values,
                    source_shape=grid_metadata["shape"],
                    source_axes=source_axes,
                    vtk_grid_shape=vtk_grid_shape,
                )
                for name, values in point_data.items()
            }

        return write_image_data_vti(
            filename=output_file,
            point_data=point_data,
            grid_shape=vtk_grid_shape,
            origin=image_origin,
            spacing=image_spacing,
            active_scalars=active_scalars,
            binary=True,
        )

    raise ValueError(f"Unsupported grid geometry: {geometry!r}.")

def write_particle_file(output_file, state):
    particle_file = output_file.with_name(
        output_file.stem + "_particles.vtp"
    )

    active_idx = np.flatnonzero(state["radii"] > 0.0)

    extra_particle_data = {
        "alive": state["alive"][active_idx].astype(np.uint8),
    }

    if state["pending"] is not None:
        extra_particle_data["pending"] = (
            state["pending"][active_idx].astype(np.uint8)
        )

    if state["coupled"] is not None:
        extra_particle_data["coupled"] = (
            state["coupled"][active_idx].astype(np.uint8)
        )

    return write_particle_points_vtp(
        filename=particle_file,
        positions=state["positions"][active_idx],
        radii=state["radii"][active_idx],
        particle_id=active_idx + 1,
        particle_type=state["particle_type"][active_idx],
        phase_id=state["phase_id"][active_idx],
        Q=state["Q"][:, active_idx],
        solute_names=state["solute_names"],
        extra_point_data=extra_particle_data,
        binary=True,
    )


def write_numpy_output(
    npz_output,
    simulation_file,
    geometry,
    state,
    grid_metadata,
    reshaped_field,
    lr_snapshot_file=None,
    lr_snapshot_time=None,
):
    npz_output.parent.mkdir(parents=True, exist_ok=True)

    npz_data = {
        "time": np.asarray(state["time"], dtype=float),
        "temperature": np.asarray(state["temperature"], dtype=float),
        "global_index": np.asarray(state["global_index"], dtype=int),
        "segment_index": np.asarray(state["segment_index"], dtype=int),
        "geometry": np.asarray(geometry),
        "grid_shape": np.asarray(grid_metadata["shape"], dtype=int),
        "points": reshaped_field["points"],
        "solute_names": np.asarray(state["solute_names"]),
        "c": np.asarray(reshaped_field["c"]),
        "inside_precipitate": reshaped_field["inside_precipitate"],
        "particle_id": reshaped_field["particle_id"],
        "cbar": np.asarray(state["cbar"], dtype=float),
        "cp": np.asarray(state["cp"], dtype=float),
        "source_file": np.asarray(str(simulation_file)),
    }

    if lr_snapshot_file is not None:
        npz_data["source_lr_file"] = np.asarray(str(lr_snapshot_file))
    if lr_snapshot_time is not None:
        npz_data["lr_snapshot_time_s"] = np.asarray(
            float(lr_snapshot_time), dtype=float
        )

    if "c_long_range" in reshaped_field:
        npz_data["c_long_range"] = np.asarray(
            reshaped_field["c_long_range"]
        )
    if "c_short_range" in reshaped_field:
        npz_data["c_short_range"] = np.asarray(
            reshaped_field["c_short_range"]
        )
    if "field_model" in reshaped_field:
        npz_data["field_model"] = np.asarray(reshaped_field["field_model"])

    # Preserve historical Mg/Si NPZ field names when those species exist.
    name_to_index = {
        str(name): index
        for index, name in enumerate(state["solute_names"])
    }
    if "Mg" in name_to_index:
        idx = name_to_index["Mg"]
        npz_data["c_mg"] = np.asarray(reshaped_field["c"][idx])
        npz_data["cbar_mg"] = np.asarray(state["cbar"][idx], dtype=float)
        npz_data["cp_mg"] = np.asarray(state["cp"][idx], dtype=float)
    if "Si" in name_to_index:
        idx = name_to_index["Si"]
        npz_data["c_si"] = np.asarray(reshaped_field["c"][idx])
        npz_data["cbar_si"] = np.asarray(state["cbar"][idx], dtype=float)
        npz_data["cp_si"] = np.asarray(state["cp"][idx], dtype=float)

    if geometry == "line":
        npz_data.update({
            "distance": np.asarray(grid_metadata["distance"], dtype=float),
            "line_start": np.asarray(grid_metadata["start"], dtype=float),
            "line_end": np.asarray(grid_metadata["end"], dtype=float),
        })

    elif geometry == "plane":
        npz_data.update({
            "origin": np.asarray(grid_metadata["origin"], dtype=float),
            "vector_u": np.asarray(grid_metadata["vector_u"], dtype=float),
            "vector_v": np.asarray(grid_metadata["vector_v"], dtype=float),
            "u": np.asarray(grid_metadata["u"], dtype=float),
            "v": np.asarray(grid_metadata["v"], dtype=float),
        })

    elif geometry == "volume":
        npz_data.update({
            "origin": np.asarray(grid_metadata["origin"], dtype=float),
            "vector_u": np.asarray(grid_metadata["vector_u"], dtype=float),
            "vector_v": np.asarray(grid_metadata["vector_v"], dtype=float),
            "vector_w": np.asarray(grid_metadata["vector_w"], dtype=float),
            "u": np.asarray(grid_metadata["u"], dtype=float),
            "v": np.asarray(grid_metadata["v"], dtype=float),
            "w": np.asarray(grid_metadata["w"], dtype=float),
        })

    np.savez_compressed(npz_output, **npz_data)


def main():
    args = parse_arguments()
    validate_arguments(args)

    simulation_file = args.simulation_file.expanduser().resolve()
    grid_file = args.grid_file.expanduser().resolve()
    output_file = (
        None
        if args.output_file is None
        else args.output_file.expanduser().resolve()
    )

    npz_output = (
        None
        if args.npz_output is None
        else args.npz_output.expanduser().resolve()
    )

    if output_file is not None and output_file == simulation_file:
        raise ValueError("The VTK output file cannot overwrite the simulation file.")

    if npz_output is not None and npz_output == simulation_file:
        raise ValueError("The NPZ output file cannot overwrite the simulation file.")

    print("=" * 72)
    print("Composition-field reconstruction")
    print("=" * 72)
    print(f"Simulation file : {simulation_file}")
    print(f"Grid file       : {grid_file}")
    print(f"VTK output file : {output_file}")
    print(f"LR snapshot file: {args.lr_file}")
    print(f"LR snapshot dir : {args.lr_dir}")
    print(f"NPZ output file : {npz_output}")

    if args.state is not None:
        print(f"Requested state : {args.state}")
    else:
        print(f"Requested time  : {args.time:.8e}")
        print(f"Time selection  : {args.time_selection}")

    print(f"Workers         : {args.workers}")
    print(f"Chunk size      : {args.chunk_size}")
    print(f"Hybrid LR interp: {args.hybrid_lr_interpolation}")
    if args.hybrid_lr_interpolation == "spectral":
        print(f"Hybrid LR factor: {args.hybrid_lr_spectral_factor}")
    print(f"Show progress   : {args.show_progress}")
    print("=" * 72)

    # --------------------------------------------------------------
    # Load requested simulation state
    # --------------------------------------------------------------
    state = load_simulation_state(
        simulation_file,
        state=args.state or "final",
        time=args.time,
        time_selection=args.time_selection,
    )

    print()
    print("Loaded simulation state")
    print(f"  Time             : {state['time']:.8e} s")
    print(f"  Global index     : {state['global_index']}")
    print(f"  Segment          : {state['segment_index']}")
    print(f"  Temperature      : {state['temperature']:.6f} K")
    print(f"  Particles        : {len(state['radii'])}")
    print(f"  Alive particles  : {np.count_nonzero(state['alive'])}")

    lr_snapshot = None
    if state.get("hybrid_transport", False):
        lr_path = None
        if args.lr_file is not None:
            lr_path = args.lr_file.expanduser().resolve()
        elif args.lr_dir is not None:
            lr_path = select_long_range_snapshot(
                args.lr_dir.expanduser().resolve(),
                state["time"],
            )

        if lr_path is not None:
            lr_snapshot = load_long_range_snapshot(
                lr_path,
                expected_box=state["box"],
                expected_solute_names=state["solute_names"],
            )

            mismatch = abs(float(lr_snapshot["time_s"]) - float(state["time"]))
            if mismatch > args.max_lr_time_mismatch:
                raise RuntimeError(
                    "Particle-state/LR snapshot time mismatch exceeds the "
                    "allowed tolerance:\n"
                    f"  particle state : {float(state['time']):.12g} s\n"
                    f"  LR snapshot    : {float(lr_snapshot['time_s']):.12g} s\n"
                    f"  mismatch       : {mismatch:.8e} s\n"
                    f"  allowed        : {args.max_lr_time_mismatch:.8e} s"
                )

            if np.isfinite(lr_snapshot["matching_length_nm"]):
                if not np.isclose(
                    lr_snapshot["matching_length_nm"],
                    state["hybrid_matching_length"],
                    rtol=1.0e-12,
                    atol=1.0e-12,
                ):
                    raise ValueError(
                        "LR snapshot matching length does not match the "
                        "simulation state."
                    )

            print()
            print("Loaded standalone long-range snapshot")
            print(f"  File             : {lr_snapshot['path']}")
            print(f"  Time             : {lr_snapshot['time_s']:.8e} s")
            print(f"  Time mismatch    : {mismatch:.8e} s")
            print(f"  Native shape     : {lr_snapshot['field'].shape}")

    # --------------------------------------------------------------
    # Generate evaluation points
    # --------------------------------------------------------------
    points, grid_metadata = load_grid_request(grid_file)
    geometry = grid_metadata["geometry"]

    print_grid_description(
        grid_metadata=grid_metadata,
        n_points=len(points),
    )

    # --------------------------------------------------------------
    # Evaluate species-indexed concentration fields
    # --------------------------------------------------------------
    if state.get("hybrid_transport", False):
        if lr_snapshot is not None:
            long_range_field = lr_snapshot["field"]
            long_range_coordinates = None
        else:
            long_range_field = state.get("hybrid_long_range_field", None)
            long_range_coordinates = state.get(
                "hybrid_long_range_field_coordinates_nm", None
            )

        if long_range_field is None:
            raise RuntimeError(
                "The requested state is from a scale-separated run but no "
                "long-range field is available. For post-run reconstruction "
                "of arbitrary times, enable "
                "save_hybrid_long_range_field_snapshots during the simulation "
                "and supply --lr-file or --lr-dir."
            )

        field_result = evaluate_hybrid_composition_field(
            points=points,
            positions=state["positions"],
            radii=state["radii"],
            Q=state["Q"],
            solute_names=state["solute_names"],
            box=state["box"],
            alpha=1.0 / state["hybrid_matching_length"],
            long_range_field=long_range_field,
            long_range_coordinates=long_range_coordinates,
            short_cutoff=state["hybrid_short_cutoff"],
            cp=state["cp"],
            chunk_size=args.chunk_size,
            return_particle_id=True,
            show_progress=args.show_progress,
            long_range_interpolation=args.hybrid_lr_interpolation,
            spectral_upsample_factor=args.hybrid_lr_spectral_factor,
        )
    else:
        field_result = evaluate_composition_field(
            points=points,
            positions=state["positions"],
            radii=state["radii"],
            Q=state["Q"],
            cbar=state["cbar"],
            solute_names=state["solute_names"],
            box=state["box"],
            alpha=state["alpha"],
            real_lattice=state["real_lattice"],
            kvecs=state["kvecs"],
            k2=state["k2"],
            cp=state["cp"],
            chunk_size=args.chunk_size,
            return_particle_id=True,
            show_progress=args.show_progress,
            n_workers=args.workers,
            screening_length=state["screening_length"],
        )

    reshaped_field = reshape_field_result(
        result=field_result,
        grid_shape=grid_metadata["shape"],
    )

    print()
    print("Field evaluation completed")
    print(f"  Field model       : {field_result.get('field_model', 'legacy_quasistatic')}")
    if "hybrid_lr_interpolation" in field_result:
        print(f"  LR interpolation  : {field_result['hybrid_lr_interpolation']}")
        print(
            "  LR evaluation grid: "
            f"{tuple(int(v) for v in field_result['hybrid_lr_evaluation_grid_shape'])}"
        )
    for species_index, species_name in enumerate(field_result["solute_names"]):
        values = field_result["c"][species_index]
        print(
            f"  {species_name} range".ljust(21)
            + f": {np.min(values):.8e} to {np.max(values):.8e}"
        )
    print(
        "  Points in precipitates: "
        f"{np.count_nonzero(field_result['inside_precipitate'])}"
    )
    print(f"  Reshaped field   : {reshaped_field['c'].shape}")

    # --------------------------------------------------------------
    # Write optional VTK field and particle datasets
    # --------------------------------------------------------------
    if output_file is not None:
        written_file = write_reconstructed_field(
            output_file=output_file,
            geometry=geometry,
            field_result=field_result,
            reshaped_field=reshaped_field,
            grid_metadata=grid_metadata,
        )

        print()
        print(f"VTK field written to: {written_file}")

        written_particle_file = write_particle_file(
            output_file=written_file,
            state=state,
        )

        print(f"VTK particle file written to: {written_particle_file}")

    # --------------------------------------------------------------
    # Write optional NumPy output
    # --------------------------------------------------------------
    if npz_output is not None:
        write_numpy_output(
            npz_output=npz_output,
            simulation_file=simulation_file,
            geometry=geometry,
            state=state,
            grid_metadata=grid_metadata,
            reshaped_field=reshaped_field,
            lr_snapshot_file=(
                None if lr_snapshot is None else lr_snapshot["path"]
            ),
            lr_snapshot_time=(
                None if lr_snapshot is None else lr_snapshot["time_s"]
            ),
        )

        print(f"NumPy field data written to: {npz_output}")


if __name__ == "__main__":
    main()