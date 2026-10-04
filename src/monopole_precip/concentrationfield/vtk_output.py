"""
VTK XML output utilities for monopole precipitation simulations.

Supported outputs
-----------------
1. Regular scalar/vector fields:
       VTK ImageData (.vti)

2. Particle centres with radius and associated particle data:
       VTK PolyData (.vtp)

3. Arbitrary sampled points with associated field data:
       VTK PolyData (.vtp)

4. Time-dependent dataset collections:
       ParaView Data Collection (.pvd)

No vtk Python package is required.
"""

import base64
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


# ============================================================
# VTK datatype handling
# ============================================================

_VTK_DTYPE_NAMES = {
    np.dtype(np.int8): "Int8",
    np.dtype(np.uint8): "UInt8",
    np.dtype(np.int16): "Int16",
    np.dtype(np.uint16): "UInt16",
    np.dtype(np.int32): "Int32",
    np.dtype(np.uint32): "UInt32",
    np.dtype(np.int64): "Int64",
    np.dtype(np.uint64): "UInt64",
    np.dtype(np.float32): "Float32",
    np.dtype(np.float64): "Float64",
}


def _prepare_numeric_array(values):
    """
    Convert an input array to a VTK-compatible numeric NumPy array.
    """
    values = np.asarray(values)

    if values.dtype == np.bool_:
        values = values.astype(np.uint8)

    if values.dtype not in _VTK_DTYPE_NAMES:
        if np.issubdtype(values.dtype, np.integer):
            values = values.astype(np.int64)
        elif np.issubdtype(values.dtype, np.floating):
            values = values.astype(np.float64)
        else:
            raise TypeError(
                f"Unsupported VTK array dtype: {values.dtype}. "
                "Only integer, floating-point, and boolean arrays "
                "are supported."
            )

    if np.any(~np.isfinite(values)) and np.issubdtype(values.dtype, np.floating):
        raise ValueError("VTK output arrays must contain only finite values.")

    return values


def _vtk_dtype_name(values):
    """
    Return the VTK XML type name for a NumPy array.
    """
    values = _prepare_numeric_array(values)
    return _VTK_DTYPE_NAMES[values.dtype]


def _little_endian_bytes(values):
    """
    Return contiguous little-endian bytes for a numeric array.
    """
    values = _prepare_numeric_array(values)

    dtype = values.dtype

    if dtype.byteorder == ">" or (dtype.byteorder == "=" and not np.little_endian):
        dtype = dtype.newbyteorder("<")
        values = values.astype(dtype, copy=False)

    return np.ascontiguousarray(values).tobytes(order="C")


def _encode_vtk_binary(values):
    """
    Encode one array using VTK XML inline binary format.

    The first UInt64 value gives the number of raw data bytes.
    """
    payload = _little_endian_bytes(values)

    header = np.asarray(
        [len(payload)],
        dtype="<u8",
    ).tobytes()

    return base64.b64encode(header + payload).decode("ascii")


def _set_data_array_content(data_array_element, values, binary):
    """
    Set the contents of a VTK DataArray element.
    """
    values = _prepare_numeric_array(values)

    if binary:
        data_array_element.set("format", "binary")
        data_array_element.text = _encode_vtk_binary(values)
    else:
        data_array_element.set("format", "ascii")

        if np.issubdtype(values.dtype, np.integer):
            data_array_element.text = " ".join(
                str(int(value))
                for value in values.ravel(order="C")
            )
        else:
            data_array_element.text = " ".join(
                f"{float(value):.16e}"
                for value in values.ravel(order="C")
            )


def _add_data_array(
    parent,
    name,
    values,
    *,
    number_of_components=1,
    binary=True,
):
    """
    Add one VTK DataArray to an XML parent element.
    """
    values = _prepare_numeric_array(values)

    attributes = {
        "type": _vtk_dtype_name(values),
        "Name": str(name),
        "NumberOfComponents": str(int(number_of_components)),
    }

    data_array = ET.SubElement(
        parent,
        "DataArray",
        attributes,
    )

    _set_data_array_content(
        data_array,
        values,
        binary=binary,
    )

    return data_array


def _write_xml_tree(root, filename):
    """
    Write an XML tree to disk.
    """
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    tree = ET.ElementTree(root)

    try:
        ET.indent(tree, space="  ")
    except AttributeError:
        pass

    tree.write(
        filename,
        encoding="utf-8",
        xml_declaration=True,
    )


# ============================================================
# Structured-field utilities
# ============================================================

def _normalize_grid_shape(grid_shape):
    """
    Convert line, plane, or volume shapes to a three-dimensional shape.

    Examples
    --------
    (Nx,)         -> (Nx, 1, 1)
    (Nx, Ny)      -> (Nx, Ny, 1)
    (Nx, Ny, Nz)  -> unchanged
    """
    grid_shape = tuple(int(value) for value in grid_shape)

    if len(grid_shape) == 1:
        grid_shape = grid_shape + (1, 1)
    elif len(grid_shape) == 2:
        grid_shape = grid_shape + (1,)
    elif len(grid_shape) != 3:
        raise ValueError("grid_shape must contain one, two, or three dimensions.")

    if any(value <= 0 for value in grid_shape):
        raise ValueError("Every grid dimension must be positive.")

    return grid_shape


def _normalize_xyz_vector(value, name, singleton_default=None):
    """
    Convert a scalar or length-3 value to a finite three-component vector.
    """
    if np.isscalar(value):
        vector = np.full(3, float(value), dtype=float)
    else:
        vector = np.asarray(value, dtype=float)

        if vector.shape != (3,):
            raise ValueError(f"{name} must be a scalar or have shape (3,).")

    if np.any(~np.isfinite(vector)):
        raise ValueError(f"{name} must contain finite values.")

    if singleton_default is not None:
        vector = vector.copy()
        vector[vector == 0.0] = float(singleton_default)

    return vector


def _flatten_structured_array(values, grid_shape):
    """
    Flatten a scalar or vector structured-grid array in VTK point order.

    Arrays are assumed to use NumPy indexing:

        values[ix, iy, iz]

    VTK structured datasets expect x to vary fastest. NumPy Fortran-order
    flattening provides that ordering for arrays shaped (Nx, Ny, Nz).
    """
    values = np.asarray(values)

    original_ndim = len(tuple(grid_shape))
    normalized_shape = _normalize_grid_shape(grid_shape)

    if original_ndim == 1:
        expected_scalar_shape = (normalized_shape[0],)
    elif original_ndim == 2:
        expected_scalar_shape = normalized_shape[:2]
    else:
        expected_scalar_shape = normalized_shape

    if values.shape == expected_scalar_shape:
        scalar_values = values.reshape(normalized_shape)
        return scalar_values.ravel(order="F"), 1

    if (
        values.ndim == len(expected_scalar_shape) + 1
        and values.shape[:-1] == expected_scalar_shape
    ):
        number_of_components = values.shape[-1]

        if number_of_components <= 0:
            raise ValueError("Vector arrays must have at least one component.")

        component_columns = []

        for component in range(number_of_components):
            component_values = values[..., component].reshape(normalized_shape)
            component_columns.append(component_values.ravel(order="F"))

        flattened = np.column_stack(component_columns).ravel(order="C")

        return flattened, number_of_components

    raise ValueError(
        f"Field array has shape {values.shape}; expected "
        f"{expected_scalar_shape} for a scalar field or "
        f"{expected_scalar_shape + ('Ncomponents',)} for a vector field."
    )


# ============================================================
# VTK ImageData writer
# ============================================================

def write_image_data_vti(
    filename,
    point_data,
    grid_shape,
    origin,
    spacing,
    *,
    active_scalars=None,
    binary=True,
):
    """
    Write a regular line, plane, or volume field as VTK ImageData.

    The supplied arrays are stored as PointData. This is appropriate
    when the generated grid coordinates represent the actual evaluation
    points, including cell-centred sampling grids.

    Parameters
    ----------
    filename : str or Path
        Output filename. The ".vti" suffix is added if absent.

    point_data : dict
        Mapping from field names to arrays.

        Scalar examples:
            c_mg.shape == grid_shape
            c_si.shape == grid_shape

        Vector example:
            displacement.shape == grid_shape + (3,)

    grid_shape : tuple
        Grid dimensions:

            (Nx,)
            (Nx, Ny)
            (Nx, Ny, Nz)

    origin : scalar or array_like, shape (3,)
        Coordinates of the first grid point.

        For a cell-centred grid this should be the coordinate of the
        first cell centre, not the lower box boundary.

    spacing : scalar or array_like, shape (3,)
        Grid-point spacing.

        For singleton dimensions, any positive value may be used.

    active_scalars : str or None
        Name of the default scalar field displayed by ParaView.

    binary : bool, default=True
        Use compact VTK inline binary encoding. ASCII output is useful
        mainly for debugging.

    Returns
    -------
    filename : Path
        Final output filename.
    """
    filename = Path(filename)

    if filename.suffix.lower() != ".vti":
        filename = filename.with_suffix(".vti")

    if not isinstance(point_data, dict) or len(point_data) == 0:
        raise ValueError("point_data must be a non-empty dictionary.")

    original_grid_shape = tuple(int(value) for value in grid_shape)
    nx, ny, nz = _normalize_grid_shape(original_grid_shape)

    origin = _normalize_xyz_vector(origin, "origin")
    spacing = _normalize_xyz_vector(spacing, "spacing")

    if np.any(spacing <= 0.0):
        raise ValueError("spacing values must be positive.")

    extent = f"0 {nx - 1} 0 {ny - 1} 0 {nz - 1}"

    root = ET.Element(
        "VTKFile",
        {
            "type": "ImageData",
            "version": "1.0",
            "byte_order": "LittleEndian",
            "header_type": "UInt64",
        },
    )

    image_data = ET.SubElement(
        root,
        "ImageData",
        {
            "WholeExtent": extent,
            "Origin": " ".join(f"{value:.16e}" for value in origin),
            "Spacing": " ".join(f"{value:.16e}" for value in spacing),
        },
    )

    piece = ET.SubElement(
        image_data,
        "Piece",
        {
            "Extent": extent,
        },
    )

    point_data_attributes = {}

    if active_scalars is not None:
        if active_scalars not in point_data:
            raise ValueError(
                f"active_scalars={active_scalars!r} is not present in point_data."
            )

        point_data_attributes["Scalars"] = str(active_scalars)

    point_data_element = ET.SubElement(
        piece,
        "PointData",
        point_data_attributes,
    )

    for name, values in point_data.items():
        flattened, number_of_components = _flatten_structured_array(
            values,
            original_grid_shape,
        )

        _add_data_array(
            point_data_element,
            name,
            flattened,
            number_of_components=number_of_components,
            binary=binary,
        )

    ET.SubElement(piece, "CellData")

    _write_xml_tree(root, filename)

    return filename


# ============================================================
# VTK PolyData point writer
# ============================================================

def _validate_points(points):
    """
    Validate an N-by-3 coordinate array.
    """
    points = np.asarray(points, dtype=float)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3).")

    if np.any(~np.isfinite(points)):
        raise ValueError("points contains non-finite coordinates.")

    return points


def _flatten_point_array(values, n_points, name):
    """
    Validate and flatten point-associated scalar or vector data.
    """
    values = np.asarray(values)

    if values.shape == (n_points,):
        return values, 1

    if values.ndim == 2 and values.shape[0] == n_points:
        number_of_components = values.shape[1]

        if number_of_components <= 0:
            raise ValueError(f"{name} must contain at least one component.")

        return values.ravel(order="C"), number_of_components

    raise ValueError(
        f"Point array {name!r} has shape {values.shape}; expected "
        f"({n_points},) or ({n_points}, Ncomponents)."
    )


def write_point_data_vtp(
    filename,
    points,
    point_data=None,
    *,
    active_scalars=None,
    binary=True,
):
    """
    Write arbitrary points and associated data as VTK PolyData vertices.

    Parameters
    ----------
    filename : str or Path
        Output filename. The ".vtp" suffix is added if absent.

    points : ndarray, shape (N, 3)
        Point coordinates.

    point_data : dict or None
        Point-associated scalar or vector arrays.

    active_scalars : str or None
        Default scalar array selected in ParaView.

    binary : bool, default=True
        Use inline binary VTK XML data.

    Returns
    -------
    filename : Path
        Final output filename.
    """
    filename = Path(filename)

    if filename.suffix.lower() != ".vtp":
        filename = filename.with_suffix(".vtp")

    points = _validate_points(points)
    n_points = len(points)

    if point_data is None:
        point_data = {}

    if not isinstance(point_data, dict):
        raise TypeError("point_data must be a dictionary or None.")

    if active_scalars is not None and active_scalars not in point_data:
        raise ValueError(
            f"active_scalars={active_scalars!r} is not present in point_data."
        )

    root = ET.Element(
        "VTKFile",
        {
            "type": "PolyData",
            "version": "1.0",
            "byte_order": "LittleEndian",
            "header_type": "UInt64",
        },
    )

    poly_data = ET.SubElement(root, "PolyData")

    piece = ET.SubElement(
        poly_data,
        "Piece",
        {
            "NumberOfPoints": str(n_points),
            "NumberOfVerts": str(n_points),
            "NumberOfLines": "0",
            "NumberOfStrips": "0",
            "NumberOfPolys": "0",
        },
    )

    point_data_attributes = {}

    if active_scalars is not None:
        point_data_attributes["Scalars"] = str(active_scalars)

    point_data_element = ET.SubElement(
        piece,
        "PointData",
        point_data_attributes,
    )

    for name, values in point_data.items():
        flattened, number_of_components = _flatten_point_array(
            values,
            n_points,
            name,
        )

        _add_data_array(
            point_data_element,
            name,
            flattened,
            number_of_components=number_of_components,
            binary=binary,
        )

    ET.SubElement(piece, "CellData")

    points_element = ET.SubElement(piece, "Points")

    _add_data_array(
        points_element,
        "Points",
        points.ravel(order="C").astype(np.float64),
        number_of_components=3,
        binary=binary,
    )

    verts_element = ET.SubElement(piece, "Verts")

    connectivity = np.arange(n_points, dtype=np.int64)
    offsets = np.arange(1, n_points + 1, dtype=np.int64)

    _add_data_array(
        verts_element,
        "connectivity",
        connectivity,
        binary=binary,
    )

    _add_data_array(
        verts_element,
        "offsets",
        offsets,
        binary=binary,
    )

    _write_xml_tree(root, filename)

    return filename


# ============================================================
# Particle writer
# ============================================================

def write_particle_points_vtp(
    filename,
    positions,
    radii,
    *,
    particle_id=None,
    particle_type=None,
    phase_id=None,
    Q=None,
    solute_names=None,
    Q_mg=None,
    Q_si=None,
    extra_point_data=None,
    binary=True,
):
    """
    Write particle centres and radii as VTK PolyData.

    Particles are stored as points, not explicit triangulated spheres.
    In ParaView, apply the Glyph filter with a Sphere source and scale
    the glyphs using the "radius" array.

    Parameters
    ----------
    filename : str or Path
        Output filename.

    positions : ndarray, shape (N, 3)
        Particle centres.

    radii : ndarray, shape (N,)
        Particle radii.

    particle_id : ndarray, shape (N,), optional
        Particle identifiers. Defaults to 1, ..., N.

    particle_type : ndarray, shape (N,), optional
        Particle-type identifiers.

    phase_id : ndarray, shape (N,), optional
        Zero-based precipitate-phase identifiers. Written to VTK as the
        point-data array ``phase``.

    Q : ndarray, shape (S, N), optional
        Species-indexed monopole strengths. When supplied, ``solute_names``
        must contain the corresponding S species names. Arrays are written
        as ``Q_<species>`` (for example ``Q_Mg``, ``Q_Si``, or ``Q_Sc``).

    solute_names : sequence of str, optional
        Species names corresponding to the first axis of ``Q``. Required
        when ``Q`` is supplied.

    Q_mg, Q_si : ndarray, shape (N,), optional
        Legacy Mg and Si monopole strengths. These retain the historical VTK
        array names ``Q_mg`` and ``Q_si``. Do not combine them with ``Q``.

    extra_point_data : dict or None
        Additional scalar or vector point data.

    binary : bool, default=True
        Use inline binary VTK XML output.

    Returns
    -------
    filename : Path
        Final output filename.
    """
    positions = _validate_points(positions)
    radii = np.asarray(radii, dtype=float)

    n_particles = len(positions)

    if radii.shape != (n_particles,):
        raise ValueError("radii must have shape (Nparticles,).")

    if np.any(~np.isfinite(radii)) or np.any(radii < 0.0):
        raise ValueError("radii must contain finite non-negative values.")

    if particle_id is None:
        particle_id = np.arange(1, n_particles + 1, dtype=np.int64)
    else:
        particle_id = np.asarray(particle_id)

        if particle_id.shape != (n_particles,):
            raise ValueError("particle_id must have shape (Nparticles,).")

    point_data = {
        "particle_id": particle_id,
        "radius": radii,
        "diameter": 2.0 * radii,
    }

    optional_arrays = {
        "particle_type": particle_type,
        "phase": phase_id,
    }

    if Q is not None:
        if Q_mg is not None or Q_si is not None:
            raise ValueError(
                "Do not supply Q together with legacy Q_mg/Q_si arrays."
            )

        if solute_names is None:
            raise ValueError("solute_names is required when Q is supplied.")

        solute_names = tuple(str(name) for name in solute_names)
        Q = np.asarray(Q)

        if Q.ndim != 2:
            raise ValueError("Q must have shape (Nspecies, Nparticles).")

        if Q.shape[1] != n_particles:
            raise ValueError(
                "Q must have shape (Nspecies, Nparticles); "
                f"got {Q.shape} for {n_particles} particles."
            )

        if len(solute_names) != Q.shape[0]:
            raise ValueError(
                "solute_names must contain one name for each row of Q."
            )

        if any(not name for name in solute_names):
            raise ValueError("solute_names must contain non-empty names.")

        if len(set(solute_names)) != len(solute_names):
            raise ValueError("solute_names must be unique.")

        for species_index, species_name in enumerate(solute_names):
            field_name = f"Q_{species_name}"
            if field_name in point_data or field_name in optional_arrays:
                raise ValueError(
                    f"Species name {species_name!r} produces the reserved "
                    f"VTK field name {field_name!r}."
                )
            optional_arrays[field_name] = Q[species_index]
    else:
        optional_arrays["Q_mg"] = Q_mg
        optional_arrays["Q_si"] = Q_si

    for name, values in optional_arrays.items():
        if values is None:
            continue

        values = np.asarray(values)

        if values.shape != (n_particles,):
            raise ValueError(f"{name} must have shape (Nparticles,).")

        point_data[name] = values

    if extra_point_data is not None:
        if not isinstance(extra_point_data, dict):
            raise TypeError("extra_point_data must be a dictionary or None.")

        for name, values in extra_point_data.items():
            if name in point_data:
                raise ValueError(
                    f"extra_point_data contains the reserved name {name!r}."
                )

            point_data[name] = values

    return write_point_data_vtp(
        filename=filename,
        points=positions,
        point_data=point_data,
        active_scalars="radius",
        binary=binary,
    )


# ============================================================
# ParaView time-collection writer
# ============================================================

def write_pvd_collection(
    filename,
    datasets,
):
    """
    Write a ParaView Data Collection file for time-dependent datasets.

    Parameters
    ----------
    filename : str or Path
        Output ".pvd" filename.

    datasets : iterable
        Entries may be either:

            (time, dataset_filename)

        or:

            {
                "time": time,
                "filename": dataset_filename,
                "group": optional_group_name,
                "part": optional_part_number,
            }

    Returns
    -------
    filename : Path
        Final PVD filename.
    """
    filename = Path(filename)

    if filename.suffix.lower() != ".pvd":
        filename = filename.with_suffix(".pvd")

    filename.parent.mkdir(parents=True, exist_ok=True)

    root = ET.Element(
        "VTKFile",
        {
            "type": "Collection",
            "version": "1.0",
            "byte_order": "LittleEndian",
        },
    )

    collection = ET.SubElement(root, "Collection")

    for entry in datasets:
        if isinstance(entry, dict):
            time_value = float(entry["time"])
            dataset_filename = Path(entry["filename"])
            group = str(entry.get("group", ""))
            part = int(entry.get("part", 0))
        else:
            time_value, dataset_filename = entry
            time_value = float(time_value)
            dataset_filename = Path(dataset_filename)
            group = ""
            part = 0

        if not np.isfinite(time_value):
            raise ValueError("Dataset times must be finite.")

        relative_filename = os.path.relpath(
            dataset_filename,
            start=filename.parent,
        )

        ET.SubElement(
            collection,
            "DataSet",
            {
                "timestep": f"{time_value:.16e}",
                "group": group,
                "part": str(part),
                "file": relative_filename,
            },
        )

    _write_xml_tree(root, filename)

    return filename