import json
from pathlib import Path

import numpy as np


def _as_point(value, name):
    point = np.asarray(value, dtype=float)

    if point.shape != (3,):
        raise ValueError(f"{name} must contain exactly three coordinates.")

    if np.any(~np.isfinite(point)):
        raise ValueError(f"{name} contains non-finite coordinates.")

    return point


def _cell_center_coordinates(n):
    """
    Return n coordinates uniformly distributed over [0, 1] using
    cell centres.
    """
    n = int(n)

    if n <= 0:
        raise ValueError("Grid dimensions must be positive integers.")

    return (np.arange(n, dtype=float) + 0.5) / n


def _endpoint_coordinates(n):
    """
    Return n coordinates uniformly distributed over [0, 1], including
    both endpoints.
    """
    n = int(n)

    if n < 2:
        raise ValueError("A line must contain at least two points.")

    return np.linspace(0.0, 1.0, n)


def make_line_grid(start, end, n_points):
    """
    Construct uniformly spaced points along a line, including both
    endpoints.
    """
    start = _as_point(start, "start")
    end = _as_point(end, "end")

    u = _endpoint_coordinates(n_points)
    points = start[None, :] + u[:, None] * (end - start)[None, :]

    distance = u * np.linalg.norm(end - start)

    metadata = {
        "geometry": "line",
        "shape": (len(u),),
        "start": start,
        "end": end,
        "distance": distance,
    }

    return points, metadata


def make_plane_grid(origin, vector_u, vector_v, shape):
    """
    Construct a cell-centred grid on a plane.

    x(u, v) = origin + u * vector_u + v * vector_v
    """
    origin = _as_point(origin, "origin")
    vector_u = _as_point(vector_u, "vector_u")
    vector_v = _as_point(vector_v, "vector_v")

    if len(shape) != 2:
        raise ValueError("Plane shape must contain [Nu, Nv].")

    Nu, Nv = map(int, shape)

    u = _cell_center_coordinates(Nu)
    v = _cell_center_coordinates(Nv)

    U, V = np.meshgrid(u, v, indexing="ij")

    points = (
        origin[None, None, :]
        + U[..., None] * vector_u[None, None, :]
        + V[..., None] * vector_v[None, None, :]
    )

    metadata = {
        "geometry": "plane",
        "shape": (Nu, Nv),
        "origin": origin,
        "vector_u": vector_u,
        "vector_v": vector_v,
        "u": u,
        "v": v,
    }

    return points.reshape(-1, 3), metadata


def make_volume_grid(origin, vector_u, vector_v, vector_w, shape):
    """
    Construct a cell-centred grid within a parallelepiped.

    x(u, v, w) = (
        origin
        + u * vector_u
        + v * vector_v
        + w * vector_w
    )
    """
    origin = _as_point(origin, "origin")
    vector_u = _as_point(vector_u, "vector_u")
    vector_v = _as_point(vector_v, "vector_v")
    vector_w = _as_point(vector_w, "vector_w")

    if len(shape) != 3:
        raise ValueError("Volume shape must contain [Nu, Nv, Nw].")

    Nu, Nv, Nw = map(int, shape)

    u = _cell_center_coordinates(Nu)
    v = _cell_center_coordinates(Nv)
    w = _cell_center_coordinates(Nw)

    U, V, W = np.meshgrid(u, v, w, indexing="ij")

    points = (
        origin[None, None, None, :]
        + U[..., None] * vector_u[None, None, None, :]
        + V[..., None] * vector_v[None, None, None, :]
        + W[..., None] * vector_w[None, None, None, :]
    )

    metadata = {
        "geometry": "volume",
        "shape": (Nu, Nv, Nw),
        "origin": origin,
        "vector_u": vector_u,
        "vector_v": vector_v,
        "vector_w": vector_w,
        "u": u,
        "v": v,
        "w": w,
    }

    return points.reshape(-1, 3), metadata


def load_grid_request(filename):
    """
    Read a JSON grid request and construct the corresponding points.
    """
    filename = Path(filename)

    with filename.open("r", encoding="utf-8") as stream:
        request = json.load(stream)

    geometry = str(request["geometry"]).strip().lower()

    if geometry == "line":
        return make_line_grid(
            start=request["start"],
            end=request["end"],
            n_points=request["n_points"],
        )

    if geometry == "plane":
        return make_plane_grid(
            origin=request["origin"],
            vector_u=request["vector_u"],
            vector_v=request["vector_v"],
            shape=request["shape"],
        )

    if geometry == "volume":
        return make_volume_grid(
            origin=request["origin"],
            vector_u=request["vector_u"],
            vector_v=request["vector_v"],
            vector_w=request["vector_w"],
            shape=request["shape"],
        )

    raise ValueError(
        f"Unknown geometry {geometry!r}. "
        "Expected 'line', 'plane', or 'volume'."
    )
