"""Readers for the DaVis XML exports used by the stereo-PIV example.

Two export files are understood:

``MarkPositionTable.xml``
    The calibration marks of every camera: for each mark the *world*
    position on the calibration plate (mm) and the *raw* (distorted)
    pixel position in the image.

``Calibration.xml``
    The camera models DaVis fitted itself.  Only the reported
    reprojection RMS per camera is read here, so that the fit done by
    OpenPIV can be compared against the DaVis reference.

Both readers work on plain XML text, so they can be unit tested with
small synthetic fixtures and do not require the DaVis software.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "CameraMarks",
    "PlateInfo",
    "VectorField",
    "parse_calibration_rms",
    "parse_mark_position_table",
    "parse_plate_info",
    "parse_scales",
    "read_vector_field",
    "vector_field_grid",
]


@dataclass(frozen=True)
class CameraMarks:
    """Calibration marks of a single camera.

    Attributes
    ----------
    camera:
        DaVis camera number.
    object_points:
        ``(3, n)`` world positions of the marks in mm.
    image_points:
        ``(2, n)`` raw (distorted) pixel positions of the marks.
    """

    camera: int
    object_points: np.ndarray
    image_points: np.ndarray

    def __post_init__(self) -> None:
        if self.object_points.shape[0] != 3:
            raise ValueError("object_points must have shape (3, n)")
        if self.image_points.shape[0] != 2:
            raise ValueError("image_points must have shape (2, n)")
        if self.object_points.shape[1] != self.image_points.shape[1]:
            raise ValueError("object_points and image_points must have the same length")

    @property
    def count(self) -> int:
        """Number of marks."""
        return int(self.object_points.shape[1])

    def planes(self) -> dict[float, np.ndarray]:
        """Indices of the marks grouped by their world ``z`` plane."""
        z = self.object_points[2]
        return {float(value): np.flatnonzero(np.isclose(z, value)) for value in np.unique(z)}


@dataclass(frozen=True)
class PlateInfo:
    """Geometry of the calibration plate as exported by DaVis."""

    identifier: str
    mark_spacing_mm: float
    mark_size_mm: float
    plate_thickness_mm: float
    plane_to_plane_distance_mm: float


def _parse_float(value: str | None) -> float:
    if value is None:
        raise ValueError("missing numeric attribute")
    return float(value)


def parse_mark_position_table(path: str | Path) -> dict[int, CameraMarks]:
    """Parse a DaVis ``MarkPositionTable.xml`` export.

    Parameters
    ----------
    path:
        Path of the XML file.

    Returns
    -------
    dict
        ``camera number -> CameraMarks``.
    """
    root = ET.parse(Path(path)).getroot()
    marks: dict[int, CameraMarks] = {}
    for camera_el in root.findall("Camera"):
        number = int(camera_el.attrib["CameraNumber"])
        world: list[list[float]] = [[], [], []]
        raw: list[list[float]] = [[], []]
        for view in camera_el.findall("View"):
            for mark in view.findall("Mark"):
                world_el = mark.find("WorldPos")
                raw_el = mark.find("RawPos")
                if world_el is None or raw_el is None:
                    continue
                world[0].append(_parse_float(world_el.attrib.get("x")))
                world[1].append(_parse_float(world_el.attrib.get("y")))
                world[2].append(_parse_float(world_el.attrib.get("z")))
                raw[0].append(_parse_float(raw_el.attrib.get("x")))
                raw[1].append(_parse_float(raw_el.attrib.get("y")))
        if not world[0]:
            continue
        marks[number] = CameraMarks(
            camera=number,
            object_points=np.array(world, dtype=np.float64),
            image_points=np.array(raw, dtype=np.float64),
        )
    if not marks:
        raise ValueError(f"no camera with marks found in {path}")
    return marks


def parse_calibration_rms(path: str | Path) -> dict[int, float]:
    """Parse the per-camera reprojection RMS from a DaVis ``Calibration.xml``.

    Returns
    -------
    dict
        ``camera number -> RMS in pixels``.  Cameras without a reported fit
        error are omitted.
    """
    root = ET.parse(Path(path)).getroot()
    result: dict[int, float] = {}
    for mapper in root.findall(".//CoordinateMapper"):
        identifier = mapper.attrib.get("CameraIdentifier")
        if identifier is None:
            continue
        fit_error = mapper.find(".//CommonParameters/FitError")
        if fit_error is None:
            continue
        rms = fit_error.attrib.get("RMS")
        if rms is None:
            continue
        result[int(identifier)] = float(rms)
    return result


def parse_plate_info(path: str | Path) -> PlateInfo:
    """Parse the calibration plate geometry from ``MarkPositionTable.xml``."""
    root = ET.parse(Path(path)).getroot()
    return PlateInfo(
        identifier=root.attrib.get("CalibrationPlateIdentifier", ""),
        mark_spacing_mm=_parse_float(root.attrib.get("DistanceMarkToMarkInMm", "nan")),
        mark_size_mm=_parse_float(root.attrib.get("MarkSizeMm", "nan")),
        plate_thickness_mm=_parse_float(root.attrib.get("PlateThicknessMm", "nan")),
        plane_to_plane_distance_mm=_parse_float(root.attrib.get("PlaneToPlaneDistanceMm", "nan")),
    )


@dataclass(frozen=True)
class VectorField:
    """A DaVis vector field (``.vc7``) with its world grid.

    The components are stored in the *corrected image* frame of DaVis:
    ``u`` points to the right, ``v`` points **down** the image and ``w``
    is the out-of-plane component.  DaVis' corrected image has
    ``y = offset_y - step * row`` (the ``_SCALE_Y`` factor is negative),
    i.e. the world Y axis points *up* while the image row grows
    downwards.  The world-frame vertical velocity is therefore
    ``V_world = -v``.

    Attributes
    ----------
    u, v, w:
        The velocity components in m/s, shaped ``(ny, nx)``.
    x, y:
        The world coordinates (mm) of the grid columns and rows.
    valid:
        Mask of the vectors DaVis actually computed (the invalid ones are
        stored as zero).
    """

    u: np.ndarray
    v: np.ndarray
    w: np.ndarray
    x: np.ndarray
    y: np.ndarray
    valid: np.ndarray

    @property
    def u_world(self) -> np.ndarray:
        """Horizontal velocity in the world frame (m/s)."""
        return self.u

    @property
    def v_world(self) -> np.ndarray:
        """Vertical velocity in the world frame (m/s).

        The sign is flipped with respect to :attr:`v`: the DaVis
        corrected image counts its rows downwards, the world Y axis
        points upwards.
        """
        return -self.v


def vector_field_grid(
    scale_x: float, offset_x: float, scale_y: float, offset_y: float,
    shape: tuple[int, int], spacing_px: float = 16.0,
) -> tuple[np.ndarray, np.ndarray]:
    """World grid of a DaVis vector field.

    Parameters
    ----------
    scale_x, offset_x:
        The ``_SCALE_X`` factor (mm per pixel) and offset (mm).
    scale_y, offset_y:
        The ``_SCALE_Y`` factor and offset.
    shape:
        ``(ny, nx)`` of the field.
    spacing_px:
        Grid spacing in pixels (16 for the 4x64/75% DaVis recipe).

    Returns
    -------
    (x, y)
        The world coordinates (mm) of the columns and of the rows.  The
        rows follow the image: the last row is the smallest ``y`` when
        ``scale_y`` is negative.
    """
    ny, nx = shape
    step = abs(scale_x) * spacing_px
    x = offset_x + step * np.arange(nx)
    y = offset_y + np.sign(scale_y) * step * np.arange(ny)
    return x, y


def parse_scales(attributes: Mapping[str, str]) -> dict:
    """Extract the scale factors of a DaVis vector field.

    Parameters
    ----------
    attributes:
        The attribute dictionary of a ``.vc7`` file; every entry is a
        multi line string whose first line holds the value and the
        second one the offset.

    Returns
    -------
    dict
        ``scale`` (velocity, m/s per unit), ``scale_x``, ``offset_x``,
        ``scale_y`` and ``offset_y``.
    """

    def _numbers(*names: str) -> tuple[float, float]:
        for name in names:
            if name in attributes:
                break
        else:
            raise KeyError(f"none of {names} in the vector field attributes")
        lines = str(attributes[name]).split("\n")
        tokens = lines[0].split()
        value = float(tokens[0])
        if len(tokens) > 1:
            # "_SCALE_X: '<value> <offset>'"
            return value, float(tokens[1])
        # "FrameScaleX0: '<value>' / '<offset>' on the next line"
        offset = float(lines[1].split()[0]) if len(lines) > 1 else 0.0
        return value, offset

    scale, _ = _numbers("_SCALE_I", "FrameScaleI0")
    scale_x, offset_x = _numbers("_SCALE_X", "FrameScaleX0")
    scale_y, offset_y = _numbers("_SCALE_Y", "FrameScaleY0")
    return {
        "scale": scale,
        "scale_x": scale_x,
        "offset_x": offset_x,
        "scale_y": scale_y,
        "offset_y": offset_y,
    }


def read_vector_field(path: str | Path, spacing_px: float = 16.0) -> VectorField:
    """Read a DaVis vector field (``.vc7``) with its world grid.

    The velocities are converted to m/s with the ``_SCALE_I`` factor of
    the file; the world grid is rebuilt from the ``_SCALE_X`` /
    ``_SCALE_Y`` factors and offsets.

    Parameters
    ----------
    path:
        The ``.vc7`` file.
    spacing_px:
        Grid spacing in pixels, 16 for the ``4x64x64_75`` DaVis recipe.

    Returns
    -------
    VectorField
    """
    try:
        import readim7
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "reading DaVis vector fields needs the readim7 package: "
            'pip install "readim7 @ git+https://github.com/alexlib/readim7.git"'
        ) from exc

    buffer_, attributes = readim7.get_Buffer_andAttributeList(str(path))
    array, _ = readim7.buffer_as_array(buffer_)
    array = np.asarray(array, dtype=np.float64)
    if array.ndim != 3 or array.shape[0] < 3:
        raise ValueError(f"{path} does not hold a 3-component vector field")

    scales = parse_scales(attributes)
    scale = scales["scale"]
    scale_x, offset_x = scales["scale_x"], scales["offset_x"]
    scale_y, offset_y = scales["scale_y"], scales["offset_y"]

    u, v, w = array[0] * scale, array[1] * scale, array[2] * scale
    x, y = vector_field_grid(scale_x, offset_x, scale_y, offset_y, u.shape, spacing_px)
    return VectorField(u=u, v=v, w=w, x=x, y=y, valid=u != 0)
