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
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "CameraMarks",
    "PlateInfo",
    "parse_calibration_rms",
    "parse_mark_position_table",
    "parse_plate_info",
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
