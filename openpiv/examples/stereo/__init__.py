"""Stereo-PIV example: calibration, rectification, disparity and 3C.

The example drives :mod:`openpiv.calibration.stereo` over a DaVis stereo
data set:

.. code-block:: bash

    python -m openpiv.examples.stereo --data-root /data/run \
        --mark-table /data/run/MarkPositionTable.xml calibrate

See the module README for the full workflow.
"""

from __future__ import annotations

from .config import ENV_PREFIX, MODEL_CHOICES, StereoConfig
from .davis import (
    CameraMarks,
    PlateInfo,
    VectorField,
    parse_calibration_rms,
    parse_mark_position_table,
    parse_plate_info,
    read_vector_field,
    vector_field_grid,
)
from .frames import ims_info, read_image, read_pair
from .pipeline import (
    MAP_NAMES,
    build_dewarp_maps,
    build_world_grid,
    common_marks,
    compare_fields,
    correct_disparity,
    fit_cameras,
    make_camera,
    reconstruct_3c,
    rectify,
    reprojection_report,
    run_2c_piv,
    triangulation_report,
)

__all__ = [
    "ENV_PREFIX",
    "MAP_NAMES",
    "MODEL_CHOICES",
    "CameraMarks",
    "PlateInfo",
    "StereoConfig",
    "VectorField",
    "build_dewarp_maps",
    "build_world_grid",
    "common_marks",
    "compare_fields",
    "correct_disparity",
    "fit_cameras",
    "ims_info",
    "make_camera",
    "parse_calibration_rms",
    "parse_mark_position_table",
    "parse_plate_info",
    "read_image",
    "read_pair",
    "read_vector_field",
    "reconstruct_3c",
    "rectify",
    "reprojection_report",
    "run_2c_piv",
    "triangulation_report",
    "vector_field_grid",
]
