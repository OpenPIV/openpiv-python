"""
=========
DLT Model
=========

This module contains an implementation of a direct linear transformation
model which is equivalent to a pinhole camera model under conditions with
no distortion. However, this condition is practically impossible to obtain
in laboratory conditions, so calibration errors are typically higher than
that of pinhole and polynomial calibration methods.

Functions
=========
    get_cam_params - Create DLT camera model parameters
    calibrate_dlt - Calibrate and return DLT coefficients and fitting error
    minimize_params - Minimize the camera model parameters
    project_points - Project 3D points to image points
    project_to_z - Project image points to 3D points
    line_intersect - Using two lines, locate where those lines intersect
    save_parameters - Save DLT camera parameters to a text file
    load_parameters - Load DLT camera parameters from a text file

Note
====
It is important to only import the submodule and not the functions that
are in the submodules. Explicitly importing a function from this submodule
could cause conflicts between other camera models due to similar naming
conventions that are normally protected behind namespaces.

"""
from ._minimization import *
from ._projection import *
from ._utils import *


__all__ = [s for s in dir() if not s.startswith("_")]