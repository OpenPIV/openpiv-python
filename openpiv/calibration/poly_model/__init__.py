"""
=======================
Polynomial Camera Model
=======================

This module contains an implementation of a polynomial camera model. This
model is implemented using 3rd order polynomials in the x and y axis and a
2nd order polynomial along the z-axis. This model can handle a multiplitude
of different distortions and is usually preferred if processing algorithms
later on do not heavily utilize triangulation. Additionally, it is important
that the calibration markers cover as much of the image(s) as possible to
limit artifacts from extrapolation.

Functions
=========
    get_cam_params - Generate polynomial camera model parameters
    minimize_params - Minimize the camera model parameters
    project_points - Project 3D points to image points
    project_to_z - Project image points to 3D points
    save_parameters - Save polynomial camera parameters to a text file
    load_parameters - Load polynomial camera parameters from a text file

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