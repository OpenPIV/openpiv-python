"""
=======================
Polynomial Camera Model
=======================

This module contains an implementation of a polynomial camera model. This model is
implented using 3rd order polynomials in the x and y axis and a 2nd order polynomial
along the z-axis. This model can handle a multiplitude of different distortions and
is usually preferred if processing algorithms later on do not heavily utilize
trangulation. Additionally, it is important that the calibration markers cover as
much of the image(s) as possible to limit artifacts from extrapolation.

Public Functions
================
    generate_camera_params - Create a pinhole camera data structure
    project_points - Project lab coordinates to image coordinates
    project_to_z - Project image coordinates to lab coordinates at specified Z-plane
    minimize_polynomial - Optimize polynomial camera parameters
    
Private Functions
=================
    _check_parameters - Check polynomial camera parameters

Note
====
It is important to only import the submodule and not the functions that are in the
submodules. Explicitly importing a function from this submodule could cause
conflicts between other camera models due to similar naming conventions that are
normaly protected behind namespaces.

"""
from ._check_params import *
from ._minimization import *
from ._projection import *
from ._utils import *


__all__ = [s for s in dir() if not s.startswith("_")]