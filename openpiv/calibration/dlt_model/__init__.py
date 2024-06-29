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
    camera - Create an instance of a DLT camera model
    calibrate_dlt - Calibrate and return DLT coefficients and fitting error
    line_intersect - Using two lines, locate where those lines intersect
    multi_line_intersect - Using multiple lines, approximate their intersection

Note
====
It is important to only import the submodule and not the functions that
are in the submodules. Explicitly importing a function from this submodule
could cause conflicts between other camera models due to similar naming
conventions that are normally protected behind namespaces.

"""
from ._camera import *
from ._epipolar_geom import *
from ._utils import *


__all__ = [s for s in dir() if not s.startswith("_")]