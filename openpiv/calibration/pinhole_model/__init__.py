"""
====================
Pinhole Camera Model
====================

This module contains an implementation of the pinhole camera model. This
model is an approximation of how light rays are captured by a camera. Under
ideal circumstances, lab coordinates can be mapped to image sensor
coordinates (also known as pixel coordinates). However, cameras are not
usually ideal and are placed arbitrarily in the lab space. This means that
the lab coordinates have to be transformed into normalized camera 
coordinates to remove this arbitrary translation. The normalized camera
coordinates are calculated as explained in readme pinhole camera model file.


Functions
=========
    calibrate_intrinsics - Calculate the intrinsic parameters using Zang's algorithm
    camera - Create an instance of a Pinhole camera model
    calibrate_dlt - Calibrate and return DLT coefficients and fitting error
    line_intersect - Using two lines, locate where those lines intersect
    multi_line_intersect - Using multiple lines, approximate their intersection

"""
from ._camera import *
from ._epipolar_geom import *
from ._zang import *


__all__ = [s for s in dir() if not s.startswith("_")]