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
    get_cam_params - Generate pinhole camera model parameters
    get_rotation_matrix - Calculate the orthogonal rotation matrix
    minimize_params - Minimize the camera model parameters
    project_points - Project 3D points to image points
    project_to_z - Project image points to 3D points
    line_intersect - Using two lines, locate where those lines intersect
    multi_line_intersect - Using multiple lines, approximate their intersection
    save_parameters - Save pinhole camera parameters to a text file
    load_parameters - Load pinhole camera parameters from a text file

"""
from ._camera import *
from ._epipolar_geom import *
from ._utils import *


__all__ = [s for s in dir() if not s.startswith("_")]