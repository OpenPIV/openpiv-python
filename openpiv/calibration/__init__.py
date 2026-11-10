"""
==================
Camera Calibration
==================

This submodule contains functions and routines to calibrate a camera
system.

DLT Model
=========
DLTCamera - Create an instance of a DLT camera model

calibrate_dlt - Calibrate and return DLT coefficients and fitting error

line_intersect - Using two lines, locate where those lines intersect

multi_line_intersect - Using multiple lines, approximate their intersection

Pinhole Model
=============
calibrate_intrinsics - Calculate the intrinsic parameters using Zhang's algorithm

PinCam - Create an instance of a Pinhole camera model

calibrate_zhang - Initialize and refine planar-target calibration

bundle_adjust - Fit cameras and shared target poses into a single camera system

align_reference_plate - Align fitted cameras with a reference capture

line_intersect - Using two lines, locate where those lines intersect

multi_line_intersect - Using multiple lines, approximate their intersection

Polynomial Model
================
PolyCamera - Create an instance of a Soloff camera model

line_intersect - Using two lines, locate where those lines intersect

multi_line_intersect - Using multiple lines, approximate their intersection

Target Detection
================
detect_markers - Detect marker centers via template correlation and subpixel fitting

detect_checkerboard - Detect checkerboard corners with subpixel refinement

Calibration Grid
================
get_asymmetric_grid - Create an asymmetric rectangular calibration grid

get_simple_grid - Create a simple rectangular calibration grid

Match Calibration Points
========================
find_median_point - Find the detected point nearest the field median

find_middle_point - Find the detected point nearest the field mean

find_nearest_points - Find detected points nearest a supplied position

get_pairs_auto - Match a symmetric grid using its known board size

get_pairs_marching - Match symmetric or two-layer asymmetric grids by marching

get_pairs_proj - Match marker pairs using a rough projection estimate

show_calibration_image - Plot calibration image and markers

Utils
=====
homogenize - Homogenize array by appending ones to the last axis

get_image_mapping - Calculate the mappings to rectify a 2D image

get_los_error - Calculate the RMS error of a line of sight (LOS) at a z plane

get_reprojection_error - Calculate the root mean square (RMS) error

get_rmse - Calculate the root mean square error of the residuals

plot_epipolar_line - Plot a 3D representation of epipolar lines

"""
from ._camera import BaseCamera
