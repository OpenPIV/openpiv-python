"""
==================
Camera Calibration
==================

This submodule contains functions and routines to calibrate a camera
system.

DLT Model
=========
    camera - Create an instance of a DLT camera model
    calibrate_dlt - Calibrate and return DLT coefficients and fitting error
    line_intersect - Using two lines, locate where those lines intersect
    multi_line_intersect - Using multiple lines, approximate their intersection
    
Pinhole Model
=============
    calibrate_intrinsics - Calculate the intrinsic parameters using Zang's algorithm
    camera - Create an instance of a Pinhole camera model
    calibrate_dlt - Calibrate and return DLT coefficients and fitting error
    line_intersect - Using two lines, locate where those lines intersect
    multi_line_intersect - Using multiple lines, approximate their intersection
   
Polynomial Model
================
    camera - Create an instance of a Soloff camera model
    multi_line_intersect - Using multiple lines, approximate their intersection

Marker Detection (Marker Grid)
==============================
    detect_markers_template - Detect markers via template correlation
    detect_markers_blobs - Detect markers via labeling blobs
    get_circular_template - Generate a circular template
    get_cross_template - Generate a cross template
    preprocess_image - Preprocess calibration image

Calibration Grid
================
    get_asymmetric_grid - Create an asymmetric rectangular calibration grid
    get_simple_grid - Create a simple rectangular calibration grid
    
Match Calibration Points
========================
    find_corners - Locate 4 or 6 corners of a calibration grid
    find_nearest_points - Find the closest point to a cursor
    get_pairs_dlt - Match marker pairs using 4 corners and the DLT algorithm
    get_pairs_proj - Match marker pairs using a rough projection estimate
    reorder_image_points - Reorder image points in ascending order
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