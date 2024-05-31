"""
==================
Camera Calibration
==================

This submodule contains functions and routines to calibrate a camera
system.

DLT Model
=============
    get_cam_params - Create DLT camera model parameters
    calibrate_dlt - Calibrate and return DLT coefficients and fitting error
    minimize_params - Minimize the camera model parameters
    project_points - Project 3D points to image points
    
Pinhole Model
=============
    get_cam_params - Generate pinhole camera model parameters
    get_rotation_matrix - Calculate the orthogonal rotation matrix
    minimize_params - Minimize the camera model parameters
    project_points - Project 3D points to image points
    project_to_z - Project image points to 3D points
    line_intersect - Using two lines, locate where those lines intersect
    save_parameters - Save pinhole camera parameters to a text file
    load_parameters - Load pinhole camera parameters from a text file
   
Polynomial Model
================
    get_cam_params - Generate polynomial camera model parameters
    minimize_params - Minimize the camera model parameters
    project_points - Project 3D points to image points
    project_to_z - Project image points to 3D points
    save_parameters - Save polynomial camera parameters to a text file
    load_parameters - Load polynomial camera parameters from a text file

Marker Detection (Marker Grid)
==============================
    preprocess_image - Preprocess calibration image
    get_circular_template - Generate a circular template
    get_cross_template - Generate a cross template
    detect_markers_template - Detect markers via template correlation
    detect_markers_blobs - Detect markers via labeling blobs

Calibration Grid
================
    get_simple_grid - Create a simple rectangular calibration grid
    
Match Calibration Points
========================
    find_corners - Locate 4 or 6 corners of a calibration grid
    find_nearest_points - Find the closest point to a cursor
    reorder_image_points - Reorder image points in ascending order
    show_calibration_image - Plot calibration image and markers
    get_pairs_proj - Match marker pairs using a rough projection estimate
    
Utils
=====
    get_reprojection_error - Calculate the root mean square (RMS) error
    get_los_error - Calculate the RMS error of a line of sight (LOS) at a z plane
    get_image_mapping - Calculate the mappings to rectify a 2D image
    plot_epipolar_line - Plot a 3D representation of epipolar lines

"""