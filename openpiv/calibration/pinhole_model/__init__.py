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
coordinates are calculated as such:

$$ P =
\begin{vmatrix}
X \\ Y \\ Z \\
\end{vmatrix}
$$

$$ 
\begin{vmatrix}
x_c \\ y_c \\ z_c
\end{vmatrix}
= P * R^{-1} - R^{-1} * T
$$

$$
x_n = \frac{x_c}{z_c} \\
y_n = \frac{y_c}{z_c} 
$$

where

$$ 
R = 
\begin{vmatrix}
r_{11} & r_{12} & r_{13} \\
r_{21} & r_{22} & r_{23} \\
r_{31} & r_{32} & r_{33} \\
\end{vmatrix}
$$ 
and
$$ 
T = 
\begin{vmatrix}
T_x \\ T_y \\ T_z
\end{vmatrix}
$$

where letters denoting R and T define the 3x3 rotation matix and the
translation vector respectively, which are commonly associated with
the extrinsic matrix. 

Since there are usually additional irregularities such as lens 
distortion and intermediate medias, additional steps to mitigate them are
necessary. Without correcting these irregularities, results from triangulation
or tomographic reconstruction are often meaningless. To circumnavigate this
issue, two distortion models are implemented: polynomial and brown
distortion models. The brown distortion model covers nearly all tangential
and radial distortion components and is relatively simple. However, this
distortion model cannot account for scheimpflug lenses or intermediate
medias. Under these circumstances, it is wise to use the polynomial
distortion model. This distortion model utilized second degree polynomials
for distorting and undistorting normalized camera coordinates. Since we are
working with polynomials, most distortions can be minimized including the use
of scheimpflug lenses and distortions caused by intermediate medias. All
distortion correction methods are applied directly to the normalized camera
coordinates.

Finally, the normalized pixel coordinates are scaled using the intrinsic
matrix. The intrinsic matrix is composed of focal depths (fx, fy), measured
in pixels, and the principal point (cx, cy). One can estimate the focal
depths of fx and fy by dividing the focal number in mm with its associated
pixel size. For instance, if a lens had a focal number of 20mm and the pixels
are 5 μm, then the fx/fy would be approximately 20/0.005 = 4000 pixels. The
following transformation applies the intrinsic matrix to normalized camera
coordinates:

$$ 
\begin{vmatrix}
x \\ y \\ 1
\end{vmatrix} = 
\begin{vmatrix}
f_x & 0 & c_x \\
0 & f_y & c_y \\
0 & 0 & 1
\end{vmatrix} *
\begin{vmatrix}
x_n \\ y_n \\ 1
\end{vmatrix}
$$ 

Once a camera systen is calibrated on an individual basis, it may be
beneficial to calculate 3D triangulation errors. This requires projecting
pixels along a light ray to a specific Z-plane. Since the distortion model
operate with normalized camera coordinates, it is vital that the image
coordinates are properly normalized.

$$ 
\begin{vmatrix}
x_n \\ y_n \\ 1
\end{vmatrix} = 
\begin{vmatrix}
f_x & 0 & c_x \\
0 & f_y & c_y \\
0 & 0 & 1 \\
\end{vmatrix}^{-1} *
\begin{vmatrix}
x \\
y \\
1 \\
\end{vmatrix}
$$ 

After normalization, the camera coordinates are distorted and turned into
direction vectors.

$$ 
\begin{vmatrix}
dx \\
dy \\
dz \\
\end{vmatrix} = R *
\begin{vmatrix}
x_n \\y_n \\ 1
\end{vmatrix}
$$ 

Finally, the lab coordinates can be calculated as Ax + b where x is the
direction vector, A is the distance from the physical camera to the
projection plane, and b is the translation of the physical camera in 
respect to the calibration markers.

$$
a = \frac{(Z - T_z)}{d_z} \\
X = a*d_x + T_x \\
Y = a*d_y + T_y \\
Z =  Z
$$

Functions
=========
    calibrate_intrinsics - Calculate the instrinsic parameters using Zang's method
    generate_camera_params - Create a pinhole camera data structure
    get_rotation_matrix - Calculate a 3x3 rotation matrix from Euler angles
    project_points - Project lab coordinates to image coordinates
    project_to_z - Project image coordinates to lab coordinates
    minimize_camera_params - Optimize pinhole camera parameters
    line_intersect - Calculate where two rays intersect
    save_parameters - Save pinhole camera parameters to a text file
    load_parameters - Load pinhole camera parameters from a text file

"""
from ._epipolar_geom import *
from ._minimization import *
from ._projection import *
from ._utils import *
#from ._zang import *


__all__ = [s for s in dir() if not s.startswith("_")]