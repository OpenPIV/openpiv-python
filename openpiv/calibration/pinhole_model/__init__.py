"""
====================
Pinhole Camera Model
====================

This module contains an implementation of the pinhole camera model. This model
is an approximation of how light rays are captured by a camera. Under ideal 
circumstances, world, or lab, coordinates can be mapped to image sensor
coordinates, also known as pixel coordinates. However, cameras are not usually
ideal and are placed arbitrarily in the lab space. This means that the lab
coordinates have to be transformed into normalized camera coordinates to remove
this arbitrary translation. The normalized camera coordinates are calculated
as such:

    |X|
P = |Y|
    |Z|

|x_c|   
|y_c| = P * R^-1 - R^-1 * T
| h |   

or

|x_c|   |r11 r21 r31|^-1  |X|   |r11 r21 r31|^-1  |tx|
|y_c| = |r12 r22 r32|  .  |Y| - |r12 r22 r32|  .  |ty|
| h |   |r13 r23 r33|     |Z|   |r13 r23 r33|     |tz|

x_n = x_c / h
y_n = y_c / h

where letters denoting r/R and t/T define the 3x3 rotation matix and translation 
vector respectively, which are commonly associated with the extrinsic matrix. 

Since there are usually additional irregularities such as lens 
distortion and intermediate medias, additional steps to mitigate them are
necessary. Without correcting these irregularities, results from traingulation or
tomographic reconstruction are often meaningless. To circumnavigate this issue, two
distortion models are implemented: polynomial and brown distortion models. The brown
distortion model covers nearly all tangential andradial distortion components and is
relatively simple. However, this distortion model cannot account for scheimpflug 
lenses or intermediate medias. Under these circumstances, it is wise to use the 
polynomial distortion model. This distortion model utilized second degree 
polynomials for distorting and undistorting normalized camera coordinates. Since
we are working with polynomials, most distortions can be minimized including the use
of scheimpflug lenses and distortions caused by intermediate medias. All distortion
correction methods are applied directly to the normalized camera coordinates.

Finally, the normalized pixel coordinates are scaled using the intrinsic matrix. The
intrinsic matrix is composed of focal depths (fx, fy), measured in pixels, and the
principal point (cx, cy). One can estimate the focal depths of fx and fy by dividing
the focal number in mm with its associated pixel size. For instance, if a lens had a
focal number of 20mm and the pixels are 5 μm, then the fx/fy would be approximately
20/0.005 = 4000 pixels. The following transformation applies the intrinsif matrix to
normalized camera coordinates.

|x|   |fx 0  cx|   |x_n|
|y| = |0  fy cy| . |y_n|
|1|   |0  0  1 |   | 1 |

Once a camera systen is calibrated on an individual basis, it may be beneficial to 
calculate 3D triangulation errors. This requires projecting pixels along a light ray
to a specific Z-plane. Since the distortion model operate with normalized camera
coordinates, it is vital that the image coordinates are properly normalized.

|x_n|   |fx 0  cx|^-1  |x|
|y_n| = |0  fy cy|   . |y|
| 1 |   |0  0  1 |     |1|

After normalization, the camera coordinates are distorted and turned into direction
vectors.

|dx|   |r11 r21 r31|   |x_nd|
|dy| = |r12 r22 r32| . |y_nd|
|dz|   |r13 r23 r33|   | 1  |

Finally, the lab coordinates can be calculated as Ax + b where x is the direction
vector, A is the distance from the physical camera to the projection plane, and b is
the translation of the physical camera in respect to the calibration markers.

a = ((z - tz) / dz
    
X = a*dx + tx
Y = a*dy + ty
Z = np.zeros_like(X) + z

Public Functions
================
    calibrate_dlt - Calculate a homography matrix in either 2D or 3D domains
    calibrate_intrinsics - Calculate the instrinsic parameters using Zang's method
    generate_camera_params - Create a pinhole camera data structure
    get_rotation_matrix - Calculate a 3x3 rotation matrix from Euler angles
    project_points - Project lab coordinates to image coordinates
    project_to_z - Project image coordinates to lab coordinates at specified Z-plane
    minimize_camera_params - Optimize pinhole camera parameters
    line_dist - Calculate where two rays intersect
    point_line_dist - Calculate the minimum dinstance between a point and a ray
    save_parameters - Save pinhole camera parameters to a text file
    load_parameters - Load pinhole camera parameters from a text file
    
Note
====
It is important to only import the submodule and not the functions that are in the
submodules. Explicitly importing a function from this submodule could cause
conflicts between other camera models due to similar naming conventions that are
normaly protected behind namespaces.

"""
from ._check_params import *
from ._distortion import *
from ._dlt import *
from ._epipolar_geom import *
from ._minimization import *
from ._projection import *
from ._utils import *
from ._zang import *


__all__ = [s for s in dir() if not s.startswith("_")]