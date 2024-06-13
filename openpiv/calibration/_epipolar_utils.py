import numpy as np
from typing import Tuple
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt

from . import _cal_doc_utils

__all__ = [
    "plot_epipolar_line"
]


@_cal_doc_utils.docfiller
def plot_epipolar_line(
    cam_struct: dict,
    project_to_z: "function", 
    image_points: np.ndarray,
    zlims: Tuple[int, int],
    ax:"matplotlib.axes.Axes"=None,
    color=None
):
    """Plot 3D epipolar lines.
    
    Using the passed camera structure and projection function, plot the 3D
    epipolar line(s). By passing ax, multiple epipolar lines can be plotted
    as a visualization aid for camera alighnment and multi-camera system
    performance.
    
    Parameters
    ----------
    %(cam_struct)s
    %(project_to_z_func)s
    %(image_points)s
    zlims : tuple[int, int]
        The start and end values of the epipolar line.
    ax : matplotlib.axes.Axes, optional
        The axis of which to plot the epipolar line.
    color : str, optional
        The color of the epipolar line.
        
    Returns
    -------
    fig, ax : matplotlib figure, optional
        If an axis is not passed, a new figure and axis will be returned.
        
    Notes
    -----
    This function is based on a similar utilitiy in MyPTV, which is referenced below.
    https://github.com/ronshnapp/MyPTV
    
    """
    Z1, Z2 = zlims
    
    X1, Y1, Z1 = project_to_z(
        cam_struct,
        image_points,
        z = Z1
    )
    
    X2, Y2, Z2 = project_to_z(
        cam_struct,
        image_points,
        z = Z2
    )
    
    if ax is None:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
                
        if color is None:
            ax.plot3D([X1,X2], [Z1,Z2], [Y1,Y2])
        else:
            ax.plot3D([X1,X2], [Z1,Z2], [Y1,Y2], c=color)

        return fig, ax

    else:
        if color is None:
            ax.plot3D([X1,X2], [Z1,Z2], [Y1,Y2])
        else:
            ax.plot3D([X1,X2], [Z1,Z2], [Y1,Y2], c=color)
        
        return None

            
def _line_intersect(t1, r1, t2, r2):
    """Calculate where two rays intersect.
    
    Using two cameras, calculate the world coordinates where two rays
    intersect. This is done through an analytical solution based on 
    direction vectors (r) and camera origins/translations (O). 
    
    Parameters
    ----------
    t1, t2 : np.ndarray
        Three element numpy arrays for camera origin/translation.
    r1, r2 : np.ndarray
        Three element numpy arrays for direction vectors.
        
    Returns
    -------
    dist : float
        The minimum dinstance between the two rays.
    coord : np.ndarray
        The world coordinate that is nearest to the two rays intersecting.
    
    Notes
    -----
    Function taken from MyPTV; all rights reserved. The direct link to this
    repository is provided below.
    https://github.com/ronshnapp/MyPTV
    
    """
    r1r2 = r1[0]*r2[0] + r1[1]*r2[1] + r1[2]*r2[2]
    r12 = r1[0]**2 + r1[1]**2 + r1[2]**2
    r22 = r2[0]**2 + r2[1]**2 + r2[2]**2
    
    dO = t2-t1
    B = [r1[0]*dO[0] + r1[1]*dO[1] + r1[2]*dO[2],
         r2[0]*dO[0] + r2[1]*dO[1] + r2[2]*dO[2]]
    
    # invert matrix to get coefficients a and b
    try:
        a = (-r22*B[0] + r1r2*B[1])/(r1r2**2 - r12 * r22)
        b = (-r1r2*B[0] + r12*B[1])/(r1r2**2 - r12 * r22)
    except:
        a, b = 0.0, 0.0
    
    # now use a and b to calculate the minimum distance
    l1, l2 = t1 + a*r1, t2 + b*r2 # lines
    dist = sum((l1 - l2)**2)**0.5 # minimum distance
    coord = (l1 + l2)*0.5 # coordinate location
    
    return dist, coord