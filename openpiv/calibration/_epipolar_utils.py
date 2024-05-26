import numpy as np
from typing import Tuple
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt

from .._doc_utils import (docstring_decorator,
                          doc_cam_struct)

__all__ = [
    "plot_epipolar_line"
]


@docstring_decorator(doc_cam_struct)
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
    cam_struct : dict
        {0}
    project_to_z_func : function
        Projection function with the following signiture:
        res = func(cam_struct, image_points, Z).
    image_points: 2D np.ndarray
        A numpy array containing [x, y] image points.
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
