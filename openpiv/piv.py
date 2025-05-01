import numpy as np
import matplotlib.pyplot as plt

from openpiv import pyprocess, tools, validation, filters
from importlib_resources import files

import matplotlib.animation as animation

"""This module contains high-level PIV processing functions that combine
various steps of the PIV analysis into convenient workflows."""

__licence_ = """
Copyright (C) 2011  www.openpiv.net
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""


def simple_piv(im1, im2, window_size=32, overlap=16, search_area_size=32, 
               dt=1.0, plot=True, validation_method=None, s2n_thresh=1.3):
    """
    Simplified PIV analysis on a pair of images with optional validation.

    Parameters
    ----------
    im1 : str or numpy.ndarray
        First image - can be a file path or a numpy array
    im2 : str or numpy.ndarray
        Second image - can be a file path or a numpy array
    window_size : int, optional
        Size of the interrogation window, default is 32 pixels
    overlap : int, optional
        Overlap of interrogation windows, default is 16 pixels
    search_area_size : int, optional
        Size of the search area, default is 32 pixels
    dt : float, optional
        Time interval between images, default is 1.0
    plot : bool, optional
        Whether to display a quiver plot of the results, default is True
    validation_method : str, optional
        Method for validation: 'sig2noise' or 'global_std' or None
    s2n_thresh : float, optional
        Signal-to-noise threshold for validation, default is 1.3

    Returns
    -------
    x : 2d np.ndarray
        x-coordinates of the velocity vectors
    y : 2d np.ndarray
        y-coordinates of the velocity vectors
    u : 2d np.ndarray
        u velocity component
    v : 2d np.ndarray
        v velocity component
    s2n : 2d np.ndarray
        signal-to-noise ratio for each vector
    """
    # Load images if they are file paths
    if isinstance(im1, str):
        im1 = tools.imread(im1)
        im2 = tools.imread(im2)

    # Perform PIV analysis
    u, v, s2n = pyprocess.extended_search_area_piv(
        im1.astype(np.int32), im2.astype(np.int32), 
        window_size=window_size,
        overlap=overlap, 
        search_area_size=search_area_size
    )
    
    # Get coordinates
    x, y = pyprocess.get_coordinates(
        image_size=im1.shape,
        search_area_size=search_area_size, 
        overlap=overlap
    )

    # Validate vectors if requested
    if validation_method == 'sig2noise':
        valid = s2n > s2n_thresh
    elif validation_method == 'global_std':
        valid = validation.global_std(u, v)
    else:
        # Default validation using bottom 5% of s2n values
        valid = s2n > np.percentile(s2n, 5)
    
    # Replace outliers
    if np.any(~valid):
        u, v = filters.replace_outliers(u, v, ~valid)

    # Plot if requested
    if plot:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(im1, cmap=plt.get_cmap("gray"), alpha=0.5, origin="upper")
        ax.quiver(x[valid], y[valid], u[valid], -v[valid], scale=70,
                  color='r', width=.005)
        plt.title('Velocity field (dt={:.3f})'.format(dt))
        plt.show()
        
    # Transform coordinates to match windef and tools.display_vector_field
    x, y, u, v = tools.transform_coordinates(x, y, u, v)

    return x, y, u, v, s2n


def piv_example(plot_animation=True, plot_quiver=True):
    """
    Demonstrate PIV analysis using example vortex data.
    
    This function loads example images from the package data, performs
    PIV analysis, and displays the results.
    
    Parameters
    ----------
    plot_animation : bool, optional
        Whether to display an animation of the image pair, default is True
    plot_quiver : bool, optional
        Whether to display quiver plots of the results, default is True
        
    Returns
    -------
    x : 2d np.ndarray
        x-coordinates of the velocity vectors
    y : 2d np.ndarray
        y-coordinates of the velocity vectors
    u : 2d np.ndarray
        u velocity component
    v : 2d np.ndarray
        v velocity component
    """
    # Load example images
    im1 = files('openpiv.data').joinpath('test1/exp1_001_a.bmp')
    im2 = files('openpiv.data').joinpath('test1/exp1_001_b.bmp')    

    frame_a = tools.imread(im1)
    frame_b = tools.imread(im2)

    # Create animation of the image pair
    if plot_animation:
        images = [frame_a, frame_b]
        fig, ax = plt.subplots()
        
        ims = []
        for i in range(2):
            im = ax.imshow(images[i % 2], animated=True, cmap="gray")
            ims.append([im])

        ani = animation.ArtistAnimation(fig, ims, interval=500, blit=False,
                                      repeat_delay=0)
        plt.title('Image pair animation')
        plt.show()

    # Perform PIV analysis
    window_size = 32
    search_area_size = 64
    overlap = 8
    
    u, v, s2n = pyprocess.extended_search_area_piv(
        frame_a.astype(np.int32), frame_b.astype(np.int32), 
        window_size=window_size,
        search_area_size=search_area_size,
        overlap=overlap
    )
    
    x, y = pyprocess.get_coordinates(
        image_size=frame_a.shape,
        search_area_size=search_area_size, 
        overlap=overlap
    )

    # Plot results
    if plot_quiver:
        fig, ax = plt.subplots(1, 2, figsize=(11, 8))
        ax[0].imshow(frame_a, cmap=plt.get_cmap("gray"), alpha=0.8)
        ax[0].quiver(x, y, u, -v, scale=50, color="r")
        ax[0].set_title('Original orientation')
        
        ax[1].quiver(x, y[::-1, :], u, -1*v, scale=50, color="b")
        ax[1].set_title('Flipped y-axis')
        ax[1].set_aspect(1)
        plt.tight_layout()
        plt.show()

    # Transform coordinates for consistency with other functions
    x, y, u, v = tools.transform_coordinates(x, y, u, v)
    
    return x, y, u, v


def process_pair(frame_a, frame_b, window_size=32, overlap=16, 
                search_area_size=32, dt=1.0, validation_method='sig2noise',
                s2n_threshold=1.3, filter_method='localmean', 
                filter_kernel_size=1, plot=False):
    """
    Complete PIV processing workflow for a single image pair.
    
    This function performs PIV analysis with validation and filtering.
    
    Parameters
    ----------
    frame_a : numpy.ndarray
        First image
    frame_b : numpy.ndarray
        Second image
    window_size : int, optional
        Size of the interrogation window, default is 32 pixels
    overlap : int, optional
        Overlap of interrogation windows, default is 16 pixels
    search_area_size : int, optional
        Size of the search area, default is 32 pixels
    dt : float, optional
        Time interval between images, default is 1.0
    validation_method : str, optional
        Method for validation: 'sig2noise', 'global_std', or None
    s2n_threshold : float, optional
        Signal-to-noise threshold for validation, default is 1.3
    filter_method : str, optional
        Method for outlier replacement: 'localmean', 'disk', or 'distance'
    filter_kernel_size : int, optional
        Size of the kernel for outlier replacement, default is 1
    plot : bool, optional
        Whether to display a quiver plot of the results, default is False
        
    Returns
    -------
    x : 2d np.ndarray
        x-coordinates of the velocity vectors
    y : 2d np.ndarray
        y-coordinates of the velocity vectors
    u : 2d np.ndarray
        u velocity component
    v : 2d np.ndarray
        v velocity component
    mask : 2d np.ndarray
        Mask of invalid vectors
    """
    # Perform PIV analysis
    u, v, s2n = pyprocess.extended_search_area_piv(
        frame_a.astype(np.int32), frame_b.astype(np.int32), 
        window_size=window_size,
        overlap=overlap, 
        search_area_size=search_area_size
    )
    
    # Get coordinates
    x, y = pyprocess.get_coordinates(
        image_size=frame_a.shape,
        search_area_size=search_area_size, 
        overlap=overlap
    )

    # Validate vectors
    if validation_method == 'sig2noise':
        mask = s2n < s2n_threshold
    elif validation_method == 'global_std':
        mask = ~validation.global_std(u, v)
    else:
        mask = np.zeros_like(u, dtype=bool)
    
    # Replace outliers
    if np.any(mask):
        u, v = filters.replace_outliers(
            u, v, mask, 
            method=filter_method,
            kernel_size=filter_kernel_size
        )
    
    # Plot if requested
    if plot:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(frame_a, cmap=plt.get_cmap("gray"), alpha=0.7, origin="upper")
        ax.quiver(x, y, u, -v, scale=50, color='r', width=.005)
        ax.set_title(f'Velocity field (dt={dt:.3f})')
        plt.tight_layout()
        plt.show()
    
    # Transform coordinates
    x, y, u, v = tools.transform_coordinates(x, y, u, v)
    
    return x, y, u, v, mask
