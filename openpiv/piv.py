import numpy as np
import matplotlib.pyplot as plt

from openpiv import pyprocess, tools
import pkg_resources as pkg

# import numpy as np

import matplotlib.animation as animation

"""This module contains image processing routines that improve
images prior to PIV processing."""

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


def simple_piv(im1, im2, plot=True):
    """
    Simplest PIV run on the pair of images using default settings

    piv(im1,im2) will create a tmp.vec file with the vector filed in pix/dt
    (dt=1) from two images, im1,im2 provided as full path filenames
    (TIF is preferable, whatever imageio can read)

    """
    if isinstance(im1, str):
        im1 = tools.imread(im1)
        im2 = tools.imread(im2)

    u, v, s2n = pyprocess.extended_search_area_piv(
        im1.astype(np.int32), im2.astype(np.int32), window_size=32,
        overlap=16, search_area_size=32
    )
    x, y = pyprocess.get_coordinates(image_size=im1.shape,
                                     search_area_size=32, overlap=16)

    valid = s2n > np.percentile(s2n, 5)

    if plot:
        _, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(im1, cmap=plt.get_cmap("gray"), alpha=0.5, origin="upper")
        ax.quiver(x[valid], y[valid], u[valid], -v[valid], scale=70,
                  color='r', width=.005)
        plt.show()

    return x, y, u, v


def piv_example():
    """
    PIV example uses examples/test5 vortex PIV data to show the main principles

    piv(im1,im2) will create a tmp.vec file with the vector filed in pix/dt
    (dt=1) from two images, im1,im2 provided as full path filenames
    (TIF is preferable)

    """
    # if im1 is None and im2 is None:
    im1 = pkg.resource_filename("openpiv", "data/test1/exp1_001_a.bmp")
    im2 = pkg.resource_filename("openpiv", "data/test1/exp1_001_b.bmp")

    frame_a = tools.imread(im1)
    frame_b = tools.imread(im2)

    # frame_a[0:32, 512 - 32:] = 255

    images = []
    images.extend([frame_a, frame_b])

    fig, ax = plt.subplots()

    # ims is a list of lists, each row is a list of artists to draw in the
    # current frame; here we are just animating one artist, the image, in
    # each frame
    ims = []
    for i in range(2):
        im = ax.imshow(images[i % 2], animated=True, cmap="gray")
        ims.append([im])

    _ = animation.ArtistAnimation(fig, ims, interval=500, blit=False,
                                  repeat_delay=0)
    plt.show()

    # import os

    vel = pyprocess.extended_search_area_piv(
        frame_a.astype(np.int32), frame_b.astype(np.int32), window_size=32,
        search_area_size=64,
        overlap=8
    )
    x, y = pyprocess.get_coordinates(image_size=frame_a.shape,
                                     search_area_size=64, overlap=8)

    fig, ax = plt.subplots(1, 2, figsize=(11, 8))
    ax[0].imshow(frame_a, cmap=plt.get_cmap("gray"), alpha=0.8)
    ax[0].quiver(x, y, vel[0], -vel[1], scale=50, color="r")
    ax[1].quiver(x, y[::-1, :], vel[0], -1*vel[1], scale=50, color="b")
    ax[1].set_aspect(1)
    # ax[1].invert_yaxis()
    plt.show()

    return x, y, vel[0], vel[1]
