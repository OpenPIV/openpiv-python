"""The openpiv.tools module is a collection of utilities and tools.
"""

__licence__ = """
Copyright (C) 2011  www.openpiv.net

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import sys
import pathlib
import multiprocessing
from typing import Any, Union, List, Optional
# import re

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pt
from natsort import natsorted
from scipy.ndimage import maximum_filter

# from builtins import range
from imageio.v3 import imread as _imread, imwrite as _imsave


def _read_image_stack(list_img: list) -> np.ndarray:
    """Load a list of images into a single integer stack."""
    if not list_img:
        raise ValueError("list_img must contain at least one image")

    frames = [np.asarray(imread(image_name)) for image_name in list_img]
    return np.stack(frames).astype(np.int32, copy=False)


def _set_figure_window_title(fig: Any, title: str) -> None:
    """Set the window title when the active backend supports it."""
    manager = getattr(fig.canvas, "manager", None)
    if manager is not None and hasattr(manager, "set_window_title"):
        manager.set_window_title(title)


def _prepare_image_for_save(arr: np.ndarray, preserve_tiff_depth: bool = False) -> np.ndarray:
    """Normalize image arrays into a format imageio/Pillow can write reliably."""
    image = np.asarray(arr)

    if np.ndim(image) > 2:
        image = rgb2gray(image)

    image = image.astype(np.float32, copy=False)

    if np.amin(image) < 0:
        image = image - np.amin(image)

    max_value = float(np.amax(image)) if image.size else 0.0

    if preserve_tiff_depth and max_value > 255:
        if max_value > np.iinfo(np.uint16).max:
            image = image / max_value * np.iinfo(np.uint16).max
        return np.clip(image, 0, np.iinfo(np.uint16).max).astype(np.uint16)

    if np.issubdtype(image.dtype, np.floating) and 0 < max_value <= 1.0:
        image = image * 255.0
    elif max_value > 255:
        image = image / max_value * 255.0

    return np.clip(image, 0, 255).astype(np.uint8)


def natural_sort(file_list: List[pathlib.Path]) -> List[pathlib.Path]:
    """ Creates naturally sorted list """
    # convert = lambda text: int(text) if text.isdigit() else text.lower()
    # alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    # return sorted(file_list, key=alphanum_key)
    return natsorted(file_list, key=str)


def sorted_unique(array: np.ndarray) -> np.ndarray:
    """Creates sorted unique array

    Parameters
    ----------
    array : np.ndarray
        Input array

    Returns
    -------
    np.ndarray
        Array with unique values sorted in ascending order
    """
    # Just use np.unique which returns sorted unique values by default
    return np.unique(array)


def display_vector_field_from_arrays(
    x: np.ndarray,
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    flags: np.ndarray,
    mask: np.ndarray,
    on_img: Optional[bool] = False,
    image_name: Optional[Union[pathlib.Path, str]] = None,
    window_size: Optional[int] = 32,
    scaling_factor: Optional[float] = 1.,
    ax: Optional[Any] = None,
    width: Optional[float] = 0.0025,
    show_invalid: Optional[bool] = True,
    **kw
):
    """ Displays quiver plot of the data in five arrays: x,y,u,v and flags


    Parameters
    ----------
    filename :  string
        the absolute path of the text file

    on_img : Bool, optional
        if True, display the vector field on top of the image provided by
        image_name

    image_name : string, optional
        path to the image to plot the vector field onto when on_img is True

    window_size : int, optional
        when on_img is True, provide the interrogation window size to fit the
        background image to the vector field

    scaling_factor : float, optional
        when on_img is True, provide the scaling factor to scale the background
        image to the vector field

    show_invalid: bool, show or not the invalid vectors, default is True


    Key arguments   : (additional parameters, optional)
        *scale*: [None | float]
        *width*: [None | float]


    See also:
    ---------
    matplotlib.pyplot.quiver


    Examples
    --------
    --- only vector field
    >>> openpiv.tools.display_vector_field('./exp1_0000.txt',scale=100,
                                           width=0.0025)

    --- vector field on top of image
    >>> openpiv.tools.display_vector_field(Path('./exp1_0000.txt'), on_img=True,
                                          image_name=Path('exp1_001_a.bmp'),
                                          window_size=32, scaling_factor=70,
                                          scale=100, width=0.0025)

    """

    if isinstance(u, np.ma.MaskedArray):
        u = u.filled(0.)
        v = v.filled(0.)

    if mask is None:
        mask = np.zeros_like(u, dtype=int)

    if flags is None:
        flags = np.zeros_like(u, dtype=int)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    if on_img is True:  # plot a background image
        im = imread(image_name)
        im = negative(im)  # plot negative of the image for more clarity
        xmax = np.amax(x) + window_size / (2 * scaling_factor)
        ymax = np.amax(y) + window_size / (2 * scaling_factor)
        ax.imshow(im, cmap="Greys_r", extent=[0.0, xmax, 0.0, ymax])

    # first mask whatever has to be masked
    u[mask.astype(bool)] = 0.
    v[mask.astype(bool)] = 0.

    # now mark the valid/invalid vectors
    invalid = flags > 0  # mask.astype("bool")
    valid = ~invalid

    # visual conversion for the data on image
    # to be consistent with the image coordinate system

    # if on_img:
    #     y = y.max() - y
    #     v *= -1

    ax.quiver(
        x[valid],
        y[valid],
        u[valid],
        v[valid],
        color="b",
        width=width,
        **kw
    )

    if show_invalid and len(invalid) > 0:
        ax.quiver(
            x[invalid],
            y[invalid],
            u[invalid],
            v[invalid],
            color="r",
            width=width,
            **kw,
        )

    # if on_img is False:
    #     ax.invert_yaxis()

    ax.set_aspect(1.)
    # fig.canvas.set_window_title('Vector field, '+str(np.count_nonzero(invalid))+' wrong vectors')

    plt.show()

    return fig, ax


def display_vector_field(
    filename: Union[pathlib.Path, str],
    on_img: Optional[bool] = False,
    image_name: Optional[Union[pathlib.Path, str]] = None,
    window_size: Optional[int] = 32,
    scaling_factor: Optional[float] = 1.,
    ax: Optional[Any] = None,
    width: Optional[float] = 0.0025,
    show_invalid: Optional[bool] = True,
    **kw
):
    """ Displays quiver plot of the data stored in the file


    Parameters
    ----------
    filename :  string
        the absolute path of the text file

    on_img : Bool, optional
        if True, display the vector field on top of the image provided by
        image_name

    image_name : string, optional
        path to the image to plot the vector field onto when on_img is True

    window_size : int, optional
        when on_img is True, provide the interrogation window size to fit the
        background image to the vector field

    scaling_factor : float, optional
        when on_img is True, provide the scaling factor to scale the background
        image to the vector field

    show_invalid: bool, show or not the invalid vectors, default is True


    Key arguments   : (additional parameters, optional)
        *scale*: [None | float]
        *width*: [None | float]


    See also:
    ---------
    matplotlib.pyplot.quiver


    Examples
    --------
    --- only vector field
    >>> openpiv.tools.display_vector_field('./exp1_0000.txt',scale=100,
                                           width=0.0025)

    --- vector field on top of image
    >>> openpiv.tools.display_vector_field(Path('./exp1_0000.txt'), on_img=True,
                                          image_name=Path('exp1_001_a.bmp'),
                                          window_size=32, scaling_factor=70,
                                          scale=100, width=0.0025)

    """

    # print(f' Loading {filename} which exists {filename.exists()}')
    a = np.loadtxt(filename)
    # parse
    x, y, u, v, flags, mask = a[:, 0], a[:,
                                         1], a[:, 2], a[:, 3], a[:, 4], a[:, 5]

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    if on_img is True:  # plot a background image
        im = imread(image_name)
        im = negative(im)  # plot negative of the image for more clarity
        xmax = np.amax(x) + window_size / (2 * scaling_factor)
        ymax = np.amax(y) + window_size / (2 * scaling_factor)
        ax.imshow(im, cmap="Greys_r", extent=[0.0, xmax, 0.0, ymax])

    # first mask whatever has to be masked
    u[mask.astype(bool)] = 0.
    v[mask.astype(bool)] = 0.

    # now mark the valid/invalid vectors
    invalid = flags > 0  # mask.astype("bool")
    valid = ~invalid

    # visual conversion for the data on image
    # to be consistent with the image coordinate system

    # if on_img:
    #     y = y.max() - y
    #     v *= -1

    ax.quiver(
        x[valid],
        y[valid],
        u[valid],
        v[valid],
        color="b",
        width=width,
        **kw
    )

    if show_invalid and len(invalid) > 0:
        ax.quiver(
            x[invalid],
            y[invalid],
            u[invalid],
            v[invalid],
            color="r",
            width=width,
            **kw,
        )

    # if on_img is False:
    #     ax.invert_yaxis()

    ax.set_aspect(1.)
    # fig.canvas.set_window_title('Vector field, '+str(np.count_nonzero(invalid))+' wrong vectors')

    plt.show()

    return fig, ax


def imread(filename, flatten=0):
    """Read an image file into a numpy array
    using imageio imread

    Parameters
    ----------
    filename :  string
        the absolute path of the image file
    flatten :   bool
        True if the image is RGB color or False (default) if greyscale

    Returns
    -------
    frame : np.ndarray
        a numpy array with grey levels


    Examples
    --------

    >>> image = openpiv.tools.imread( 'image.bmp' )
    >>> print image.shape
        (1280, 1024)


    """
    im = _imread(filename)
    if np.ndim(im) > 2:
        im = rgb2gray(im)

    return im


def rgb2gray(rgb: np.ndarray) -> np.ndarray:
    """converts rgb image to gray

    Args:
        rgb (_type_): numpy.ndarray, image size, three channels

    Returns:
        gray: numpy.ndarray of the same shape, one channel
    """
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.144])


def imsave(filename, arr):
    """Write an image file from a numpy array
    using imageio imread

    Parameters
    ----------
    filename :  string
        the absolute path of the image file that will be created
    arr : 2d np.ndarray
        a 2d numpy array with grey levels

    Example
    --------

    >>> image = openpiv.tools.imread( 'image.bmp' )
    >>> image2 = openpiv.tools.negative(image)
    >>> imsave( 'negative-image.tif', image2)

    """

    preserve_tiff_depth = str(filename).lower().endswith((".tif", ".tiff"))
    prepared = _prepare_image_for_save(arr, preserve_tiff_depth=preserve_tiff_depth)

    if preserve_tiff_depth:
        _imsave(filename, prepared, format="TIFF")
    else:
        _imsave(filename, prepared)


def convert_16bits_tif(filename, save_name):
    """convert 16 bits TIFF to an openpiv readable image

    Args:
        filename (_type_): filename of a 16 bit TIFF
        save_name (_type_): new image filename
    """
    img = np.asarray(_imread(filename))

    if img.ndim == 3:
        img = img[..., 0]

    if img.dtype != np.uint8:
        max_value = float(np.max(img)) if img.size else 0.0
        if max_value > 0:
            img = img.astype(np.float32) / max_value * 255.0
        img = img.astype(np.uint8)

    imsave(save_name, img)


def mark_background(
    threshold: float,
    list_img: list,
    filename: str
) -> np.ndarray:
    """marks background

    Args:
        threshold (float): threshold
        list_img (list of images): _description_
        filename (str): image filename to save the mask

    Returns:
        _type_: _description_
    """
    frames = _read_image_stack(list_img)
    threshold_sum = float(threshold) * frames.shape[0]
    background = np.where(frames.sum(axis=0) >= threshold_sum, 255, 0).astype(np.uint8)
    imsave(filename, background)
    print("done with background")
    return background


def mark_background2(list_img, filename):
    frames = _read_image_stack(list_img)
    background = frames.min(axis=0).astype(np.uint8)
    imsave(filename, background)
    print("done with background")
    return background


def edges(list_img, filename):
    back = mark_background(30, list_img, filename)
    edges = canny(back, sigma=3)
    imsave(filename, edges)


def find_reflexions(list_img, filename):
    background = mark_background2(list_img, filename)
    reflexion = np.where(background > 253, 255, 0).astype(np.uint8)
    imsave(filename, reflexion)
    print("done with reflexions")
    return reflexion


def find_boundaries(threshold, list_img1, list_img2, filename, picname):
    print("mark1..")
    mark1 = np.where(_read_image_stack(list_img1).sum(axis=0) >= float(threshold) * len(list_img1), 255, 0).astype(np.uint8)
    print("[DONE]")
    print((mark1.shape))
    print("mark2..")
    mark2 = np.where(_read_image_stack(list_img2).sum(axis=0) >= float(threshold) * len(list_img2), 255, 0).astype(np.uint8)
    print("[DONE]")
    print("computing boundary")
    print((mark2.shape))

    difference = mark1 != mark2
    neighborhood_difference = maximum_filter(difference.astype(np.uint8), size=5) > 0
    list_bound = np.where(mark1 == 0, 125, 0).astype(np.uint8)
    list_bound[neighborhood_difference] = 255
    list_bound[[0, -1], :] = 255
    list_bound[:, [0, -1]] = 255

    with open(filename, "w", encoding="utf-8") as file_handle:
        for i in range(list_bound.shape[0]):
            for j in range(list_bound.shape[1]):
                file_handle.write(f"{i}\t{j}\t{int(list_bound[i, j])}\n")

    print("[DONE]")
    imsave(picname, list_bound)
    return list_bound


def save(
    filename: Union[pathlib.Path, str],
    x: np.ndarray,
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    flags: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    fmt: str = "%.4e",
    delimiter: str = "\t",
) -> None:
    """Save flow field to an ascii file.

    Parameters
    ----------
    filename : string
        the path of the file where to save the flow field

    x : 2d np.ndarray
        a two dimensional array containing the x coordinates of the
        interrogation window centers, in pixels.

    y : 2d np.ndarray
        a two dimensional array containing the y coordinates of the
        interrogation window centers, in pixels.

    u : 2d np.ndarray
        a two dimensional array containing the u velocity components,
        in pixels/seconds.

    v : 2d np.ndarray
        a two dimensional array containing the v velocity components,
        in pixels/seconds.

    flags : 2d np.ndarray
        a two dimensional integers array where elements corresponding to
        vectors: 0 - valid, 1 - invalid (, 2 - interpolated)
        default: None, will create all valid 0

    mask: 2d np.ndarray boolean, marks the image masked regions (dynamic and/or static)
        default: None - will be all False


    fmt : string
        a format string. See documentation of numpy.savetxt
        for more details.

    delimiter : string
        character separating columns

    Examples
    --------

    openpiv.tools.save('field_001.txt', x, y, u, v, flags, mask,  fmt='%6.3f',
                        delimiter='\t')

    """
    if isinstance(u, np.ma.MaskedArray):
        u = u.filled(0.)
        v = v.filled(0.)

    if mask is None:
        mask = np.zeros_like(u, dtype=int)

    if flags is None:
        flags = np.zeros_like(u, dtype=int)

    # build output array
    out = np.vstack([m.flatten() for m in [x, y, u, v, flags, mask]])

    # save data to file.
    np.savetxt(
        filename,
        out.T,
        fmt=fmt,
        delimiter=delimiter,
        header="x"
        + delimiter
        + "y"
        + delimiter
        + "u"
        + delimiter
        + "v"
        + delimiter
        + "flags"
        + delimiter
        + "mask",
    )


def display(message):
    """Display a message to standard output.

    Parameters
    ----------
    message : string
        a message to be printed

    """
    sys.stdout.write(message)
    sys.stdout.write("\n")
    sys.stdout.flush()


class Multiprocesser:
    def __init__(self,
                 data_dir: pathlib.Path,
                 pattern_a: str,
                 pattern_b: Optional[str] = None,
                 ) -> None:
        """A class to handle and process large sets of images.

        This class is responsible of loading image datasets
        and processing them. It has parallelization facilities
        to speed up the computation on multicore machines.

        It currently support only image pair obtained from
        conventional double pulse piv acquisition. Support
        for continuos time resolved piv acquistion is in the
        future.


        Parameters
        ----------
        data_dir : str
            the path where image files are located

        pattern_a : str
            a shell glob pattern to match the first (A) frames.

        pattern_b : str
            a shell glob pattern to match the second (B) frames.

        Options:
                pattern_a = 'image_*_a.bmp'
                pattern_b = 'image_*_b.bmp'

            or
                pattern_a = '000*.tif'
                pattern_b = '(1+2),(2+3)'
                will create PIV of these pairs: 0001.tif+0002.tif, 0002.tif+0003.tif ...
            or
                pattern_a = '000*.tif'
                pattern_b = '(1+3),(2+4)'
                will create PIV of these pairs: 0001.tif+0003.tif, 0002.tif+0004.tif ...
            or
                pattern_a = '000*.tif'
                pattern_b = '(1+2),(3+4)'
                will create PIV of these pairs: 0001.tif+0002.tif, 0003.tif+0004.tif ...


        Examples
        --------
        >>> multi = openpiv.tools.Multiprocesser( '/home/user/images', 'image_*_a.bmp', 'image_*_b.bmp')

        """
        # load lists of images

        # print('Inside Multiprocesser')
        # print(f'data_dir = {data_dir}')
        # print(f'pattern_a = {pattern_a}')
        # print(f' dir exists: {data_dir.exists()}')

        self.files_a = natural_sort(list(data_dir.glob(pattern_a)))

        # print(f'List of files:')
        # print(f'{self.files_a}')

        if pattern_b == '(1+2),(2+3)':
            self.files_b = self.files_a[1:]
            self.files_a = self.files_a[:-1]
        elif pattern_b == '(1+3),(2+4)':
            self.files_b = self.files_a[2:]
            self.files_a = self.files_a[:-2]
        elif pattern_b == '(1+2),(3+4)':
            self.files_b = self.files_a[1::2]
            self.files_a = self.files_a[0::2]
        else:
            self.files_b = natural_sort(list(data_dir.glob(pattern_b)))

        # number of images
        self.n_files = len(self.files_a)

        # check if everything was fine
        if not len(self.files_a) == len(self.files_b):
            print(self.files_a)
            print(self.files_b)

            raise ValueError(
                'Something failed loading the image file. There should be an equal number of "a" and "b" files.'
            )

        if len(self.files_a) == 0:
            raise ValueError(
                "Something failed loading the image file. No images were found. Please check directory and image template name."
            )

    def run(self, func, n_cpus=1):
        """Start to process images.

        Parameters
        ----------

        func : python function which will be executed for each
            image pair. See tutorial for more details.

        n_cpus : int
            the number of processes to launch in parallel.
            For debugging purposes use n_cpus=1

        """

        # create a list of tasks to be executed.
        image_pairs = [
            (file_a, file_b, i)
            for file_a, file_b, i in zip(
                self.files_a, self.files_b, range(self.n_files)
            )
        ]

        # for debugging purposes always use n_cpus = 1,
        # since it is difficult to debug multiprocessing stuff.
        if n_cpus > 1:
            pool = multiprocessing.Pool(processes=n_cpus)
            res = pool.map(func, image_pairs)
        else:
            for image_pair in image_pairs:
                func(image_pair)


def negative(image):
    """ Return the negative of an image

    Parameter
    ----------
    image : 2d np.ndarray of grey levels

    Returns
    -------
    (255-image) : 2d np.ndarray of grey levels

    """
    return 255 - image


def display_windows_sampling(x, y, window_size, skip=0, method="standard"):
    """ Displays a map of the interrogation points and windows


    Parameters
    ----------
    x : 2d np.ndarray
        a two dimensional array containing the x coordinates of the
        interrogation window centers, in pixels.

    y : 2d np.ndarray
        a two dimensional array containing the y coordinates of the
        interrogation window centers, in pixels.

    window_size : the interrogation window size, in pixels

    skip : the number of windows to skip on a row during display.
           Recommended value is 0 or 1 for standard method, can be more for random method
           -1 to not show any window

    method : can be only <standard> (uniform sampling and constant window size)
                         <random> (pick randomly some windows)

    Examples
    --------

    >>> openpiv.tools.display_windows_sampling(x, y, window_size=32, skip=0, method='standard')


    """

    fig = plt.figure()
    if skip < 0 or skip + 1 > len(x[0]) * len(y):
        _set_figure_window_title(fig, "interrogation points map")
        plt.scatter(x, y, color="g")  # plot interrogation locations
    else:
        nb_windows = len(x[0]) * len(y) // (skip + 1)
        # standard method --> display uniformly picked windows
        if method == "standard":
            # plot interrogation locations (green dots)
            plt.scatter(x, y, color="g")
            _set_figure_window_title(fig, "interrogation window map")
            # plot the windows as red squares
            for i in range(len(x[0])):
                for j in range(len(y)):
                    if j % 2 == 0:
                        if i % (skip + 1) == 0:
                            x1 = x[0][i] - window_size / 2
                            y1 = y[j][0] - window_size / 2
                            plt.gca().add_patch(
                                pt.Rectangle(
                                    (x1, y1),
                                    window_size,
                                    window_size,
                                    facecolor="r",
                                    alpha=0.5,
                                )
                            )
                    else:
                        if i % (skip + 1) == 1 or skip == 0:
                            x1 = x[0][i] - window_size / 2
                            y1 = y[j][0] - window_size / 2
                            plt.gca().add_patch(
                                pt.Rectangle(
                                    (x1, y1),
                                    window_size,
                                    window_size,
                                    facecolor="r",
                                    alpha=0.5,
                                )
                            )
        # random method --> display randomly picked windows
        elif method == "random":
            plt.scatter(x, y, color="g")  # plot interrogation locations
            _set_figure_window_title(
                fig,
                "interrogation window map, showing randomly "
                + str(nb_windows)
                + " windows"
            )
            for i in range(nb_windows):
                k = np.random.randint(len(x[0]))  # pick a row and column index
                l = np.random.randint(len(y))
                x1 = x[0][k] - window_size / 2
                y1 = y[l][0] - window_size / 2
                plt.gca().add_patch(
                    pt.Rectangle(
                        (x1, y1), window_size, window_size, facecolor="r", alpha=0.5
                    )
                )
        else:
            raise ValueError(
                "method not valid: choose between standard and random")
    plt.draw()
    plt.show()


def transform_coordinates(x, y, u, v):
    """ Converts coordinate systems from/to the image based / physical based

    Input/Output: x,y,u,v

        image based is 0,0 top left, x = columns to the right, y = rows downwards
        and so u,v

        physical or right hand one is that leads to the positive vorticity with
        the 0,0 origin at bottom left to be counterclockwise

    """
    # Handle both 1D and 2D arrays
    if y.ndim == 1:
        y = y[::-1]
    else:
        y = y[::-1, :]

    v *= -1
    return x, y, u, v
