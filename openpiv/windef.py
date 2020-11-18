# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 14:04:04 2019

@author: Theo
@modified: Alex, Erich
"""


import os
import numpy as np
import scipy.ndimage as scn
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt

from openpiv.tools import imread, Multiprocesser, save, display_vector_field
from openpiv import validation, filters
from openpiv import preprocess, scaling
from openpiv.pyprocess import extended_search_area_piv, get_coordinates, \
    get_field_shape
from openpiv import smoothn


def piv(settings):
    """ the func fuction is the "frame" in which the PIV evaluation is done """

    def func(args):
        """A function to process each image pair."""

        # this line is REQUIRED for multiprocessing to work
        # always use it in your custom function

        file_a, file_b, counter = args

        # counter2=str(counter2)
        #####################
        # Here goes you code
        #####################

        " read images into numpy arrays"
        frame_a = imread(os.path.join(settings.filepath_images, file_a))
        frame_b = imread(os.path.join(settings.filepath_images, file_b))

        # Miguel: I just had a quick look, and I do not understand the reason
        # for this step.
        #  I propose to remove it.
        # frame_a = (frame_a*1024).astype(np.int32)
        # frame_b = (frame_b*1024).astype(np.int32)

        " crop to ROI"
        if settings.ROI == "full":
            frame_a = frame_a
            frame_b = frame_b
        else:
            frame_a = frame_a[
                settings.ROI[0]:settings.ROI[1],
                settings.ROI[2]:settings.ROI[3]
            ]
            frame_b = frame_b[
                settings.ROI[0]:settings.ROI[1],
                settings.ROI[2]:settings.ROI[3]
            ]
        if settings.dynamic_masking_method == "edge" or "intensity":
            frame_a = preprocess.dynamic_masking(
                frame_a,
                method=settings.dynamic_masking_method,
                filter_size=settings.dynamic_masking_filter_size,
                threshold=settings.dynamic_masking_threshold,
            )
            frame_b = preprocess.dynamic_masking(
                frame_b,
                method=settings.dynamic_masking_method,
                filter_size=settings.dynamic_masking_filter_size,
                threshold=settings.dynamic_masking_threshold,
            )

        # "first pass"
        x, y, u, v, sig2noise_ratio = first_pass(
            frame_a,
            frame_b,
            settings.windowsizes[0],
            settings.overlap[0],
            settings.iterations,
            correlation_method=settings.correlation_method,
            subpixel_method=settings.subpixel_method,
            do_sig2noise=settings.extract_sig2noise,
            sig2noise_method=settings.sig2noise_method,
            sig2noise_mask=settings.sig2noise_mask,
        )

        "validation using gloabl limits and std and local median"
        """MinMaxU : two elements tuple
            sets the limits of the u displacment component
            Used for validation.

        MinMaxV : two elements tuple
            sets the limits of the v displacment component
            Used for validation.

        std_threshold : float
            sets the  threshold for the std validation

        median_threshold : float
            sets the threshold for the median validation

        filter_method : string
            the method used to replace the non-valid vectors
            Methods:
                'localmean',
                'disk',
                'distance',

        max_filter_iteration : int
            maximum of filter iterations to replace nans

        filter_kernel_size : int
            size of the kernel used for the filtering"""

        mask = np.full_like(x, False)
        if settings.validation_first_pass is True:
            u, v, mask_g = validation.global_val(
                u, v, settings.MinMax_U_disp, settings.MinMax_V_disp
            )
            u, v, mask_s = validation.global_std(
                u, v, std_threshold=settings.std_threshold
            )
            u, v, mask_m = validation.local_median_val(
                u,
                v,
                u_threshold=settings.median_threshold,
                v_threshold=settings.median_threshold,
                size=settings.median_size,
            )
            if (
                settings.extract_sig2noise is True
                and settings.iterations == 1
                and settings.do_sig2noise_validation is True
            ):
                u, v, mask_s2n = validation.sig2noise_val(
                    u, v, sig2noise_ratio,
                    threshold=settings.sig2noise_threshold
                )
                mask = mask + mask_g + mask_m + mask_s + mask_s2n
            else:
                mask = mask + mask_g + mask_m + mask_s
        "filter to replace the values that where marked by the validation"
        if settings.iterations > 1:
            u, v = filters.replace_outliers(
                u,
                v,
                method=settings.filter_method,
                max_iter=settings.max_filter_iteration,
                kernel_size=settings.filter_kernel_size,
            )
            "adding masks to add the effect of all the validations"
            if settings.smoothn is True:
                u, dummy_u1, dummy_u2, dummy_u3 = smoothn.smoothn(
                    u, s=settings.smoothn_p
                )
                v, dummy_v1, dummy_v2, dummy_v3 = smoothn.smoothn(
                    v, s=settings.smoothn_p
                )
        elif settings.iterations == 1 and settings.replace_vectors is True:
            u, v = filters.replace_outliers(
                u,
                v,
                method=settings.filter_method,
                max_iter=settings.max_filter_iteration,
                kernel_size=settings.filter_kernel_size,
            )
            "adding masks to add the effect of all the validations"
            if settings.smoothn is True:
                u, v = filters.replace_outliers(
                    u,
                    v,
                    method=settings.filter_method,
                    max_iter=settings.max_filter_iteration,
                    kernel_size=settings.filter_kernel_size,
                )
                u, dummy_u1, dummy_u2, dummy_u3 = smoothn.smoothn(
                    u, s=settings.smoothn_p
                )
                v, dummy_v1, dummy_v2, dummy_v3 = smoothn.smoothn(
                    v, s=settings.smoothn_p
                )

        i = 1
        "all the following passes"
        for i in range(2, settings.iterations + 1):
            x, y, u, v, sig2noise_ratio, mask = multipass_img_deform(
                frame_a,
                frame_b,
                settings.windowsizes[i - 1],
                settings.overlap[i - 1],
                settings.iterations,
                i,
                x,
                y,
                u,
                v,
                correlation_method=settings.correlation_method,
                subpixel_method=settings.subpixel_method,
                deformation_method=settings.deformation_method,
                do_sig2noise=settings.extract_sig2noise,
                sig2noise_method=settings.sig2noise_method,
                sig2noise_mask=settings.sig2noise_mask,
                MinMaxU=settings.MinMax_U_disp,
                MinMaxV=settings.MinMax_V_disp,
                std_threshold=settings.std_threshold,
                median_threshold=settings.median_threshold,
                median_size=settings.median_size,
                filter_method=settings.filter_method,
                max_filter_iteration=settings.max_filter_iteration,
                filter_kernel_size=settings.filter_kernel_size,
                interpolation_order=settings.interpolation_order,
            )
            # If the smoothing is active, we do it at each pass
            if settings.smoothn is True:
                u, dummy_u1, dummy_u2, dummy_u3 = smoothn.smoothn(
                    u, s=settings.smoothn_p
                )
                v, dummy_v1, dummy_v2, dummy_v3 = smoothn.smoothn(
                    v, s=settings.smoothn_p
                )

        if (
            settings.extract_sig2noise is True
            and i == settings.iterations
            and settings.iterations != 1
            and settings.do_sig2noise_validation is True
        ):
            u, v, mask_s2n = validation.sig2noise_val(
                u, v, sig2noise_ratio, threshold=settings.sig2noise_threshold
            )
            mask = mask + mask_s2n
        if settings.replace_vectors is True:
            u, v = filters.replace_outliers(
                u,
                v,
                method=settings.filter_method,
                max_iter=settings.max_filter_iteration,
                kernel_size=settings.filter_kernel_size,
            )
        "pixel/frame->pixel/sec"
        u = u / settings.dt
        v = v / settings.dt
        "scales the results pixel-> meter"
        x, y, u, v = scaling.uniform(x, y, u, v,
                                     scaling_factor=settings.scaling_factor)
        "save to a file"
        save(
            x,
            y,
            u,
            v,
            sig2noise_ratio,
            mask,
            os.path.join(save_path, "field_A%03d.txt" % counter),
            delimiter="\t",
        )
        "some messages to check if it is still alive"

        "some other stuff that one might want to use"
        if settings.show_plot is True or settings.save_plot is True:
            plt.close("all")
            plt.ioff()
            Name = os.path.join(save_path, "Image_A%03d.png" % counter)
            display_vector_field(
                os.path.join(save_path, "field_A%03d.txt" % counter),
                scale=settings.scale_plot,
            )
            if settings.save_plot is True:
                plt.savefig(Name)
            if settings.show_plot is True:
                plt.show()

        print("Image Pair " + str(counter + 1))

    "Below is code to read files and create a folder to store the results"
    save_path = os.path.join(
        settings.save_path,
        "Open_PIV_results_"
        + str(settings.windowsizes[settings.iterations - 1])
        + "_"
        + settings.save_folder_suffix,
    )
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    task = Multiprocesser(
        data_dir=settings.filepath_images,
        pattern_a=settings.frame_pattern_a,
        pattern_b=settings.frame_pattern_b,
    )
    task.run(func=func, n_cpus=1)


def create_deformation_field(
        frame, x, y, u, v, interpolation_order=1, kx=3, ky=3):
    """
    Deform an image by window deformation where a new grid is defined based
    on the grid and displacements of the previous pass and pixel values are
    interpolated onto the new grid.

    Parameters
    ----------
    frame : 2d np.ndarray, dtype=np.int32
        an two dimensions array of integers containing grey levels of
        the first frame.

    x : 2d np.ndarray
        a two dimensional array containing the x coordinates of the
        interrogation window centers, in pixels.

    y : 2d np.ndarray
        a two dimensional array containing the y coordinates of the
        interrogation window centers, in pixels.

    u : 2d np.ndarray
        a two dimensional array containing the u velocity component,
        in pixels/seconds.

    v : 2d np.ndarray
        a two dimensional array containing the v velocity component,
        in pixels/seconds.

    interpolation_order: scalar
        the degree of the frame interpolation (deformation) of the mesh

    kx : scalar
         the degree of the interpolation of the B-splines over the x-axis
         of a rectangular mesh

    ky : scalar
         the degree of the interpolation of the B-splines over the
         y-axis of a rectangular mesh

    Returns
    -------
    frame_def:
        a deformed image based on the meshgrid and displacements of the
        previous pass
    """
    frame = frame.astype(np.float32)
    y1 = y[:, 0]  # extract first coloumn from meshgrid
    #  y1 = y1[::-1]  # flip
    x1 = x[0, :]  # extract first row from meshgrid
    side_x = np.arange(0, np.size(frame[0, :]), 1)  # extract the image grid
    side_y = np.arange(0, np.size(frame[:, 0]), 1)

    # interpolating displacements onto a new meshgrid
    ip = RectBivariateSpline(y1, x1, u, kx=kx, ky=ky)
    ut = ip(side_y, side_x)
    # the way how to use the interpolation functions differs from matlab

    ip2 = RectBivariateSpline(y1, x1, v, kx=kx, ky=ky)
    vt = ip2(side_y, side_x)
    x, y = np.meshgrid(side_x, side_y)
    return x, y, ut, vt


def deform_windows(frame, x, y, u, v, interpolation_order=1, kx=3, ky=3):
    """
    Deform an image by window deformation where a new grid is defined based
    on the grid and displacements of the previous pass and pixel values are
    interpolated onto the new grid.

    Parameters
    ----------
    frame : 2d np.ndarray, dtype=np.int32
        an two dimensions array of integers containing grey levels of
        the first frame.

    x : 2d np.ndarray
        a two dimensional array containing the x coordinates of the
        interrogation window centers, in pixels.

    y : 2d np.ndarray
        a two dimensional array containing the y coordinates of the
        interrogation window centers, in pixels.

    u : 2d np.ndarray
        a two dimensional array containing the u velocity component,
        in pixels/seconds.

    v : 2d np.ndarray
        a two dimensional array containing the v velocity component,
        in pixels/seconds.

    interpolation_order: scalar
        the degree of the frame interpolation (deformation) of the mesh

    kx : scalar
         the degree of the interpolation of the B-splines over the x-axis
         of a rectangular mesh

    ky : scalar
         the degree of the interpolation of the B-splines over the
         y-axis of a rectangular mesh

    Returns
    -------
    frame_def:
        a deformed image based on the meshgrid and displacements of the
        previous pass
    """
    x, y, ut, vt = create_deformation_field(frame, x, y, u, v, kx=kx, ky=ky)
    frame_def = scn.map_coordinates(
        frame, ((y + vt, x + ut,)), order=interpolation_order, mode='nearest')

    return frame_def


def first_pass(
    frame_a,
    frame_b,
    window_size,
    overlap,
    iterations,
    correlation_method="circular",
    subpixel_method="gaussian",
    do_sig2noise=False,
    sig2noise_method="peak2peak",
    sig2noise_mask=2,
):
    """
    First pass of the PIV evaluation.

    This function does the PIV evaluation of the first pass. It returns
    the coordinates of the interrogation window centres, the displacment
    u and v for each interrogation window as well as the mask which indicates
    wether the displacement vector was interpolated or not.


    Parameters
    ----------
    frame_a : 2d np.ndarray
        the first image

    frame_b : 2d np.ndarray
        the second image

    window_size : int
         the size of the interrogation window

    overlap : int
        the overlap of the interrogation window, typically it is window_size/2

    subpixel_method: string
        the method used for the subpixel interpolation.
        one of the following methods to estimate subpixel location of the peak:
        'centroid' [replaces default if correlation map is negative],
        'gaussian' [default if correlation map is positive],
        'parabolic'

    Returns
    -------
    x : 2d np.array
        array containg the x coordinates of the interrogation window centres

    y : 2d np.array
        array containg the y coordinates of the interrogation window centres

    u : 2d np.array
        array containing the u displacement for every interrogation window

    u : 2d np.array
        array containing the u displacement for every interrogation window

    """

    # Alex: in windef_refactoring we created a pyprocess.py that holds
    # all the vectorized solutions from windef, using the same principles
    # now the first_pass is just wraping the extended_search_piv

    # We replace this part with the extended_search

    # cor_win_1 = moving_window_array(frame_a, window_size, overlap)
    # cor_win_2 = moving_window_array(frame_b, window_size, overlap)
    # """Filling the interrogation window. They windows are arranged
    # in a 3d array with number of interrogation windows
    # this way is much faster then using a loop"""

    # correlation = correlate_windows(cor_win_1, cor_win_2,
    #                                 correlation_method=correlation_method)
    # "do the correlation"
    # # create a dummy for the loop to fill
    # disp = np.zeros((np.size(correlation, 0), 2))
    # for i in range(0, np.size(correlation, 0)):
    #     """ determine the displacment on subpixel level """
    #     disp[i, :] = find_subpixel_peak_position(
    #         correlation[i, :, :], subpixel_method=subpixel_method
    #     )
    # "this loop is doing the displacment evaluation for each window "

    if do_sig2noise is False or iterations != 1:
        sig2noise_method = None  # this indicates to get out nans

    u, v, s2n = extended_search_area_piv(
        frame_a,
        frame_b,
        window_size=window_size,
        overlap=overlap,
        search_area_size=window_size,
        width=sig2noise_mask,
        subpixel_method=subpixel_method,
        sig2noise_method=sig2noise_method,
    )

    shapes = np.array(get_field_shape(frame_a.shape, window_size, overlap))
    u = u.reshape(shapes)
    v = v.reshape(shapes)
    s2n = s2n.reshape(shapes)
    x, y = get_coordinates(frame_a.shape, window_size, overlap)

    return x, y, u, v, s2n


def multipass_img_deform(
    frame_a,
    frame_b,
    window_size,
    overlap,
    iterations,
    current_iteration,
    x_old,
    y_old,
    u_old,
    v_old,
    correlation_method="circular",
    subpixel_method="gaussian",
    deformation_method="symmetric",
    do_sig2noise=False,
    sig2noise_method="peak2peak",
    sig2noise_mask=2,
    MinMaxU=(-100, 50),
    MinMaxV=(-50, 50),
    std_threshold=5,
    median_threshold=2,
    median_size=1,
    filter_method="localmean",
    max_filter_iteration=10,
    filter_kernel_size=2,
    interpolation_order=3,
):
    """
    Multi pass of the PIV evaluation.

    This function does the PIV evaluation of the second and other passes.
    It returns the coordinates of the interrogation window centres,
    the displacement u, v for each interrogation window as well as
    the mask which indicates
    wether the displacement vector was interpolated or not.


    Parameters
    ----------
    frame_a : 2d np.ndarray
        the first image

    frame_b : 2d np.ndarray
        the second image

    window_size : tuple of ints
         the size of the interrogation window

    overlap : tuple of ints
        the overlap of the interrogation window, e.g. window_size/2

    x_old : 2d np.ndarray
        the x coordinates of the vector field of the previous pass

    y_old : 2d np.ndarray
        the y coordinates of the vector field of the previous pass

    u_old : 2d np.ndarray
        the u displacement of the vector field of the previous pass

    v_old : 2d np.ndarray
        the v displacement of the vector field of the previous pass

    subpixel_method: string
        the method used for the subpixel interpolation.
        one of the following methods to estimate subpixel location of the peak:
        'centroid' [replaces default if correlation map is negative],
        'gaussian' [default if correlation map is positive],
        'parabolic'

    MinMaxU : two elements tuple
        sets the limits of the u displacment component
        Used for validation.

    MinMaxV : two elements tuple
        sets the limits of the v displacment component
        Used for validation.

    std_threshold : float
        sets the  threshold for the std validation

    median_threshold : float
        sets the threshold for the median validation

    filter_method : string
        the method used to replace the non-valid vectors
        Methods:
            'localmean',
            'disk',
            'distance',

    max_filter_iteration : int
        maximum of filter iterations to replace nans

    filter_kernel_size : int
        size of the kernel used for the filtering

    interpolation_order : int
        the order of the spline interpolation used for the image deformation

    Returns
    -------
    x : 2d np.array
        array containg the x coordinates of the interrogation window centres

    y : 2d np.array
        array containg the y coordinates of the interrogation window centres

    u : 2d np.array
        array containing the u displacement for every interrogation window

    u : 2d np.array
        array containing the u displacement for every interrogation window

    mask : 2d np.array
        array containg the mask values (bool) which contains information if
        the vector was filtered

    """

    x, y = get_coordinates(np.shape(frame_a), window_size, overlap)

    "calculate the y and y coordinates of the interrogation window centres"
    """The interpolation function dont like meshgrids as input. Hence, the
    edges must be extracted to provide the sufficient input. x_old and y_old
    are the coordinates of the old grid. x_int and y_int are the coordinates
    of the new grid"""

    y_old = y_old[:, 0]
    # y_old = y_old[::-1]
    x_old = x_old[0, :]
    y_int = y[:, 0]
    # y_int = y_int[::-1]
    x_int = x[0, :]

    # interpolating the displacements from the old grid onto the new grid
    # y befor x because of numpy works row major
    ip = RectBivariateSpline(y_old, x_old, u_old, kx=2, ky=2)
    u_pre = ip(y_int, x_int)

    ip2 = RectBivariateSpline(y_old, x_old, v_old, kx=2, ky=2)
    v_pre = ip2(y_int, x_int)
    # I added another method to the windowdeformation, 'symmetric' splits the
    # onto both frames, takes more effort due to additional interpolation,
    # however should deliver better results
    if deformation_method == "symmetric":
        # this one is doing the image deformation (see above)
        x_new, y_new, ut, vt = create_deformation_field(
            frame_a, x, y, u_pre, v_pre)
        frame_a = scn.map_coordinates(
            frame_a, ((y_new - vt / 2, x_new - ut / 2)),
            order=interpolation_order, mode='nearest')
        frame_b = scn.map_coordinates(
            frame_b, ((y_new + vt / 2, x_new + ut / 2)),
            order=interpolation_order, mode='nearest')
    elif deformation_method == "second image":
        frame_b = deform_windows(
            frame_b, x, y, u_pre, v_pre,
            interpolation_order=interpolation_order)
    else:
        raise Exception("Deformation method is not valid.")

    if (
        do_sig2noise is True and
        current_iteration == iterations and
        iterations != 1
    ):
        sig2noise_method = sig2noise_method
    else:
        sig2noise_method = None

    u, v, s2n = extended_search_area_piv(
        frame_a,
        frame_b,
        window_size=window_size,
        overlap=overlap,
        search_area_size=window_size,
        width=sig2noise_mask,
        subpixel_method=subpixel_method,
        sig2noise_method=sig2noise_method,
    )

    shapes = np.array(get_field_shape(frame_a.shape, window_size, overlap))
    u = u.reshape(shapes)
    v = v.reshape(shapes)
    s2n = s2n.reshape(shapes)

    # adding the recent displacment on to the displacment of the previous pass
    u += u_pre
    v -= v_pre

    # validation using gloabl limits and local median
    u, v, mask_g = validation.global_val(u, v, MinMaxU, MinMaxV)
    u, v, mask_s = validation.global_std(u, v, std_threshold=std_threshold)
    u, v, mask_m = validation.local_median_val(
        u,
        v,
        u_threshold=median_threshold,
        v_threshold=median_threshold,
        size=median_size,
    )

    # adding masks to add the effect of alle the validations
    mask = mask_g + mask_m + mask_s

    # mask=np.zeros_like(u)
    # filter to replace the values that where marked by the validation
    if current_iteration != iterations:
        # filter to replace the values that where marked by the validation
        u, v = filters.replace_outliers(
            u,
            v,
            method=filter_method,
            max_iter=max_filter_iteration,
            kernel_size=filter_kernel_size,
        )

    return x, y, u, v, s2n, mask


class Settings(object):
    pass


if __name__ == "__main__":
    """ Run windef.py as a script:

    python windef.py

    """

    settings = Settings()

    "Data related settings"
    # Folder with the images to process
    settings.filepath_images = "./examples/test1/"
    # Folder for the outputs
    settings.save_path = "./examples/test1/"
    # Root name of the output Folder for Result Files
    settings.save_folder_suffix = "Test_4"
    # Format and Image Sequence
    settings.frame_pattern_a = 'exp1_001_a.bmp'
    settings.frame_pattern_b = 'exp1_001_b.bmp'

    "Region of interest"
    # (50,300,50,300) #Region of interest: (xmin,xmax,ymin,ymax) or 'full' for
    # full image
    settings.ROI = "full"

    "Image preprocessing"
    # 'None' for no masking, 'edges' for edges masking, 'intensity' for
    # intensity masking
    # WARNING: This part is under development so better not to use MASKS
    settings.dynamic_masking_method = "None"
    settings.dynamic_masking_threshold = 0.005
    settings.dynamic_masking_filter_size = 7

    "Processing Parameters"
    settings.correlation_method = "circular"  # 'circular' or 'linear'
    settings.iterations = 3  # select the number of PIV passes
    # add the interroagtion window size for each pass.
    # For the moment, it should be a power of 2
    settings.windowsizes = (
        128,
        64,
        32,
    )  # if longer than n iteration the rest is ignored
    # The overlap of the interroagtion window for each pass.
    settings.overlap = (64, 32, 16)  # This is 50% overlap
    # Has to be a value with base two. In general window size/2 is a good
    # choice.
    # methode used for subpixel interpolation:
    # 'gaussian','centroid','parabolic'
    settings.subpixel_method = "gaussian"
    # 'symmetric' or 'second image', 'symmetric' splits the deformation
    # both images, while 'second image' does only deform the second image.
    settings.deformation_method = 'symmetric'  # 'symmetric' or 'second image'
    # order of the image interpolation for the window deformation
    settings.interpolation_order = 3
    settings.scaling_factor = 1  # scaling factor pixel/meter
    settings.dt = 1  # time between to frames (in seconds)
    "Signal to noise ratio options (only for the last pass)"
    # It is possible to decide if the S/N should be computed (for the last
    # pass) or not
    # 'True' or 'False' (only for the last pass)
    settings.extract_sig2noise = True
    # method used to calculate the signal to noise ratio 'peak2peak' or
    # 'peak2mean'
    settings.sig2noise_method = "peak2peak"
    # select the width of the masked to masked out pixels next to the main peak
    settings.sig2noise_mask = 2
    # If extract_sig2noise==False the values in the signal to noise ratio
    # output column are set to NaN
    "vector validation options"
    # choose if you want to do validation of the first pass: True or False
    settings.validation_first_pass = True
    # only effecting the first pass of the interrogation the following passes
    # in the multipass will be validated
    "Validation Parameters"
    # The validation is done at each iteration based on three filters.
    # The first filter is based on the min/max ranges. Observe that these
    # values are defined in
    # terms of minimum and maximum displacement in pixel/frames.
    settings.MinMax_U_disp = (-30, 30)
    settings.MinMax_V_disp = (-30, 30)
    # The second filter is based on the global STD threshold
    settings.std_threshold = 10  # threshold of the std validation
    # The third filter is the median test (not normalized at the moment)
    settings.median_threshold = 3  # threshold of the median validation
    # On the last iteration, an additional validation can be done based on the
    # S/N.
    settings.median_size = 1  # defines the size of the local median
    "Validation based on the signal to noise ratio"
    # Note: only available when extract_sig2noise==True and only for the last
    # pass of the interrogation
    # Enable the signal to noise ratio validation. Options: True or False
    settings.do_sig2noise_validation = False  # This is time consuming
    # minmum signal to noise ratio that is need for a valid vector
    settings.sig2noise_threshold = 1.2
    "Outlier replacement or Smoothing options"
    # Replacment options for vectors which are masked as invalid by the
    # validation
    # Choose: True or False
    settings.replace_vectors = True  # Enable the replacement.
    settings.smoothn = True  # Enables smoothing of the displacement field
    settings.smoothn_p = 0.5  # This is a smoothing parameter
    # select a method to replace the outliers: 'localmean', 'disk', 'distance'
    settings.filter_method = "localmean"
    # maximum iterations performed to replace the outliers
    settings.max_filter_iteration = 4
    settings.filter_kernel_size = 2  # kernel size for the localmean method
    "Output options"
    # Select if you want to save the plotted vectorfield: True or False
    settings.save_plot = True
    # Choose wether you want to see the vectorfield or not :True or False
    settings.show_plot = False
    settings.scale_plot = (
        100  # select a value to scale the quiver plot of the vectorfield
    )
    # run the script with the given settings

    piv(settings)
