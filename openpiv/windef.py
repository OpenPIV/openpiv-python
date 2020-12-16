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
from skimage.measure import points_in_poly
from skimage.util import invert


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

        if hasattr(settings, 'invert') and settings.invert is True:
            frame_a = invert(frame_a)
            frame_b = invert(frame_b)

        if settings.dynamic_masking_method in ("edge", "intensity"):
            frame_a, mask_a = preprocess.dynamic_masking(
                frame_a,
                method=settings.dynamic_masking_method,
                filter_size=settings.dynamic_masking_filter_size,
                threshold=settings.dynamic_masking_threshold,
            )
            frame_b, mask_b = preprocess.dynamic_masking(
                frame_b,
                method=settings.dynamic_masking_method,
                filter_size=settings.dynamic_masking_filter_size,
                threshold=settings.dynamic_masking_threshold,
            )

        # "first pass"
        x, y, u, v, s2n = first_pass(
            frame_a,
            frame_b,
            settings.windowsizes[0],
            settings.overlap[0],
            settings.num_iterations,
            correlation_method=settings.correlation_method,
            subpixel_method=settings.subpixel_method,
            do_sig2noise=settings.extract_sig2noise,
            sig2noise_method=settings.sig2noise_method,
            sig2noise_mask=settings.sig2noise_mask,
            normalized_correlation=settings.normalized_correlation
        )

        # " Image masking "
        if hasattr(settings, 'image_mask') and settings.image_mask:
            image_mask = np.logical_and(mask_a, mask_b)
            mask_coords = preprocess.mask_coordinates(image_mask)
            # mark those points on the grid of PIV inside the mask
            xymask = points_in_poly(np.c_[y.flatten(), x.flatten()], mask_coords)
            tmp = np.zeros_like(x,dtype=bool)
            tmp.flat[xymask] = True
            u = np.ma.masked_array(u, mask=tmp)
            v = np.ma.masked_array(v, mask=tmp)

        """
        validation using gloabl limits and std and local median

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
        """

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
            mask = mask + mask_g + mask_m + mask_s

            if (
                settings.extract_sig2noise is True
                and settings.num_iterations == 1
                and settings.do_sig2noise_validation is True
            ):
                u, v, mask_s2n = validation.sig2noise_val(
                    u, v, s2n,
                    threshold=settings.sig2noise_threshold
                )
                mask += mask_s2n
                
        
        # "filter to replace the values that where marked by the validation"
        if settings.num_iterations == 1 and settings.replace_vectors is True:
            # for multi-pass we cannot have holes in the data
            # after the first pass
            u, v = filters.replace_outliers(
                u,
                v,
                method=settings.filter_method,
                max_iter=settings.max_filter_iteration,
                kernel_size=settings.filter_kernel_size,
            )
        elif settings.num_iterations > 1: # don't even check if it's true or false
            u, v = filters.replace_outliers(
                u,
                v,
                method=settings.filter_method,
                max_iter=settings.max_filter_iteration,
                kernel_size=settings.filter_kernel_size,
            )

            # "adding masks to add the effect of all the validations"
        if settings.smoothn is True:
            u, dummy_u1, dummy_u2, dummy_u3 = smoothn.smoothn(
                u, s=settings.smoothn_p
            )
            v, dummy_v1, dummy_v2, dummy_v3 = smoothn.smoothn(
                v, s=settings.smoothn_p
            )

        # plt.figure()
        # plt.quiver(x,y,u,v)
        # plt.gca().invert_yaxis()
        # plt.title('after first pass')
        # plt.show()
    
        """ Multi pass """
        
        for i in range(1, settings.num_iterations):
            x, y, u, v, s2n = multipass_img_deform(
                frame_a,
                frame_b,
                settings.windowsizes[i],
                settings.overlap[i],
                settings.num_iterations,
                i,
                x,
                y,
                u,
                v,
                correlation_method=settings.correlation_method,
                subpixel_method=settings.subpixel_method,
                deformation_method=settings.deformation_method,
                sig2noise_method=settings.sig2noise_method,
                sig2noise_mask=settings.sig2noise_mask,
                sig2noise_threshold=settings.sig2noise_threshold,
                interpolation_order=settings.interpolation_order,
                normalized_correlation=settings.normalized_correlation
            )

            mask = np.full_like(x, False, dtype=bool)
            # validation every step
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
            mask = mask + mask_s + mask_m + mask_g

            if settings.sig2noise_method is not None:
                u, v, mask_s2n = validation.sig2noise_val(
                    u, v, s2n, threshold=settings.sig2noise_threshold
                )
                mask = mask + mask_s2n
            
            # we have to replace outliers
            u, v = filters.replace_outliers(
                u,
                v,
                method=settings.filter_method,
                max_iter=settings.max_filter_iteration,
                kernel_size=settings.filter_kernel_size,
            )

            # If the smoothing is active, we do it at each pass
            # but not the last one
            if settings.smoothn is True and i < settings.num_iterations-1: 
                u, dummy_u1, dummy_u2, dummy_u3 = smoothn.smoothn(
                    u, s=settings.smoothn_p
                )
                v, dummy_v1, dummy_v2, dummy_v3 = smoothn.smoothn(
                    v, s=settings.smoothn_p
                )

            # if (
            #     settings.extract_sig2noise is True
            #     and i == settings.iterations
            #     and settings.iterations != 1
            #     and settings.do_sig2noise_validation is True
            # ):
            if settings.sig2noise_method is not None:
                u, v, mask_s2n = validation.sig2noise_val(
                    u, v, s2n,
                    threshold=settings.sig2noise_threshold
                )
                mask = mask + mask_s2n

            # replace also every loop
            u, v = filters.replace_outliers(
                u,
                v,
                method=settings.filter_method,
                max_iter=settings.max_filter_iteration,
                kernel_size=settings.filter_kernel_size,
            )

        # "pixel/frame->pixel/sec"
        u = u / settings.dt
        v = v / settings.dt
        
        # "scales the results pixel-> meter"
        x, y, u, v = scaling.uniform(x, y, u, v,
                                     scaling_factor=settings.scaling_factor)
        
        # "save to a file"
        save(
            x,
            y,
            u,
            v,
            s2n,
            mask,
            os.path.join(save_path, "field_A%03d.txt" % counter),
            delimiter="\t",
        )
        
        # "some other stuff that one might want to use"
        if settings.show_plot is True or settings.save_plot is True:
            plt.close("all")
            plt.ioff()
            Name = os.path.join(save_path, "Image_A%03d.png" % counter)
            fig, _ = display_vector_field(
                os.path.join(save_path, "field_A%03d.txt" % counter),
                scale=settings.scale_plot,
            )
            if settings.save_plot is True:
                fig.savefig(Name)
            if settings.show_plot is True:
                plt.show()

        print("Image Pair " + str(counter + 1))

    # "Below is code to read files and create a folder to store the results"
    save_path = os.path.join(
        settings.save_path,
        "Open_PIV_results_"
        + str(settings.windowsizes[settings.num_iterations-1])
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
    x, y, ut, vt = \
        create_deformation_field(frame,
                                 x, y, u, v,
                                 interpolation_order=interpolation_order,
                                 kx=kx, ky=ky)
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
    normalized_correlation=False,
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

#     if do_sig2noise is False or iterations != 1:
#         sig2noise_method = None  # this indicates to get out nans

    u, v, s2n = extended_search_area_piv(
        frame_a,
        frame_b,
        window_size=window_size,
        overlap=overlap,
        search_area_size=window_size,
        width=sig2noise_mask,
        subpixel_method=subpixel_method,
        sig2noise_method=sig2noise_method,
        correlation_method=correlation_method,
        normalized_correlation=normalized_correlation
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
    normalized_correlation=False,
    subpixel_method="gaussian",
    deformation_method="symmetric",
    sig2noise_method="peak2peak",
    sig2noise_threshold=1.0,
    sig2noise_mask=2,
    interpolation_order=1,
    mask_coords=[],
):
    """
    Multi pass of the PIV evaluation.

    This function does the PIV evaluation of the second and other passes.
    It returns the coordinates of the interrogation window centres,
    the displacement u, v for each interrogation window as well as
    the signal to noise ratio array (which is full of NaNs if opted out)


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
        in case of the image mask - u_old and v_old are MaskedArrays

    v_old : 2d np.ndarray
        the v displacement of the vector field of the previous pass

    subpixel_method: string
        the method used for the subpixel interpolation.
        one of the following methods to estimate subpixel location of the peak:
        'centroid' [replaces default if correlation map is negative],
        'gaussian' [default if correlation map is positive],
        'parabolic'

    interpolation_order : int
        the order of the spline interpolation used for the image deformation

    mask_coords : list of x,y coordinates (pixels) of the image mask,
        default is an empty list

    Returns
    -------
    x : 2d np.array
        array containg the x coordinates of the interrogation window centres

    y : 2d np.array
        array containg the y coordinates of the interrogation window centres

    u : 2d np.array
        array containing the horizontal displacement for every interrogation
        window [pixels]

    u : 2d np.array
        array containing the vertical displacement for every interrogation
        window it returns values in [pixels]

    s2n : 2D np.array of signal to noise ratio values

    """

    y_old = y_old[:, 0]
    x_old = x_old[0, :]

    # calculate the y and y coordinates of the interrogation window centres.
    # The interpolation function dont like meshgrids as input. Hence, the
    # edges must be extracted to provide the sufficient input. x_old and y_old
    # are the coordinates of the old grid. x_int and y_int are the coordinates
    # of the new grid

    x, y = get_coordinates(np.shape(frame_a),
                           window_size,
                           overlap)

    # reapply the image mask to the new coordinates
    if len(mask_coords) > 1:  # not an empty list means there is a mask
        xymask = points_in_poly(np.c_[y.flatten(), x.flatten()], mask_coords)
        mask = np.zeros_like(x, dtype=bool)
        mask.flat[xymask] = 1

    y_int = y[:, 0]
    x_int = x[0, :]


    # interpolating the displacements from the old grid onto the new grid
    # y befor x because of numpy works row major
    ip = RectBivariateSpline(y_old, x_old, u_old, kx=2, ky=2)
    u_pre = ip(y_int, x_int)

    ip2 = RectBivariateSpline(y_old, x_old, v_old, kx=2, ky=2)
    v_pre = ip2(y_int, x_int)

    if len(mask_coords) > 1:
        u_pre = np.ma.masked_array(u_pre, mask=mask)
        v_pre = np.ma.masked_array(v_pre, mask=mask)

    # if settings.show_plot:
    #     plt.figure()
    #     plt.quiver(x_old, y_old, u_old, v_old,color='b')
    #     plt.quiver(x_int, y_int, u_pre, v_pre,color='r',lw=2)
    #     plt.gca().invert_yaxis()
    #     plt.title('inside deform')
    #     plt.gca().set_aspect(1.)

    # @TKauefer added another method to the windowdeformation, 'symmetric'
    # splits the onto both frames, takes more effort due to additional
    # interpolation however should deliver better results

    # old_frame_a = frame_a.copy()
    # old_frame_b = frame_b.copy()

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

    # if settings.show_plot:
    #     plt.figure()
    #     plt.imshow(frame_a-old_frame_a)
    #     plt.figure()
    #     plt.imshow(frame_b-old_frame_b)

    # if do_sig2noise is True 
    #     sig2noise_method = sig2noise_method
    # else:
    #     sig2noise_method = None

    # so we use here default circular not normalized correlation:
    u, v, s2n = extended_search_area_piv(
        frame_a,
        frame_b,
        window_size=window_size,
        overlap=overlap,
        width=sig2noise_mask,
        subpixel_method=subpixel_method,
        sig2noise_method=sig2noise_method,  # if it's None, it's not used
        correlation_method=correlation_method,
        normalized_correlation=normalized_correlation,
    )

    shapes = np.array(get_field_shape(frame_a.shape,
                                      window_size,
                                      overlap))
    u = u.reshape(shapes)
    v = v.reshape(shapes)
    s2n = s2n.reshape(shapes)

    #     plt.figure()
    #     plt.quiver(x_int, y_int, u, v,color='r')
    #     plt.quiver(x_int, y_int, u_pre, v_pre,color='b')
    #     plt.gca().invert_yaxis()

    # adding the recent displacment on to the displacment of the previous pass
    u += u_pre
    v -= v_pre

    # reapply image mask just to be sure
    if len(mask_coords) > 1:
        u = np.ma.masked_array(u, mask=mask)
        v = np.ma.masked_array(v, mask=mask)

    # if settings.show_plot:
    #     plt.figure()
    #     plt.quiver(x_int, y_int, u, v,color='r')
    #     plt.quiver(x_int, y_int, u_pre, v_pre,color='b')
    #     plt.gca().invert_yaxis()
    #     plt.gca().set_aspect(1)

    return x, y, u, v, s2n


class Settings(object):
    def __init__(self):
        # "Data related settings"
        # Folder with the images to process
        self.filepath_images = "."
        # Folder for the outputs
        self.save_path = "./res"
        # Root name of the output Folder for Result Files
        self.save_folder_suffix = "Test_4"
        # Format and Image Sequence
        self.frame_pattern_a = 'exp1_001_a.bmp'
        self.frame_pattern_b = 'exp1_001_b.bmp'

        # "Region of interest"
        # (50,300,50,300) #Region of interest: (xmin,xmax,ymin,ymax) or 'full'
        # for full image
        self.ROI = "full"

        # "Image preprocessing"
        # 'None' for no masking, 'edges' for edges masking, 'intensity' for
        # intensity masking
        # WARNING: This part is under development so better not to use MASKS
        self.dynamic_masking_method = "None"
        self.dynamic_masking_threshold = 0.005
        self.dynamic_masking_filter_size = 7

        # "Processing Parameters"
        self.correlation_method = "circular"  # 'circular' or 'linear'
        self.normalized_correlation = False

        # add the interroagtion window size for each pass.
        # For the moment, it should be a power of 2
        self.windowsizes = (
            64,
            32,
            16,
        )  # if longer than n iteration the rest is ignored
        # The overlap of the interroagtion window for each pass.
        self.overlap = (32, 16, 8)  # This is 50% overlap
        # Has to be a value with base two. In general window size/2 is a good
        # choice.

        self.iterations = len(self.windowsizes)  # select the number of PIV
        # passes

        # methode used for subpixel interpolation:
        # 'gaussian','centroid','parabolic'
        self.subpixel_method = "gaussian"
        # 'symmetric' or 'second image', 'symmetric' splits the deformation
        # both images, while 'second image' does only deform the second image.
        self.deformation_method = 'symmetric'  # 'symmetric' or 'second image'
        # order of the image interpolation for the window deformation
        self.interpolation_order = 3
        self.scaling_factor = 1  # scaling factor pixel/meter
        self.dt = 1  # time between to frames (in seconds)

        # "Signal to noise ratio options (only for the last pass)"
        # It is possible to decide if the S/N should be computed (for the last
        # pass) or not
        # 'True' or 'False' (only for the last pass)
        self.extract_sig2noise = False
        # method used to calculate the signal to noise ratio 'peak2peak' or
        # 'peak2mean'
        self.sig2noise_method = "peak2peak" # or "peak2mean" or "None"
        # select the width of the masked to masked out pixels next to the main
        # peak
        self.sig2noise_mask = 2
        # If extract_sig2noise==False the values in the signal to noise ratio
        # output column are set to NaN

        # "vector validation options"
        # choose if you want to do validation of the first pass: True or False
        self.validation_first_pass = True
        # only effecting the first pass of the interrogation the following
        # passes
        # in the multipass will be validated

        # "Validation Parameters"
        # The validation is done at each iteration based on three filters.
        # The first filter is based on the min/max ranges. Observe that these
        # values are defined in
        # terms of minimum and maximum displacement in pixel/frames.
        self.MinMax_U_disp = (-30, 30)
        self.MinMax_V_disp = (-30, 30)
        # The second filter is based on the global STD threshold
        self.std_threshold = 10  # threshold of the std validation
        # The third filter is the median test (not normalized at the moment)
        self.median_threshold = 3  # threshold of the median validation
        # On the last iteration, an additional validation can be done based on
        # the S/N.
        self.median_size = 1  # defines the size of the local median

        # "Validation based on the signal to noise ratio"
        # Note: only available when extract_sig2noise==True and only for the
        # last pass of the interrogation
        # Enable the signal to noise ratio validation. Options: True or False
        # self.do_sig2noise_validation = False  # This is time consuming
        # minmum signal to noise ratio that is need for a valid vector
        self.sig2noise_threshold = 1.05

        # "Outlier replacement or Smoothing options"
        # Replacment options for vectors which are masked as invalid by the
        # validation
        # Choose: True or False
        self.replace_vectors = True  # Enable the replacement.
        self.smoothn = True  # Enables smoothing of the displacement field
        self.smoothn_p = 0.5  # This is a smoothing parameter
        # select a method to replace the outliers:
        # 'localmean', 'disk', 'distance'
        self.filter_method = "localmean"
        # maximum iterations performed to replace the outliers
        self.max_filter_iteration = 4
        self.filter_kernel_size = 2  # kernel size for the localmean method

        # "Output options"
        # Select if you want to save the plotted vectorfield: True or False
        self.save_plot = True
        # Choose wether you want to see the vectorfield or not :True or False
        self.show_plot = True
        self.scale_plot = 100  # select a value to scale the quiver plot of
        # the vectorfield run the script with the given settings


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
    settings.normalized_correlation = False
    settings.num_iterations = 3  # select the number of PIV passes
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
