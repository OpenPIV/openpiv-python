# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 14:04:04 2019

@author: Theo
"""



import numpy as np
from numpy.fft import rfft2, irfft2, fftshift
import scipy.ndimage as scn
from scipy.interpolate import RectBivariateSpline


from openpiv import process, validation, filters, pyprocess
import pyprocess_windef # local import


def correlation_func(cor_win_1, cor_win_2, window_size,correlation_method='circular'):
    '''This function is doing the cross-correlation. Right now circular cross-correlation
    That means no zero-padding is done
    the .real is to cut off possible imaginary parts that remains due to finite numerical accuarcy
     '''
    if correlation_method=='linear':
        cor_win_1=cor_win_1-cor_win_1.mean(axis=(1,2)).reshape(cor_win_1.shape[0],1,1)
        cor_win_2=cor_win_2-cor_win_2.mean(axis=(1,2)).reshape(cor_win_1.shape[0],1,1)
        cor_win_1[cor_win_1<0]=0
        cor_win_2[cor_win_2<0]=0

     
        corr = fftshift(irfft2(np.conj(rfft2(cor_win_1,s=(2*window_size,2*window_size))) *
                                  rfft2(cor_win_2,s=(2*window_size,2*window_size))).real, axes=(1, 2))
        corr=corr[:,window_size//2:3*window_size//2,window_size//2:3*window_size//2]
        
    else:
        corr = fftshift(irfft2(np.conj(rfft2(cor_win_1)) *
                                  rfft2(cor_win_2)).real, axes=(1, 2))
    return corr

def frame_interpolation(frame, x, y, u, v, interpolation_order=1):
    '''This one is doing the image deformation also known as window deformation
    Therefore, the pixel values of the old image are interpolated on a new grid that is defined
    by the grid of the previous pass and the displacment evaluated by the previous pass
    '''
    '''
    The interpolation function dont like meshgrids as input. Hence, the the edges
    must be extracted to provide the sufficient input, also the y coordinates need
    to be inverted since the image origin is in the upper left corner and the
    y-axis goes downwards. The x-axis goes to the right.
    '''
    frame=frame.astype(np.float32)
    y1 = y[:, 0] # extract first coloumn from meshgrid
    y1 = y1[::-1] #flip 
    x1 = x[0, :] #extract first row from meshgrid
    side_x = np.arange(0, np.size(frame[0, :]), 1) #extract the image grid
    side_y = np.arange(0, np.size(frame[:, 0]), 1)

    ip = RectBivariateSpline(y1, x1, u) #interpolate the diplacement on the image grid
    ut = ip(side_y, side_x)# the way how to use the interpolation functions differs
                            #from matlab 
    ip2 = RectBivariateSpline(y1, x1, v)
    vt = ip2(side_y, side_x)
    
    '''This lines are interpolating the displacement from the interrogation window
    grid onto the image grid. The result is displacment meshgrid with the size of the image.
    '''
    x, y = np.meshgrid(side_x, side_y)#create a meshgrid 
    frame_def = scn.map_coordinates(
        frame, ((y-vt, x+ut,)), order=interpolation_order,mode='nearest')
    #deform the image by using the map coordinates function
    '''This spline interpolation is doing the image deformation. This one likes meshgrids
    new grid is defined by the old grid + the displacement.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
    This function returns the deformed image.
    '''
    #print('stop')
    return frame_def

def first_pass(frame_a, frame_b, window_size, overlap,iterations,correlation_method='circular', subpixel_method='gaussian',do_sig2noise=False, sig2noise_method='peak2peak', sig2noise_mask=2):
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
        the overlap of the interrogation window normal for example window_size/2

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

    cor_win_1 = pyprocess.moving_window_array(frame_a, window_size, overlap)
    cor_win_2 = pyprocess.moving_window_array(frame_b, window_size, overlap)
    '''Filling the interrogation window. They windows are arranged
    in a 3d array with number of interrogation window *window_size*window_size
    this way is much faster then using a loop'''

    correlation = correlation_func(cor_win_1, cor_win_2, window_size,correlation_method=correlation_method)
    'do the correlation'
    disp = np.zeros((np.size(correlation, 0), 2))#create a dummy for the loop to fill
    for i in range(0, np.size(correlation, 0)):
        ''' determine the displacment on subpixel level '''
        disp[i, :] = pyprocess_windef.find_subpixel_peak_position_windef(
            correlation[i, :, :], subpixel_method=subpixel_method)
    'this loop is doing the displacment evaluation for each window '

    shapes = np.array(process.get_field_shape(
        np.shape(frame_a), window_size, overlap))
    u = disp[:, 1].reshape(shapes)
    v = disp[:, 0].reshape(shapes)
    'reshaping the interrogation window to vector field shape'
    
    x, y = pyprocess_windef.get_coordinates_windef(np.shape(frame_a), window_size, overlap)
    'get coordinates for to map the displacement'
    if do_sig2noise==True and iterations==1:
        sig2noise_ratio=pyprocess_windef.sig2noise_ratio_windef(correlation, sig2noise_method=sig2noise_method, width=sig2noise_mask)
        sig2noise_ratio = sig2noise_ratio.reshape(shapes)
    else:
        sig2noise_ratio=np.full_like(u,np.nan)
    return x, y, u, v, sig2noise_ratio

def multipass_img_deform(frame_a, frame_b, window_size, overlap,iterations,current_iteration, x_old, y_old, u_old, v_old,correlation_method='circular',
                         subpixel_method='gaussian', do_sig2noise=False, sig2noise_method='peak2peak', sig2noise_mask=2,
                         MinMaxU=(-100, 50), MinMaxV=(-50, 50), std_threshold=5, median_threshold=2,median_size=1, filter_method='localmean',
                         max_filter_iteration=10, filter_kernel_size=2, interpolation_order=3):
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

    window_size : tuple of ints
         the size of the interrogation window

    overlap : tuple of ints
        the overlap of the interrogation window normal for example window_size/2

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
        the order of the spline inpolation used for the image deformation

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

    x, y = pyprocess_windef.get_coordinates_windef(np.shape(frame_a), window_size, overlap)
    'calculate the y and y coordinates of the interrogation window centres'
    y_old = y_old[:, 0]
    y_old = y_old[::-1]
    x_old = x_old[0, :]
    y_int = y[:, 0]
    y_int = y_int[::-1]
    x_int = x[0, :]
    '''The interpolation function dont like meshgrids as input. Hence, the the edges
    must be extracted to provide the sufficient input. x_old and y_old are the 
    are the coordinates of the old grid. x_int and y_int are the coordiantes
    of the new grid'''

    ip = RectBivariateSpline(y_old, x_old, u_old)
    u_pre = ip(y_int, x_int)
    ip2 = RectBivariateSpline(y_old, x_old, v_old)
    v_pre = ip2(y_int, x_int)
    ''' interpolating the displacements from the old grid onto the new grid
    y befor x because of numpy works row major
    '''

    frame_b_deform = frame_interpolation(
        frame_b, x, y, u_pre, -v_pre, interpolation_order=interpolation_order)
    '''this one is doing the image deformation (see above)'''

    cor_win_1 = pyprocess.moving_window_array(frame_a, window_size, overlap)
    cor_win_2 = pyprocess.moving_window_array(
        frame_b_deform, window_size, overlap)
    '''Filling the interrogation window. They windows are arranged
    in a 3d array with number of interrogation window *window_size*window_size
    this way is much faster then using a loop'''

    correlation = correlation_func(cor_win_1, cor_win_2, window_size,correlation_method=correlation_method)
    'do the correlation'
    disp = np.zeros((np.size(correlation, 0), 2))
    for i in range(0, np.size(correlation, 0)):
        ''' determine the displacment on subpixel level  '''
        disp[i, :] = pyprocess_windef.find_subpixel_peak_position_windef(
            correlation[i, :, :], subpixel_method=subpixel_method)
    'this loop is doing the displacment evaluation for each window '

    'reshaping the interrogation window to vector field shape'
    shapes = np.array(process.get_field_shape(
        np.shape(frame_a), window_size, overlap))
    u = disp[:, 1].reshape(shapes)
    v = disp[:, 0].reshape(shapes)

    'adding the recent displacment on to the displacment of the previous pass'
    u = u+u_pre
    v = v+v_pre
    'validation using gloabl limits and local median'
    u, v, mask_g = validation.global_val(u, v, MinMaxU, MinMaxV)
    u, v, mask_s = validation.global_std(u, v, std_threshold=std_threshold)
    u, v, mask_m = validation.local_median_val(u, v, u_threshold=median_threshold, v_threshold=median_threshold, size=median_size)
    mask = mask_g+mask_m+mask_s
    'adding masks to add the effect of alle the validations'
    #mask=np.zeros_like(u)
    'filter to replace the values that where marked by the validation'
    if current_iteration != iterations:
        'filter to replace the values that where marked by the validation'
        u, v = filters.replace_outliers(
                    u, v, method=filter_method, max_iter=max_filter_iteration, kernel_size=filter_kernel_size) 
    if do_sig2noise==True and current_iteration==iterations and iterations!=1:
        sig2noise_ratio=pyprocess_windef.sig2noise_ratio_windef(correlation, sig2noise_method=sig2noise_method, width=sig2noise_mask)
        sig2noise_ratio = sig2noise_ratio.reshape(shapes)
    else:
        sig2noise_ratio=np.full_like(u,np.nan)

    return x, y, u, v,sig2noise_ratio, mask
