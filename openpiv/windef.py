# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 14:04:04 2019

@author: Theo
"""


import os
import numpy as np
from numpy.fft import rfft2, irfft2, fftshift
import scipy.ndimage as scn
from scipy.interpolate import RectBivariateSpline
from openpiv import process, validation, filters, pyprocess, tools, preprocess,scaling
from openpiv import smoothn
import matplotlib.pyplot as plt

def piv(settings):
    

#    '''the func fuction is the "frame" in which the PIV evaluation is done'''

    def func(args):
        """A function to process each image pair."""

        # this line is REQUIRED for multiprocessing to work
        # always use it in your custom function

        file_a, file_b, counter = args

        # counter2=str(counter2)
        #####################
        # Here goes you code
        #####################

        ' read images into numpy arrays'
        frame_a = tools.imread(os.path.join(settings.filepath_images, file_a))
        frame_b = tools.imread(os.path.join(settings.filepath_images, file_b))

        
        ## Miguel: I just had a quick look, and I do not understand the reason for this step.
        #  I propose to remove it.
        #frame_a = (frame_a*1024).astype(np.int32)
        #frame_b = (frame_b*1024).astype(np.int32)

        ' crop to ROI'
        if settings.ROI=='full':
            frame_a=frame_a
            frame_b=frame_b
        else:     
            frame_a =  frame_a[settings.ROI[0]:settings.ROI[1],settings.ROI[2]:settings.ROI[3]]
            frame_b =  frame_b[settings.ROI[0]:settings.ROI[1],settings.ROI[2]:settings.ROI[3]]
        if settings.dynamic_masking_method=='edge' or 'intensity':    
            frame_a = preprocess.dynamic_masking(frame_a,method=settings.dynamic_masking_method,filter_size=settings.dynamic_masking_filter_size,threshold=settings.dynamic_masking_threshold)
            frame_b = preprocess.dynamic_masking(frame_b,method=settings.dynamic_masking_method,filter_size=settings.dynamic_masking_filter_size,threshold=settings.dynamic_masking_threshold)

        '''%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'''
        'first pass'
        x, y, u, v, sig2noise_ratio = first_pass(frame_a,frame_b,settings.windowsizes[0], settings.overlap[0],settings.iterations,
                                      correlation_method=settings.correlation_method, subpixel_method=settings.subpixel_method, do_sig2noise=settings.extract_sig2noise,
                                      sig2noise_method=settings.sig2noise_method, sig2noise_mask=settings.sig2noise_mask,)
    
        'validation using gloabl limits and std and local median'
        '''MinMaxU : two elements tuple
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
            size of the kernel used for the filtering'''

        mask=np.full_like(x,False)
        if settings.validation_first_pass==True:    
            u, v, mask_g = validation.global_val( u, v, settings.MinMax_U_disp, settings.MinMax_V_disp)
            u,v, mask_s = validation.global_std( u, v, std_threshold = settings.std_threshold )
            u, v, mask_m = validation.local_median_val( u, v, u_threshold=settings.median_threshold, v_threshold=settings.median_threshold, size=settings.median_size )
            if settings.extract_sig2noise==True and settings.iterations==1 and settings.do_sig2noise_validation==True:
                u,v, mask_s2n = validation.sig2noise_val( u, v, sig2noise_ratio, threshold = settings.sig2noise_threshold)
                mask=mask+mask_g+mask_m+mask_s+mask_s2n
            else:
                mask=mask+mask_g+mask_m+mask_s
        'filter to replace the values that where marked by the validation'
        if settings.iterations>1:
             u, v = filters.replace_outliers( u, v, method=settings.filter_method, max_iter=settings.max_filter_iteration, kernel_size=settings.filter_kernel_size)
             'adding masks to add the effect of all the validations'
             if settings.smoothn==True:
                  u,dummy_u1,dummy_u2,dummy_u3=smoothn.smoothn(u,s=settings.smoothn_p)
                  v,dummy_v1,dummy_v2,dummy_v3=smoothn.smoothn(v,s=settings.smoothn_p)        
        elif settings.iterations==1 and settings.replace_vectors==True:    
             u, v = filters.replace_outliers( u, v, method=settings.filter_method, max_iter=settings.max_filter_iteration, kernel_size=settings.filter_kernel_size)
             'adding masks to add the effect of all the validations'
             if settings.smoothn==True:
                  u, v = filters.replace_outliers( u, v, method=settings.filter_method, max_iter=settings.max_filter_iteration, kernel_size=settings.filter_kernel_size)
                  u,dummy_u1,dummy_u2,dummy_u3=smoothn.smoothn(u,s=settings.smoothn_p)
                  v,dummy_v1,dummy_v2,dummy_v3=smoothn.smoothn(v,s=settings.smoothn_p)        
     




        i = 1
        'all the following passes'
        for i in range(2, settings.iterations+1):
            x, y, u, v, sig2noise_ratio, mask = multipass_img_deform(frame_a, frame_b, settings.windowsizes[i-1], settings.overlap[i-1],settings.iterations,i,
                                                    x, y, u, v, correlation_method=settings.correlation_method,
                                                    subpixel_method=settings.subpixel_method, do_sig2noise=settings.extract_sig2noise,
                                                    sig2noise_method=settings.sig2noise_method, sig2noise_mask=settings.sig2noise_mask,
                                                    MinMaxU=settings.MinMax_U_disp,
                                                    MinMaxV=settings.MinMax_V_disp,std_threshold=settings.std_threshold,
                                                    median_threshold=settings.median_threshold,median_size=settings.median_size,filter_method=settings.filter_method,
                                                    max_filter_iteration=settings.max_filter_iteration, filter_kernel_size=settings.filter_kernel_size,
                                                    interpolation_order=settings.interpolation_order)
            # If the smoothing is active, we do it at each pass
            if settings.smoothn==True:
                 u,dummy_u1,dummy_u2,dummy_u3= smoothn.smoothn(u,s=settings.smoothn_p)
                 v,dummy_v1,dummy_v2,dummy_v3= smoothn.smoothn(v,s=settings.smoothn_p)        
   
        
        '''%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'''
        if settings.extract_sig2noise==True and i==settings.iterations and settings.iterations!=1 and settings.do_sig2noise_validation==True:
            u,v, mask_s2n = validation.sig2noise_val( u, v, sig2noise_ratio, threshold = settings.sig2noise_threshold)
            mask=mask+mask_s2n
        if settings.replace_vectors==True:
            u, v = filters.replace_outliers( u, v, method=settings.filter_method, max_iter=settings.max_filter_iteration, kernel_size=settings.filter_kernel_size)
        'pixel/frame->pixel/sec'
        u=u/settings.dt
        v=v/settings.dt
        'scales the results pixel-> meter'
        x, y, u, v = scaling.uniform(x, y, u, v, scaling_factor = settings.scaling_factor )     
        'save to a file'
        save(x, y, u, v,sig2noise_ratio, mask ,os.path.join(save_path,'field_A%03d.txt' % counter), delimiter='\t')
        'some messages to check if it is still alive'
        

        'some other stuff that one might want to use'
        if settings.show_plot==True or settings.save_plot==True:
            plt.close('all')
            plt.ioff()
            Name = os.path.join(save_path, 'Image_A%03d.png' % counter)
            display_vector_field(os.path.join(save_path, 'field_A%03d.txt' % counter), scale=settings.scale_plot)
            if settings.save_plot==True:
                plt.savefig(Name)
            if settings.show_plot==True:
                plt.show()

        print('Image Pair ' + str(counter+1))
        
    'Below is code to read files and create a folder to store the results'
    save_path=os.path.join(settings.save_path,'Open_PIV_results_'+str(settings.windowsizes[settings.iterations-1])+'_'+settings.save_folder_suffix)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    task = tools.Multiprocesser(
        data_dir=settings.filepath_images, pattern_a=settings.frame_pattern_a, pattern_b=settings.frame_pattern_b)
    task.run(func=func, n_cpus=1)


def correlation_func(cor_win_1, cor_win_2, window_size,correlation_method='circular'):
    '''This function is doing the cross-correlation. Right now circular cross-correlation
    That means no zero-padding is done
    the .real is to cut off possible imaginary parts that remains due to finite numerical accuarcy
     '''
    if correlation_method=='linear':
        cor_win_1 = cor_win_1-cor_win_1.mean(axis=(1,2)).reshape(cor_win_1.shape[0],1,1)
        cor_win_2 = cor_win_2-cor_win_2.mean(axis=(1,2)).reshape(cor_win_1.shape[0],1,1)
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
        frame, ((y+vt, x+ut,)), order=interpolation_order,mode='nearest')
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

    correlation = correlation_func(cor_win_1, cor_win_2, window_size, correlation_method=correlation_method)
    'do the correlation'
    disp = np.zeros((np.size(correlation, 0), 2))#create a dummy for the loop to fill
    for i in range(0, np.size(correlation, 0)):
        ''' determine the displacment on subpixel level '''
        disp[i, :] = find_subpixel_peak_position(
            correlation[i, :, :], subpixel_method=subpixel_method)
    'this loop is doing the displacment evaluation for each window '

    shapes = np.array(process.get_field_shape(
        frame_a.shape, window_size, overlap))
    u = disp[:, 1].reshape(shapes)
    v = -disp[:, 0].reshape(shapes)
    'reshaping the interrogation window to vector field shape'
    
    x, y = get_coordinates(frame_a.shape, window_size, overlap)
    'get coordinates for to map the displacement'
    if do_sig2noise==True and iterations==1:
        sig2noise_ratio = sig2noise_ratio_function(correlation, sig2noise_method=sig2noise_method, width=sig2noise_mask)
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
        disp[i, :] = find_subpixel_peak_position(
            correlation[i, :, :], subpixel_method=subpixel_method)
    'this loop is doing the displacment evaluation for each window '

    'reshaping the interrogation window to vector field shape'
    shapes = np.array(process.get_field_shape(
        np.shape(frame_a), window_size, overlap))
    u = disp[:, 1].reshape(shapes)
    v = -disp[:, 0].reshape(shapes)

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
        sig2noise_ratio=sig2noise_ratio_function(correlation, sig2noise_method=sig2noise_method, width=sig2noise_mask)
        sig2noise_ratio = sig2noise_ratio.reshape(shapes)
    else:
        sig2noise_ratio=np.full_like(u,np.nan)

    return x, y, u, v,sig2noise_ratio, mask


def save( x, y, u, v, sig2noise_ratio, mask, filename, fmt='%8.4f', delimiter='\t' ):
    """Save flow field to an ascii file.
    
    Parameters
    ----------
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
        
    mask : 2d np.ndarray
        a two dimensional boolen array where elements corresponding to
        invalid vectors are True.
        
    filename : string
        the path of the file where to save the flow field
        
    fmt : string
        a format string. See documentation of numpy.savetxt
        for more details.
    
    delimiter : string
        character separating columns
        
    Examples
    --------
    
    >>> openpiv.tools.save( x, y, u, v, 'field_001.txt', fmt='%6.3f', delimiter='\t')
    
    """
    # build output array
    out = np.vstack( [m.ravel() for m in [x, y, u, v,sig2noise_ratio, mask] ] )
            
    # save data to file.
    np.savetxt( filename, out.T, fmt=fmt, delimiter=delimiter, header='x'+delimiter+'y'+delimiter+'u'+delimiter+'v'+delimiter+'s2n'+delimiter+'mask' )
    
def display_vector_field( filename, on_img=False, image_name='None', window_size=32, scaling_factor=1,skiprows=1, **kw):
    """ Displays quiver plot of the data stored in the file 
    
    
    Parameters
    ----------
    filename :  string
        the absolute path of the text file

    on_img : Bool, optional
        if True, display the vector field on top of the image provided by image_name

    image_name : string, optional
        path to the image to plot the vector field onto when on_img is True

    window_size : int, optional
        when on_img is True, provide the interogation window size to fit the background image to the vector field

    scaling_factor : float, optional
        when on_img is True, provide the scaling factor to scale the background image to the vector field
    
    Key arguments   : (additional parameters, optional)
        *scale*: [None | float]
        *width*: [None | float]
    
    
    See also:
    ---------
    matplotlib.pyplot.quiver
    
        
    Examples
    --------
    --- only vector field
    >>> openpiv.tools.display_vector_field('./exp1_0000.txt',scale=100, width=0.0025) 

    --- vector field on top of image
    >>> openpiv.tools.display_vector_field('./exp1_0000.txt', on_img=True, image_name='exp1_001_a.bmp', window_size=32, scaling_factor=70, scale=100, width=0.0025)
    
    """
    
    a = np.loadtxt(filename)
    fig=plt.figure()
    if on_img: # plot a background image
        im = fig.imread(image_name)
        im = fig.negative(im) #plot negative of the image for more clarity
        fig.imsave('neg.tif', im)
        im = fig.imread('neg.tif')
        xmax = np.amax(a[:,0])+window_size/(2*scaling_factor)
        ymax = np.amax(a[:,1])+window_size/(2*scaling_factor)
        implot = plt.imshow(im, origin='lower', cmap="Greys_r",extent=[0.,xmax,0.,ymax])
    invalid = a[:,5].astype('bool')
    fig.canvas.set_window_title('Vector field, '+str(np.count_nonzero(invalid))+' wrong vectors')
    valid = ~invalid
    plt.quiver(a[invalid,0],a[invalid,1],a[invalid,2],a[invalid,3],color='r',width=0.001,headwidth=3,**kw)
    plt.quiver(a[valid,0],a[valid,1],a[valid,2],a[valid,3],color='b',width=0.001,headwidth=3,**kw)
    plt.draw()


def get_field_shape(image_size, window_size, overlap):
    """Compute the shape of the resulting flow field.
    Given the image size, the interrogation window size and
    the overlap size, it is possible to calculate the number
    of rows and columns of the resulting flow field.
    Parameters
    ----------
    image_size: two elements tuple
        a two dimensional tuple for the pixel size of the image
        first element is number of rows, second element is
        the number of columns.
    window_size: int
        the size of the interrogation window.
    overlap: int
        the number of pixel by which two adjacent interrogation
        windows overlap.
    Returns
    -------
    field_shape : two elements tuple
        the shape of the resulting flow field
    """

    return ((image_size[0] - window_size) // (window_size - overlap) + 1,
            (image_size[1] - window_size) // (window_size - overlap) + 1)



def get_coordinates(image_size, window_size, overlap):
        """Compute the x, y coordinates of the centers of the interrogation windows.

        Parameters
        ----------
        image_size: two elements tuple
            a two dimensional tuple for the pixel size of the image
            first element is number of rows, second element is 
            the number of columns.

        window_size: int
            the size of the interrogation windows.

        overlap: int
            the number of pixel by which two adjacent interrogation
            windows overlap.


        Returns
        -------
        x : 2d np.ndarray
            a two dimensional array containing the x coordinates of the 
            interrogation window centers, in pixels.

        y : 2d np.ndarray
            a two dimensional array containing the y coordinates of the 
            interrogation window centers, in pixels.

        """

        # get shape of the resulting flow field
        '''%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        The get_field_shape function calculates how many interrogation windows
        fit in the image in each dimension output is a 
        tuple (amount of interrogation windows in y, amount of interrogation windows in x)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        The get coordinates function calculates the coordinates of the center of each 
        interrogation window using bases on the to field_shape returned by the
        get field_shape function, the window size and the overlap. It returns a meshgrid
        of the interrogation area centers.
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        '''

        field_shape = get_field_shape(image_size, window_size, overlap)

        # compute grid coordinates of the interrogation window centers
        x = np.arange(field_shape[1])*(window_size-overlap) + (window_size)/2.0
        y = np.arange(field_shape[0])*(window_size-overlap) + (window_size)/2.0

        return np.meshgrid(x, y[::-1])


def find_subpixel_peak_position(corr, subpixel_method='gaussian'):
        """
        Find subpixel approximation of the correlation peak.

        This function returns a subpixels approximation of the correlation
        peak by using one of the several methods available. If requested,
        the function also returns the signal to noise ratio level evaluated
        from the correlation map.

        Parameters
        ----------
        corr : np.ndarray
            the correlation map.

        subpixel_method : string
             one of the following methods to estimate subpixel location of the peak:
             'centroid' [replaces default if correlation map is negative],
             'gaussian' [default if correlation map is positive],
             'parabolic'.

        Returns
        -------
        subp_peak_position : two elements tuple
            the fractional row and column indices for the sub-pixel
            approximation of the correlation peak.
        """

        # initialization
        default_peak_position = (
                np.floor(corr.shape[0] / 2.), np.floor(corr.shape[1] / 2.))
        '''this calculates the default peak position (peak of the autocorrelation).
        It is window_size/2. It needs to be subtracted to from the peak found to determin the displacment
        '''
        #default_peak_position = (0,0)

        # the peak locations
        peak1_i, peak1_j, dummy = pyprocess.find_first_peak(corr)
        '''
        The find_first_peak function returns the coordinates of the correlation peak
        and the value of the peak. Here only the coordinates are needed.
        '''

        try:
            # the peak and its neighbours: left, right, down, up
            c = corr[peak1_i,   peak1_j]
            cl = corr[peak1_i - 1, peak1_j]
            cr = corr[peak1_i + 1, peak1_j]
            cd = corr[peak1_i,   peak1_j - 1]
            cu = corr[peak1_i,   peak1_j + 1]

            # gaussian fit
            if np.any(np.array([c, cl, cr, cd, cu]) < 0) and subpixel_method == 'gaussian':
                subpixel_method = 'centroid'

            try:
                if subpixel_method == 'centroid':
                    subp_peak_position = (((peak1_i - 1) * cl + peak1_i * c + (peak1_i + 1) * cr) / (cl + c + cr),
                                          ((peak1_j - 1) * cd + peak1_j * c + (peak1_j + 1) * cu) / (cd + c + cu))

                elif subpixel_method == 'gaussian':
                    subp_peak_position = (peak1_i + ((np.log(cl) - np.log(cr)) / (2 * np.log(cl) - 4 * np.log(c) + 2 * np.log(cr))),
                                          peak1_j + ((np.log(cd) - np.log(cu)) / (2 * np.log(cd) - 4 * np.log(c) + 2 * np.log(cu))))

                elif subpixel_method == 'parabolic':
                    subp_peak_position = (peak1_i + (cl - cr) / (2 * cl - 4 * c + 2 * cr),
                                          peak1_j + (cd - cu) / (2 * cd - 4 * c + 2 * cu))

            except:
                subp_peak_position = default_peak_position

        except IndexError:
            subp_peak_position = default_peak_position

            '''This block is looking for the neighbouring pixels. The subpixelposition is calculated based one
            the correlation values. Different methods can be choosen.
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            This function returns the displacement in u and v
            '''
        return subp_peak_position[0] - default_peak_position[0], subp_peak_position[1] - default_peak_position[1]
    


def sig2noise_ratio_function(corr, sig2noise_method='peak2peak', width=2):
    """
    Computes the signal to noise ratio from the correlation map.

    The signal to noise ratio is computed from the correlation map with
    one of two available method. It is a measure of the quality of the
    matching between to interogation windows.

    Parameters
    ----------
    corr : 2d np.ndarray
        the correlation map.

    sig2noise_method: string
        the method for evaluating the signal to noise ratio value from
        the correlation map. Can be `peak2peak`, `peak2mean` or None
        if no evaluation should be made.

    width : int, optional
        the half size of the region around the first
        correlation peak to ignore for finding the second
        peak. [default: 2]. Only used if ``sig2noise_method==peak2peak``.

    Returns
    -------
    sig2noise : np.ndarray 
        the signal to noise ratio from the correlation map.

    """

    corr_max1=np.zeros(corr.shape[0])
    corr_max2=np.zeros(corr.shape[0])
    peak1_i=np.zeros(corr.shape[0])
    peak1_j=np.zeros(corr.shape[0])
    peak2_i=np.zeros(corr.shape[0])
    peak2_j = np.zeros(corr.shape[0])
    for i in range(0,corr.shape[0]):
        # compute first peak position
        peak1_i[i], peak1_j[i], corr_max1[i] = pyprocess.find_first_peak(corr[i,:,:])
        if sig2noise_method == 'peak2peak':
            # now compute signal to noise ratio
            
                # find second peak height
                peak2_i[i], peak2_j[i], corr_max2[i] = pyprocess.find_second_peak(
                    corr[i,:,:], int(peak1_i[i]), int(peak1_j[i]), width=width)
        
                # if it's an empty interrogation window
                # if the image is lacking particles, totally black it will correlate to very low value, but not zero
                # if the first peak is on the borders, the correlation map is also
                # wrong
                if corr_max1[i] < 1e-3 or (peak1_i[i] == 0 or peak1_j[i] == corr.shape[1] or peak1_j[i] == 0 or peak1_j[i] == corr.shape[2] or
                                        peak2_i[i] == 0 or peak2_j[i] == corr.shape[1] or peak2_j[i] == 0 or peak2_j[i] == corr.shape[2]):
                    # return zero, since we have no signal.
                    corr_max1[i]=0
        
    
        elif sig2noise_method == 'peak2mean':
            # find mean of the correlation map
            corr_max2 = corr.mean(axis=(1,2))

        else:
            raise ValueError('wrong sig2noise_method')

    # avoid dividing by zero
    corr_max2[corr_max2==0]=np.nan    
    sig2noise = corr_max1 / corr_max2
    sig2noise[sig2noise==np.nan]=0

    return sig2noise



class Settings(object):
    pass


if __name__ == "__main__":
    """ Run windef.py as a script: 

    python windef.py 

    """



    settings = Settings()


    'Data related settings'
    # Folder with the images to process
    settings.filepath_images = './examples/test1/'
    # Folder for the outputs
    settings.save_path = './examples/test1/'
    # Root name of the output Folder for Result Files
    settings.save_folder_suffix = 'Test_4'
    # Format and Image Sequence
    settings.frame_pattern_a = 'exp1_001_a.bmp'
    settings.frame_pattern_b = 'exp1_001_b.bmp'

    'Region of interest'
    # (50,300,50,300) #Region of interest: (xmin,xmax,ymin,ymax) or 'full' for full image
    settings.ROI = 'full'

    'Image preprocessing'
    # 'None' for no masking, 'edges' for edges masking, 'intensity' for intensity masking
    # WARNING: This part is under development so better not to use MASKS
    settings.dynamic_masking_method = 'None'
    settings.dynamic_masking_threshold = 0.005
    settings.dynamic_masking_filter_size = 7

    'Processing Parameters'
    settings.correlation_method='circular'  # 'circular' or 'linear'
    settings.iterations =1  # select the number of PIV passes
    # add the interroagtion window size for each pass. 
    # For the moment, it should be a power of 2 
    settings.windowsizes = (128, 64, 32) # if longer than n iteration the rest is ignored
    # The overlap of the interroagtion window for each pass.
    settings.overlap = (64, 32, 16) # This is 50% overlap
    # Has to be a value with base two. In general window size/2 is a good choice.
    # methode used for subpixel interpolation: 'gaussian','centroid','parabolic'
    settings.subpixel_method = 'gaussian'
    # order of the image interpolation for the window deformation
    settings.interpolation_order = 3
    settings.scaling_factor = 1  # scaling factor pixel/meter
    settings.dt = 1  # time between to frames (in seconds)
    'Signal to noise ratio options (only for the last pass)'
    # It is possible to decide if the S/N should be computed (for the last pass) or not
    settings.extract_sig2noise = True  # 'True' or 'False' (only for the last pass)
    # method used to calculate the signal to noise ratio 'peak2peak' or 'peak2mean'
    settings.sig2noise_method = 'peak2peak'
    # select the width of the masked to masked out pixels next to the main peak
    settings.sig2noise_mask = 2
    # If extract_sig2noise==False the values in the signal to noise ratio
    # output column are set to NaN
    'vector validation options'
    # choose if you want to do validation of the first pass: True or False
    settings.validation_first_pass = True
    # only effecting the first pass of the interrogation the following passes
    # in the multipass will be validated
    'Validation Parameters'
    # The validation is done at each iteration based on three filters.
    # The first filter is based on the min/max ranges. Observe that these values are defined in
    # terms of minimum and maximum displacement in pixel/frames.
    settings.MinMax_U_disp = (-30, 30)
    settings.MinMax_V_disp = (-30, 30)
    # The second filter is based on the global STD threshold
    settings.std_threshold = 10  # threshold of the std validation
    # The third filter is the median test (not normalized at the moment)
    settings.median_threshold = 3  # threshold of the median validation
    # On the last iteration, an additional validation can be done based on the S/N.
    settings.median_size=1 #defines the size of the local median
    'Validation based on the signal to noise ratio'
    # Note: only available when extract_sig2noise==True and only for the last
    # pass of the interrogation
    # Enable the signal to noise ratio validation. Options: True or False
    settings.do_sig2noise_validation = False # This is time consuming
    # minmum signal to noise ratio that is need for a valid vector
    settings.sig2noise_threshold = 1.2
    'Outlier replacement or Smoothing options'
    # Replacment options for vectors which are masked as invalid by the validation
    settings.replace_vectors = True # Enable the replacment. Chosse: True or False
    settings.smoothn=True #Enables smoothing of the displacemenet field
    settings.smoothn_p=0.5 # This is a smoothing parameter
    # select a method to replace the outliers: 'localmean', 'disk', 'distance'
    settings.filter_method = 'localmean'
    # maximum iterations performed to replace the outliers
    settings.max_filter_iteration = 4
    settings.filter_kernel_size = 2  # kernel size for the localmean method
    'Output options'
    # Select if you want to save the plotted vectorfield: True or False
    settings.save_plot = True
    # Choose wether you want to see the vectorfield or not :True or False
    settings.show_plot = False
    settings.scale_plot = 100 # select a value to scale the quiver plot of the vectorfield
    # run the script with the given settings

    piv(settings)
