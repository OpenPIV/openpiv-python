# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 12:01:29 2019

@author: Theo
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from openpiv import tools, validation, filters, scaling, preprocess
import tools_windef
import windef
from smoothn import smoothn

def PIV_windef(settings):
    

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
        x, y, u, v, sig2noise_ratio = windef.first_pass(frame_a,frame_b,settings.windowsizes[0], settings.overlap[0],settings.iterations,
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
                  u,dummy_u1,dummy_u2,dummy_u3=smoothn(u,s=settings.smoothn_p)
                  v,dummy_v1,dummy_v2,dummy_v3=smoothn(v,s=settings.smoothn_p)        
        elif settings.iterations==1 and settings.replace_vectors==True:    
             u, v = filters.replace_outliers( u, v, method=settings.filter_method, max_iter=settings.max_filter_iteration, kernel_size=settings.filter_kernel_size)
             'adding masks to add the effect of all the validations'
             if settings.smoothn==True:
                  u, v = filters.replace_outliers( u, v, method=settings.filter_method, max_iter=settings.max_filter_iteration, kernel_size=settings.filter_kernel_size)
                  u,dummy_u1,dummy_u2,dummy_u3=smoothn(u,s=settings.smoothn_p)
                  v,dummy_v1,dummy_v2,dummy_v3=smoothn(v,s=settings.smoothn_p)        
     




        i = 1
        'all the following passes'
        for i in range(2, settings.iterations+1):
            x, y, u, v, sig2noise_ratio, mask = windef.multipass_img_deform(frame_a, frame_b, settings.windowsizes[i-1], settings.overlap[i-1],settings.iterations,i,
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
                 u,dummy_u1,dummy_u2,dummy_u3=smoothn(u,s=settings.smoothn_p)
                 v,dummy_v1,dummy_v2,dummy_v3=smoothn(v,s=settings.smoothn_p)        
   
        
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
        tools_windef.save_windef(x, y, u, v,sig2noise_ratio, mask ,os.path.join(save_path,'field_A%03d.txt' % counter), delimiter='\t')
        'some messages to check if it is still alive'
        

        'some other stuff that one might want to use'
        if settings.show_plot==True or settings.save_plot==True:
            plt.close('all')
            plt.ioff()
            Name = os.path.join(save_path, 'Image_A%03d.png' % counter)
            tools_windef.display_vector_field_windef(os.path.join(save_path, 'field_A%03d.txt' % counter), scale=settings.scale_plot)
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
