import numpy as np
from openpiv import pyprocess

def get_coordinates_windef(image_size, window_size, overlap):
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

        field_shape = pyprocess.get_field_shape(image_size, window_size, overlap)

        # compute grid coordinates of the interrogation window centers
        x = np.arange(field_shape[1])*(window_size-overlap) + (window_size)/2.0
        y = np.arange(field_shape[0])*(window_size-overlap) + (window_size)/2.0

        return np.meshgrid(x, y[::-1])


def find_subpixel_peak_position_windef(corr, subpixel_method='gaussian'):
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
    


def sig2noise_ratio_windef(corr, sig2noise_method='peak2peak', width=2):
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