from openpiv import tools, scaling, process, validation, filters,preprocess
import os
import numpy as np
import matplotlib.pyplot as plt

from time import time
import warnings


scaling_factor = 100

# we can run it from any folder
path = os.path.dirname(os.path.abspath(__file__))


frame_a  = tools.imread( os.path.join(path,'../test2/2image_00.tif'))
frame_b  = tools.imread( os.path.join(path,'../test2/2image_01.tif'))

#no background removal will be performed so 'mark' is initialized to 1 everywhere
mark = np.zeros(frame_a.shape, dtype=np.int32)
for I in range(mark.shape[0]):
    for J in range(mark.shape[1]):
        mark[I,J]=1

#main algorithm
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    x,y,u,v, mask=process.WiDIM( frame_a.astype(np.int32), frame_b.astype(np.int32), mark, min_window_size=16, overlap_ratio=0.0, coarse_factor=2, dt=0.02, validation_method='mean_velocity', trust_1st_iter=1, validation_iter=1, tolerance=0.7, nb_iter_max=3, sig2noise_method='peak2peak')

#display results
x, y, u, v = scaling.uniform(x, y, u, v, scaling_factor = scaling_factor )

tools.save(x, y, u, v, mask, '2image_00.txt' )

tools.display_vector_field('2image_00.txt',on_img=True, image_name=os.path.join(path,'../test2/2image_00.tif'), window_size=16, scaling_factor=scaling_factor, scale=200, width=0.001)

#further validation can be performed to eliminate the few remaining wrong vectors
