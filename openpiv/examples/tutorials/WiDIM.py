import openpiv.tools
import openpiv.process
import openpiv.scaling
import numpy as np
from time import time
import warnings


scaling_factor = 100

frame_a  = openpiv.tools.imread( '2image_00.tif' )
frame_b  = openpiv.tools.imread( '2image_01.tif' )

#no background removal will be performed so 'mark' is initialized to 1 everywhere
mark = np.zeros(frame_a.shape, dtype=np.int32)
for I in range(mark.shape[0]):
    for J in range(mark.shape[1]):
        mark[I,J]=1

#main algorithm
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    x,y,u,v, mask=openpiv.process.WiDIM( frame_a, frame_b, mark, min_window_size=16, overlap_ratio=0.0, coarse_factor=2, dt=0.02, validation_method='mean_velocity', trust_1st_iter=1, validation_iter=1, tolerance=0.7, nb_iter_max=3, sig2noise_method='peak2peak')

#display results
x, y, u, v = openpiv.scaling.uniform(x, y, u, v, scaling_factor = scaling_factor )

openpiv.tools.save(x, y, u, v, mask, '2image_00.txt' )

openpiv.tools.display_vector_field('2image_00.txt',on_img=True, image_name='2image_00.tif', window_size=16, scaling_factor=scaling_factor, scale=200, width=0.001)

#further validation can be performed to eliminate the few remaining wrong vectors
