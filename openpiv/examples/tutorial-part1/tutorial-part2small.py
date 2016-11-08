import sys
sys.path.append('/Users/alex/Documents/OpenPIV/alexlib/openpiv-python')

import openpiv.tools
import openpiv.process
import openpiv.scaling
import numpy as np

from pylab import * 

frame_a  = openpiv.tools.imread('exp1_001_a.bmp')
frame_b  = openpiv.tools.imread('exp1_001_b.bmp' )


frame_a = frame_a[:128,:128]
frame_b = frame_b[:128,:128]

from pylab import *
imshow(np.c_[frame_a,frame_b],cmap=cm.gray)
show()


u, v, sig2noise = openpiv.process.extended_search_area_piv( 
                                                           frame_a.astype(np.int32), 
frame_b.astype(np.int32), 
window_size=24, overlap=12, 
dt=0.02, search_area_size=32, 
sig2noise_method='peak2peak' )

x, y = openpiv.process.get_coordinates( image_size=frame_a.shape, window_size=24, overlap=12 )



quiver(x,y,u,v)
show()


u, v= openpiv.pyprocess.piv( frame_a, frame_b, corr_method='fft', 
                            window_size=32, overlap=12, dt=0.02, sig2noise_method='peak2peak' )
x, y = openpiv.pyprocess.get_coordinates( image_size=frame_a.shape, window_size=24, overlap=12 )

quiver(x,y,u,v)
show()


u, v= openpiv.pyprocess.piv( frame_a, frame_b, corr_method='direct', 
                            window_size=32, overlap=12, dt=0.02, sig2noise_method='peak2peak' )
x, y = openpiv.pyprocess.get_coordinates( image_size=frame_a.shape, 
                                         window_size=24, overlap=12 )

quiver(x,y,u,v)
show()