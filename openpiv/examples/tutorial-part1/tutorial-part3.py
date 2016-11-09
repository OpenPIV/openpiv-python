#!/usr/bin/env python
import sys

if 'OpenPIV' not in sys.path:
    sys.path.append('/Users/alex/Documents/OpenPIV/alexlib/openpiv-python')

from openpiv import tools, validation, process, filters, scaling, pyprocess
import numpy as np
import matplotlib.pyplot as plt



frame_a  = tools.imread( 'test001.png' )
frame_b  = tools.imread( 'test002.png')


print frame_a.shape 
print frame_a.dtype


u, v, sig2noise = process.extended_search_area_piv(frame_a.astype(np.int32), 
frame_b.astype(np.int32), window_size=24, overlap=12, dt=0.02, 
search_area_size=48, sig2noise_method='peak2peak' )

x, y = process.get_coordinates( image_size=frame_a.shape, window_size=24, overlap=12 )

plt.figure()
plt.quiver(x,y,u,v)
plt.show()


u, v, sig2noise = pyprocess.piv(frame_a, frame_b, window_size=24, overlap=12, dt=0.02, 
search_size=48, sig2noise_method='peak2peak' )

x, y = process.get_coordinates( image_size=frame_a.shape, window_size=24, overlap=12 )

plt.figure()
plt.quiver(x,y,u,v)
plt.show()