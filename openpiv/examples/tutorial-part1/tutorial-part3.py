#!/usr/bin/env python
import sys

if 'OpenPIV' not in sys.path:
    sys.path.append('/Users/alex/Documents/OpenPIV/alexlib/openpiv-python')

from openpiv import tools, process, pyprocess
import numpy as np
import matplotlib.pyplot as plt



frame_a  = tools.imread( 'test001.png' )
frame_b  = tools.imread( 'test002.png')

frame_a = frame_a[:32,:32]
frame_b = frame_b[:32,:32]

plt.figure()
plt.imshow(np.c_[frame_a,frame_b],cmap=plt.cm.gray)
plt.show()


u, v = process.extended_search_area_piv(frame_a.astype(np.int32), 
frame_b.astype(np.int32), window_size=24, overlap=0, dt=0.02, 
search_area_size=32)

x, y = process.get_coordinates( image_size=frame_a.shape, window_size=24, overlap=12 )

plt.figure()
plt.quiver(x,y,u,v)
plt.show()

del u,v

u, v = pyprocess.piv(frame_a, frame_b, window_size=24, search_size=24, corr_method='direct')

x, y = process.get_coordinates( image_size=frame_a.shape, window_size=24, overlap=0)

plt.figure()
plt.quiver(x,y,u,v)
plt.show()