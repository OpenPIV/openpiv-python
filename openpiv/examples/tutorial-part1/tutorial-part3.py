#!/usr/bin/env python
import sys

if 'OpenPIV' not in sys.path:
    sys.path.append('/Users/alex/Documents/OpenPIV/alexlib/openpiv-python')

from openpiv import tools, process, pyprocess
import numpy as np
import matplotlib.pyplot as plt

from skimage.transform import resize
from skimage import img_as_ubyte


frame_a  = tools.imread( 'test001.png')
frame_b  = tools.imread( 'test002.png')


# frame_a = img_as_ubyte(resize(frame_a,(1024,256)))
# frame_b = img_as_ubyte(resize(frame_b,(1024,256)))

frame_a = frame_a[:512,-200:]
frame_b = frame_b[:512,-200:]

#frame_a = frame_a[:256,:]
#frame_b = frame_b[:256,:]

plt.figure()
plt.imshow(np.c_[frame_a,frame_b],cmap=plt.cm.gray)
plt.show()


u, v = process.extended_search_area_piv(frame_a.astype(np.int32), 
frame_b.astype(np.int32), window_size=24, overlap=12, search_area_size=32, dt = 1.)

plt.figure()
plt.quiver(u,v)
plt.axis('equal')
plt.show()

u1, v1 = pyprocess.piv(frame_a, frame_b, window_size=32, search_size=48, overlap=24)

# x, y = process.get_coordinates( image_size=frame_a.shape, window_size=24, overlap=0)

plt.figure()
plt.quiver(u1,v1,v1)
plt.axis('equal')
plt.show()


u2, v2 = pyprocess.piv(frame_a, frame_b, corr_method = 'direct', window_size=32)

# x, y = process.get_coordinates( image_size=frame_a.shape, window_size=24, overlap=0)

plt.figure()
plt.quiver(u2,v2,v2)
plt.axis('equal')
plt.show()