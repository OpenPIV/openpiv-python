---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.5.0
  kernelspec:
    display_name: Python [conda env:openpiv]
    language: python
    name: conda-env-openpiv-py
---

```python
# %load red_Cell.py
from openpiv import tools, pyprocess, scaling, filters, \
                    validation, process
import numpy as np
import matplotlib.pyplot as plt

import imageio
from pylab import *
%matplotlib inline

from skimage import img_as_uint

frame_a  = tools.imread('../test3/Y4-S3_Camera000398.tif')  
frame_b  = tools.imread('../test3/Y4-S3_Camera000399.tif')
```

```python
# for whatever reason the shape of frame_a is (3, 284, 256)
# so we first tranpose to the RGB image and then convert to the gray scale

# frame_a = img_as_uint(rgb2gray(frame_a))
# frame_b = img_as_uint(rgb2gray(frame_b))
plt.imshow(np.c_[frame_a,frame_b],cmap=plt.cm.gray)
```

```python
# Use Cython version: process.pyx

u, v, sig2noise = process.extended_search_area_piv( frame_a.astype(np.int32), frame_b.astype(np.int32), window_size=32, overlap=8, dt=.1, sig2noise_method='peak2peak' )
x, y = process.get_coordinates( image_size=frame_a.shape, window_size=32, overlap=8 )

u, v, mask = validation.sig2noise_val( u, v, sig2noise, threshold = 1.3 )
u, v = filters.replace_outliers( u, v, method='localmean', max_iter=10, kernel_size=2)
x, y, u, v = scaling.uniform(x, y, u, v, scaling_factor = 96.52 )

tools.save(x, y, u, v, mask, 'Y4-S3_Camera000398_a.txt' )
```

```python
# Use Python version, pyprocess:

u, v, sig2noise = pyprocess.extended_search_area_piv( frame_a.astype(np.int32), frame_b.astype(np.int32), window_size=32, overlap=8, dt=.1, sig2noise_method='peak2peak' )
x, y = pyprocess.get_coordinates( image_size=frame_a.shape, window_size=32, overlap=8 )
u, v, mask = validation.sig2noise_val( u, v, sig2noise, threshold = 1.3 )
u, v = filters.replace_outliers( u, v, method='localmean', max_iter=10, kernel_size=2)
x, y, u, v = scaling.uniform(x, y, u, v, scaling_factor = 96.52 )

tools.save(x, y, u, v, mask, 'Y4-S3_Camera000398_b.txt' )
```

```python
# "natural" view without image
fig,ax = plt.subplots(2,1,figsize=(6,12))
ax[0].invert_yaxis()
ax[0].quiver(x,y,u,v)
ax[0].set_title(' Sort of natural view ')

ax[1].quiver(x,y,u,-v)
ax[1].set_title('Quiver with 0,0 origin needs `negative` v for display');
# plt.quiver(x,y,u,v)
```

```python
tools.display_vector_field('Y4-S3_Camera000398_a.txt',on_img=True,image_name='../test3/Y4-S3_Camera000398.tif',scaling_factor=96.52)
```

```python
tools.display_vector_field('Y4-S3_Camera000398_a.txt')
```

```python
tools.display_vector_field('Y4-S3_Camera000398_b.txt')
```

```python
x,y,u,v, mask = process.WiDIM(frame_a.astype(np.int32), frame_b.astype(np.int32), ones_like(frame_a).astype(np.int32), min_window_size=32, overlap_ratio=0.25, coarse_factor=0, dt=0.1, validation_method='mean_velocity', trust_1st_iter=0, validation_iter=0, tolerance=0.7, nb_iter_max=1, sig2noise_method='peak2peak')
```

```python
tools.save(x, y, u, v, zeros_like(u), 'Y4-S3_Camera000398_widim1.txt' )
```

```python
x,y,u,v, mask = process.WiDIM(frame_a.astype(np.int32), frame_b.astype(np.int32), ones_like(frame_a).astype(np.int32), min_window_size=16, overlap_ratio=0.25, coarse_factor=2, dt=0.1, validation_method='mean_velocity', trust_1st_iter=1, validation_iter=2, tolerance=0.7, nb_iter_max=4, sig2noise_method='peak2peak')
```

```python
tools.save(x, y, u, v, zeros_like(u), 'Y4-S3_Camera000398_widim2.txt' )
```

```python
tools.display_vector_field('Y4-S3_Camera000398_widim1.txt', widim=True, scale=300, width=0.005)
tools.display_vector_field('Y4-S3_Camera000398_widim2.txt', widim=True, scale=300, width=0.005)
tools.display_vector_field('Y4-S3_Camera000398_a.txt', scale=2, width=0.005,scaling_factor=96.52)
tools.display_vector_field('Y4-S3_Camera000398_b.txt', scale=2, width=0.005,scaling_factor=96.52)
```
