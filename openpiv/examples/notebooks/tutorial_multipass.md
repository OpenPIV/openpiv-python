---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.4.2
  kernelspec:
    display_name: Python [conda env:openpiv] *
    language: python
    name: conda-env-openpiv-py
---

```python
# %load red_Cell.py
from openpiv import tools, pyprocess, scaling, filters, \
                    validation, process
import numpy as np
from skimage import data
import matplotlib.pyplot as plt

from scipy import ndimage
from skimage import feature
# from PIL import Image

import imageio
from pylab import *
%matplotlib inline

from skimage.color import rgb2gray
from skimage import img_as_uint

frame_a  = imageio.imread('../test3/Y4-S3_Camera000398.tif')  
frame_b  = imageio.imread('../test3/Y4-S3_Camera000399.tif')
```

```python
frame_a
```

```python
# for whatever reason the shape of frame_a is (3, 284, 256)
# so we first tranpose to the RGB image and then convert to the gray scale

# frame_a = img_as_uint(rgb2gray(frame_a))
# frame_b = img_as_uint(rgb2gray(frame_b))
plt.imshow(np.c_[frame_a,frame_b],cmap=plt.cm.gray)
```

```python

```

```python
# frame_a.dtype
```

```python
u, v, sig2noise = process.extended_search_area_piv( frame_a.astype(np.int32), frame_b.astype(np.int32), window_size=32, overlap=8, dt=.1, sig2noise_method='peak2peak' )
#u, v, sig2noise = pyprocess.extended_search_area_piv( frame_a, frame_b, window_size=32, overlap=8, dt=.1, sig2noise_method='peak2peak' )
x, y = pyprocess.get_coordinates( image_size=frame_a.shape, window_size=32, overlap=8 )
x, y = process.get_coordinates( image_size=frame_a.shape, window_size=32, overlap=8 )
```

```python
u, v, mask = validation.sig2noise_val( u, v, sig2noise, threshold = 1.3 )
u, v = filters.replace_outliers( u, v, method='localmean', max_iter=10, kernel_size=2)
x, y, u, v = scaling.uniform(x, y, u, v, scaling_factor = 96.52 )
```

```python
plt.figure()
plt.quiver(x,y,u,v)
```

```python
tools.save(x, y, u, v, mask, 'Y4-S3_Camera000398.txt' )
tools.display_vector_field('Y4-S3_Camera000398.txt', scale=3, width=0.0125)
# frame_vectors  = io.imshow(vectors)
```

```python
x,y,u,v, mask = process.WiDIM(frame_a.astype(np.int32), frame_b.astype(np.int32), ones_like(frame_a).astype(np.int32), min_window_size=32, overlap_ratio=0.25, coarse_factor=0, dt=0.1, validation_method='mean_velocity', trust_1st_iter=0, validation_iter=0, tolerance=0.7, nb_iter_max=1, sig2noise_method='peak2peak')
```

```python
tools.save(x, y, u, v, zeros_like(u), 'Y4-S3_Camera000398.txt' )
tools.display_vector_field('Y4-S3_Camera000398.txt', scale=300, width=0.005)
```

```python
x,y,u,v, mask = process.WiDIM(frame_a.astype(np.int32), frame_b.astype(np.int32), ones_like(frame_a).astype(np.int32), min_window_size=16, overlap_ratio=0.25, coarse_factor=2, dt=0.1, validation_method='mean_velocity', trust_1st_iter=1, validation_iter=2, tolerance=0.7, nb_iter_max=4, sig2noise_method='peak2peak')
```

```python
tools.save(x, y, u, v, zeros_like(u), 'Y4-S3_Camera000398.txt' )
tools.display_vector_field('Y4-S3_Camera000398.txt', scale=300, width=0.005)
```

```python

```
