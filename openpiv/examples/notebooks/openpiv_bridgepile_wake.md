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

# OpenPIV on the bridgepile_wake

See the post on LinkedIn by Stefano Brizzolara
https://www.linkedin.com/posts/stefano-brizzolara-6a8501198_rheinfall-flowvisualization-ugcPost-6672832128742408192-lRub

```python
from openpiv import tools, process, validation, filters, scaling 

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

import imageio
```


```python
frame_a  = tools.imread( '~/Downloads/bridgepile_wake/frame0001.tif' )
frame_b  = tools.imread( '~/Downloads/bridgepile_wake/frame0011.tif' )
```

```python
fig,ax = plt.subplots(1,2,figsize=(12,10))
ax[0].imshow(frame_a,cmap=plt.cm.gray)
ax[1].imshow(frame_b,cmap=plt.cm.gray)
```


```python
winsize = 48 # pixels
searchsize = 96  # pixels, search in image B
overlap = 24 # pixels
dt = 1./30 # sec, assume 30 fps


u0, v0, sig2noise = process.extended_search_area_piv( frame_a.astype(np.int32), frame_b.astype(np.int32), window_size=winsize, overlap=overlap, dt=dt, search_area_size=searchsize, sig2noise_method='peak2peak' )
```

```python
x, y = process.get_coordinates( image_size=frame_a.shape, window_size=winsize, overlap=overlap )
```

```python
u1, v1, mask = validation.sig2noise_val( u0, v0, sig2noise, threshold = 1.15)
```

```python
u2, v2 = filters.replace_outliers( u1, v1, method='localmean', max_iter=10, kernel_size=2)
```

```python
x, y, u3, v3 = scaling.uniform(x, y, u2, v2, scaling_factor = 1. )
```

```python
tools.save(x, y, u3, v3, mask, 'exp1_001.txt' )
```

```python
# tools.display_vector_field('exp1_001.txt', scaling_factor=100., width=0.0025)
```

```python
# If you need a larger view:
fig, ax = plt.subplots(figsize=(12,12))
tools.display_vector_field('exp1_001.txt', ax=ax, scaling_factor=1.0, scale=3500, width=0.0045, on_img=True, image_name='~/Downloads/bridgepile_wake/frame0001.tif');
```

```python

```

```python

```
