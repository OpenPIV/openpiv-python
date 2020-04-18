# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python [conda env:openpiv] *
#     language: python
#     name: conda-env-openpiv-py
# ---

# %% [markdown]
# # OpenPIV tutorial 1
#
#
# In this tutorial we read the pair of images using `imread`, compare them visually 
# and process using OpenPIV. Here the import is using directly the basic functions and methods

# %%
from openpiv import tools, process, validation, filters, scaling 

import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

import imageio


# %%
frame_a  = tools.imread( '../test1/exp1_001_a.bmp' )
frame_b  = tools.imread( '../test1/exp1_001_b.bmp' )

# %%
fig,ax = plt.subplots(1,2,figsize=(12,10))
ax[0].imshow(frame_a,cmap=plt.cm.gray)
ax[1].imshow(frame_b,cmap=plt.cm.gray)


# %%
winsize = 24 # pixels
searchsize = 64  # pixels, search in image B
overlap = 12 # pixels
dt = 0.02 # sec


u0, v0, sig2noise = process.extended_search_area_piv( frame_a.astype(np.int32), frame_b.astype(np.int32), window_size=winsize, overlap=overlap, dt=dt, search_area_size=searchsize, sig2noise_method='peak2peak' )

# %%
x, y = process.get_coordinates( image_size=frame_a.shape, window_size=winsize, overlap=overlap )

# %%
u1, v1, mask = validation.sig2noise_val( u0, v0, sig2noise, threshold = 1.3 )

# %%
u2, v2 = filters.replace_outliers( u1, v1, method='localmean', max_iter=10, kernel_size=2)

# %%
x, y, u3, v3 = scaling.uniform(x, y, u2, v2, scaling_factor = 96.52 )

# %%
tools.save(x, y, u3, v3, mask, 'exp1_001.txt' )

# %%
tools.display_vector_field('exp1_001.txt', scale=50, width=0.0025)

# %%
# If you need a larger view:

fig, ax = plt.subplots(figsize=(12,12))
tools.display_vector_field('exp1_001.txt', ax=ax, scaling_factor=96.52, scale=50, width=0.0025, on_img=True, image_name='../test1/exp1_001_a.bmp');

# %%
