import pathlib
from importlib_resources import files
import numpy as np
import matplotlib.pyplot as plt
from openpiv import tools, scaling, pyprocess, validation, filters,preprocess

# we can run it from any folder
path = files('openpiv') / "data" / "test4" 

im_a  = tools.imread(  path / "Camera1-0101.tif")
im_b  = tools.imread( path / "Camera1-0102.tif")

plt.imshow(np.c_[im_a,im_b],cmap='gray')

# let's crop the region of interest
frame_a =  im_a[380:1980,0:1390]
frame_b =  im_b[380:1980,0:1390]
plt.imshow(np.c_[frame_a,frame_b],cmap='gray')

# Process the original cropped image and see the OpenPIV result:

# typical parameters:
window_size = 32 #pixels
overlap = 16 # pixels
search_area_size = 64 # pixels 
frame_rate = 40 # fps

# process again with the masked images, for comparison# process once with the original images
u, v, sig2noise = pyprocess.extended_search_area_piv(
                                                       frame_a.astype(np.int32) , frame_b.astype(np.int32), 
                                                       window_size = window_size,
                                                       overlap = overlap, 
                                                       dt=1./frame_rate, 
                                                       search_area_size = search_area_size, 
                                                       sig2noise_method = 'peak2peak')
x, y = pyprocess.get_coordinates( image_size = frame_a.shape, search_area_size = search_area_size, overlap = overlap )
flags_g = validation.global_val( u, v, (-300.,300.),(-300.,300.))
flags_s2n = validation.sig2noise_val(sig2noise, threshold = 1.1 )
flags = flags_g | flags_s2n
u, v = filters.replace_outliers( u, v, flags, method='localmean', max_iter = 3, kernel_size = 3)
x, y, u, v = scaling.uniform(x, y, u, v, scaling_factor = 96.52 )
x, y, u, v = tools.transform_coordinates(x, y, u, v)
# save to a file
tools.save(path / 'test.txt', x, y, u, v, flags, fmt='%9.6f', delimiter='\t')
tools.display_vector_field( path / 'test.txt', scale=50, width=0.002)



# masking using not optimal choice of the methods or parameters:
masked_a, _ = preprocess.dynamic_masking(frame_a,method='edges',filter_size=7,threshold=0.005)
masked_b, _ = preprocess.dynamic_masking(frame_b,method='intensity',filter_size=3,threshold=0.005)
plt.imshow(np.c_[masked_a,masked_b],cmap='gray')



# masking using optimal (manually tuned) set of parameters and the right method:
masked_a, _ = preprocess.dynamic_masking(frame_a,method='edges',filter_size=7,threshold=0.01)
masked_b, _ = preprocess.dynamic_masking(frame_b,method='edges',filter_size=7,threshold=0.01)
plt.imshow(np.c_[masked_a,masked_b],cmap='gray')



# Process the masked cropped image and see the OpenPIV result:

# process again with the masked images, for comparison# process once with the original images
u, v, sig2noise = pyprocess.extended_search_area_piv(
                                                       masked_a.astype(np.int32) , masked_b.astype(np.int32), 
                                                       window_size = window_size,
                                                       overlap = overlap, 
                                                       dt=1./frame_rate, 
                                                       search_area_size = search_area_size, 
                                                       sig2noise_method = 'peak2peak')
x, y = pyprocess.get_coordinates( image_size = masked_a.shape, search_area_size = search_area_size, overlap = overlap )
flags_g = validation.global_val( u, v, (-300.,300.),(-300.,300.))
flags_s2n = validation.sig2noise_val( sig2noise, threshold = 1.1)
flags = flags_g | flags_s2n
u, v = filters.replace_outliers( u, v, flags, method='localmean', max_iter = 3, kernel_size = 3)
x, y, u, v = scaling.uniform(x, y, u, v, scaling_factor = 96.52 )
x, y, u, v = tools.transform_coordinates(x, y, u, v)
# save to a file
tools.save(path / 'test_masked.txt', x, y, u, v, flags, None, fmt='%9.6f', delimiter='\t')
tools.display_vector_field( path / 'test_masked.txt', scale=50, width=0.002)

