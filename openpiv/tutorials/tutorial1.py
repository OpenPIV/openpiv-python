from importlib_resources import files
import numpy as np
from openpiv import tools, pyprocess, scaling, validation, filters

# we can run it from any folder
path = files('openpiv') / "data" / "test1" 


frame_a  = tools.imread( path / "exp1_001_a.bmp" )
frame_b  = tools.imread( path / "exp1_001_b.bmp" )

frame_a = frame_a.astype(np.int32)
frame_b = frame_b.astype(np.int32)

u, v, sig2noise = pyprocess.extended_search_area_piv( frame_a, frame_b, \
    window_size=32, overlap=16, dt=1., search_area_size=64, sig2noise_method='peak2peak' )

print(u,v,sig2noise)

x, y = pyprocess.get_coordinates( image_size=frame_a.shape, search_area_size=64, overlap=16 )
flags_s2n = validation.sig2noise_val(sig2noise, threshold = 1.2 )
flags_g = validation.global_val( u, v, (-10, 10), (-10, 10) )
flags = flags_s2n | flags_g

u, v = filters.replace_outliers( u, v, flags, method='localmean', max_iter=10, kernel_size=2)
x, y, u, v = scaling.uniform(x, y, u, v, scaling_factor = 96.52 )
x, y, u, v = tools.transform_coordinates(x, y, u, v)
tools.save(str(path / 'test_data.vec') , x, y, u, v, flags)
# tools.display_vector_field(path / 'test_data.vec', scale=75, width=0.0035)
tools.display_vector_field(
    str(path / 'test_data.vec'), 
    scale=1, 
    scaling_factor=96.52,
    width=0.0035,
    on_img=True,
    image_name = str(path / "exp1_001_a.bmp")
)