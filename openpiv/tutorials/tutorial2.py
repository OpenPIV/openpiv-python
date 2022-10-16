""" Tutorial of using window deformation multi-pass """
from importlib_resources import files
from openpiv import tools, pyprocess, validation, filters


def func( args ):
    """A function to process each image pair."""

    # this line is REQUIRED for multiprocessing to work
    # always use it in your custom function

    file_a, file_b, counter = args

    # read images into numpy arrays
    frame_a  = tools.imread( path / file_a )
    frame_b  = tools.imread( path / file_b )

    # process image pair with extended search area piv algorithm.
    u, v, sig2noise = pyprocess.extended_search_area_piv( frame_a, frame_b, \
        window_size=64, overlap=32, dt=0.02, search_area_size=128, sig2noise_method='peak2peak')
    u, v, mask = validation.sig2noise_val( u, v, sig2noise, threshold = 1.5 )
    u, v = filters.replace_outliers( u, v, method='localmean', max_iter=10, kernel_size=2)
    # get window centers coordinates
    x, y = pyprocess.get_coordinates( image_size=frame_a.shape, search_area_size=128, overlap=32 )
    # save to a file
    tools.save(x, y, u, v, mask, path / f'test2_{counter:03d}.txt' )
    tools.display_vector_field( path / f'test2_{counter:03d}.txt' )



path = files('openpiv') / "data" / "test2"
task = tools.Multiprocesser(
    data_dir = path, 
    pattern_a='2image_*0.tif',
    pattern_b='2image_*1.tif')

task.run( func = func, n_cpus=2 )


