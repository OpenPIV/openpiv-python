import openpiv.tools
import openpiv.scaling
import openpiv.process

def func( args ):
    """A function to process each image pair."""

    # this line is REQUIRED for multiprocessing to work
    # always use it in your custom function

    file_a, file_b, counter = args


    #####################
    # Here goes you code
    #####################

    # read images into numpy arrays
    frame_a  = openpiv.tools.imread( file_a )
    frame_b  = openpiv.tools.imread( file_b )

    # process image pair with extended search area piv algorithm.
    u, v, sig2noise = openpiv.process.extended_search_area_piv( frame_a, frame_b, window_size=32, overlap=16, dt=0.02, search_area_size=48, sig2noise_method='peak2peak')


    u, v, mask = openpiv.validation.sig2noise_val( u, v, sig2noise, threshold = 1.3 )

    u, v = openpiv.filters.replace_outliers( u, v, method='localmean', max_iter=10, kernel_size=2)

    # get window centers coordinates
    x, y = openpiv.process.get_coordinates( image_size=frame_a.shape, window_size=32, overlap=16 )

    # save to a file
    openpiv.tools.save(x, y, u, v, mask, 'exp1_%03d.txt' % counter)
    
    openpiv.tools.display_vector_field('exp1_%03d.txt' % counter)

task = openpiv.tools.Multiprocesser( data_dir = '.', pattern_a='2image_*0.tif', pattern_b='2image_*1.tif' )
task.run( func = func, n_cpus=1 )


