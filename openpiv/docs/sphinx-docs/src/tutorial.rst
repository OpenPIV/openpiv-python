========
Tutorial
========

This is a series of examples and tutorials which focuses on showing features and capabilities of OpenPiv, so that after reading you should be able to set up scripts for 
your own analyses. If you are looking for a complete reference to the OpenPiv api, please look at :ref:`api_reference`. It is assumed that you have Openpiv installed on your system
along with a working python environment as well as the necessary :ref:`OpenPiv dependencies <dependencies>`. For installation details on various platforms see :ref:`installation_instruction`.



In this tutorial we are going to use some example data provided with the source distribution of OpenPiv. Altough it is not necessary, you may find helpful to actually run 
the code examples as the tutorial progresses. If you downloaded a tarball file, you should find these examples under the directory openpiv/docs/examples. Similarly if you cloned the git repository.
If you cannot find them, dowload example images as well as the python source code from the :ref:`downloads <downloads>` page.


First example: how to process an image pair
===========================================

The first example shows how to process a single image pair. This is a common task and may be useful if you are studying how does a certain
algorithm behaves. We assume that the current working directory is where the two image of the first example are located. Here is the code::


    import openpiv.tools
    import openpiv.pyprocess
    import openpiv.scaling
    
    frame_a  = openpiv.tools.imread( 'exp1_001_a.bmp' )
    frame_b  = openpiv.tools.imread( 'exp1_001_b.bmp' )
    
    u, v, sig2noise = openpiv.process.extended_search_area_piv( frame_a, frame_b, window_size=24, overlap=12, dt=0.02, search_area_size=64 )
    
    u, v = opepiv.validate.sig2noise_val( u, v, sig2noise, threshold = 1.2 )
    
    u, v = openpiv.filters.replace_outliers( u, v, method='localmean', n_iter=10, kernel_size=2)
    
    x, y = openpiv.pyprocess.get_coordinates( image_size=frame_a.shape, window_size=48, overlap=32 )
    
    x, y, u, v = openpiv.scaling.uniform(x, y, u, v, scaling_factor = 96.52 )
    
    openpiv.tools.save(x, y, u, v, 'exp1_001.txt')
    
    
We first import some of the openpiv modules.::

    import openpiv.tools
    import openpiv.pyprocess
    import openpiv.scaling
    
Module ``openpiv.tools`` contains mostly contains utilities and tools, such as file I/O and multiprocessing
facilities. Module ``openpi.pyprocess`` contains a pure Python implementation of the PIV cross-correlation
algorithm and several helper functions. Last, module ``openpiv.scaling`` contains function for field scaling
and plate calibration stuff.

We then load the two image files into numpy arrays::

    frame_a  = openpiv.tools.imread( 'exp1_001_a.bmp' )
    frame_b  = openpiv.tools.imread( 'exp1_001_b.bmp' )
    
In this example we use the pure python implementation to get the velocity field from the image pair.::

    u, v = openpiv.pyprocess.piv( frame_a, frame_b, window_size=48, overlap=32, dt=0.02, sig2noise_lim=1.5 )
      
The function :py:func:`openpiv.pyprocess.piv` is a python implementation of the standard cross-correlation 
algorithm. We also provide some options to the function, namely the ``window_size``, i.e. the size of the
interrogation windows, the ``overlap`` between the windows in pixels and the time delay in seconds ``dt`` between 
the two image frames. ``sig2noise_lim`` is the lower limit for the signal to noise ratio accepted before a vector is considered
an outlier.

We then compute the coordinates of the centers of the interrogation windows using :py:func:`openpiv.pyprocess.get_coordinates`.::

    x, y = openpiv.pyprocess.get_coordinates( image_size=frame_a.shape, window_size=48, overlap=32 )
    
Note that we have provided some the same options we have given in the previuos command.

Then we apply an uniform scaling with the function :py:func:`openpiv.scaling.uniform` providing the ``scaling_factor`` value, in pixels per meters
if we want position and velocities in meters and meters/seconds or in pixels per millimeters if we want positions and velocities in millimeters and millimeters/seconds, respectively. ::

    x, y, u, v = openpiv.scaling.uniform(x, y, u, v, scaling_factor = 1236.6 )

Finally we save the data to an ascii file, for later processing, using:::

    openpiv.tools.save(x, y, u, v, 'exp1_001.txt')


Second example: how to process in batch a list of image pairs.
=================================================================

It if often the case, where several hundreds of image pairs have been sampled
in an experiment and have to be processed. For these tasks it is easier to 
launch the analysis in batch and process all the image pairs 
with the same processing parameters. OpenPiv, with its powerful python 
scripting capabilities, provides a convenient way to 
accomplish this task and offers multiprocessing facilities for machines
which have multiple cores, to speed up the computation. Since the analysis 
is an embarassingly parallel problem, the speed up that can be reached 
is quite high and almost equal to the number of core your machine has.

Compared to the previous example we have to setup some more things in the python
script we will use for the batch processing.

Let's first import the needed modules.::

  import openpiv.tools
  import openpiv.scaling
  import openpiv.pyprocess
  
We then define a python function which will be excecuted for each image pair.
Here it is:::

    def func( args ):
        """A function to process each image pair."""
        
        # this line is REQUIRED for the multiprocessing to work
        # always use it in your custom function

        file_a, file_b, counter = args
        
        
        #####################
        # Here goes you code
        #####################
        
        # read images into numpy arrays
        frame_a  = openpiv.tools.imread( file_a )
        frame_b  = openpiv.tools.imread( file_b )
            
        # process image pair with the purepython implementation
        u, v = openpiv.process.extended_search_area_piv( frame_a, frame_b, window_size=24, overlap=12, dt=0.02, search_area_size=24*3 )
        
        # get window centers coordinates
        x, y = openpiv.pyprocess.get_coordinates( image_size=frame_a.shape, window_size=24, overlap=12 )
        
        # get flow field in dimensional units: 1236.6 are pixels per millimiters so x, y, u, v will be in millimeters and millimeters/seconds
        x, y, u, v = openpiv.scaling.uniform(x, y, u, v, scaling_factor = 16.7 )
        
        # save to a file
        openpiv.tools.save(x, y, u, v, 'exp1_%03d.txt' % counter, fmt='%8.7f', delimiter='\t' )
        
The function we have written *must* accept in input a single argument. This argument is a three element tuple, which 
you have to unpack in the function as we have done with::

    file_a, file_b, counter = args

The tuple contains the two filenames of the image pair and a counter, which is needed to remember which image pair 
we are currently processing, (basically just for the output filename). After that you have unpacked the tuple into
its three elements, you can use them to load the images and do the rest.

This is just half of the job. In the same script we are going to write the following two lines of code.::

    task = openpiv.tools.Multiprocesser( data_dir = '/home/User/images', pattern_a='2image_*0.tif', pattern_b='2image_*1.tif' )
    task.run( func = func, n_cpus=8 )
    
The first line creates an instance of the :py:func:`Openpiv.tools.Multiprocesser` class. To construct the class
you have to pass three arguments: 

* ``data_dir``: the directory where image files are located
* ``pattern_a`` and ``pattern_b``: the patterns for matching image files for frames `a` and `b`.

The second line actually launch the batch process, using for each image pair the ``func`` function we have provided. Note that we have set the ``n_cpus`` option
to be equal to ``8`` becasue my machine has eight core. You should not set ``n_cpus`` higher than the number of 
core your machine has, becasue you don't get any speed up.




    

