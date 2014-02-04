========
Tutorial
========

This is a series of examples and tutorials which focuses on showing features and capabilities of OpenPIV, so that after reading you should be able to set up scripts for your own analyses. If you are looking for a complete reference to the OpenPiv api, please look at :ref:`api_reference`. It is assumed that you have Openpiv installed on your system along with a working python environment as well as the necessary :ref:`OpenPiv dependencies <dependencies>`. For installation details on various platforms see :ref:`installation_instruction`. 


In this tutorial we are going to use some example data provided with the source distribution of OpenPIV. Altough it is not necessary, you may find helpful to actually run the code examples as the tutorial progresses. If you downloaded a tarball file, you should find these examples under the directory openpiv/docs/examples. Similarly if you cloned the git repository. If you cannot find them, dowload example images as well as the python source code from the :ref:`downloads <downloads>` page.


First example: how to process an image pair
===========================================

The first example shows how to process a single image pair. This is a common task and may be useful if you are studying how does a certain algorithm behaves. We assume that the current working directory is where the two image of the first example are located. Here is the code::


    import openpiv.tools
    import openpiv.process
    import openpiv.scaling
    
    frame_a  = openpiv.tools.imread( 'exp1_001_a.bmp' )
    frame_b  = openpiv.tools.imread( 'exp1_001_b.bmp' )
    
    u, v, sig2noise = openpiv.process.extended_search_area_piv( frame_a, frame_b, window_size=24, overlap=12, dt=0.02, search_area_size=64, sig2noise_method='peak2peak' )
    
    x, y = openpiv.process.get_coordinates( image_size=frame_a.shape, window_size=24, overlap=12 )
    
    u, v, mask = openpiv.validation.sig2noise_val( u, v, sig2noise, threshold = 1.3 )
    
    u, v = openpiv.filters.replace_outliers( u, v, method='localmean', n_iter=10, kernel_size=2)
    
    x, y, u, v = openpiv.scaling.uniform(x, y, u, v, scaling_factor = 96.52 )
    
    openpiv.tools.save(x, y, u, v, 'exp1_001.txt' )
    
This code can be executed as a script, or you can type each command in an `Ipython <http://ipython.scipy.org/moin/>`_ console with pylab mode set, so that you can visualize result as they are available.  I will follow the second option and i will present the results of each command.
    
We first import some of the openpiv modules.::

    import openpiv.tools
    import openpiv.process
    import openpiv.scaling
    
Module ``openpiv.tools`` contains mostly contains utilities and tools, such as file I/O and multiprocessingvfacilities. Module ``openpiv.process`` contains advanced algorithms for PIV analysis and several helper functions. Last, module ``openpiv.scaling`` contains functions for field scaling.

We then load the two image files into numpy arrays::

    frame_a  = openpiv.tools.imread( 'exp1_001_a.bmp' )
    frame_b  = openpiv.tools.imread( 'exp1_001_b.bmp' )
    
Inspecting the attributes of one of the two images we can see that::

    frame_a.shape
    (369, 511)
    
    frame_a.dtype
    dtype('int32')
    
image has a size of 369x511 pixels and are contained in 32 bit integer arrays. Using pylab graphical capabilities it is easy to visualize  one of the two frames:::

    matshow ( frame_a, cmap=cm.Greys _r ) 
    
which results in this figure.

.. image:: ../images/image1.png
   :height: 500px
   :align: center
    
In this example we are going to use the function :py:func:`openpiv.process.extended_search_area_piv` to process the image pair.::

        u, v, sig2noise = openpiv.process.extended_search_area_piv( frame_a, frame_b, window_size=24, overlap=12, dt=0.02, search_area_size=64, sig2noise_method='peak2peak' )

      
This method  is a zero order displacement predictor cross-correlation algorithm, which cope with the problem of loss of pairs when the interrogation window is small, by increasing the search area on the second image. We also provide some options to the function, namely the ``window_size``, i.e. the size of the interrogation window  on ``frame_a``, the ``overlap`` in pixels between adjacent windows, the time delay in seconds ``dt`` between  the two image frames an te size in pixels of the extended search area on ``frame_b``. ``sig2noise_method`` specifies which method to use for the evalutaion of the signal/noise ratio. The function also returns a third array, ``sig2noise`` which contains the signal to noise ratio obtained from each cross-correlation function, intended as the ratio between the heigth of the first and second peaks.

We then compute the coordinates of the centers of the interrogation windows using :py:func:`openpiv.process.get_coordinates`.::

    x, y = openpiv.process.get_coordinates( image_size=frame_a.shape, window_size=48, overlap=32 )
    
Note that we have provided some the same options we have given in the previous command to the processing function.

We can now plot the vector plot on a new figure to inspect the result of the analysis, using::

    close()
    quiver( x, y, u, v )
 
and we obtain:

.. image:: ../images/image2.png
   :height: 500px
   :align: center

Several outliers vectors can be observed as a result of the small interrogation window size and we need to apply a validation scheme. Since we have information about the signal to noise ratio of the cross-correlation function we can apply a well know filtering scheme, classifing a vector as an outlier if its signal to noise ratio exceeds a certain threshold. To accomplish this task we use the function::

    u, v, mask = openpiv.validation.sig2noise_val( u, v, sig2noise, threshold = 1.3 )
    
with a threshold value set to ``1.3``. This function actually sets to NaN all those vector for which the signal to noise ratio is below 1.3. Therefore, the
arrays ``u`` and ``v`` contains some np.nan elements. Furthermore, we get in output a third variable ``mask``, which is a boolean array where elements corresponding to invalid vectors have been replace by Nan. The result of the filtering is shown in the following image, which we obtain with the two commands::

    figure()
    quiver( x, y, u, v ) 

.. image:: ../images/image3.png
   :height: 500px
   :align: center

The final step is to replace the missing vector. This is done which the function :py:func:`openpiv.filters.replace_outliers`, which implements an iterative image inpainting algorithm with a specified kernel. We pass to this function the two velocity components arrays,  a method type ``localmean``, the number of passes and the size of the kernel.::

    u, v = openpiv.filters.replace_outliers( u, v, method='localmean', n_iter=10, kernel_size=2 )
    
The flow field now appears much more smooth and the outlier vectors have been correctly replaced. ::

    figure()
    quiver( x, y, u, v ) 
    
.. image:: ../images/image4.png
   :height: 500px
   :align: center



The last step is to apply an uniform scaling to the flow field to get dimensional units. We use the function :py:func:`openpiv.scaling.uniform` providing the ``scaling_factor`` value, in pixels per meters if we want position and velocities in meters and meters/seconds or in pixels per millimeters if we want positions and velocities in millimeters and millimeters/seconds, respectively. ::

    x, y, u, v = openpiv.scaling.uniform(x, y, u, v, scaling_factor = 96.52 )

Finally we save the data to an ascii file, for later processing, using:::

    openpiv.tools.save(x, y, u, v, 'exp1_001.txt')


Second example: how to process in batch a list of image pairs.
==============================================================

It if often the case, where several hundreds of image pairs have been sampled in an experiment and have to be processed. For these tasks it is easier to launch the analysis in batch and process all the image pairs with the same processing parameters. OpenPiv, with its powerful python scripting capabilities, provides a convenient way to accomplish this task and offers multiprocessing facilities for machines which have multiple cores, to speed up the computation. Since the analysis is an embarassingly parallel problem, the speed up that can be reached is quite high and almost equal to the number of core your machine has.

Compared to the previous example we have to setup some more things in the python script we will use for the batch processing. 

Let's first import the needed modules.::

  import openpiv.tools
  import openpiv.scaling
  import openpiv.process
  
We then define a python function which will be excecuted for each image pair. In this function we can specify any operation to execute on each single image pair, but here, for clarity we will setup a basic analysis, without a validation/replacement step.

Here is an example of valid python function:::

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
        u, v = openpiv.process.extended_search_area_piv( frame_a, frame_b, window_size=32, overlap=16, dt=0.02, search_area_size=64 )
        
        # get window centers coordinates
        x, y = openpiv.process.get_coordinates( image_size=frame_a.shape, window_size=32, overlap=16 )
        
        # save to a file
        openpiv.tools.save(x, y, u, v, 'exp1_%03d.txt' % counter, fmt='%8.7f', delimiter='\t' )
        
The function we have specified *must* accept in input a single argument. This argument is a three element tuple, which you have to unpack inside the function body as we have done with::

    file_a, file_b, counter = args

The tuple contains the two filenames of the image pair and a counter, which is needed to remember which image pair we are currently processing, (basically just for the output filename). After that you have unpacked the tuple into its three elements, you can use them to load the images and do the rest.

The *simple* processing function we wrote is just half of the job. We still need to specify which image pairs to process and where they are located. Therefore, in the same script we add the following two lines of code.::

    task = openpiv.tools.Multiprocesser( data_dir = '.', pattern_a='2image_*0.tif', pattern_b='2image_*1.tif' )
    task.run( func = func, n_cpus=8 )
    
where we have set datadir to ``.`` because the script and the images are in the same folder. The first line creates an instance of the :py:func:`openpiv.tools.Multiprocesser` class. This class is responsible of sharing the processing work to multiple processes, so that the analysis can be executed in parallell. To construct the class you have to pass it three arguments: 

* ``data_dir``: the directory where image files are located
* ``pattern_a`` and ``pattern_b``: the patterns for matching image files for frames `a` and `b`.


.. note::
    Variables ``pattern_a`` and ``pattern_b`` are shell globbing patterns. Let 's say we have thousands of files for frame `a` in a sequence like file0001-a.tif, file0002-a.tif, file0003-a.tif, file0004-a.tif, ..., and the same for frames `b` file0001-b.tif, file0002-b.tif, file0003-b.tif, file0004-b.tif. To match these files we would set ``pattern_a = file*-a.tif`` and ``pattern_b = file*-a.tif``. Basically, the `*` is a wildcard to match 0001, 0002, 0003, ...
    

The second line actually launches the batch process, using for each image pair the ``func`` function we have provided. Note that we have set the ``n_cpus`` option to be equal to ``8`` just because my machine has eight cores. You should not set ``n_cpus`` higher than the number of core your machine has, because you would not get any speed up.




    

