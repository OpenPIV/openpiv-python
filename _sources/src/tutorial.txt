========
Tutorial
========

This tutorial focuses on the use of the openpiv python module for scripting. This tutorial only shows some of the most commonly used features of
OpenPiv. Check the complete API reference at :ref:`api_reference`

First step is to install OpenPiv. For installation details
on various platforms see :ref:`installation_instruction`.

This tutorial uses some of the example data provided with the source distribution
of OpenPiv that you can find in our `GitHub repository <https://github.com/gasagna/OpenPiv>`_.

First example: how to process an image pair
===========================================

Here is a complete working example showing how to process an image pair.  ::


    import openpiv.tools
    import openpiv.pyprocess
    import openpiv.scaling
    
    frame_a  = openpiv.tools.imread( 'exp1_001_a.bmp' )
    frame_b  = openpiv.tools.imread( 'exp1_001_b.bmp' )
    
    u, v = openpiv.pyprocess.piv( frame_a, frame_b, window_size=48, overlap=32, dt=0.02, sig2noise_lim=1.5 )
    x, y = openpiv.pyprocess.get_coordinates( image_size=frame_a.shape, window_size=48, overlap=32 )
    x, y, u, v = openpiv.scaling.uniform(x, y, u, v, scaling_factor = 1236.6 )
    
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

    

