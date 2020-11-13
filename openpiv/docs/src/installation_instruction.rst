.. _installation_instruction:

========================
Installation instruction
========================

.. _dependencies:

Dependencies
============

OpenPIV would not have been possible if other great open source projects did not
exist. We make extensive use of code and tools that other people have created, so 
you should install them before you can use OpenPIV.

The dependencies are:


* `Python <http://python.org/>`_
* `Scipy <http://numpy.scipy.org/>`_
* `Numpy <http://www.scipy.org/>`_
* `Cython <http://cython.org/>`_
* `scikit-image <http://scikit-image.org/>`_

On all platforms, the following Python distribution is recommended:

* Anaconda <https://store.continuum.io/cshop/anaconda/>   


Installation
============

Use `conda` :: 

    conda install -c conda-forge openpiv

Or use `pip` :: 

    pip install numpy cython
    pip install openpiv --pre
    
Get OpenPIV source code!
========================

At this moment the only way to get OpenPIV's source code is using git. 
`Git <http://en.wikipedia.org/wiki/Git_%28software%29>`_ Git is a distributed revision control system and 
our code is hosted at `GitHub <www.github.com>`_.

Bleeding edge development version
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you are interested in the source code you are welcome to browse out git repository
stored at https://github.com/alexlib/openpiv-python. If you want to download the source code
on your machine, for testing, you need to set up git on your computer. Please look at 
http://help.github.com/ which provide extensive help for how to set up git.

To follow the development of OpenPIV, clone our repository with the command::

    git clone http://github.com/openpiv/openpiv-python.git

and update from time to  time. You can also download a tarball containing everything.

Then add the path where the OpenPIV source are to the PYTHONPATH environment variable, so 
that OpenPIV module can be imported and used in your programs. Remeber to build the extension
with :: 

    python setup.py build_ext --inplace 
    

Experience problems?
====================
If you encountered some issues, found difficult to install OpenPIV following these instructions
please register and write on our Google groups forum https://groups.google.com/g/openpiv-users , so that we can help you and 
improve this page!





