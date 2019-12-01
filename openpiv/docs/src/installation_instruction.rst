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

<<<<<<< HEAD
* `Python 2.7 or 3.6 <http://python.org/>`_
=======
* `Python <http://python.org/>`_
>>>>>>> 2d4f48449c80e4fc0fd28dbaa87727a03ce5a992
* `Scipy <http://numpy.scipy.org/>`_
* `Numpy <http://www.scipy.org/>`_
* `Cython <http://cython.org/>`_
* `scikit-image <http://scikit-image.org/>`_

<<<<<<< HEAD
The following distributions that include Python with the required libraries are recommended for easy installations of dependencies:
=======
On all platforms, the following Python distribution is recommended:
>>>>>>> 2d4f48449c80e4fc0fd28dbaa87727a03ce5a992

* Anaconda <https://store.continuum.io/cshop/anaconda/>  
<<<<<<< HEAD
* PythonXY <https://code.google.com/p/pythonxy/>  
* WinPython <http://winpython.sourceforge.net/>  

How to install the dependencies on Linux
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
On a Linux platform installing these dependencies should not be tricky. Often, if not always, 
python is installed by default, while the other dependencies should appear in your package
manager. 

Thanks for the issue raised on our Github page, the Ubuntu installation should work as:   

    sudo apt-get install cython python-numpy python-scipy
    
    pip2.7 install OpenPIV


Using Conda 
^^^^^^^^^^^

    conda install -c conda-forge openpiv
 
Should include all the missing packages and automatically build the dependenices. 


How to install the dependencies on Windows
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
On Windows all these dependencies, as well as several other useful packages, can be installed
using one of the aforementioned distributions, e.g. Anaconda, PythonXY. Note: Install it in Custom Directories, 
without spaces in the directory names (i.e. Program Files are prohibited), e.g. `C:\Pythonxy\`


How to install the dependencies on a Mac
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The binary (32 or 64 bit) Enthought Python Distribution (EPD) or Anaconda are recommended.  Note: if you use EPD Free distribution, you need to add and install Cython from http://www.cython.org
=======
>>>>>>> 2d4f48449c80e4fc0fd28dbaa87727a03ce5a992

Installation
============

Use `conda` :: 

    conda install -c conda-forge openpiv

Or use `pip` :: 

    pip install numpy cython
    pip install openpiv --pre
    

In Python 3 the project changed name to `progressbar2` package. Install it separately using `pip`

    pip install progressbar2
    
Or using Conda:   

    conda install progressbar2
    
We will remove this requirement in the future, so don't be surprised it if just works without progressbar. 

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
    

Having problems?
================
If you encountered some issues, found difficult to install OpenPIV following these instructions
please register and write to openpiv-users@googlegroups.com , so that we can help you and 
improve this page!





