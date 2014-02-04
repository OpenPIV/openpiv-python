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

* `python <http://python.org/>`_
* `scipy <http://numpy.scipy.org/>`_
* `numpy <http://www.scipy.org/>`_
* `cython <http://cython.org/>`_

On all platforms, the binary Enthought Python Distribution (EPD) is recommended. 
Visit http://www.enthought.com

How to install the dependencies on Linux
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
On a Linux platform installing these dependencies should be trick. Often, if not always, 
python is installed by default, while the other dependencies should appear in your package
manager.

How to install the dependencies on Windows
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
On Windows all these dependencies, as well as several other useful packages, can be installed
using the Python(x,y) distribution, available at http://www.pythonxy.com/. Note: Install it in Custom Directories, 
without spaces in the directory names (i.e. Program Files are prohibited), e.g. C:\Pythonxy\


How to install the dependencies on a Mac
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The binary (32 or 64 bit) Enthought Python Distribution (EPD) is recommended.  Visit http://www.enthought.com. However, if you use EPD Free distribution, you need to install Cython from http://www.cython.org




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

    git clone http://github.com/alexlib/openpiv-python.git

and update from time to  time. You can also download a tarball containing everything.

Then add the path where the OpenPIV source are to the PYTHONPATH environment variable, so 
that OpenPIV module can be imported and used in your programs. Remeber to build the extension
with :: 

    python setup.py build 

.. Stable source distribution
.. ^^^^^^^^^^^^^^^^^^^^^^^^^^
.. If you do not want to follow the development of OpenPIV and you prefer a more stable
.. version, download the source distributions available at http://www.openpiv.sourceforge.net,
.. in the downloads page. Then unpack it and execute the following command::

..    python setupy.py install --prefix=$DIR
    
.. where ``$DIR`` is the folder you want ot install OpenPIV in. If you want to install it system
.. wide omit the ``--prefix`` option, but you should have root priviles to do so. Remember to 
.. update the PYTHONPATH environment variable if you used a custom installation directory.


.. Download pre-built binary distributions
.. =======================================

.. For Windows we provide pre-built distributions which can be used without the hassles
.. of compilation and other boring things you may not want to dig into. This is currently a work
.. in progress. Check back soon!



Having problems?
================
If you encountered some issues, found difficult to install OpenPIV following these instructions
please drop us an email to openpiv-develop@lists.sourceforge.net , so that we can help you and 
improve this page!





