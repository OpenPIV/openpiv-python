=======
Warning
=======
The OpenPIV Python version is still in its beta. This means that
it is still buggy, untested and the API may change. However testing and contributing
is very welcome, especially if you can contribute with new algorithms and features.

Development is currently done on a Linux/Mac OSX environment, but as soon as possible 
Windows will be tested. If you have access to one of these platforms
please test the code. 


=======
OpenPIV
=======
OpenPIV consists in a python module for scripting and executing the analysis of 
a set of PIV image pairs. In addition, a Qt graphical user interface is in 
development, to ease the use for those users who don't have python skills.


=======
Install
=======
Installation instructions for various platforms can be found at http://openpiv.readthedocs.org

The easiest way is to use pip:  

   >>> pip install -U openpiv
   
Or to download the package from the Github: https://github.com/alexlib/openpiv-python/archive/master.zip

Using distutils create a local (in the same directory) compilation of the Cython files:

    >>> python setup.py build_ext --inplace

Or for the global installation, use:

	>>> python setup.py install 


=============
Documentation
=============

The OpenPIV documentation is available on the project web page at http://openpiv.readthedocs.org


=============
Contributors
=============

1. Alex Liberzon  
2. Roi Gurka  
3. Zachary J. Taylor  
4. David Lasagna  
5. Mathias Aubert
