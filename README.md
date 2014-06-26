=======
Warning
=======
The OpenPIV python version is currently in alpha state. This means that
it is buggy, untested and the API may change. However testing and contributing
is very welcome, especially if you can contribute with new algorithms and features.

Development is currently done on a Linux/Mac OSX environment, but as soon as possible 
Windows will be tested. If you have access to one of these platforms
please test the code. 

The most updated version is on Alex's Github repository, please follow: 

https://github.com/alexlib/openpiv-python



=======
OpenPIV
=======
OpenPIV consists in a python module for scripting and executing the analysis of 
a set of PIV image pairs. In addition, a Qt graphical user interface is in 
development, to ease the use for those users who don't have python skills.



=======
Install
=======

It is recommended to use Github repository for the latest development branch: 

https://github.com/alexlib/openpiv-python


Installation instructions for various platforms can be found at http://www.openpiv.net/openpiv-python/

Basically we use distutils:

>>> python setup.py build_ext --inplace

should work. 

If you want to try one of the pre-compiled versions, you may try these:

Windows: https://dl.dropboxusercontent.com/u/5266698/OpenPIV/OpenPIV-0.11.win32-py2.7.msi
Mac OS X: https://dl.dropboxusercontent.com/u/5266698/OpenPIV/OpenPIV-0.11.macosx-10.9-intel.zip

We're also listed on PyPI: https://pypi.python.org/pypi/OpenPIV, so you could just try:

>>> pip install openpiv

or 

>>> easy_install openpiv


=============
Documentation
=============

The OpenPiv documentation is available on the project web page at http://www.openpiv.net/openpiv-python/


==================
Try OpenPIV online
==================

No installation is required. Just use this link - open a new account on Wakari.io and you'll have the tutorial in your browser using IPython notebook, based on Numpy/SciPy/Matplotlib/ and our OpenPIV library. 



https://www.wakari.io/sharing/bundle/openpiv/openpiv-python_tutorial




