# OpenPIV
[![Build Status](https://travis-ci.org/OpenPIV/openpiv-python.svg?branch=master)](https://travis-ci.org/OpenPIV/openpiv-python)
[![DOI](https://zenodo.org/badge/4213/OpenPIV/openpiv-python.svg)](https://zenodo.org/badge/latestdoi/4213/OpenPIV/openpiv-python)

OpenPIV consists in a Python and Cython modules for scripting and executing the analysis of 
a set of PIV image pairs. In addition, a Qt graphical user interface is in 
development, to ease the use for those users who don't have python skills.

## Warning

The OpenPIV python version is currently in alpha state. This means that
it is buggy, untested and the API may change. However testing and contributing
is very welcome, especially if you can contribute with new algorithms and features.

Development is currently done on a Linux/Mac OSX environment, but as soon as possible 
Windows will be tested. If you have access to one of these platforms
please test the code. 

## Installing

We are listed on PyPI: <https://pypi.python.org/pypi/OpenPIV>, so you could just try:

    pip install openpiv

or 

    easy_install openpiv

### To build from source

Download the package from the Github: https://github.com/OpenPIV/openpiv-python/archive/master.zip
or clone using git

    git clone https://github.com/OpenPIV/openpiv-python.git

Using distutils create a local (in the same directory) compilation of the Cython files:

    python setup.py build_ext --inplace

Or for the global installation, use:

    python setup.py install 


### Latest developments

Latest developments go into @alexlib repository <https://github.com/alexlib/openpiv-python>

## Documentation

The OpenPIV documentation is available on the project web page at <http://openpiv.readthedocs.org>


## Contributors

1. Alex Liberzon  
2. Roi Gurka  
3. Zachary J. Taylor  
4. David Lasagna  
5. Mathias Aubert

# Tutorial online (zero installation)

Run tutorial on Binder with automatic installation of openpiv (and progressbar):

[![Binder](http://mybinder.org/badge.svg)](http://mybinder.org/repo/alexlib/openpiv-python/openpiv/examples/tutorial-part1/openpiv-python-tutorial-part1.ipynb)

