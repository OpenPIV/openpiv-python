# OpenPIV
[![Build Status](https://travis-ci.org/OpenPIV/openpiv-python.svg?branch=master)](https://travis-ci.org/OpenPIV/openpiv-python)
[![Build status](https://ci.appveyor.com/api/projects/status/4ht2vwvur22jmn6b?svg=true)](https://ci.appveyor.com/project/alexlib/openpiv-python)
[![DOI](https://zenodo.org/badge/4213/OpenPIV/openpiv-python.svg)](https://zenodo.org/badge/latestdoi/4213/OpenPIV/openpiv-python)

OpenPIV consists in a Python and Cython modules for scripting and executing the analysis of 
a set of PIV image pairs. In addition, a Qt graphical user interface is in 
development, to ease the use for those users who don't have python skills.

## Warning

The OpenPIV python version is still in beta state. This means that
it still might have some bugs and the API may change. However testing and contributing
is very welcome, especially if you can contribute with new algorithms and features.

Development is currently done on a Linux/Mac OSX environment, but as soon as possible 
Windows will be tested. If you have access to one of these platforms
please test the code. 

## Test it without installation
Click the link - thanks to BinderHub, Jupyter and Conda you can now get it in your browser with zero installation:
[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/openpiv/openpiv-python-example/master?filepath=index.ipynb)


## Installing

You can use Conda <http://conda.io>:  

    conda install -c conda-forge openpiv

We are listed on PyPI: <https://pypi.python.org/pypi/OpenPIV>, so you could just try:

    pip install openpiv

or 

    easy_install openpiv
    
Note that if dependicies of Numpy, Scipy and Cython are not present, on Ubuntu Linux, install those as:

    sudo apt-get install cython python-numpy python-scipy
    pip2.7 install OpenPIV

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

1. [Alex Liberzon](http://github.com/alexlib)
2. [Roi Gurka](http://github.com/roigurka)
3. [Zachary J. Taylor](http://github.com/zjtaylor)
4. [David Lasagna](http://github.com/gasagna)
5. [Mathias Aubert](http://github.com/MathiasAubert)
6. [Pete Bachant](http://github.com/petebachant)
7. Cameron Dallas (http://github.com/CameronDallas5000)
8. Cecyl Curry (http://github.com/leycec)

