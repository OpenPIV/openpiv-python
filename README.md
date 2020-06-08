# OpenPIV
[![Build Status](https://travis-ci.org/OpenPIV/openpiv-python.svg?branch=master)](https://travis-ci.org/OpenPIV/openpiv-python)
[![Build status](https://ci.appveyor.com/api/projects/status/4ht2vwvur22jmn6b?svg=true)](https://ci.appveyor.com/project/alexlib/openpiv-python)
[![DOI](https://zenodo.org/badge/4213/OpenPIV/openpiv-python.svg)](https://zenodo.org/badge/latestdoi/4213/OpenPIV/openpiv-python)

[![Anaconda-Server Badge](https://anaconda.org/conda-forge/openpiv/badges/version.svg)](https://anaconda.org/conda-forge/openpiv)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/openpiv/badges/platforms.svg)](https://anaconda.org/conda-forge/openpiv)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/openpiv/badges/license.svg)](https://anaconda.org/conda-forge/openpiv)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/openpiv/badges/downloads.svg)](https://anaconda.org/conda-forge/openpiv)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/openpiv/badges/installer/conda.svg)](https://conda.anaconda.org/conda-forge)


OpenPIV consists in a Python and Cython modules for scripting and executing the analysis of 
a set of PIV image pairs. In addition, a Qt graphical user interface is in 
development, to ease the use for those users who don't have python skills.

## Warning

The OpenPIV python version is still in beta state. This means that
it still might have some bugs and the API may change. However, testing and contributing
is very welcome, especially if you can contribute with new algorithms and features.

Development is currently done on a Linux/Mac OSX environment, but as soon as possible 
Windows will be tested. If you have access to one of these platforms
please test the code. 

## Test it without installation
Click the link - thanks to BinderHub, Jupyter and Conda you can now get it in your browser with zero installation:
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openpiv/openpiv-python/master?filepath=openpiv%2Fexamples%2Fnotebooks%2Ftutorial1.ipynb)




## Installing

Use PyPI: <https://pypi.python.org/pypi/OpenPIV>:

    pip install cython numpy 
    pip install openpiv --pre

`--pre` because sometimes we push pre-releases

## Or `conda` 

    conda install -c conda-forge openpiv
    
    
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

## Demo notebooks 

1. [Tutorial Notebook 1](https://nbviewer.jupyter.org/github/OpenPIV/openpiv-python/blob/master/openpiv/examples/notebooks/tutorial1.ipynb)
2. [Tutorial notebook 2](https://nbviewer.jupyter.org/github/OpenPIV/openpiv-python/blob/master/openpiv/examples/notebooks/tutorial2.ipynb)
3. [Dynamic masking tutorial](https://nbviewer.jupyter.org/github/OpenPIV/openpiv-python/blob/master/openpiv/examples/notebooks/masking_tutorial.ipynb)
4. [Multipass tutorial with WiDiM](https://nbviewer.jupyter.org/github/OpenPIV/openpiv-python/blob/master/openpiv/examples/notebooks/tutorial_multipass.ipynb)
5. [Multipass with Windows Deformation](https://nbviewer.jupyter.org/github/OpenPIV/openpiv-python/blob/master/openpiv/examples/notebooks/window_deformation_comparison.ipynb)
6. [Multiple sets in one notebook](https://nbviewer.jupyter.org/github/OpenPIV/openpiv-python/blob/master/openpiv/examples/notebooks/all_test_cases_sample.ipynb)


## Contributors

1. [Alex Liberzon](http://github.com/alexlib)
2. [Roi Gurka](http://github.com/roigurka)
3. [Zachary J. Taylor](http://github.com/zjtaylor)
4. [David Lasagna](http://github.com/gasagna)
5. [Mathias Aubert](http://github.com/MathiasAubert)
6. [Pete Bachant](http://github.com/petebachant)
7. [Cameron Dallas](http://github.com/CameronDallas5000)
8. [Cecyl Curry](http://github.com/leycec)
9. [Theo Käufer](http://github.com/TKaeufer) 


Copyright statement: `smoothn.py` is a Python version of `smoothn.m` originally created by D. Garcia [https://de.mathworks.com/matlabcentral/fileexchange/25634-smoothn], written by Prof. Lewis and available on Github [https://github.com/profLewis/geogg122/blob/master/Chapter5_Interpolation/python/smoothn.py]. We include a version of it in the `openpiv` folder for convenience and preservation. We are thankful to the original authors for releasing their work as an open source. OpenPIV license does not relate to this code. Please communicate with the authors regarding their license. 

## How to cite this work
OpenPIV/openpiv-python:  http://doi.org/10.5281/zenodo.3566451


