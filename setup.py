import sys
import glob
import numpy

try:
    from setuptools import setup
    from setuptools.extension import Extension
except ImportError:
    print("Couldn't import setuptools. Falling back to distutils.")
    from distutils.core import setup

#
# Force `setup_requires` stuff like Cython to be installed before proceeding
#
from setuptools.dist import Distribution
Distribution(dict(setup_requires='Cython'))

try:
    from Cython.Distutils import build_ext
except ImportError:
    print("Could not import Cython.Distutils. Install `cython` and rerun.")
    sys.exit(1)
    


# from distutils.core import setup, Extension
# from Cython.Distutils import build_ext



# Build extensions 
module1 = Extension(    name         = "openpiv.process",
                        sources      = ["openpiv/src/process.pyx"],
                        include_dirs = [numpy.get_include()],
                    )
                    
module2 = Extension(    name         = "openpiv.lib",
                        sources      = ["openpiv/src/lib.pyx"],
                        include_dirs = [numpy.get_include()],
                    )

module_test = Extension("hello", ["hello.pyx"])

# a list of the extension modules that we want to distribute
ext_modules = [module1, module2, module_test]


# Package data are those filed 'strictly' needed by the program
# to function correctly.  Images, default configuration files, et cetera.
package_data =  [ 'data/defaults-processing-parameters.cfg', 
                  'data/ui_resources.qrc',
                  'data/images/*.png',
                  'data/icons/*.png',
                ]


# data files are other files which are not required by the program but 
# we want to ditribute as well, for example documentation.
data_files = [ ('openpiv/openpiv/tutorial-part1', glob.glob('openpiv/openpiv/tutorial-part1/*') ),
               ('openpiv/openpiv/masking_tutorial', glob.glob('openpiv/openpiv/masking_tutorial/*') ),
               ('openpiv/docs/openpiv/examples/example1', glob.glob('openpiv/docs/examples/example1/*') ),
               ('openpiv/docs/openpiv/examples/gurney-flap', glob.glob('openpiv/docs/examples/gurney-flap/*') ),
               ('openpiv/docs/openpiv', ['README.md'] ),
               ('openpiv/data/ui', glob.glob('openpiv/data/ui/*.ui') ),
             ]


# packages that we want to distribute. THis is how
# we have divided the openpiv package.
packages = ['openpiv', 'openpiv.ui']


# script are executable files that will be run to load the gui or something else.
scripts = ['openpiv/tutorial-part1/tutorial-part1.py', 'openpiv/masking_tutorial/masking_tutorial.py']




setup(  name = "OpenPIV",
        version = "0.19",
        author = "The OpenPIV contributors",
        author_email = "openpiv2008@gmail.com",
        description = "An open source software for PIV data analysis",
        license = "GPL v3",
        url = "http://www.openpiv.net",
        long_description =  """OpenPIV is a set of open source algorithms and methods
                            for the state-of-the-art experimental tool
                            of Particle Image Velocimetry (PIV) which 
                            are free, open, and easy to operate.""",
                            
        ext_modules = ext_modules, 
        packages = packages,
        cmdclass = {'build_ext': build_ext},
        scripts = scripts,
        package_data = {'': package_data},
        data_files = data_files,
        setup_requires = ['Cython'],
        test_suite='test',
        )

