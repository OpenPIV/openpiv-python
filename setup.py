import sys
import glob
import numpy


try:
    from setuptools import setup
    from setuptools import Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension
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

# a list of the extension modules that we want to distribute
ext_modules = [module1, module2]


# data files are other files which are not required by the program but 
# we want to ditribute as well, for example documentation.
data_files = [('test1',glob.glob('openpiv/examples/test1/*')),
               ('test2',glob.glob('openpiv/examples/test2/*')),
               ('test3',glob.glob('openpiv/examples/test3/*')),
               ('test4',glob.glob('openpiv/examples/test4/*')),
               ('notebooks',glob.glob('openpiv/examples/notebooks/*')),
               ('tutorials',glob.glob('openpiv/examples/tutorials/*'))]
# [ ('test', [glob.glob('openpiv/examples/test1/*')]),
               # ('readme', ['README.md']),
            # ]


# packages that we want to distribute. THis is how
# we have divided the openpiv package.


setup(  name = "OpenPIV",
        version="0.21.3",
        author = "OpenPIV contributors",
        author_email = "openpiv-users@googlegroups.com",
        description = "An open source software for PIV data analysis",
        license = "GNU General Public License v3 (GPLv3)",
        url = "http://www.openpiv.net",
        long_description =  """OpenPIV is a set of open source algorithms and methods
                            for the state-of-the-art experimental tool
                            of Particle Image Velocimetry (PIV) which 
                            are free, open, and easy to operate.""",
                            
        ext_modules = ext_modules, 
        packages = ['openpiv'],
        cmdclass = {'build_ext': build_ext},
        data_files = data_files,
        install_requires = ['scipy','numpy','cython','scikit-image >= 0.12.0','progressbar2 >= 3.8.1'],
        classifiers = [
        # PyPI-specific version type. The number specified here is a magic constant
        # with no relation to this application's version numbering scheme. *sigh*
        'Development Status :: 4 - Beta',

        # Sublist of all supported Python versions.
        'Programming Language :: Python :: 3.7',

        # Sublist of all supported platforms and environments.
        'Environment :: Console',
        'Environment :: MacOS X',
        'Environment :: Win32 (MS Windows)',
        'Environment :: X11 Applications',

        # Miscellaneous metadata.
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering',
    ]
)

