import sys
import glob
import numpy 

from setuptools import setup
from setuptools.extension import Extension
# we do not need Cython if we distribute C files
# from Cython.Build import cythonize


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


setup(  name = "OpenPIV",
        version="0.21.2c",
        author = "OpenPIV contributors",
        author_email = "openpiv-users@googlegroups.com",
        description = "An open source software for PIV data analysis",
        license = "GNU General Public License v3 (GPLv3)",
        url = "http://www.openpiv.net",
        long_description =  """OpenPIV is a set of open source algorithms and methods
                            for the state-of-the-art experimental tool
                            of Particle Image Velocimetry (PIV) which 
                            are free, open, and easy to operate.""",
        # ext_modules=cythonize("openpiv/*.pyx", include_path=[numpy.get_include()]),
        ext_modules=[
        Extension("process", ["openpiv/process.c"],
                  include_dirs=[numpy.get_include()]),
        Extension("lib", ["openpiv/lib.c"],
                  include_dirs=[numpy.get_include()]),
    ],
        include_dirs=[numpy.get_include()],
        packages = ['openpiv'],
        data_files = data_files,
        install_requires = ['numpy','scipy','cython','scikit-image >= 0.12.0','progressbar2 >= 3.8.1',\
            'pygments','future'],
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

