import os

# from setuptools import setup, find_packages
# from setuptools.extension import Extension

from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy


extensions = [
    Extension("openpiv.process",["./openpiv/process.pyx"],include_dirs = [numpy.get_include()]),
    Extension("openpiv.lib",["./openpiv/lib.pyx"], include_dirs = [numpy.get_include()])
    ]

extensions = cythonize(extensions,include_path = [numpy.get_include()])

setup(
    name = "OpenPIV",
    version ='0.21.5',
    cmdclass = {'build_ext': build_ext},
    ext_modules = extensions,
    # packages=find_packages(),
    include_package_data=True,
    setup_requires=[
        'setuptools',
        'cython>=0.29.14',
        'numpy>=1.17.4'
    ],
    install_requires=[
        'imageio',
        'matplotlib>=3',
        'scikit-image',
        'progressbar2',
        'scipy>=1.3',
    ],
    classifiers = [
        # PyPI-specific version type. The number specified here is a magic constant
        # with no relation to this application's version numbering scheme. *sigh*
        'Development Status :: 4 - Beta',

        # Sublist of all supported Python versions.
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',

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