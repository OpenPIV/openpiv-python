from os import path

from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy


#extensions = [
#    Extension("openpiv.widim", ["./openpiv/widim.pyx"],
#              include_dirs=[numpy.get_include()])]

#extensions = cythonize(extensions, include_path=[numpy.get_include()])


# read the contents of your README file
this_directory = path.abspath(path.dirname(__file__))
# with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
with open(path.join(this_directory, 'README.md')) as f:
    long_description = f.read()

setup(
    name="OpenPIV",
    version='0.23.5',
    cmdclass={'build_ext': build_ext},
    ext_modules=extensions,
    packages=find_packages(),
    include_package_data=True,
    setup_requires=[
        'setuptools',
        'numpy'
    ],
    install_requires=[
        'numpy',
        'imageio',
        'matplotlib>=3',
        'scikit-image',
        'scipy',
        'natsort',
        'GitPython',
        'pytest',
        'tqdm'
    ],
    classifiers=[
        # PyPI-specific version type. The number specified here is a magic
        # constant
        # with no relation to this application's version numbering scheme.
        # *sigh*
        'Development Status :: 4 - Beta',

        # Sublist of all supported Python versions.
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',

        # Sublist of all supported platforms and environments.
          'Operating System :: MacOS :: MacOS X',
          'Operating System :: Microsoft :: Windows',
          'Operating System :: POSIX',

        # Miscellaneous metadata.
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering',
    ],
    # long_description=long_description,
    # long_description_content_type='text/markdown'
)
