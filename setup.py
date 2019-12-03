import sys
import glob

try:
    from setuptools import setup, find_packages
    from setuptools import Extension
    from setuptools.command.build_ext import build_ext as _build_ext
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension
#
# Force `setup_requires` stuff like Cython to be installed before proceeding
#
from setuptools.dist import Distribution
Distribution(dict(setup_requires='Cython'))


from distutils.core import setup
from Cython.Build import cythonize

# try:
#     from Cython.Distutils import build_ext
# except ImportError:
#     print("Could not import Cython.Distutils. Install `cython` and rerun.")
#     sys.exit(1)


class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())



# Build extensions 
ext_modules = cythonize(["openpiv/process.pyx","openpiv/lib.pyx"])


# packages that we want to distribute. THis is how
# we have divided the openpiv package.


setup(  name = "OpenPIV",
        version="0.21.4",
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
        packages=find_packages(),
        include_package_data=True,
        cmdclass = {'build_ext': build_ext},
        install_requires = ['scipy','numpy','cython','scikit-image >= 0.12.0','progressbar2 >= 3.8.1','pygments','future'],
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

