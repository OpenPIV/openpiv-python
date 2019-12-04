from distutils.core import setup
from distutils.extension import Extension

try:
	from Cython.Build import cythonize
	from Cython.Distutils import build_ext
	USE_CYTHON = True
except ImportError:
	USE_CYTHON = False

ext = '.pyx' if USE_CYTHON else '.c'

ext_modules=[
    Extension("openpiv.process", ["openpiv/process"+ext]),
    Extension("openpiv.lib", ["openpiv/lib"+ext]),
]


if USE_CYTHON:    
    extensions = cythonize(ext_modules)


setup(
  name = 'OpenPIV',
  version = '0.21.4',
  include_package_data = True ,
  ext_modules = ext_modules,
  install_requires = ['numpy','scipy','matplotlib','scikit-image','pytest'],
)

