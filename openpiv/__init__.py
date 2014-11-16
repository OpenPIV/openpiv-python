import os.path as _osp

# the root directory of the package when installed.
# this is an os independent trick to know where the
# package is installed, so we can access data files
# we have provided with the python module
__root__ = _osp.abspath(_osp.dirname(__file__))

# this is the location of the default parameters file
# which we have distributed with openpiv.
__default_config_file__ = _osp.join( __root__, 'data/defaults-processing-parameters.cfg' )


# import default modules
import openpiv.preprocess
import openpiv.tools
import openpiv.pyprocess
import openpiv.scaling
import openpiv.validation
import openpiv.filters
# import openpiv.ui
import openpiv.process
import openpiv.lib
import openpiv.preprocess
