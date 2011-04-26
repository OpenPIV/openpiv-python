from distutils.core import setup, Extension
import glob
import numpy

# Build extensions 
module1 = Extension(    name         = "openpiv.process",
                        sources      = ["openpiv/src/process.c"],
                        include_dirs = [numpy.get_include()],
                    )

# a list of the extension modules that we want to distribute
ext_modules = [module1]


# Package data are those filed 'strictly' needed by the program
# to function correctly.  Images, default configuration files, et cetera.
package_data =  [ 'data/defaults-processing-parameters.cfg', 
                  'data/ui_resources.qrc',
                  'data/images/*.png',
                  'data/icons/*.png',
                ]


# data files are other files which are not required by the program but 
# we want to ditribute as well, for example documentation.
data_files = [ ('share/docs/openpiv/examples/example1', glob.glob('openpiv/docs/examples/example1/*') ),
               ('share/docs/openpiv/examples/gurney-flap', glob.glob('openpiv/docs/examples/gurney-flap/*') ),
               ('share/docs/openpiv', ['README'] ),
               ('share/openpiv/ui', glob.glob('openpiv/data/ui/*.ui') ),
             ]


# packages that we want ot distribute. THis is how
# we have divided the openpiv package.
packages = ['openpiv', 'openpiv.ui']


# script are executable files that will be run to load the gui or something else.
scripts = ['openpiv/openpiv']



setup(  name = "OpenPiv",
        version = "0.1",
        author = "The OpenPiv contributors",
        author_email = "openpiv@openpiv.net",
        description = "A software for PIV data analysis",
        license = "GPL v3",
        url = "www.openpiv.net",
        long_description =  """OpenPIV is an initiative of scientists to
                            develop a software, algorithms and methods
                            for the state-of-the-art experimental tool
                            of Particle Image Velocimetry (PIV) which 
                            are free, open source, and easy to operate.""",
                            
        ext_modules = ext_modules, 
        
        packages = packages,
        
        scripts = scripts,
                
        package_data = {'': package_data},
        
        data_files = data_files
        
        )

