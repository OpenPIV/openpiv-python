class ProcessParameters( UserDict ):
    def __init__ (self, config_file='' ):
        """
        This class provide a consistent way to set and get all the
        parameters needed to process piv image files. It uses a 
        ConfigParser.SafeConfigParser, which bring some cool features, 
        like reading parameters from INI files.  COnfigParser is a 
        built-in python module. See its documentation for more details.
        
        To instantiate this class you may want to provide as an argument 
        a specific configuration file, but default values are already loaded
        as you create an instance.
        
        The optional file should be formatted like an INI file, with a unique 
        section named 'options'. Example:
        
        [options]
        initial_window_size = 32
        final_window_size = 32
        ...
    
        Parameters
        ----------
        
        config_file : string, optional
            an optional configuration file containing user setting 
            processing parameters.
            
        """
        
        # instantiate ancestor's class
        configparser = ConfigParser.SafeConfigParser()
        
        # read configuration files
        configparser.read( [openpiv.__default_config_file__, config_file] )
        
        UserDict.__init__(self, configparser.items('options') )
        
        # cast parameters to the right type.
        # this trick is necessary because the ConfigParser
        # class only understands strings
        for k, v in self.iteritems():
            try:
                self[k] = int(v)
            except ValueError:
                try:
                    self[k] = float(v)
                except ValueError:
                    pass
    
    def pretty_print ( self ):
        """
        Pretty print all the processing parameters.
        """
        for k, v in self.iteritems():
            print "%s = %s" % ( k.rjust(30), repr(v).ljust(30) )

class Hdf5Database( ):
    """
    A class for writing/reading PIV data to an hdf5 file.
    
    This is currently work in progress, because it is not easy to 
    make parallel write to an hdf file with h5py.
    """
    def __init__ ( self, database, mode='create' ):
        """
        A database can be opened in read/create/modes.
        """

        if mode in [ 'create', 'c', 'w']:
            try:
                self._fh = h5py.File( database, mode='w-' )
            except:
                msg = "A file names %s already exists in folder %s. Delete it first if you really want to." % ( os.path.basename(database), os.path.dirname(database) )
                raise IOError(msg)
        elif mode in [ 'read', 'r']:
            self._fh = h5py.File( os.path.abspath(database), mode='r' )
        else:
            raise ValueError('wrong mode. create of read are accepted')
            
        # two groups for the two velocity components
        self._fh.create_group('/u')
        self._fh.create_group('/v')

    def write_coordinates( self, x, y ):
        """
        Write two datasets with the coordinates of the PIV vectors
        """
        self._fh.create_dataset ( 'x', data = x )
        self._fh.create_dataset ( 'y', data = y )

    def write_velocity_field ( self, i, u, v ):
        """
        Write datasets for the two velocity components.
        """
        self._fh.create_dataset ( '/u/field%05d' % i, data = u )
        self._fh.create_dataset ( '/v/field%05d' % i, data = v )

    def close ( self ):
        """Close file"""
        self._fh.close()


