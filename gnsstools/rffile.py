import configparser
import numpy as np

class RFFile:
    def __init__(self, configfile):
        # Initialise from config file
        config = configparser.ConfigParser()
        config.read(configfile)

        self.filepath  = config.get       ('DEFAULT', 'filepath')
        self.samp_freq = config.getfloat  ('DEFAULT', 'samp_freq')
        self.is_complex = config.getboolean('DEFAULT', 'iscomplex')

        # Find data type
        data_size = config.getint  ('DEFAULT', 'data_size')
        if data_size   == 8:
            self.data_type = np.int8
        elif data_size == 16:
            self.data_type = np.int16
        else:
            raise ValueError(f"Data type of {data_size} bit(s) is not valid.")
        
        self.file_id = None

        return

    def readFileByTime(self, time_length):
        """
        Read content of file given an amount of time to read.

        Parameters
        ----------
        time_length : int
            Amount of time to read in milliseconds.
        
        Returns
        -------
        data : numpy.array
            Data from file read.

        """

        if self.is_complex:
            chunck = int(2 * (time_length*1e-3) * self.samp_freq)
        else:
            chunck = int((time_length*1e-3) * self.samp_freq)
        
        # Read data from file
        with open(self.filepath, 'rb') as fin:
            data = np.fromfile(fin, self.data_type, count=chunck)

        # Re-organise if needed
        if self.is_complex:        
            data_real      = data[0::2]
            data_imaginary = data[1::2]
            data           = data_real+ 1j * data_imaginary

        return data

    def readFileByValues(self, nb_values, skip=0, keep_open=False):
        """
        Read content of file given a number of values to read.

        Parameters
        ----------
        nb_values : int
            Number of values to read. 
        skip : int
            Number of values to skip before reading.

        Returns
        -------
        data : numpy.array
            Data from file read.

        """

        if self.is_complex:
            chunck = int(2 * nb_values)
            offset = int(np.dtype(self.data_type).itemsize * skip * 2)
        else:
            chunck = nb_values
            offset = int(np.dtype(self.data_type).itemsize * skip)
        
        # Read data from file
        if self.file_id is None:
            fid = open(self.filepath, 'rb')
        else: 
            fid = self.file_id
        data = np.fromfile(fid, self.data_type, offset=offset, count=chunck)

        if keep_open:
            self.file_id = fid
        else:
            fid.close()

        # Re-organise if needed
        if self.is_complex:        
            data_real      = data[0::2]
            data_imaginary = data[1::2]
            data           = data_real+ 1j * data_imaginary

        return data