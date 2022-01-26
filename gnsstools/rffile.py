import configparser
import numpy as np

class RFFile:
    def __init__(self, configfile):
        # Initialise from config file
        config = configparser.ConfigParser()
        config.read(configfile)

        self.filepath  = config.get       ('DEFAULT', 'filepath')
        self.samp_freq = config.getfloat  ('DEFAULT', 'samp_freq')
        self.iscomplex = config.getboolean('DEFAULT', 'iscomplex')

        # Find data type
        data_size = config.getint  ('DEFAULT', 'data_size')
        if data_size   == 8:
            self.data_type = np.int8
        elif data_size == 16:
            self.data_type = np.int16
        else:
            raise ValueError(f"Data type of {data_size} bit(s) is not valid.")

        return

    def readFile(self, time_length):
        # Amount of data to read from file
        if complex:
            chunck = int(2 * time_length * self.samp_freq)
        else:
            chunck = int(time_length * self.samp_freq)
        
        # Read data from file
        with open(self.filepath, 'rb') as fin:
            data = np.fromfile(fin, self.data_type, count=chunck)

        # Re-organise if needed
        if self.iscomplex:        
            data_real      = data[0::2]
            data_imaginary = data[1::2]
            data           = data_real+ 1j * data_imaginary

        return data