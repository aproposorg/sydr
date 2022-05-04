import configparser
import numpy as np

class RFSignal:
    def __init__(self, configfile):
        # Initialise from config file
        config = configparser.ConfigParser()
        config.read(configfile)

        self.filepath          = config.get       ('RF_FILE', 'filepath')
        self.samplingFrequency = config.getfloat  ('RF_FILE', 'sampling_frequency')
        self.isComplex         = config.getboolean('RF_FILE', 'iscomplex')
        self.interFrequency    = config.getfloat  ('RF_FILE', 'intermediate_frequency')

        # Find data type
        dataSize = config.getint  ('RF_FILE', 'data_size')
        if dataSize   == 8:
            self.dataType = np.int8
        elif dataSize == 16:
            self.dataType = np.int16
        else:
            raise ValueError(f"Data type of {dataSize} bit(s) is not valid.")
        
        self.file_id = None

        return

    def readFile(self, timeLength, skip=0, keep_open=False):
        """
        Read content of file given an amount of time to read.

        Args:
            timeLength (int): Amount of time to read in milliseconds.
        
        Returns
            data (numpy.array): Data from file read.

        """

        if self.isComplex:
            chunck = int(2 * (timeLength*1e-3) * self.samplingFrequency)
            offset = int(np.dtype(self.dataType).itemsize * skip * 2)
        else:
            chunck = int((timeLength*1e-3) * self.samplingFrequency)
            offset = int(np.dtype(self.dataType).itemsize * skip)
        
        # Read data from file
        if self.file_id is None:
            fid = open(self.filepath, 'rb')
        else: 
            fid = self.file_id
        data = np.fromfile(fid, self.dataType, offset=offset, count=chunck)

        if keep_open:
            self.file_id = fid
        else:
            fid.close()

        # Re-organise if needed
        if self.isComplex:        
            data_real      = data[0::2]
            data_imaginary = data[1::2]
            data           = data_real+ 1j * data_imaginary

        return data

    def readFileBySamples(self, nb_values, skip=0, keep_open=False):
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

        if self.isComplex:
            chunck = int(2 * nb_values)
            offset = int(np.dtype(self.data_type).itemsize * skip * 2)
            #offset = np.dtype(self.data_type).itemsize * skip
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
        if self.isComplex:        
            data_real      = data[0::2]
            data_imaginary = data[1::2]
            data           = data_real+ 1j * data_imaginary

        return data

    def closeFile(self):
        if self.file_id is not None:
            self.file_id.close()
            self.file_id = None
        else:
            raise Warning("File was already close.")
        

        return

    def getCurrentSampleIndex(self):
        if not self.file_id is None:
            if self.isComplex:
                return int(self.file_id.tell() / 2)
            else:
                return int(self.file_id.tell())
        else:
            raise Warning("Signal file not open, cannot return current cursor position.")
            return -1
