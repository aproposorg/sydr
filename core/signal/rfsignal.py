import configparser
import numpy as np

class RFSignal:

    CHUNCK_SIZE_MS = 120 # Number of milliseconds per loaded chunck, ~100ms was found optimal given the tested hardware
    
    chunck : np.array
    chunckMsCounter : int

    samplesPerMs : int

    def __init__(self, configuration:dict):
        """
        Constructor for RFSignal class. 
        
        Args:
            configuration (dict): Configuration dictionnary.

        Returns:
            None
        
        Raises:
            None
        """
        
        # Read configuration
        self.filepath          = str  (configuration['filepath'])
        self.samplingFrequency = float(configuration['sampling_frequency'])
        self.isComplex         = bool (configuration['is_complex'])
        self.interFrequency    = float(configuration['intermediate_frequency'])
        
        dataSize = int(configuration['data_size'])
        
        # Find file data type
        if dataSize   == 8:
            self.fileDataType = np.int8
        elif dataSize == 16:
            self.fileDataType = np.int16
        else:
            raise ValueError(f"Data type of {dataSize} bit(s) is not valid.")
        
        # Check if I/Q or real data
        if self.isComplex:
            self.dtype = np.complex128
        else:
            self.dtype = self.fileDataType

        self.file_id         = None
        self.samplesPerMs    = int(self.samplingFrequency * 1e-3)
        self.chunck          = np.empty((1, self.CHUNCK_SIZE_MS * self.samplesPerMs))
        self.chunckMsCounter = self.CHUNCK_SIZE_MS

        return
    
    # -----------------------------------------------------------------------------------------------------------------
    
    def getMilliseconds(self, nbMilliseconds:int):
        """
        Return the next millisecond(s) of data. The amount of millisecond requested is assumed to be a multiple of the 
        CHUNCK_SIZE_MS variable. 

        Args:
            nbMilliseconds (int) : Number of milliseconds to requested. 

        Returns:
            chunck (np.array) : RF data array.

        Raises:
            ValueError: The number of millisecond requested should be a multiple of the chunck size for optimal read.

        """

        if self.CHUNCK_SIZE_MS % nbMilliseconds:
            raise ValueError(f"The number of millisecond requested should be a multiple of the chunck size for " + 
                             f"optimal read ({nbMilliseconds} not multiple of {self.CHUNCK_SIZE_MS}).")
        
        # Check if new data needs to be loaded
        if self.chunckMsCounter == self.CHUNCK_SIZE_MS:
            self.chunck = self.readFile(timeLength=self.CHUNCK_SIZE_MS, keep_open=True)
            self.chunckMsCounter = 0

        startIdx = self.chunckMsCounter*self.samplesPerMs
        stopIdx = self.chunckMsCounter*self.samplesPerMs + self.samplesPerMs * nbMilliseconds

        self.chunckMsCounter += nbMilliseconds

        return self.chunck[startIdx:stopIdx]
    
    # -----------------------------------------------------------------------------------------------------------------

    def readFile(self, timeLength, skip=0, keep_open=False):
        """
        Read content of file given an amount of time to read.

        Args:
            timeLength (int): Amount of time to read in milliseconds.
        
        Returns
            data (numpy.array): Data from file read.

        Raises:
            None

        """

        if self.isComplex:
            chunck = int(2 * (timeLength*1e-3) * self.samplingFrequency)
            offset = int(np.dtype(self.fileDataType).itemsize * skip * 2)
        else:
            chunck = int((timeLength*1e-3) * self.samplingFrequency)
            offset = int(np.dtype(self.fileDataType).itemsize * skip)
        
        # Read data from file
        if self.file_id is None:
            fid = open(self.filepath, 'rb')
        else: 
            fid = self.file_id
        data = np.fromfile(fid, self.fileDataType, offset=offset, count=chunck)

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
    
    # -----------------------------------------------------------------------------------------------------------------

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
    
    # -----------------------------------------------------------------------------------------------------------------

    def closeFile(self):
        if self.file_id is not None:
            self.file_id.close()
            self.file_id = None
        else:
            raise Warning("File was already close.")
        

        return
    
    # -----------------------------------------------------------------------------------------------------------------

    def getCurrentSampleIndex(self):
        if not self.file_id is None:
            if self.isComplex:
                return int(self.file_id.tell() / 2)
            else:
                return int(self.file_id.tell())
        else:
            raise Warning("Signal file not open, cannot return current cursor position.")
            return -1
        
    # =================================================================================================================
