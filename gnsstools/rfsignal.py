import configparser
import numpy as np

class RFSignal():

    def __init__(self, configFile):
        config = configparser.ConfigParser()
        config.read(configFile)
        
        self.samplingFrequency = config.getfloat('RF', 'sampling_frequency')
        self.isComplex         = config.getboolean('RF', 'iscomplex')
        self.bitQuantization   = config.getint('RF', 'bit_quantization')

        if self.bitQuantization <= 8:
            self.dataType = np.int8
        elif self.bitQuantization >= 16:
            self.dataType = np.int16
        else:
            raise ValueError(f"Data type of {self.dataType} bit(s) is not valid.")