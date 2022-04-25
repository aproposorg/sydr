import configparser
from enum import Enum
import numpy as np

import gnsstools.ca as ca

class SignalType(Enum):
    GPS_L1_CA = 0

class GNSSSignal:
    def __init__(self, configfile, signalType:SignalType):

        config = configparser.ConfigParser()
        config.read(configfile)
        
        self.signalType    = signalType
        self.name          = config.get     ("DEFAULT", 'NAME')
        self.carrierFreq   = config.getfloat("DEFAULT", 'CARRIER_FREQ')
        self.codeBits      = config.getfloat("DEFAULT", 'CODE_BIT')
        self.codeFrequency = config.getfloat("DEFAULT", 'CODE_FREQ')

        self.code_ms   = int(self.code_freq / self.code_bit / 1e3)

        self.configFile = configfile

        return

    def getCode(self, prn, samplingFrequency=None):
        if self.signal_type == SignalType.GPS_L1_CA:
            code = ca.code(prn, 0, 0, 1, self.code_bit)
        else:
            raise ValueError(f"Signal type {self.signal_type} does not exist.")

        # Raise to samping frequency
        if samplingFrequency:
            code = self.getUpsampledCode(code, samplingFrequency)

        return code

    def getUpsampledCode(self, code, samp_freq):
        ts = 1/samp_freq             # Sampling period
        tc = 1/self.code_freq # C/A code period
        
        # Number of points per code 
        samples_per_code = self.getSamplesPerCode(samp_freq)
        
        # Index with the sampling frequencies
        idx = np.trunc(ts * np.array(range(samples_per_code)) / tc).astype(int)
        
        # Upsample the original code
        code_upsampled = code[idx]

        return code_upsampled

    def getSamplesPerCode(self, samplingFrequency):
        return round(samplingFrequency / (self.code_freq / self.code_bit))