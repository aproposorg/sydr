import configparser
import os
from enum import Enum
import numpy as np

import core.signal.ca as ca
from core.utils.enumerations import GNSSSystems, GNSSSignalType


# =============================================================================

class GNSSSignal:
    def __init__(self, configfile, signalType:GNSSSignalType):

        if not os.path.exists(configfile):
            raise ValueError(f"File '{configfile}' does not exist.")

        config = configparser.ConfigParser()
        config.read(configfile)
        
        self.configFile    = configfile
        self.config        = config
        self.signalType    = signalType

        # DEFAULT
        self.name             = config.get     ("DEFAULT", 'name')
        self.carrierFrequency = config.getfloat("DEFAULT", 'carrier_frequency')
        self.codeBits         = config.getfloat("DEFAULT", 'code_bits')
        self.codeFrequency    = config.getfloat("DEFAULT", 'code_frequency')

        self.codeMs     = int(self.codeFrequency / self.codeBits / 1e3)

        return
    
    # -------------------------------------------------------------------------

    def getCode(self, prn, samplingFrequency=None):
        if self.signalType == GNSSSignalType.GPS_L1_CA:
            code = ca.code(prn, 0, 0, 1, self.codeBits)
        else:
            raise ValueError(f"Signal type {self.signalType} does not exist.")

        # Raise to samping frequency
        if samplingFrequency:
            code = self.getUpsampledCode(code, samplingFrequency)

        return code

    # -------------------------------------------------------------------------

    def getUpsampledCode(self, code, samplingFrequency):
        ts = 1/samplingFrequency     # Sampling period
        tc = 1/self.codeFrequency    # C/A code period
        
        # Number of points per code 
        samples_per_code = self.getSamplesPerCode(samplingFrequency)
        
        # Index with the sampling frequencies
        idx = np.trunc(ts * np.array(range(samples_per_code)) / tc).astype(int)
        
        # Upsample the original code
        codeUpsampled = code[idx]

        return codeUpsampled
    
    # -------------------------------------------------------------------------

    def getSamplesPerCode(self, samplingFrequency):
        return round(samplingFrequency / (self.codeFrequency / self.codeBits))
    
    # -------------------------------------------------------------------------

    def getSystem(self):

        if self.signalType is GNSSSignalType.GPS_L1_CA:
            return GNSSSystems.GPS
        
        # TODO Do the other signals

# =============================================================================

