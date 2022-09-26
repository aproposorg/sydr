import configparser
import os
import numpy as np

import core.signal.ca as ca
from core.utils.enumerations import GNSSSystems, GNSSSignalType

# =============================================================================

class GNSSSignal:
    """
    Class for GNSS signal parameters.
    """

    def __init__(self, configFilePath, signalType:GNSSSignalType):
        """
        Class constructor.

        Args:
            configFilePath (str): Path to receiver '.ini' file. 
            signalType (GNSSSignalType): Type of GNSS signals, see enumeration definition.
        """

        config = configparser.ConfigParser()
        config.read(configFilePath)
        
        self.configFile    = configFilePath
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
        """
        Return satellite PRN code. If a sampling frequency is provided, the
        code returned code will be upsampled to the frequency. 

        Args:
            samplingFrequency (float, optional): Sampling frequency of the code

        Returns:
            code (ndarray): array of PRN code.

        Raises:
            ValueError: If self.signalType not recognised. 
        """

        if self.signalType == GNSSSignalType.GPS_L1_CA:
            code = ca.code(prn, 0, 0, 1, self.codeBits)
        else:
            raise ValueError(f"Signal type {self.signalType} does not exist.")

        # Raise to samping frequency
        if samplingFrequency:
            code = self.getUpsampledCode(code, samplingFrequency)

        return code

    # -------------------------------------------------------------------------

    def getUpsampledCode(self, code, samplingFrequency:float):
        """
        Return upsampled version of code given a sampling frequency.

        Args:
            code (ndarray): Code to upsample.
            samplingFrequency (float): Sampling frequency of the code

        Returns:
            codeUpsampled (ndarray): Code upsampled.
        """
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

    def getSamplesPerCode(self, samplingFrequency:float):
        """
        Return the number of samples per code given a sampling frequency.

        Args:
            samplingFrequency (float): Sampling frequency of the code.

        """
        return round(samplingFrequency / (self.codeFrequency / self.codeBits))
    
    # -------------------------------------------------------------------------

    def getSystem(self):
        """
        Return the system of the signal.

        Returns:
            GNSSSystems
        """

        if self.signalType is GNSSSignalType.GPS_L1_CA:
            return GNSSSystems.GPS
        
        # TODO Do the other signals

# =============================================================================

