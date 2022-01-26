import configparser
import numpy as np

import gnsstools.ca as ca

class GNSSSignal:
    def __init__(self, configfile, signal_type):
        # Initialise from config file
        config = configparser.ConfigParser()
        config.read(configfile)

        self.signal_type = signal_type
        self.code_bit  = config.getfloat(signal_type, 'CODE_BIT')
        self.code_freq = config.getfloat(signal_type, 'CODE_FREQ')

        return

    def getCode(self, prn):
        if self.signal_type == 'GPS_L1_CA':
            prn_code = ca.code(prn, 0, 0, 1, self.code_bit)
        else:
            raise ValueError(f"Signal type {self.signal_type} does not exist.")    

        return prn_code

    def getUpsampledCode(self, samp_freq, code):
        ts = 1/samp_freq             # Sampling period
        tc = 1/self.code_freq # C/A code period
        
        # Number of points per code 
        samples_per_code = round(samp_freq / (self.code_freq / self.code_bit))
        
        # Index with the sampling frequencies
        idx = np.trunc(ts * np.array(range(samples_per_code)) / tc).astype(int)
        
        # Upsample the original code
        code_upsampled = code[idx]

        return code_upsampled