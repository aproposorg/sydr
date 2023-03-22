
import numpy as np

import core.signal.ca as ca
from core.utils.constants import GPS_L1CA_CODE_SIZE_BITS, GPS_L1CA_CODE_FREQ

# =============================================================================

def GenerateGPSGoldCode(prn, samplingFrequency=None):
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

    # Generate gold code
    code = ca.code(prn, 0, 0, 1, GPS_L1CA_CODE_SIZE_BITS)

    # Raise to samping frequency
    if samplingFrequency:
        code = UpsampleCode(code, samplingFrequency)

    return code

# =============================================================================

def UpsampleCode(code, samplingFrequency:float):
    """
    Return upsampled version of code given a sampling frequency.

    Args:
        code (ndarray): Code to upsample.
        samplingFrequency (float): Sampling frequency of the code

    Returns:
        codeUpsampled (ndarray): Code upsampled.
    """
    ts = 1/samplingFrequency     # Sampling period
    tc = 1/GPS_L1CA_CODE_FREQ    # C/A code period
    
    # Number of points per code 
    samples_per_code = getSamplesPerCode(samplingFrequency)
    
    # Index with the sampling frequencies
    idx = np.trunc(ts * np.array(range(samples_per_code)) / tc).astype(int)
    
    # Upsample the original code
    codeUpsampled = code[idx]

    return codeUpsampled

# =============================================================================

def getSamplesPerCode(samplingFrequency:float):
    """
    Return the number of samples per code given a sampling frequency.

    Args:
        samplingFrequency (float): Sampling frequency of the code.

    """
    return round(samplingFrequency / (GPS_L1CA_CODE_FREQ / GPS_L1CA_CODE_SIZE_BITS))

# =============================================================================