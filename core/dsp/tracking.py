

import numpy as np
from enum import Enum, unique, IntEnum

# =====================================================================================================================

@unique
class TrackingFlags(IntEnum):
    """
    Tracking flags to represent the current stage of tracking. They are to be intepreted in binary format, to allow 
    multiple state represesented in one decimal number. 
    Similar to states in https://developer.android.com/reference/android/location/GnssMeasurement 
    """

    UNKNOWN       = 0    # 0000 0000 No tracking
    CODE_LOCK     = 1    # 0000 0001 Code found (after first tracking?)
    BIT_SYNC      = 2    # 0000 0010 First bit identified 
    SUBFRAME_SYNC = 4    # 0000 0100 First subframe found
    TOW_DECODED   = 8    # 0000 1000 Time Of Week decoded from navigation message
    EPH_DECODED   = 16   # 0001 0000 Ephemeris from navigation message decoded
    TOW_KNOWN     = 32   # 0010 0000 Time Of Week known (retrieved from Assisted Data), to be set if TOW_DECODED set.
    EPH_KNOWN     = 64   # 0100 0000 Ephemeris known (retrieved from Assisted Data), to be set if EPH_DECODED set.
    FINE_LOCK     = 128  # 1000 0000 Fine tracking lock

    def __str__(self):
        return str(self.name)

# =====================================================================================================================

def generateReplica(time:np.array, nbSamples:int, carrierFrequency:float, remCarrier:float, ):

    # Generate replica and mix signal
    time = time[0:nbSamples+1]
    temp = -(carrierFrequency * 2.0 * np.pi * time) + remCarrier

    remCarrier = temp[nbSamples] % (2 * np.pi)
    replica = np.exp(1j * temp[:nbSamples])

    return replica, remCarrier

# =====================================================================================================================

def getCorrelator(iSignal:np.array, qSignal:np.array, correlatorSpacing:float, code:np.array, remainingCode:float,\
                  codeStep:float, nbSamples:int):
        """
        Return the I and Q correlation of an RF signal with a sampled code.
        """

        idx = np.ceil(
              np.linspace(remainingCode + correlatorSpacing, nbSamples * codeStep + remainingCode + correlatorSpacing, \
                nbSamples, endpoint=False)).astype(int)
        tmpCode = code[idx]

        iCorr  = np.sum(tmpCode * iSignal)
        qCorr  = np.sum(tmpCode * qSignal)

        return iCorr, qCorr

# =====================================================================================================================

def LoopFiltersCoefficients(loopNoiseBandwidth:float, dampingRatio:float, loopGain:float):
        """
        Return the loop filters coefficients. See reference [Borre, 2007].

        Args:
            loopNoiseBandwidth (float): Loop Noise Bandwith parameter
            dampingRatio (float): Damping Ratio parameter, a.k.a. zeta
            loopGain (float): Loop Gain parameter
        
        Returns
            tau1 (float): Loop filter coefficient (1st)
            tau2 (float): Loop filter coefficient (2nd)

        Raises:
            None
        """

        Wn = loopNoiseBandwidth * 8.0 * dampingRatio / (4.0 * dampingRatio**2 +1)
        
        tau1 = loopGain / Wn**2
        tau2 = 2.0 * dampingRatio / Wn

        return tau1, tau2

# =====================================================================================================================

def EPL(rfData:np.array, code:np.array, samplingFrequency:float, carrierFrequency:float, remainingCarrier:float, \
        remainingCode:float, codeStep:float, correlatorsSpacing:tuple):
    
    rfData = np.squeeze(rfData)
    
    nbSamples = len(rfData)
    correlatorResults = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    for idx in range(nbSamples):
        # Generate replica
        temp = -(carrierFrequency * 2.0 * np.pi * (idx/samplingFrequency)) + remainingCarrier
        replica = np.exp(1j * temp)

        # Mix replica and RF signal
        signal = replica * rfData[idx]
        iSignal = np.real(signal)
        qSignal = np.imag(signal)

        # Perform correlation
        for i in range(len(correlatorsSpacing)):
            codeIdx = int(np.ceil(remainingCode + correlatorsSpacing[i] + idx*codeStep))
            correlatorResults[i*2]   += code[codeIdx] * iSignal
            correlatorResults[i*2+1] += code[codeIdx] * qSignal
    
    return correlatorResults

# =====================================================================================================================

def DLL_NNEML(iEarly:float, qEarly:float, iLate:float, qLate:float, NCO_code:float, NCO_codeError:float, \
              tau1:float, tau2:float, pdi:float):
    """
    Delay Lock Loop implementation, using a Normalize Noncoherent Early Minus Late (NNEML) discriminator.
    See reference [Borre, 2023], p.65
    """

    newCodeError = (np.sqrt(iEarly**2 + qEarly**2) - np.sqrt(iLate**2 + qLate**2)) / \
                   (np.sqrt(iEarly**2 + qEarly**2) + np.sqrt(iLate**2 + qLate**2))
        
    # Update NCO code
    NCO_code += tau2 / tau1 * (newCodeError - NCO_codeError)
    NCO_code += pdi / tau1 * newCodeError

    NCO_codeError = newCodeError

    return NCO_code, NCO_codeError

# =====================================================================================================================

def PLL_costa(iPrompt:float, qPrompt:float, NCO_carrier:float, NCO_carrierError:float, 
              tau1:float, tau2:float, pdi:float):
    """
    Phase Lock Loop implementation, using a Costas discriminator. 
    See reference [Borre, 2023], p.59
    """

    newCarrierError = np.arctan(qPrompt / iPrompt) / 2.0 / np.pi

    # Update NCO frequency
    NCO_carrier += tau2 / tau1 * (newCarrierError - NCO_carrierError)
    NCO_carrier += pdi / tau1 * newCarrierError

    NCO_carrierError = newCarrierError

    return NCO_carrier, NCO_carrierError

# =====================================================================================================================
