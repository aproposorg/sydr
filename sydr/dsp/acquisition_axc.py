import numpy as np
import evoapproxlib as eal
import operator

from sydr.signal.gnsssignal import UpsampleCode

from sydr.utils.misc import shift, quantize, multiply_axc

# =====================================================================================================================

def SerialSearch_AxC(rfdata:np.array, interFrequency:float, code:np.array, dopplerRange:tuple, dopplerStep:int, 
                     samplingFrequency:float, samplesPerCode:int, axc_mult, n_bits:int):
    """
    """

    rfdata = np.squeeze(rfdata)

    frequencyBins = interFrequency + np.arange(-dopplerRange, dopplerRange+1, dopplerStep)
    phasePoints = np.array(range(samplesPerCode)) * 2 * np.pi / samplingFrequency

    # Doppler shift loop
    idxFreq = 0
    correlationMap = np.zeros((len(frequencyBins), len(code)))
    for idxFreq in range(len(frequencyBins)):
        
        i_carrier = np.sin(frequencyBins[idxFreq] * phasePoints)
        q_carrier = np.cos(frequencyBins[idxFreq] * phasePoints)

        # Quantize carrier
        i_carrier, _ = quantize(i_carrier, n_bits)
        q_carrier, _ = quantize(q_carrier, n_bits)
        
        # Approximate multiplication
        i_signal = np.zeros_like(i_carrier)
        q_signal = np.zeros_like(q_carrier)
        for i in range(len(i_carrier)):
            i_signal[i] = axc_mult(i_carrier[i], rfdata[i])
            q_signal[i] = axc_mult(q_carrier[i], rfdata[i])

        for idxCode in range(len(code)):    
            _code = shift(code, idxCode)
            _code = UpsampleCode(_code, samplingFrequency)

            # Multiply signal with code
            i_corr = i_signal * _code
            q_corr = q_signal * _code

            correlationMap[idxFreq, idxCode] += np.sum(i_corr)**2 + np.sum(q_corr)**2
    
    return correlationMap