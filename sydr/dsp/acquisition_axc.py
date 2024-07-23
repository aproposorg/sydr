import numpy as np
from sydr.signal.gnsssignal import UpsampleCode

from sydr.utils.misc import shift, quantize, multiply_axc

# =====================================================================================================================

def SerialSearch_AxC(rfdata:np.array, code:np.array, dopplerRange:tuple, dopplerStep:int, samplingFrequency:float, 
                 samplesPerCode:int, axc_mult, n_bits):
    """
    """

    frequencyBins = np.arange(-dopplerRange, dopplerRange+1, dopplerStep)
    phasePoints = np.array(range(samplesPerCode)) * 2 * np.pi / samplingFrequency
    
    correlationMap = np.zeros((len(frequencyBins), len(code)))

    # Doppler shift loop
    idxFreq = 0
    for freq in frequencyBins:
        # Code shift loop
        for idxCode in range(len(code)):
            
            carrier = np.exp(-1j * freq * phasePoints)

            _code = shift(code, idxCode)
            _code = UpsampleCode(_code, samplingFrequency)

            # NOT AXC MULTPLICATION (Should be changed?)
            # This multiplication only involve float -1 to 1 (carrier) and integer (-1 / 1)
            # Technically the multiplication only implies a sign inversion in the phase of the carrier, thus not the
            # computation cost should be low and does not need to be approximated? 
            signal = np.multiply(carrier, _code) 

            # Quantize to get integers for axc multiplication
            # TODO Implemente an NCO with LUT to avoid quantization and speed up this process
            signal = quantize(signal, n_bits)

            i_signal = multiply_axc(np.real(rfdata), signal)
            q_signal = multiply_axc(np.imag(rfdata), signal)

            # Correlation
            correlationMap[idxFreq, idxCode] += np.sum(i_signal)**2 + np.sum(q_signal)**2

        idxFreq += 1
    
    correlationMap = np.squeeze(np.squeeze(correlationMap))
    
    return correlationMap