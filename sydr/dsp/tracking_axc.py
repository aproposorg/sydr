
import numpy as np

from sydr.utils.misc import shift, quantize, multiply_axc


def EPL_AxC(rfdata:np.array, code:np.array, samplingFrequency:float, carrierFrequency:float, remainingCarrier:float, \
        remainingCode:float, codeStep:float, correlatorsSpacing:tuple, axc_mult, n_bits:int):
    
    rfdata = np.squeeze(rfdata)
        
    nbSamples = len(rfdata)
    correlatorResults = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Generate replica
    
    #carrierFrequency = 9547426.3420105

    time = np.arange(0.0, nbSamples) / samplingFrequency
    phase = -(carrierFrequency * 2.0 * np.pi * time) + remainingCarrier

    # Carrier generation (for some reason cos and sin are reverse ...)
    i_carrier = np.cos(phase)
    q_carrier = np.sin(phase)

    # # Quantize carrier
    i_carrier, _ = quantize(i_carrier, n_bits)
    q_carrier, _ = quantize(q_carrier, n_bits)
    
    # # Approximate multiplication
    # i_signal = i_carrier * rfdata
    # q_signal = q_carrier * rfdata

    i_signal = np.zeros_like(i_carrier)
    q_signal = np.zeros_like(q_carrier)
    for i in range(len(i_carrier)):
        i_signal[i] = axc_mult(i_carrier[i], rfdata[i])
        q_signal[i] = axc_mult(q_carrier[i], rfdata[i])

    # Perform correlation
    for i in range(len(correlatorsSpacing)):
        shift = remainingCode + correlatorsSpacing[i]
        codeIdx = np.ceil(np.linspace(shift, codeStep * nbSamples + shift, nbSamples, endpoint=False)).astype(int)
        correlatorResults[i*2]   = np.sum(code[codeIdx] * i_signal)
        correlatorResults[i*2+1] = np.sum(code[codeIdx] * q_signal)
    
    return correlatorResults
