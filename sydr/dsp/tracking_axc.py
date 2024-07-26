
import numpy as np

from sydr.utils.misc import shift, quantize, multiply_axc


def EPL_AxC(rfdata:np.array, code:np.array, samplingFrequency:float, carrierFrequency:float, remainingCarrier:float, \
        remainingCode:float, codeStep:float, correlatorsSpacing:tuple, axc_mult, n_bits:int):
    
    rfdata = np.squeeze(rfdata)
    
    nbSamples = len(rfdata)
    correlatorResults = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    for idx in range(nbSamples):
        # # Generate replica
        temp = -(carrierFrequency * 2.0 * np.pi * (idx/samplingFrequency)) + remainingCarrier
        # replica = np.exp(1j * temp)

        # # Mix replica and RF signal
        # signal = replica * rfdata[idx]
        # iSignal = np.real(signal)
        # qSignal = np.imag(signal)

        i_carrier = np.sin(temp)
        q_carrier = np.cos(temp)

        # Quantize carrier
        i_carrier, _ = quantize(i_carrier, n_bits)
        q_carrier, _ = quantize(q_carrier, n_bits)
   
        i_signal = axc_mult(i_carrier, rfdata)
        q_signal = axc_mult(q_carrier, rfdata)

        # Perform correlation
        for i in range(len(correlatorsSpacing)):
            codeIdx = int(np.ceil(remainingCode + correlatorsSpacing[i] + idx*codeStep))
            correlatorResults[i*2]   += code[codeIdx] * i_signal
            correlatorResults[i*2+1] += code[codeIdx] * q_signal
    
    return correlatorResults
