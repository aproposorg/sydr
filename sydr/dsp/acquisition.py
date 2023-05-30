
import numpy as np
import matplotlib.pyplot as plt 

from sydr.signal.gnsssignal import UpsampleCode

# =====================================================================================================================

def PCPS(rfData:np.array, interFrequency:float, samplingFrequency:float, codeFFT:np.array, dopplerRange:tuple, 
         dopplerStep:int, samplesPerCode:int, coherentIntegration:int=1, nonCoherentIntegration:int=1):
    """
    Implementation of the Parallel Code Phase Search (PCPS) method [Borre, 2007]. This method perform the correlation 
    of the code in the frequency domain using FFTs. It produces a 2D correlation map over the frequency and code 
    dimensions.

    Args:
        rfData (numpy.array): Data sample to be used.
        interFrequency (float) : Intermediate Frequency used in RF signal.
        codeFFT (numpy.array): Primary code of the GNSS signal, transformed using a FFT. 
        dopplerRange (int): Frequency bound (-/+) for acquisition search.
        dopplerStep (int) : Frequency step for acquisition search.
        codeBins (np.array): Code bins 

    Returns:
        correlationMap (numpy.array): 2D correlation results.

    Raises:
        None
    """

    rfData = np.squeeze(rfData)

    phasePoints = np.array(range(coherentIntegration * samplesPerCode)) * 2 * np.pi / samplingFrequency
    frequencyBins = np.arange(-dopplerRange, dopplerRange+1, dopplerStep)

    # Search loop
    correlationMap = np.zeros((len(frequencyBins), samplesPerCode))
    noncoh_sum     = np.zeros((1, samplesPerCode))
    coh_sum        = np.zeros((1, samplesPerCode))
    idx = 0
    for freq in frequencyBins:
        freq = interFrequency - freq

        # Generate carrier replica
        signal_carrier = np.exp(-1j * freq * phasePoints)

        # Non-Coherent Integration 
        noncoh_sum = noncoh_sum * 0.0
        for idx_noncoh in range(0, nonCoherentIntegration):
            # Select only require part of the dataset
            iq_signal = rfData[idx_noncoh*coherentIntegration*samplesPerCode:(idx_noncoh+1)*coherentIntegration*samplesPerCode]
            # Mix with carrier
            iq_signal = np.multiply(signal_carrier, iq_signal)
            
            # Coherent Integration
            coh_sum = noncoh_sum * 0.0
            for idx_coh in range(0, coherentIntegration):
                # Perform FFT
                iq_fft = np.fft.fft(iq_signal[idx_coh*samplesPerCode:(idx_coh+1)*samplesPerCode])

                # Correlation with C/A code
                iq_conv = np.multiply(iq_fft, codeFFT)

                # Inverse FFT (go back to time domain)
                coh_sum = coh_sum + np.fft.ifft(iq_conv)

            # Absolute values
            noncoh_sum = noncoh_sum + abs(coh_sum)
        
        correlationMap[idx, :] = abs(noncoh_sum)
        idx += 1
    correlationMap = np.squeeze(np.squeeze(correlationMap))

    return correlationMap

# =====================================================================================================================

def TwoCorrelationPeakComparison(correlationMap:np.array, samplesPerCode:int, samplesPerCodeChip:int):
    """ 
    Perform analysis on correlation map, finding the the highest peak and comparing its correlation value to the one 
    from the second highest peak.

    Args:
        correlationMap (numpy.array): 2D-array from correlation method.
        samplesPerCode (int): Number of samples per code.
        samplesPerCodeChip (int): Number of code samples per code chip
    
    Returns:
        idxHighestPeak (tuple): Indices of the highest correlation peak. 
        acquisitionMetric (float): Ratio between the highest and second highest peaks.
    
    Raises:
        None

    """
    
    # Find first correlation peak
    idxHighestPeak = np.unravel_index(correlationMap.argmax(), correlationMap.shape)
    idxHighestPeak = [int(idxHighestPeak[0]), int(idxHighestPeak[1])] #Weird type otherwise
    peak_1 = correlationMap[idxHighestPeak[0], idxHighestPeak[1]]

    # Find second correlation peak
    exclude = list((int(idxHighestPeak[1] - samplesPerCodeChip), int(idxHighestPeak[1] + samplesPerCodeChip)))

    if exclude[0] < 1:
        code_range = list(range(exclude[1], samplesPerCode - 1))
    elif exclude[1] >= samplesPerCode:
        code_range = list(range(0, exclude[0]))
    else:
        code_range = list(range(0, exclude[0])) + list(range(exclude[1], samplesPerCode - 1))
    peak_2 = np.amax(correlationMap[idxHighestPeak[0], code_range])
    
    acquisitionMetric = peak_1 / peak_2

    return idxHighestPeak, acquisitionMetric

# =====================================================================================================================

def SerialSearch(rfdata:np.array, code:np.array, dopplerRange:tuple, dopplerStep:int, samplingFrequency:float, samplesPerCode:int):
    """
    """

    frequencyBins = np.arange(-dopplerRange, dopplerRange+1, dopplerStep)
    phasePoints = np.array(range(samplesPerCode)) * 2 * np.pi / samplingFrequency
    
    correlationMap = np.zeros((len(frequencyBins), len(code)))

    # Doppler shift loop
    idxFreq = 0
    for freq in frequencyBins:
        carrier = np.exp(-1j * -freq * phasePoints)
        signal = np.multiply(rfdata, carrier)

        # _phase = -freq * phasePoints;
        # i_signal = np.multiply(rfdata, np.sin(_phase))
        # q_signal = np.multiply(rfdata, np.cos(_phase))

        # Code shift loop
        for idxCode in range(len(code)):
            _code = shift(code, idxCode)
            _code = UpsampleCode(_code, samplingFrequency)

            i_signal = np.multiply(np.real(signal), _code)
            q_signal = np.multiply(np.imag(signal), _code)
            # i_signal = np.multiply(i_signal, _code)
            # q_signal = np.multiply(q_signal, _code)

            # Correlation
            correlationMap[idxFreq, idxCode] += np.sum(i_signal)**2 + np.sum(q_signal)**2

        idxFreq += 1
    
    correlationMap = np.squeeze(np.squeeze(correlationMap))
    
    return correlationMap

# =====================================================================================================================

def TwoCorrelationPeakComparison_SS(correlationMap:np.array):
    """ 
    Perform analysis on correlation map, finding the the highest peak and comparing its correlation value to the one 
    from the second highest peak.

    Args:
        correlationMap (numpy.array): 2D-array from correlation method.
        samplesPerCode (int): Number of samples per code.
        samplesPerCodeChip (int): Number of code samples per code chip
    
    Returns:
        idxHighestPeak (tuple): Indices of the highest correlation peak. 
        acquisitionMetric (float): Ratio between the highest and second highest peaks.
    
    Raises:
        None

    """
    
    # Find first correlation peak
    idxHighestPeak = np.unravel_index(correlationMap.argmax(), correlationMap.shape)
    idxHighestPeak = [int(idxHighestPeak[0]), int(idxHighestPeak[1])] #Weird type otherwise
    peak_1 = correlationMap[idxHighestPeak[0], idxHighestPeak[1]]

    # Find second correlation peak
    _map = np.copy(correlationMap[idxHighestPeak[0]-1:idxHighestPeak[0]+2, idxHighestPeak[1]-1:idxHighestPeak[1]+2])
    correlationMap[idxHighestPeak[0]-1:idxHighestPeak[0]+2, idxHighestPeak[1]-1:idxHighestPeak[1]+2] = 0.0 # Remove value from search
    peak_2 = np.amax(correlationMap)

    # Put back the previous value
    correlationMap[idxHighestPeak[0]-1:idxHighestPeak[0]+2, idxHighestPeak[1]-1:idxHighestPeak[1]+2] = _map
    
    acquisitionMetric = peak_1 / peak_2

    return idxHighestPeak, acquisitionMetric


# preallocate empty array and assign slice by chrisaycock
def shift(arr, num, fill_value=np.nan):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = arr[-num:]
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = arr[-num:]
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result
