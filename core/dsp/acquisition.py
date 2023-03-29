
import numpy as np

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
    frequencyBins = np.arange(-dopplerRange, dopplerRange, dopplerStep)

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
    peak_1 = np.amax(correlationMap)
    idx = np.where(correlationMap == peak_1)
    idxHighestPeak = (int(idx[0]), int(idx[1]))

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
