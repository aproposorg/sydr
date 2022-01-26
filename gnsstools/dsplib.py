
import numpy as np
from numpy.core.function_base import linspace
from scipy import signal as sp
import matplotlib.pyplot as plt
from matplotlib import cm

fig_size = (10,8)

#------------------------------------------------------------------------------

def readFile(filepath, sampling_freq, time_length, data_type, complex=False):

    # Amount of data to read from file
    if complex:
        chunck = int(2 * time_length * sampling_freq)
    else:
        chunck = int(time_length * sampling_freq)
    
    # Read data from file
    with open(filepath, 'rb') as fin:
        data = np.fromfile(fin, data_type, count=chunck)

    # Re-organise if needed
    if complex:        
        data_real      = data[0::2]
        data_imaginary = data[1::2]
        data           = data_real+ 1j * data_imaginary

    return data

#------------------------------------------------------------------------------

def plotTimeDomain(data, sampling_freq, show=True, save_filepath=None):

    time_length = len(data) / sampling_freq
    timeplot = linspace(0, time_length, len(data))

    plt.figure(figsize=fig_size)
    plt.title("Time domain")

    plt.subplot(2,1,1)
    plt.title("In-Phase")
    plt.plot(timeplot, np.real(data))
    plt.grid()
    
    plt.subplot(2,1,2)
    plt.title("Quadrature")
    plt.plot(timeplot, np.imag(data))
    plt.grid()

    if save_filepath is not None:
        plt.savefig(save_filepath)
    if show:
        plt.show()

    return 

#------------------------------------------------------------------------------

def plotFreqDomain(data, sampling_freq, xlim=None, show=True, save_filepath=None):

    size_data = len(data)

    # Compute FFT and shift to center
    data_fft = np.fft.fft(data)

    # Compute the frequencies of the FFT
    freq_fft = np.arange(-sampling_freq/2, sampling_freq/2, sampling_freq/size_data)
    freq_fft = np.fft.fftfreq(len(data_fft)) * sampling_freq

    signal_mag   = np.abs(data_fft)
    signal_phase = np.angle(data_fft)

    # Plot 
    plt.figure(figsize=fig_size)
    plt.title("Frequency domain")

    plt.subplot(2,1,1)
    plt.title('Magnitude')
    plt.plot(freq_fft, signal_mag, '.-')
    plt.xlim(xlim)
    plt.grid()
    
    plt.subplot(2,1,2)
    plt.title('Phase')
    plt.plot(freq_fft, signal_phase, '.-')
    plt.xlim(xlim)
    plt.grid()

    if save_filepath is not None:
        plt.savefig(save_filepath)
    if show:
        plt.show()

    return

#------------------------------------------------------------------------------

def plotPSD(data, sampling_freq, noverlap, xlim=None, show=True, save_filepath=None):

    data_welch = data - np.mean(data)
    window     = sp.hanning(16384, True)

    # Compute Welch (parameters from Kai Borre Matlab software)
    f, Pxxf = sp.welch(data_welch, sampling_freq, window=window, noverlap=noverlap)
    
    plt.figure(figsize=fig_size)
    plt.title("Power Spectral Density (Welch)")
    plt.semilogy(f, Pxxf, '.-')
    plt.xlim(xlim)
    plt.grid()

    if save_filepath is not None:
        plt.savefig(save_filepath)
    if show:
        plt.show()

    return

#------------------------------------------------------------------------------

def acquireSignalL1CA(data, fs, signal_if, prnCA, freq_bins, coh_integration=1, noncoh_integration=1, plots=True, show=True, save_filepath=None):
    # Parallel code phase search
    # Base on Kai Borre Matlab code

    lc = 1023.0     # Number bit per C/A code 
    fc = 1.023e6    # C/A code frequency 
    ts = 1/fs       # Sampling period
    samples_per_code = round(fs / (fc / lc))
    samples_per_code_chip = round(fs / fc)

    # Upsample basic code to the sampling frequency
    caCode = getUpsampledCode(fs, fc, lc, prnCA)
    # Get code FFT
    caCode_fft = np.conj(np.fft.fft(caCode))

    # Data for carrier generation
    phasePoints = np.array(range(len(data))) * 2 * np.pi * ts

    # Search loop
    acquisition = np.zeros((len(freq_bins), samples_per_code))
    noncoh_sum  = np.zeros((1, samples_per_code))
    idx = 0
    for freq in freq_bins:
        freq = signal_if - freq

        # Generate local replica
        carrier_sin = np.sin(freq * phasePoints)
        carrier_cos = np.cos(freq * phasePoints)

        # Remove the carrier
        in_phase   = np.multiply(carrier_sin, data)
        quadrature = np.multiply(carrier_cos, data)
        iq_signal_total = in_phase + 1j*quadrature

        # Non-Coherent Integration 
        noncoh_sum = np.zeros((1, samples_per_code))
        for idx_noncoh in range(0, noncoh_integration):
            
            iq_signal = iq_signal_total[idx_noncoh*coh_integration*samples_per_code:(idx_noncoh+1)*coh_integration*samples_per_code]

            # Coherent Integration
            coh_sum = np.zeros((1, samples_per_code))
            for idx_coh in range(0,coh_integration):
                # Perform FFT
                iq_fft = np.fft.fft(iq_signal[idx_coh*samples_per_code:(idx_coh+1)*samples_per_code])

                # Correlation with C/A code
                iq_conv = np.multiply(iq_fft, caCode_fft)

                # Inverse FFT (go back to time domain)
                coh_sum = coh_sum + np.fft.ifft(iq_conv)

            # Absolute values
            noncoh_sum = noncoh_sum + abs(coh_sum)
        
        acquisition[idx, :] = abs(noncoh_sum)
        idx += 1
    acquisition = np.squeeze(acquisition)

    ## Get acquisition metric 
    # Find first correlation peak
    peak_1 = np.amax(acquisition)
    idx = np.where(acquisition == peak_1)
    coarse_freq = freq_bins[int(idx[0])]
    coarse_code = int(idx[1]) * lc / fs

    # Find second correlation peak
    exclude = list((int(idx[1] - samples_per_code_chip), int(idx[1] + samples_per_code_chip)))

    if exclude[0] < 1:
        code_range = list(range(exclude[1], samples_per_code-1))
    elif exclude[1] >= samples_per_code:
        code_range = list(range(0, exclude[0]))
    else:
        code_range = list(range(0, exclude[0])) + list(range(exclude[1], samples_per_code-1))
    peak_2 = np.amax(acquisition[idx[0], code_range])
    acq_metric = peak_1 / peak_2

    if plots:
        fig, ax = plotCorrelation(acquisition, samples_per_code, freq_bins)
        if save_filepath is not None:
            fig.savefig(save_filepath)
            fig.clear()
            plt.close(fig)
        if show:
            fig.show()
    
    return acquisition, acq_metric, coarse_freq, coarse_code

def plotCorrelation(acquisition, samples_per_code, freq_bins):
    x, y = np.meshgrid(np.linspace(0, 1023, samples_per_code), freq_bins)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(x, y, acquisition, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    return fig, ax
    
def getUpsampledCode(fs, fc, lc, code):
    ts = 1/fs       # Sampling period
    tc = 1/fc       # C/A code period
    
    # Number of points per code 
    samples_per_code = round(fs / (fc / lc))
    
    # Index with the sampling frequencies
    idx = np.trunc(ts * np.array(range(samples_per_code)) / tc).astype(int)
    
    # Upsample the original code
    code_upsampled = code[idx]

    return code_upsampled
