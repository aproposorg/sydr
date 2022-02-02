import configparser
import numpy as np

from gnsstools.gnsssignal import GNSSSignal
from gnsstools.rffile import RFFile

class Acquisition:
    
    def __init__(self, configfile, prn, signal:GNSSSignal):
        config = configparser.ConfigParser()
        config.read(configfile)
        
        # Read contents
        self.samp_freq  = config.getfloat('DEFAULT', 'samp_freq')
        self.inter_freq = config.getfloat('DEFAULT', 'inter_freq')

        self.method             = config.get     ('ACQUISITION', 'method')
        self.doppler_range      = config.getfloat('ACQUISITION', 'doppler_range')
        self.doppler_steps      = config.getfloat('ACQUISITION', 'doppler_steps')
        self.coh_integration    = config.getint  ('ACQUISITION', 'coh_integration')
        self.noncoh_integration = config.getint  ('ACQUISITION', 'noncoh_integration')

        self.prn    = prn
        self.signal = signal
        
        return

    def acquire(self, data_file:RFFile, method=None):
        """
        Perform selected acquisition method.
        """
        # Switch to default parameters
        if method==None:
            method = self.method

        # Select appropriate function
        if method == 'PCPS':
            results = self.doPCPS(data_file)
        else:
            raise ValueError(f"Acquisition method {method} is not defined.")

        return results

    def doPCPS(self, data_file:RFFile):
        ts = 1/self.samp_freq       # Sampling period
        samples_per_code = round(self.samp_freq / (self.signal.code_freq / self.signal.code_bit))
        samples_per_code_chip = round(self.samp_freq  / self.signal.code_freq)

        prn_code = self.signal.getCode(self.prn)
        prn_code = self.signal.getUpsampledCode(self.samp_freq, prn_code)

        # Get code FFT
        caCode_fft = np.conj(np.fft.fft(prn_code))

        # Load data
        time_to_read = self.coh_integration * self.noncoh_integration
        data = data_file.readFile(time_to_read)

        # Data for carrier generation
        phasePoints = np.array(range(self.coh_integration*samples_per_code)) * 2 * np.pi * ts

        # Search loop
        freq_bins = np.arange(-self.doppler_range, self.doppler_range, self.doppler_steps)
        acq_corr = np.zeros((len(freq_bins), samples_per_code))
        noncoh_sum  = np.zeros((1, samples_per_code))
        idx = 0
        for freq in freq_bins:
            #print(f"{freq} ...")
            freq = self.inter_freq - freq

            # Generate carrier replica
            signal_carrier = np.exp(-1j*freq*phasePoints)

            # Non-Coherent Integration 
            noncoh_sum = np.zeros((1, samples_per_code))
            for idx_noncoh in range(0, self.noncoh_integration):
                # Select only require part of the dataset
                iq_signal = data[idx_noncoh*self.coh_integration*samples_per_code:(idx_noncoh+1)*self.coh_integration*samples_per_code]
                # Mix with carrier
                iq_signal = np.multiply(signal_carrier, iq_signal)
                
                # Coherent Integration
                coh_sum = np.zeros((1, samples_per_code))
                for idx_coh in range(0, self.coh_integration):
                    # Perform FFT
                    iq_fft = np.fft.fft(iq_signal[idx_coh*samples_per_code:(idx_coh+1)*samples_per_code])

                    # Correlation with C/A code
                    iq_conv = np.multiply(iq_fft, caCode_fft)

                    # Inverse FFT (go back to time domain)
                    coh_sum = coh_sum + np.fft.ifft(iq_conv)

                # Absolute values
                noncoh_sum = noncoh_sum + abs(coh_sum)
            
            acq_corr[idx, :] = abs(noncoh_sum)
            idx += 1
        acq_corr = np.squeeze(acq_corr)

        # Get acquisition metric 
        ## Find first correlation peak
        peak_1 = np.amax(acq_corr)
        idx = np.where(acq_corr == peak_1)
        coarse_freq = freq_bins[int(idx[0])]
        coarse_code = int(idx[1]) * self.signal.code_bit / np.size(acq_corr, axis=1)

        ## Find second correlation peak
        exclude = list((int(idx[1] - samples_per_code_chip), int(idx[1] + samples_per_code_chip)))

        if exclude[0] < 1:
            code_range = list(range(exclude[1], samples_per_code-1))
        elif exclude[1] >= samples_per_code:
            code_range = list(range(0, exclude[0]))
        else:
            code_range = list(range(0, exclude[0])) + list(range(exclude[1], samples_per_code-1))
        peak_2 = np.amax(acq_corr[idx[0], code_range])
        acq_metric = peak_1 / peak_2
        
        # Save results
        self.correlation_map = np.squeeze(acq_corr)
        self.acq_metric      = acq_metric
        self.coarse_freq     = coarse_freq       
        self.coarse_code     = coarse_code

        return 0
    
    def doSparseFFT(self, data_file:RFFile, prn, signal:GNSSSignal):

        
        return


    