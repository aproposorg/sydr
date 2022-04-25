from abc import ABC, abstractmethod, abstractproperty
import configparser
from typing import overload
from unittest import result
import numpy as np

from gnsstools.gnsssignal import GNSSSignal
from gnsstools.rffile import RFFile
from gnsstools.rfsignal import RFSignal

class Acquisition:
    
    def __init__(self, configfile, prn, signal:GNSSSignal):
        config = configparser.ConfigParser()
        config.read(configfile)
        
        # Read contents
        self.samp_freq  = config.getfloat('RF_FILE', 'samp_freq')
        self.inter_freq = config.getfloat('RF_FILE', 'inter_freq')

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
        data = data_file.readFileByTime(time_to_read)

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
        coarse_freq      = self.inter_freq - freq_bins[int(idx[0])]
        coarse_doppler   = - freq_bins[int(idx[0])]
        coarse_code      = int(np.round(idx[1]))
        coarse_code_norm = coarse_code * self.signal.code_bit / np.size(acq_corr, axis=1)

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
        self.correlationMap = np.squeeze(acq_corr)
        self.acqMetric      = acq_metric
        self.coarseFreq     = coarse_freq 
        self.coarseDoppler  = coarse_doppler
        self.coarseCode     = coarse_code
        self.coarseCodeNorm = coarse_code_norm

        return 0
    
    def doSparseFFT(self, data_file:RFFile, prn, signal:GNSSSignal):


        return


class AcquisitionAbstract(ABC):

    def __init__(self, rfConfig):
        
        # RF parameters
        self.rfConfig           = rfConfig
        self.samplingFrequency  = self.rfConfig.samplingFrequency
        self.samplingPeriod     = 1 / self.rfConfig.samplingFrequency

        # Results
        self.estimatedDoppler     = np.nan
        self.estimatedCode        = np.nan
        self.acquisitionMetric    = np.nan
        self.estimatedFrequency   = np.nan

        self.isAcquired = False

        return

    # -------------------------------------------------------------------------
    # ABSTRACT PROPERTIES
    
    # -------------------------------------------------------------------------
    # ACSTRACT METHODS

    @abstractmethod
    def acquire(self):
        """
        Public method to be called to perform the acquisition.
        """
        pass

    def setSignal(self, signalConfig:GNSSSignal, svid):
        """
        Define parameters to acquire the wanted signal.
        """
        self.signalConfig       = signalConfig
        self.samplesPerCode     = round(self.samplingFrequency / (signalConfig.code_freq / signalConfig.code_bit))
        self.samplesPerCodeChip = round(signalConfig.samp_freq  / signalConfig.signal.code_freq)

        return

    def setSatellite(self, svid):
        self.svid = svid
        self.code = self.signalConfig.getCode(svid, self.samplingFrequency)   

        return

    def getEstimation(self):
        return self.estimatedFrequency, self.estimatedCode

    # -------------------------------------------------------------------------
    # SEARCH METHODS

    # -------------------------------------------------------------------------
    # ACQUISITION METRICS METHODS

    def twoCorrelationPeakComparison(self, correlationMap):
        """ 
        Perform analysis on correlation map, finding the the highest peak and 
        comparing its correlation value to the one from the second highest 
        peak.

        Args:
            correlationMap (numpy.array): 2D-array from correlation method.
        
        Returns:
            estimatedDoppler (float): Estimated Doppler for the signal.
            estimatedCode (float): Estimated code phase for the signal.
            acquisitionMetric (float): Ratio between the highest and second highest peaks.
        
        Raises:
            None

        """
        
        # Find first correlation peak
        peak_1 = np.amax(correlationMap)
        idx = np.where(correlationMap == peak_1)
        estimatedDoppler   = - self.frequencyBins[int(idx[0])]
        estimatedCode      = int(np.round(idx[1]))

        # Find second correlation peak
        exclude = list((int(idx[1] - self.samplesPerCodeChip), int(idx[1] + self.samplesPerCodeChip)))

        if exclude[0] < 1:
            code_range = list(range(exclude[1], self.samplesPerCode - 1))
        elif exclude[1] >= self.samplesPerCode:
            code_range = list(range(0, exclude[0]))
        else:
            code_range = list(range(0, exclude[0])) + list(range(exclude[1], self.samplesPerCode - 1))
        peak_2 = np.amax(correlationMap[idx[0], code_range])
        
        acquisitionMetric = peak_1 / peak_2

        return estimatedDoppler, estimatedCode, acquisitionMetric

# =============================================================================
class Acquisition_PCPS(AcquisitionAbstract):

    def __init__(self, rfConfig:RFSignal, signalConfig:GNSSSignal):
        super().__init__(rfConfig)
        self.setSignal(signalConfig)

        # Load parameters
        config = configparser.ConfigParser()
        config.read(signalConfig.configFile)

        self.name               = 'PCPS'
        self.doppler_range      = config.getfloat('ACQUISITION', 'doppler_range')
        self.doppler_steps      = config.getfloat('ACQUISITION', 'doppler_steps')
        self.coh_integration    = config.getint  ('ACQUISITION', 'coh_integration')
        self.noncoh_integration = config.getint  ('ACQUISITION', 'noncoh_integration')
        self.metric_Threshold   = config.getfloat('ACQUISITION', 'metric_Threshold')

        self.frequencyBins = np.arange(-self.signalConfig.doppler_range, \
                                        self.signalConfig.doppler_range, \
                                        self.signalConfig.doppler_steps)
        
        return
    
    def setSatellite(self, svid):
        super().setSatellite(svid)
        self.codeFFT = np.conj(np.fft.fft(self.code))
        return

    def run(self, rfData):

        # Perform PCPS loop
        correlationMap = self.PCPS(rfData)

        # Analyse results
        results = self.twoCorrelationPeakComparison(correlationMap)

        self.estimatedDoppler     = results[0]
        self.estimatedCode        = results[1]
        self.acquisitionMetric    = results[2]
        self.estimatedFrequency   = self.inter_freq - self.coarseDoppler

        if self.acquisitionMetric > self.signalConfig.acquisitionThreshold:
            self.isAcquired = True

        return

    def PCPS(self, rfData):
        """
        Implementation of the Parallel Code Phase Search (PCPS) method 
        [Borre, 2007]. This method perform the correlation of the code in the 
        frequency domain using FFTs. It produces a 2D correlation map over
        the frequency and code dimensions.

        Args:
            data (numpy.array): Data sample to be used.

        Returns:
            correlationMap (numpy.array): 2D correlation results.

        Raises:
            None
        """
        phasePoints = np.array(range(self.coh_integration * self.samplesPerCode)) * 2 * np.pi * self.samplingPeriod
        # Search loop
        correlationMap = np.zeros((len(self.frequencyBins), self.samplesPerCode))
        noncoh_sum  = np.zeros((1, self.samplesPerCode))
        idx = 0
        for freq in self.frequencyBins:
            freq = self.inter_freq - freq

            # Generate carrier replica
            signal_carrier = np.exp(-1j * freq * phasePoints)

            # Non-Coherent Integration 
            noncoh_sum = np.zeros((1, self.samplesPerCode))
            for idx_noncoh in range(0, self.noncoh_integration):
                # Select only require part of the dataset
                iq_signal = rfData[idx_noncoh*self.coh_integration*self.samplesPerCode:(idx_noncoh+1)*self.coh_integration*self.samplesPerCode]
                # Mix with carrier
                iq_signal = np.multiply(signal_carrier, iq_signal)
                
                # Coherent Integration
                coh_sum = np.zeros((1, self.samplesPerCode))
                for idx_coh in range(0, self.coh_integration):
                    # Perform FFT
                    iq_fft = np.fft.fft(iq_signal[idx_coh*self.samplesPerCode:(idx_coh+1)*self.samplesPerCode])

                    # Correlation with C/A code
                    iq_conv = np.multiply(iq_fft, self.codeFFT)

                    # Inverse FFT (go back to time domain)
                    coh_sum = coh_sum + np.fft.ifft(iq_conv)

                # Absolute values
                noncoh_sum = noncoh_sum + abs(coh_sum)
            
            correlationMap[idx, :] = abs(noncoh_sum)
            idx += 1
        correlationMap = np.squeeze(correlationMap)

        return correlationMap