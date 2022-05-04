# -*- coding: utf-8 -*-
# ============================================================================
# Class for acquisition using the PCPS method.
# Author: Antoine GRENIER (TAU)
# Date: 2022.05.04
# References: 
# =============================================================================
# PACKAGES
import configparser
import numpy as np
from gnsstools.gnsssignal import GNSSSignal
from gnsstools.rfsignal import RFSignal
from gnsstools.acquisition.abstract import AcquisitionAbstract
# =============================================================================
class Acquisition(AcquisitionAbstract):
    """
    TODO
    """

    def __init__(self, rfConfig:RFSignal, signalConfig:GNSSSignal):
        """
        TODO
        """
        super().__init__(rfConfig, signalConfig)

        # Read acquisition parameters for signal
        config = configparser.ConfigParser()
        config.read(signalConfig.configFile)

        self.name              = 'PCPS'
        self.dopplerRange      = config.getfloat('ACQUISITION', 'doppler_range')
        self.dopplerSteps      = config.getfloat('ACQUISITION', 'doppler_steps')
        self.cohIntegration    = config.getint  ('ACQUISITION', 'coh_integration')
        self.nonCohIntegration = config.getint  ('ACQUISITION', 'noncoh_integration')
        self.metricThreshold   = config.getfloat('ACQUISITION','metric_threshold')
        
        self.samplesPerCode     = round(self.rfConfig.samplingFrequency / (signalConfig.codeFrequency / signalConfig.codeBits))
        self.samplesPerCodeChip = round(self.rfConfig.samplingFrequency / signalConfig.codeFrequency)
        self.samplingPeriod     = 1 / self.rfConfig.samplingFrequency

        self.frequencyBins = np.arange(-self.dopplerRange, \
                                        self.dopplerRange, \
                                        self.dopplerSteps)
        
        #self.estimatedCode      = np.nan
        #self.estimatedFrequency = np.nan
        self.estimatedDoppler   = np.nan
        self.acquisitionMetric  = np.nan
        
        self.isAcquired = False
        
        return

    # -------------------------------------------------------------------------
    
    def setSatellite(self, svid):
        """
        TODO
        """
        super().setSatellite(svid)

        self.codeFFT = np.conj(np.fft.fft(self.code))

        return

    # -------------------------------------------------------------------------

    def run(self, rfData):
        """
        TODO
        """

        # Perform PCPS loop
        correlationMap = self.PCPS(rfData)

        # Analyse results
        results = self.twoCorrelationPeakComparison(correlationMap)

        self.estimatedDoppler     = results[0]
        self.estimatedCode        = results[1]
        self.acquisitionMetric    = results[2]
        self.estimatedFrequency   = self.rfConfig.interFrequency + self.estimatedDoppler 

        if self.acquisitionMetric > self.metricThreshold:
            self.isAcquired = True

        return

    # -------------------------------------------------------------------------

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

        phasePoints = np.array(range(self.cohIntegration * self.samplesPerCode)) * 2 * np.pi * self.samplingPeriod
        # Search loop
        correlationMap = np.zeros((len(self.frequencyBins), self.samplesPerCode))
        noncoh_sum  = np.zeros((1, self.samplesPerCode))
        idx = 0
        for freq in self.frequencyBins:
            freq = self.rfConfig.interFrequency - freq

            # Generate carrier replica
            signal_carrier = np.exp(-1j * freq * phasePoints)

            # Non-Coherent Integration 
            noncoh_sum = np.zeros((1, self.samplesPerCode))
            for idx_noncoh in range(0, self.nonCohIntegration):
                # Select only require part of the dataset
                iq_signal = rfData[idx_noncoh*self.cohIntegration*self.samplesPerCode:(idx_noncoh+1)*self.cohIntegration*self.samplesPerCode]
                # Mix with carrier
                iq_signal = np.multiply(signal_carrier, iq_signal)
                
                # Coherent Integration
                coh_sum = np.zeros((1, self.samplesPerCode))
                for idx_coh in range(0, self.cohIntegration):
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

    # -------------------------------------------------------------------------

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
        estimatedDoppler   = -self.frequencyBins[int(idx[0])]
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
    
    # END OF CLASS


