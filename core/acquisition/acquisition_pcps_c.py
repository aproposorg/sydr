# -*- coding: utf-8 -*-
# ============================================================================
# Class for acquisition using the PCPS method with C functions
# Author: Antoine GRENIER (TAU) and Hans Jakob DAMSGAARD (TAU)
# Date: 2023.01.31
# References: 
# =============================================================================
# PACKAGES
import configparser
import logging
import pickle
import sqlite3
import numpy as np
import ctypes
from numpy.ctypeslib import ndpointer
from core.signal.gnsssignal import GNSSSignal
from core.signal.rfsignal import RFSignal
from core.acquisition.acquisition_pcps import Acquisition as AcquisitionPCPS
# =============================================================================
class Acquisition(AcquisitionPCPS):
    """
    TODO
    """

    def __init__(self, rfSignal:RFSignal, gnssSignal:GNSSSignal):
        """
        TODO
        """
        super().__init__(rfSignal, gnssSignal)

        # Initialize connection to external library
        _lib = ctypes.cdll.LoadLibrary('./core/c_functions/acquisition.so')
        self._setSatellite = _lib.setSatellite
        self._setSatellite.argtypes = [...]
        self._setSatellite.restype = None

        self._PCPS = _lib.PCPS
        self._PCPS.argtypes = [...]
        self._PCPS.restype = None

        self._twoCorrelationPeakComparison = _lib.twoCorrelationPeakComparison
        self._twoCorrelationPeakComparison.argtypes = [ndpointer(ctypes.c_double, ndim=2, flags='C_CONTIGUOUS'),
                                                       ctypes.c_size_t,
                                                       ndpointer(ctypes.c_double, ndim=1, flags='C_CONTIGUOUS'),
                                                       ctypes.c_size_t,
                                                       ctypes.c_longlong,
                                                       ctypes.c_longlong,
                                                       ctypes.c_double,
                                                       ndpointer(ctypes.c_double),
                                                       ndpointer(ctypes.c_double),
                                                       ndpointer(ctypes.c_double),
                                                       ndpointer(ctypes.c_longlong),
                                                       ndpointer(ctypes.c_longlong),
                                                       ndpointer(ctypes.c_longlong)]
        self._twoCorrelationPeakComparison.restype = None

        return

    # -------------------------------------------------------------------------
    
    def setSatellite(self, svid):
        """
        TODO
        """
        super(AcquisitionPCPS, self).setSatellite(svid)

        self.codeFFT = np.conj(np.fft.fft(self.code))

        return

    # -------------------------------------------------------------------------

    def run(self, rfData):
        """
        TODO
        """

        # Perform PCPS loop
        self.correlationMap = self.PCPS(rfData)

        # Analyse results
        self.twoCorrelationPeakComparison(self.correlationMap)

        if self.acquisitionMetric > self.metricThreshold:
            self.isAcquired = True
        
        logging.getLogger(__name__).debug(f"svid={self.svid}, freq={self.estimatedDoppler: .1f}, code={self.estimatedCode:.1f}, threshold={self.acquisitionMetric:.2f}")

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
        noncoh_sum     = np.zeros((1, self.samplesPerCode))
        coh_sum        = np.zeros((1, self.samplesPerCode))
        idx = 0
        for freq in self.frequencyBins:
            freq = self.rfSignal.interFrequency - freq

            # Generate carrier replica
            signal_carrier = np.exp(-1j * freq * phasePoints)

            # Non-Coherent Integration 
            noncoh_sum = noncoh_sum * 0.0
            for idx_noncoh in range(0, self.nonCohIntegration):
                # Select only require part of the dataset
                iq_signal = rfData[idx_noncoh*self.cohIntegration*self.samplesPerCode:(idx_noncoh+1)*self.cohIntegration*self.samplesPerCode]
                # Mix with carrier
                iq_signal = np.multiply(signal_carrier, iq_signal)
                
                # Coherent Integration
                coh_sum = noncoh_sum * 0.0
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
        correlationMap = np.squeeze(np.squeeze(correlationMap))

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
        acquisitionMetric, estimatedDoppler = np.empty((1,)), np.empty((1,))
        estimatedFrequency, estimatedCode = np.empty((1,)), np.empty((1,), dtype='int64')
        idxEstimatedFrequency, idxEstimatedCode = np.empty((1,), dtype='int64'), np.empty((1,), dtype='int64')
        self._twoCorrelationPeakComparison(correlationMap, correlationMap.shape[1],\
                                           self.frequencyBins, correlationMap.shape[0],\
                                           self.samplesPerCode, self.samplesPerCodeChip,\
                                           self.rfSignal.interFrequency, acquisitionMetric,\
                                           estimatedDoppler, estimatedFrequency,\
                                           estimatedCode, idxEstimatedFrequency, idxEstimatedCode)
        self.estimatedDoppler = estimatedDoppler[0]
        self.estimatedCode = estimatedCode[0]
        self.acquisitionMetric = acquisitionMetric[0]
        self.idxEstimatedFrequency = idxEstimatedFrequency[0]
        self.idxEstimatedCode = idxEstimatedCode[0]
        self.estimatedFrequency = estimatedFrequency[0]
        return

    # END OF CLASS


