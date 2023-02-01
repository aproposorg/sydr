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
        self._setSatellite.argtypes = [ndpointer(ctypes.c_double, ndim=1, flags='C_CONTIGUOUS'),
                                       ctypes.c_size_t,
                                       ndpointer(np.cdouble, ndim=1, flags='C_CONTIGUOUS')]
        self._setSatellite.restype = None

        self._PCPS = _lib.PCPS
        self._PCPS.argtypes = [ndpointer(np.cdouble, ndim=1, flags='C_CONTIGUOUS'),
                               ndpointer(np.cdouble, ndim=1, flags='C_CONTIGUOUS'),
                               ctypes.c_longlong,
                               ctypes.c_longlong,
                               ctypes.c_longlong,
                               ctypes.c_double,
                               ctypes.c_double,
                               ndpointer(ctypes.c_double, ndim=1, flags='C_CONTIGUOUS'),
                               ctypes.c_size_t,
                               ndpointer(ctypes.c_double, ndim=2, flags='C_CONTIGUOUS')]
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
        codeFFT = np.empty((len(self.code)//2,), dtype='complex128')
        self._setSatellite(np.ascontiguousarray(self.code), len(self.code),\
                           np.ascontiguousarray(codeFFT))
        self.codeFFT = codeFFT
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
        correlationMap = np.empty((len(self.frequencyBins), self.samplesPerCode//2))
        self._PCPS(np.ascontiguousarray(rfData), np.ascontiguousarray(self.codeFFT),\
                   self.cohIntegration, self.nonCohIntegration, self.samplesPerCode,\
                   self.samplingPeriod, self.rfSignal.interFrequency,\
                   np.ascontiguousarray(self.frequencyBins), len(self.frequencyBins),\
                   np.ascontiguousarray(correlationMap))
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
        self._twoCorrelationPeakComparison(np.ascontiguousarray(correlationMap), correlationMap.shape[1],\
                                           np.ascontiguousarray(self.frequencyBins), correlationMap.shape[0],\
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


