# -*- coding: utf-8 -*-
# ============================================================================
# Class for tracking using Early-Prompt-Late method with C functions
# Author: Antoine GRENIER (TAU) and Hans Jakob DAMSGAARD (TAU)
# Date: 2023.01.31
# References: 
# =============================================================================
# PACKAGES
import configparser
import logging
import numpy as np
import ctypes
from numpy.ctypeslib import ndpointer
from core.signal.gnsssignal import GNSSSignal
from core.signal.rfsignal import RFSignal
from core.tracking.tracking_epl import Tracking as TrackingEPL
# =============================================================================
class Tracking(TrackingEPL):
    """
    Tracking class using EPL technique with C implementation.

    Assumes a compiled library of tracking functions is available at
    `./core/c_functions/tracking.so`
    """

    def __init__(self, rfSignal:RFSignal, gnssSignal:GNSSSignal):
        """
        TODO
        """
        super().__init__(rfSignal, gnssSignal)

        # Initialize connection to external library
        _lib = ctypes.cdll.LoadLibrary('./core/c_functions/tracking.so')
        self._getCorrelator = _lib.getCorrelator
        self._getCorrelator.argtypes = [ndpointer(ctypes.c_double, ndim=1, flags='C_CONTIGUOUS'),
                                        ndpointer(ctypes.c_double, ndim=1, flags='C_CONTIGUOUS'),
                                        ndpointer(ctypes.c_int, ndim=1, flags='C_CONTIGUOUS'),
                                        ctypes.c_size_t,
                                        ctypes.c_double,
                                        ctypes.c_double,
                                        ctypes.c_double,
                                        ndpointer(ctypes.c_double, ndim=1, flags='C_CONTIGUOUS'),
                                        ndpointer(ctypes.c_double, ndim=1, flags='C_CONTIGUOUS')]
        self._getCorrelator.restype = None

        self._generateReplica = _lib.generateReplica
        self._generateReplica.argtypes = [ndpointer(ctypes.c_double, ndim=1, flags='C_CONTIGUOUS'),
                                          ctypes.c_size_t,
                                          ctypes.c_double,
                                          ctypes.c_double,
                                          ndpointer(ctypes.cdouble, ndim=1, flags='C_CONTIGUOUS')]
        self._generateReplica.restype = None

        # RUN

    # -------------------------------------------------------------------------
    # METHODS

    def getCorrelator(self, correlatorSpacing):
        iCorr, qCorr = np.empty((1,)), np.empty((1,))
        self._getCorrelator(self.iSignal, self.qSignal, self.code, len(self.code), \
                            self.codePhaseStep, self.remCodePhase, correlatorSpacing, \
                            iCorr, qCorr)
        return iCorr[0], qCorr[0]

    # -------------------------------------------------------------------------

    def generateReplica(self):
        remCarrierPhase, replica = np.empty((1,)), np.empty((self.samplesRequired,), dtype='complex128')
        self._generateReplica(self.time[:self.samplesRequired+1], self.samplesRequired,\
                              self.carrierFrequency, self.remCarrierPhase, \
                              remCarrierPhase, replica)
        self.remCarrierPhase = remCarrierPhase[0]
        return replica

    # -------------------------------------------------------------------------

    def run(self, rfData):
        pass

    # -------------------------------------------------------------------------

    def getLoopCoefficients(self, loopNoiseBandwidth, dumpingRatio, loopGain):
        return super().getLoopCoefficients(loopNoiseBandwidth, dumpingRatio, loopGain)

    # -------------------------------------------------------------------------

    def getPrompt(self):
        return super().getPrompt()

    # -------------------------------------------------------------------------

    def getDatabaseDict(self):
        return super().getDatabaseDict()

    # -------------------------------------------------------------------------
    # END OF CLASS


