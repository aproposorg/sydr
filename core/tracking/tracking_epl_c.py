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
from core.signal.gnsssignal import GNSSSignal, GNSSSignalType
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
                                        ndpointer(ctypes.c_double),
                                        ndpointer(ctypes.c_double)]
        self._getCorrelator.restype = None

        self._generateReplica = _lib.generateReplica
        self._generateReplica.argtypes = [ndpointer(ctypes.c_double, ndim=1, flags='C_CONTIGUOUS'),
                                          ctypes.c_size_t,
                                          ctypes.c_double,
                                          ctypes.c_double,
                                          ndpointer(ctypes.c_double),
                                          ndpointer(np.cdouble, ndim=1, flags='C_CONTIGUOUS')]
        self._generateReplica.restype = None

        self._generateCarrier = _lib.generateCarrier
        self._generateCarrier.argtypes = [ndpointer(np.cdouble, ndim=1, flags='C_CONTIGUOUS'),
                                          ndpointer(np.cdouble, ndim=1, flags='C_CONTIGUOUS'),
                                          ctypes.c_size_t,
                                          ndpointer(ctypes.c_double, ndim=1, flags='C_CONTIGUOUS'),
                                          ndpointer(ctypes.c_double, ndim=1, flags='C_CONTIGUOUS')]
        
        self._delayLockLoop = _lib.delayLockLoop
        self._delayLockLoop.argtypes = [ctypes.c_double,
                                        ctypes.c_double,
                                        ctypes.c_double,
                                        ctypes.c_double,
                                        ctypes.c_double,
                                        ctypes.c_double,
                                        ctypes.c_double,
                                        ctypes.c_double,
                                        ctypes.c_double,
                                        ctypes.c_double,
                                        ndpointer(ctypes.c_double),
                                        ndpointer(ctypes.c_double),
                                        ndpointer(ctypes.c_double)]
        self._delayLockLoop.restype = None

        self._phaseLockLoop = _lib.phaseLockLoop
        self._phaseLockLoop.argtypes = [ctypes.c_double,
                                        ctypes.c_double,
                                        ctypes.c_double,
                                        ctypes.c_double,
                                        ctypes.c_double,
                                        ctypes.c_double,
                                        ctypes.c_double,
                                        ctypes.c_double,
                                        ndpointer(ctypes.c_double),
                                        ndpointer(ctypes.c_double),
                                        ndpointer(ctypes.c_double)]
        self._phaseLockLoop.restype = None

        self._getLoopCoefficients = _lib.getLoopCoefficients
        self._getLoopCoefficients.argtypes = [ctypes.c_double,
                                              ctypes.c_double,
                                              ctypes.c_double,
                                              ndpointer(ctypes.c_double),
                                              ndpointer(ctypes.c_double)]
        self._getLoopCoefficients.restype = None

        super().__init__(rfSignal, gnssSignal)

        return

    # -------------------------------------------------------------------------
    # METHODS

    def getCorrelator(self, correlatorSpacing):
        """
        TODO
        """
        iCorr, qCorr = np.empty((1,)), np.empty((1,))
        self._getCorrelator(np.ascontiguousarray(self.iSignal), np.ascontiguousarray(self.qSignal),\
                            np.ascontiguousarray(self.code), self.samplesRequired, self.codePhaseStep,\
                            self.remCodePhase, correlatorSpacing, iCorr, qCorr)
        return iCorr[0], qCorr[0]

    # -------------------------------------------------------------------------

    def generateReplica(self):
        """
        TODO
        """
        remCarrierPhase, replica = np.empty((1,)), np.empty((self.samplesRequired,), dtype='complex128')
        self._generateReplica(np.ascontiguousarray(self.time[:self.samplesRequired+1]), self.samplesRequired,\
                              self.carrierFrequency, self.remCarrierPhase, remCarrierPhase,\
                              np.ascontiguousarray(replica))
        self.remCarrierPhase = remCarrierPhase[0]
        return replica

    # -------------------------------------------------------------------------

    def run(self, rfData):
        """
        TODO
        """
        replica = self.generateReplica()
        iSignal, qSignal = np.empty((len(replica),)), np.empty((len(replica),))
        self._generateCarrier(np.ascontiguousarray(rfData), np.ascontiguousarray(replica), len(rfData),\
                              np.ascontiguousarray(iSignal), np.ascontiguousarray(qSignal))
        self.iSignal = iSignal
        self.qSignal = qSignal
        
        # Build correlators (Early-Prompt-Late)
        iEarly , qEarly  = self.getCorrelator(self.correlatorSpacing[0])
        iPrompt, qPrompt = self.getCorrelator(self.correlatorSpacing[1])
        iLate  , qLate   = self.getCorrelator(self.correlatorSpacing[2])

        self.correlatorResults = [iEarly, qEarly, iPrompt, qPrompt, iLate, qLate]
        
        # Delay Lock Loop (DLL)
        self.delayLockLoop(iEarly, qEarly, iLate, qLate)
        
        # Phase Lock Loop (PLL)
        self.phaseLockLoop(iPrompt, qPrompt)

        # Get remaining phase
        idx = np.linspace(self.remCodePhase, self.samplesRequired * self.codePhaseStep + self.remCodePhase, \
                          self.samplesRequired, endpoint=False)
        self.remCodePhase = idx[self.samplesRequired-1] + self.codePhaseStep - self.gnssSignal.codeBits

        self.codePhaseStep = self.codeFrequency / self.rfSignal.samplingFrequency
        self.samplesRequired = int(np.ceil((self.gnssSignal.codeBits - self.remCodePhase) / self.codePhaseStep))
        
        # it will create a log of lines in the logfile if uncommented
        #logging.getLogger(__name__).debug(f"svid={self.svid}, iprompt={iPrompt: 10.2f}, qprompt={iPrompt: 10.2f}, DLL={self.dll: 5.3f}, PLL={self.dll: 5.3f}")

        return

    # -------------------------------------------------------------------------

    def setSatellite(self, svid):
        super().setSatellite(svid)
        self.code = self.code.astype(np.int32) # Conversion from float to int32 to fit the C arguments
        return 

    # -------------------------------------------------------------------------

    # def delayLockLoop(self, iEarly, qEarly, iLate, qLate):
    #     """
    #     TODO
    #     """
    #     codeNCO, codeError, codeFrequency = np.empty((1,)), np.empty((1,)), np.empty((1,))
    #     self._delayLockLoop(iEarly, qEarly, iLate, qLate, self.dllTau1, self.dllTau2,\
    #                         self.pdiCode, self.codeNCO, self.codeError, self.gnssSignal.codeFrequency,\
    #                         codeNCO, codeError, codeFrequency)
    #     self.codeNCO = codeNCO[0]
    #     self.codeError = codeError[0]
    #     self.codeFrequency = codeFrequency[0]
    #     self.dll = self.codeNCO
    #     return

    # # -------------------------------------------------------------------------

    # def phaseLockLoop(self, iPrompt, qPrompt):
    #     """
    #     TODO
    #     """
    #     carrierNCO, carrierError, carrierFrequency = np.empty((1,)), np.empty((1,)), np.empty((1,))
    #     self._phaseLockLoop(iPrompt, qPrompt, self.pllTau1, self.pllTau2, self.pdiCarrier,\
    #                         self.carrierNCO, self.carrierError, self.initialFrequency,\
    #                         carrierNCO, carrierError, carrierFrequency)
    #     self.carrierNCO = carrierNCO[0]
    #     self.carrierError = carrierError[0]
    #     self.carrierFrequency = carrierFrequency[0]
    #     self.pll = self.carrierNCO
    #     return

    # # -------------------------------------------------------------------------

    # def getLoopCoefficients(self, loopNoiseBandwidth, dumpingRatio, loopGain):
    #     """
    #     TODO
    #     """
    #     tau1, tau2 = np.empty((1,)), np.empty((1,))
    #     self._getLoopCoefficients(loopNoiseBandwidth, dumpingRatio, loopGain, tau1, tau2)
    #     return tau1[0], tau2[0]

    # -------------------------------------------------------------------------

    def getPrompt(self):
        return super().getPrompt()

    # -------------------------------------------------------------------------

    def getDatabaseDict(self):
        return super().getDatabaseDict()

    # -------------------------------------------------------------------------
    # END OF CLASS


