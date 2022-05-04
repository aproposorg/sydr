# -*- coding: utf-8 -*-
# ============================================================================
# Abstract class for tracking process.
# Author: Antoine GRENIER (TAU)
# Date: 2022.05.04
# References: 
# =============================================================================
# PACKAGES
from abc import ABC, abstractmethod
import numpy as np
from gnsstools.gnsssignal import GNSSSignal
from gnsstools.rfsignal import RFSignal
# =============================================================================
class TrackingAbstract(ABC):
    """ Abstract class for defining tracking processes.

    Attributes:
        TODO
        
    """
    carrierFrequency  : float
    codeFrequency     : float
    samplesRequired   : int
    correlatorSpacing : list
    correlatorResults : list

    # -------------------------------------------------------------------------

    @abstractmethod
    def __init__(self, rfConfig:RFSignal, signalConfig:GNSSSignal):
        """
        TODO
        """

        self.rfConfig  = rfConfig
        self.signalConfig = signalConfig

        return
    
    # -------------------------------------------------------------------------
    # METHODS

    def getCorrelator(self, correlatorSpacing):
        """
        TODO
        """

        idx = np.ceil(np.linspace(self.remCodePhase + correlatorSpacing, \
                self.samplesRequired * self.codePhaseStep + self.remCodePhase + correlatorSpacing, \
                self.samplesRequired, endpoint=False)).astype(int)
        tmpCode = self.code[idx]

        iCorr  = np.sum(tmpCode * self.iSignal)
        qCorr  = np.sum(tmpCode * self.qSignal)

        return iCorr, qCorr

    # -------------------------------------------------------------------------
    
    def setInitialValues(self, estimatedFrequency):
        """
        TODO
        """

        self.initialFrequency = estimatedFrequency
        self.carrierFrequency = estimatedFrequency
        
        return
    # -------------------------------------------------------------------------

    def setSatellite(self, svid):
        """
        TODO
        """
        
        self.svid = svid
        code = self.signalConfig.getCode(svid)
        self.code = np.r_[code[-1], code, code[0]]
        
        return 

    # -------------------------------------------------------------------------

    def getSamplesRequired(self):
        return self.samplesRequired

    # -------------------------------------------------------------------------
    # ACSTRACT METHODS
    
    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def delayLockLoop(self):
        pass

    @abstractmethod
    def phaseLockLoop(self):
        pass

    # END OF CLASS

