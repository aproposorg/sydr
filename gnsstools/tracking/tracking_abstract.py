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
from gnsstools.signal.gnsssignal import GNSSSignal
from gnsstools.signal.rfsignal import RFSignal

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
    pll : float
    dll : float

    # -------------------------------------------------------------------------

    @abstractmethod
    def __init__(self, rfSignal:RFSignal, gnssSignal:GNSSSignal):
        """
        TODO
        """

        self.rfSignal  = rfSignal
        self.gnssSignal = gnssSignal

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

    def generateReplica(self):

        # Generate replica and mix signal
        time = self.time[0:self.samplesRequired+1]
        temp = -(self.carrierFrequency * 2.0 * np.pi * time) + self.remCarrierPhase

        self.remCarrierPhase = temp[self.samplesRequired] % (2 * np.pi)
        
        replica = np.exp(1j * temp[:self.samplesRequired])

        return replica

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
        code = self.gnssSignal.getCode(svid)
        self.code = np.r_[code[-1], code, code[0]]
        
        return 

    # -------------------------------------------------------------------------

    def getSamplesRequired(self):
        return self.samplesRequired

    def getCorrelatorResults(self):
        return self.correlatorResults

    def getCarrierFrequency(self):
        return self.carrierFrequency
    
    def getCodeFrequency(self):
        return self.codeFrequency

    def getDLL(self):
        return self.dll

    def getPLL(self):
        return self.pll 

    def getPrompt(self):
        iPrompt = self.correlatorResults[2*self.correlatorPrompt]
        qPrompt = self.correlatorResults[2*self.correlatorPrompt+1]
        return iPrompt, qPrompt

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
    
    @abstractmethod
    def getPrompt(self):
        pass

    # END OF CLASS

