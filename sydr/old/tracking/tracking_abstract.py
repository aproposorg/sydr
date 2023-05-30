# -*- coding: utf-8 -*-
# ============================================================================
# Abstract class for tracking process.
# Author: Antoine GRENIER (TAU)
# Date: 2022.05.04
# References: 
# =============================================================================
# PACKAGES
from abc import ABC, abstractmethod
from enum import Enum, unique
import numpy as np
from sydr.signal.gnsssignal import GNSSSignal
from sydr.signal.rfsignal import RFSignal

# =============================================================================

@unique
class TrackingFlags(Enum):
    """
    Tracking flags to be represent the current stage of tracking.
    They are to be intepreted in binary format, to allow multiple state 
    represesented in one decimal number. 
    Similar to states in https://developer.android.com/reference/android/location/GnssMeasurement 
    """

    UNKNOWN       = 0    # 0000 0000 No tracking
    CODE_LOCK     = 1    # 0000 0001 Code found (after acquisition)
    BIT_SYNC      = 2    # 0000 0010 First bit identified 
    SUBFRAME_SYNC = 4    # 0000 0100 First subframe found
    TOW_DECODED   = 8    # 0000 1000 Time Of Week decoded from navigation message
    EPH_DECODED   = 16   # 0001 0000 Ephemeris from navigation message decoded
    TOW_KNOWN     = 32   # 0010 0000 Time Of Week known (retrieved from Assisted Data), to be set if TOW_DECODED set.
    EPH_KNOWN     = 64   # 0100 0000 Ephemeris known (retrieved from Assisted Data), to be set if EPH_DECODED set.
    FINE_LOCK     = 128  # 1000 0000 Fine tracking lock

    def __str__(self):
        return str(self.name)

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

    samplesRequired : int
    time : np.array

    # -------------------------------------------------------------------------

    @abstractmethod
    def __init__(self, rfSignal:RFSignal, gnssSignal:GNSSSignal):
        """
        TODO
        """

        self.rfSignal  = rfSignal
        self.gnssSignal = gnssSignal

        self.remCodePhase = 0.0

        self.codeFrequency   = self.gnssSignal.codeFrequency
        self.codePhaseStep   = self.gnssSignal.codeFrequency / self.rfSignal.samplingFrequency
        self.samplesRequired = int(np.ceil((self.gnssSignal.codeBits - self.remCodePhase) / self.codePhaseStep))

        self.time = np.arange(0, self.samplesRequired+2) / self.rfSignal.samplingFrequency

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

    @abstractmethod
    def getDatabaseDict(self):
        mdict = {
            "type" : "tracking"
        }
        return mdict

    # END OF CLASS

