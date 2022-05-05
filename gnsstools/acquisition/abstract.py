# -*- coding: utf-8 -*-
# ============================================================================
# Abstract class for acquisition process.
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
class AcquisitionAbstract(ABC):

    estimatedFrequency: float
    estimatedCode     : float
    correlationMap    : np.array

    @abstractmethod
    def __init__(self, rfConfig:RFSignal, signalConfig:GNSSSignal):
        """
        TODO
        """

        self.rfConfig     = rfConfig
        self.signalConfig = signalConfig

        return

    # -------------------------------------------------------------------------
    # METHODS

    def setSatellite(self, svid):
        """
        TODO
        """

        self.svid = svid
        self.code = self.signalConfig.getCode(svid, self.rfConfig.samplingFrequency)   

        return
    
    # -------------------------------------------------------------------------

    def getEstimation(self):
        return self.estimatedFrequency, self.estimatedCode

    # -------------------------------------------------------------------------

    def getCorrelationMap(self):
        return self.correlationMap
    
    # -------------------------------------------------------------------------
    # ABSTRACT METHODS

    @abstractmethod
    def run(self):
        pass

    # END OF CLASS

