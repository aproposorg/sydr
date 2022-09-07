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
from core.signal.gnsssignal import GNSSSignal
from core.signal.rfsignal import RFSignal
# =============================================================================
class AcquisitionAbstract(ABC):

    estimatedFrequency: float
    estimatedCode     : float
    acquisitionMetric : float
    
    correlationMap    : np.array
    frequencyBins     : np.array
    idxEstimatedFrequency : float
    idxEstimatedCode : float

    @abstractmethod
    def __init__(self, rfSignal:RFSignal, gnssSignal:GNSSSignal):
        """
        TODO
        """

        self.rfSignal   = rfSignal
        self.gnssSignal = gnssSignal

        return

    # -------------------------------------------------------------------------
    # METHODS

    def setSatellite(self, svid):
        """
        TODO
        """

        self.svid = svid
        self.code = self.gnssSignal.getCode(svid, self.rfSignal.samplingFrequency)   

        return
    
    # -------------------------------------------------------------------------

    def getEstimation(self):
        return self.estimatedFrequency, self.estimatedCode

    # -------------------------------------------------------------------------

    def getCorrelationMap(self):
        return self.correlationMap

    # -------------------------------------------------------------------------

    def getMetric(self):
        return self.acquisitionMetric

    # -------------------------------------------------------------------------
    # ABSTRACT METHODS

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def getDatabaseDict(self):
        pass

    # END OF CLASS
