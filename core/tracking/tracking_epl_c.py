# -*- coding: utf-8 -*-
# ============================================================================
# Class for tracking using Early-Prompt-Late method with C callings
# Author: Antoine GRENIER (TAU)
# Date: 2023.01.20
# References: 
# =============================================================================
# PACKAGES
import configparser
import logging
import numpy as np
from core.signal.gnsssignal import GNSSSignal
from core.signal.rfsignal import RFSignal
from core.tracking.tracking_epl import Tracking as TrackingEPL
# =============================================================================
class Tracking(TrackingEPL):
    """
    Tracking class using EPL technique with C implementation.
    """

    def generateReplica(self):
        return super().generateReplica()

    def getCorrelator(self, correlatorSpacing):
        return super().getCorrelator(correlatorSpacing)
    
    # -------------------------------------------------------------------------
    # END OF CLASS


