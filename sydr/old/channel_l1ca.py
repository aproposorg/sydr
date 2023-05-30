# -*- coding: utf-8 -*-
# ============================================================================
# Abstract class for tracking process.
# Author: Antoine GRENIER (TAU)
# Date: 2022.05.04
# References: 
# =============================================================================
# PACKAGES
import multiprocessing
import numpy as np
from logging import Logger
from sydr.signal.gnsssignal import GNSSSignal
from sydr.decoding.message_lnav import LNAV
from sydr.signal.rfsignal import RFSignal
from sydr.channel.channel_abstract import ChannelAbstract, ChannelStatus
from sydr.tracking.tracking_epl import Tracking
#from sydr.tracking.tracking_epl_c import Tracking
from sydr.acquisition.acquisition_pcps import Acquisition
from sydr.utils.circularbuffer import CircularBuffer
# =============================================================================

class ChannelL1CA(ChannelAbstract):
    def __init__(self, cid:int, gnssSignal:GNSSSignal, rfSignal:RFSignal, timeInSamples:int, \
        queue:multiprocessing.Queue, pipe):
        super().__init__(cid, rfSignal, gnssSignal, timeInSamples, queue, pipe)

        self.acquisition = Acquisition(self.rfSignal, self.gnssSignal)
        self.tracking    = Tracking(self.rfSignal, self.gnssSignal)
        self.decoding    = LNAV()

        self.dataRequiredAcquisition = int(self.gnssSignal.codeMs * \
            self.acquisition.nonCohIntegration * \
            self.acquisition.cohIntegration * self.rfSignal.samplingFrequency * 1e-3)
        
        self.dataRequiredTracking = int(self.gnssSignal.codeMs * self.rfSignal.samplingFrequency * 1e-3)

        self.buffer = CircularBuffer(np.max([self.dataRequiredTracking, self.dataRequiredAcquisition]))

        return
    
# =============================================================================

class ChannelStatusL1CA(ChannelStatus):
    def __init__(self):
        super().__init__()
        self.subframeFlags = [False, False, False, False, False]

# =============================================================================