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
from core.signal.gnsssignal import GNSSSignal
from core.decoding.message_lnav import LNAV
from core.signal.rfsignal import RFSignal
from core.channel.channel_abstract import ChannelAbstract
from core.tracking.tracking_epl import Tracking
from core.acquisition.acquisition_pcps import Acquisition
from core.utils.circularbuffer import CircularBuffer
# =============================================================================

class ChannelL1CA(ChannelAbstract):
    def __init__(self, cid:int, gnssSignal:GNSSSignal, rfSignal:RFSignal, timeInSamples:int, \
        queue:multiprocessing.Queue, event:multiprocessing.Event, pipe):
        super().__init__(cid, rfSignal, gnssSignal, timeInSamples, queue, event, pipe)

        self.acquisition = Acquisition(self.rfSignal, self.gnssSignal)
        self.tracking    = Tracking(self.rfSignal, self.gnssSignal)
        self.decoding    = LNAV()

        self.dataRequiredAcquisition = int(self.gnssSignal.codeMs * \
            self.acquisition.nonCohIntegration * \
            self.acquisition.cohIntegration * self.rfSignal.samplingFrequency * 1e-3)
        
        self.dataRequiredTracking = int(self.gnssSignal.codeMs * self.rfSignal.samplingFrequency * 1e-3)

        self.buffer = CircularBuffer(np.max([self.dataRequiredTracking, self.dataRequiredAcquisition]))

        return
    
    