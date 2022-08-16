# -*- coding: utf-8 -*-
# ============================================================================
# Abstract class for tracking process.
# Author: Antoine GRENIER (TAU)
# Date: 2022.05.04
# References: 
# =============================================================================
# PACKAGES
import numpy as np
from gnsstools.gnsssignal import GNSSSignal
from ..message.lnav import LNAV
from gnsstools.rfsignal import RFSignal
from gnsstools.channel.abstract import ChannelAbstract
from gnsstools.tracking.tracking_epl import Tracking
from gnsstools.acquisition.acquisition_pcps import Acquisition
from gnsstools.utils.circularbuffer import CircularBuffer
# =============================================================================

class Channel(ChannelAbstract):
    def __init__(self, cid:int, gnssSignal:GNSSSignal, rfSignal:RFSignal, timeInSamples):
        super().__init__(cid, rfSignal, gnssSignal, timeInSamples)

        self.acquisition = Acquisition(self.rfSignal, self.gnssSignal)
        self.tracking    = Tracking(self.rfSignal, self.gnssSignal)
        self.decoding    = LNAV()

        self.dataRequiredAcquisition = int(self.gnssSignal.codeMs * \
            self.acquisition.nonCohIntegration * \
            self.acquisition.cohIntegration * self.rfSignal.samplingFrequency * 1e-3)
        
        self.dataRequiredTracking = int(self.gnssSignal.codeMs * self.rfSignal.samplingFrequency * 1e-3)

        # Buffer keeps in memory the last ms for each channel
        # Buffer size is based on the maximum amont of data required, most 
        # probably from acquisition.
        # self.bufferMaxSize = np.max([self.dataRequiredTracking, self.dataRequiredAcquisition])
        # self.buffer = np.empty(self.bufferMaxSize, dtype=np.complex128)
        # self.buffer[:] = np.nan
        # self.bufferSize = 0
        # self.isBufferFull = False

        self.buffer = CircularBuffer(np.max([self.dataRequiredTracking, self.dataRequiredAcquisition]))

        return
    
    