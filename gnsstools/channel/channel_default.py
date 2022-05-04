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
from gnsstools.rfsignal import RFSignal
from gnsstools.channel.abstract import ChannelAbstract
from gnsstools.tracking.tracking_epl import Tracking
from gnsstools.acquisition.acquisition_pcps import Acquisition
# =============================================================================

class Channel(ChannelAbstract):
    def __init__(self, cid:int, signalConfig:GNSSSignal, rfConfig:RFSignal):
        super().__init__(cid, rfConfig, signalConfig)

        self.acquisition = Acquisition(self.rfConfig, self.signalConfig)
        self.tracking    = Tracking(self.rfConfig, self.signalConfig)

        self.dataRequiredAcquisition = int(self.signalConfig.codeMs * \
            self.acquisition.nonCohIntegration * \
            self.acquisition.cohIntegration * self.rfConfig.samplingFrequency * 1e-3)
        
        self.dataRequiredTracking = int(self.signalConfig.codeMs * self.rfConfig.samplingFrequency * 1e-3)

        # Buffer keeps in memory the last ms for each channel
        # Buffer size is based on the maximum amont of data required, most 
        # probably from acquisition.
        self.bufferMaxSize = np.max([self.dataRequiredTracking, self.dataRequiredAcquisition])
        self.buffer = np.empty(self.bufferMaxSize, dtype=np.complex128)
        self.buffer[:] = np.nan
        self.bufferSize = 0
        self.isBufferFull = False

        return
    
    