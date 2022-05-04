# -*- coding: utf-8 -*-
# ============================================================================
# Abstract class for channel definition.
# Author: Antoine GRENIER (TAU)
# Date: 2022.05.04
# References: 
# =============================================================================
# PACKAGES
from abc import ABC, abstractmethod
import numpy as np
import copy
from gnsstools.acquisition.abstract import AcquisitionAbstract
from gnsstools.gnsssignal import GNSSSignal
from gnsstools.rfsignal import RFSignal
from gnsstools.tracking.abstract import TrackingAbstract
from gnsstools.utils import ChannelState
# =============================================================================
class ChannelAbstract(ABC):

    state                  : ChannelState
    acquisition            : AcquisitionAbstract
    tracking               : TrackingAbstract
    dataRequiredAcquisition: int
    dataRequiredAcquisition: int
    buffer                 : np.array
    trackingResults        : list

    @abstractmethod
    def __init__(self, cid:int, rfConfig:RFSignal, signalConfig:GNSSSignal):
        self.cid          = cid
        self.signalConfig = signalConfig
        self.rfConfig     = rfConfig
        self.state        = ChannelState.IDLE

        self.trackingResults = []

        return
    
    # -------------------------------------------------------------------------

    def run(self, rfData, numberOfms=1):

        self.shiftBuffer(rfData, int(numberOfms*self.rfConfig.samplingFrequency*1e-3))
        if not self.isBufferFull:
            return

        # IDLE
        # Initialise the acquisition
        if self.state == ChannelState.IDLE:
            print(f"WARNING: Tracking channel {self.cid} is in IDLE.")
            return
        
        # ACQUIRING
        # Find coarse parameters of the signal
        if self.state == ChannelState.ACQUIRING:
            self.acquisition.run(self.buffer[-self.dataRequiredAcquisition:])

            if self.acquisition.isAcquired:
                frequency, code = self.acquisition.getEstimation()
                self.tracking.setInitialValues(frequency)
                self.currentSample = code + 1
                self.switchState(ChannelState.TRACKING)
        
        # TRACKING
        # Fine alignement of the signal replica  
        if self.state == ChannelState.TRACKING:
            samplesRequired = self.tracking.getSamplesRequired()
            while self.currentSample <= (self.bufferMaxSize - samplesRequired):
                self.tracking.run(self.buffer[self.currentSample:self.currentSample + samplesRequired])
                self.currentSample += self.tracking.samplesRequired
                samplesRequired = self.tracking.getSamplesRequired()
                
                self.trackingResults.append(copy.copy(self.tracking))

        # # DECODING
        # if self.tracking.preambuleFound:
        #     print("HOW decoding")

        # if self.tracking.frameFound:
        #     print("Frame decoding")

        return

    # -------------------------------------------------------------------------

    def setSatellite(self, svid):
        # Update the configuration
        self.svid = svid

        # Update the methods
        self.acquisition.setSatellite(svid)
        self.tracking.setSatellite(svid)

        # Set state to acquisition
        self.switchState(ChannelState.ACQUIRING)

        return

    # -------------------------------------------------------------------------

    def switchState(self, newState):
        self.state = newState
        return

    # -------------------------------------------------------------------------
    
    def getState(self):
        return self.state

    # -------------------------------------------------------------------------

    def shiftBuffer(self, data, shift:int):
        bufferShifted = np.empty_like(self.buffer)
        bufferShifted[self.bufferMaxSize-shift:] = data
        bufferShifted[:self.bufferMaxSize-shift] = self.buffer[shift:]
        self.buffer = bufferShifted
        
        # Update buffer size
        if not self.isBufferFull:
            self.bufferSize += shift
            if self.bufferSize >= self.bufferMaxSize:
                self.isBufferFull = True
        else:
            self.currentSample -= shift

        return
    
    # -------------------------------------------------------------------------
    # END OF CLASS
