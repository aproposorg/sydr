# -*- coding: utf-8 -*-
# ============================================================================
# Abstract class for channel definition.
# Author: Antoine GRENIER (TAU)
# Date: 2022.05.04
# References: 
# =============================================================================
# PACKAGES
from abc import ABC, abstractmethod
from random import sample
from typing import List
import numpy as np
import copy
from gnsstools.acquisition.abstract import AcquisitionAbstract
from gnsstools.gnsssignal import GNSSSignal
from ..message.abstract import NavigationMessageAbstract
#from ..message.abstract import NavigationMessageAbstract
from gnsstools.rfsignal import RFSignal
from gnsstools.tracking.abstract import TrackingAbstract
from enum import Enum, IntFlag, unique, auto

from gnsstools.utils.circularbuffer import CircularBuffer

# =============================================================================
class ChannelConfig:
    cid : int
    gnssSignal : GNSSSignal
    rfSignal : RFSignal
    pass
    
# =============================================================================
@unique
class ChannelState(Enum):
    OFF           = 0
    IDLE          = 1
    ACQUIRING     = 2
    TRACKING      = 3

# =============================================================================
class ChannelAbstract(ABC):

    state                  : ChannelState
    acquisition            : AcquisitionAbstract
    tracking               : TrackingAbstract
    decoding               : NavigationMessageAbstract
    dataRequiredAcquisition: int
    timeInSamples          : int                        # Number of samples since the receiver started, needed to synchronise the channels together

    buffer                 : CircularBuffer
    currentSample          : int
    unprocessedSamples     : int

    iPrompt : List

    isAcquired     : bool
    isTracking     : bool
    isFineTracking : bool

    @abstractmethod
    def __init__(self, cid:int, rfSignal:RFSignal, gnssSignal:GNSSSignal, timeInSamples:int):
        self.cid          = cid
        self.gnssSignal   = gnssSignal
        self.rfSignal     = rfSignal
        self.state        = ChannelState.IDLE
        self.timeInSamples= timeInSamples

        self.currentSample = 0
        self.unprocessedSamples = 0

        self.iPrompt = []

        self.isAcquired = False
        self.isTracking = False

        return
    
    # -------------------------------------------------------------------------

    def run(self, rfData):
        
        self.buffer.shift(rfData)

        if not self.buffer.isFull():
            return
        
        self.unprocessedSamples += len(rfData)

        if self.state == ChannelState.IDLE:
            print(f"WARNING: Tracking channel {self.cid} is in IDLE.")
        elif self.state == ChannelState.ACQUIRING:
            buffer = self.buffer.getBuffer()
            self.acquisition.run(buffer[-self.dataRequiredAcquisition:])
            self.isAcquired = self.acquisition.isAcquired

            if self.isAcquired:
                frequency, code = self.acquisition.getEstimation()
                self.tracking.setInitialValues(frequency)
                samplesRequired = self.tracking.getSamplesRequired()

                # We take double the amount required to be sure one full code will fit
                self.currentSample = self.buffer.getBufferMaxSize() - 2 * samplesRequired + (code + 1)
                self.unprocessedSamples = self.buffer.getBufferMaxSize() - self.currentSample
            else:
                self.switchState(ChannelState.IDLE)
            
        elif self.state == ChannelState.TRACKING:
            # Track
            self.doTracking()

            # If decoding
            self.doDecoding()
        else:
            raise ValueError(f"Channel state {self.state} is not valid.")

        return

    # -------------------------------------------------------------------------

    def doTracking(self):
        samplesRequired = self.tracking.getSamplesRequired()
            
        while self.unprocessedSamples >= samplesRequired:

            buffer = self.buffer.getSlice(self.currentSample, samplesRequired)
            self.timeInSamples += samplesRequired
            
            # Run tracking
            self.tracking.run(buffer)

            # Update the index for samples
            self.currentSample = (self.currentSample + samplesRequired) % self.buffer.getBufferMaxSize()
            self.unprocessedSamples -= samplesRequired

            # Update for next loop
            samplesRequired = self.tracking.getSamplesRequired()
            buffer = self.buffer.getSlice(self.currentSample, samplesRequired)
        return

    # -------------------------------------------------------------------------

    def doDecoding(self):
        
        # Gather last measurements
        iPrompt, qPrompt = self.tracking.getPrompt()
        self.decoding.addMeasurement(self.timeInSamples, iPrompt)

        # Run decoding
        self.decoding.run()

        return

    # -------------------------------------------------------------------------

    def setSatellite(self, svid):
        # Update the configuration
        self.svid = svid

        # Update the methods
        self.acquisition.setSatellite(svid)
        self.tracking.setSatellite(svid)
        self.decoding.setSatellite(svid)

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

    def getAcquisitionEstimation(self):
        frequency, code = self.acquisition.getEstimation()
        acquisitionMetric = self.acquisition.getMetric()
        return frequency, code, acquisitionMetric

    # -------------------------------------------------------------------------

    def getTrackingEstimation(self):
        i, q = self.tracking.getPrompt()
        frequency = self.tracking.getCarrierFrequency()
        code = self.tracking.getCodeFrequency()
        dll  = self.tracking.getDLL()
        pll  = self.tracking.getPLL()
        return frequency, code, i, q, dll, pll

    # -------------------------------------------------------------------------

    def getConfig(self):

        config = ChannelConfig()
        config.cid = self.cid
        config.gnssSignal = self.gnssSignal
        config.rfSignal = self.rfSignal

        return
    
    # -------------------------------------------------------------------------
    # END OF CLASS



