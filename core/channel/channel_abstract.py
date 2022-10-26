# -*- coding: utf-8 -*-
# ============================================================================
# Abstract class for channel definition.
# Author: Antoine GRENIER (TAU)
# Date: 2022.05.04
# References: 
# =============================================================================
# PACKAGES
from abc import ABC, abstractmethod
import logging
from typing import List

from core.acquisition.acquisition_abstract import AcquisitionAbstract
from core.signal.gnsssignal import GNSSSignal
from core.decoding.message_abstract import NavigationMessageAbstract
from core.signal.rfsignal import RFSignal
from core.tracking.tracking_abstract import TrackingAbstract
from enum import Enum, unique

from core.utils.circularbuffer import CircularBuffer

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
    cid    : int  # Channel ID
    svid   : int  # Satellite ID

    state                  : ChannelState
    acquisition            : AcquisitionAbstract
    tracking               : TrackingAbstract
    decoding               : NavigationMessageAbstract
    dataRequiredAcquisition: int
    timeInSamples          : int                        # Number of samples since the receiver started, needed to synchronise the channels together
    samplesSinceFirstTOW   : int # Number of samples since the first TOW found
    tow                    : int 

    buffer                 : CircularBuffer
    currentSample          : int
    unprocessedSamples     : int

    iPrompt : List

    isAcquired         : bool
    isTOWDecoded       : bool
    isEphemerisDecoded : bool

    @abstractmethod
    def __init__(self, cid:int, rfSignal:RFSignal, gnssSignal:GNSSSignal, timeInSamples:int):
        self.cid          = cid
        self.gnssSignal   = gnssSignal
        self.rfSignal     = rfSignal
        self.state        = ChannelState.IDLE
        self.timeInSamples= timeInSamples
        self.samplesSinceFirstTOW = -1
        self.dbid = -1

        self.codeSinceLastTOW = 0

        self.currentSample = 0
        self.unprocessedSamples = 0

        self.iPrompt = []

        self.isAcquired = False
        self.isTOWDecoded = False
        self.isEphemerisDecoded = False

        self.tow = 0
        self.week = 0

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
            
            # Run tracking
            self.tracking.run(buffer)

            self.timeInSamples += samplesRequired
            if self.isTOWDecoded:
                # TODO Change to add the number of epoch processed at each round
                self.codeSinceLastTOW += 1

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

        # Check if ephemeris decoded
        # TODO Add condition if a new message is available
        if not self.isEphemerisDecoded and self.decoding.isEphemerisDecoded:
            self.isEphemerisDecoded = True
            self.week = self.decoding.weekNumber

        if self.decoding.isTOWDecoded and self.tow != self.decoding.tow:
            self.isTOWDecoded = True
            self.tow = self.decoding.tow
            self.codeSinceLastTOW = 0

        return

    # -------------------------------------------------------------------------

    def getTimeSinceTOW(self):
        """
        Time since the last TOW in milliseconds.
        """
        timeSinceTOW = 0
        timeSinceTOW += self.codeSinceLastTOW * self.gnssSignal.codeMs # Add number of code since TOW
        timeSinceTOW += self.unprocessedSamples / (self.rfSignal.samplingFrequency/1e3) # Add number of unprocessed samples 
        return timeSinceTOW

    # -------------------------------------------------------------------------

    def setSatellite(self, svid, dbid):
        # Update the configuration
        self.svid = svid
        self.dbid = dbid

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



