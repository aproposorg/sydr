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
import multiprocessing
from queue import Empty

from core.acquisition.acquisition_abstract import AcquisitionAbstract
from core.signal.gnsssignal import GNSSSignal
from core.decoding.message_abstract import NavigationMessageAbstract
from core.signal.rfsignal import RFSignal
from core.tracking.tracking_abstract import TrackingAbstract
from enum import Enum, unique

from core.utils.circularbuffer import CircularBuffer

TIMEOUT = 120 # Seconds

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

    def __str__(self):
        return str(self.name)

# =============================================================================
class ChannelAbstract(ABC, multiprocessing.Process):
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
    
    # Multiprocessing
    rfQueue                : multiprocessing.Queue
    event                  : multiprocessing.Event
    daemon                 : bool # Start as a daemon process (see multiprocessing documentation)

    iPrompt : List

    isAcquired         : bool
    isTOWDecoded       : bool
    isEphemerisDecoded : bool

    @abstractmethod
    def __init__(self, cid:int, rfSignal:RFSignal, gnssSignal:GNSSSignal, timeInSamples:int, \
        queue:multiprocessing.Queue, event:multiprocessing.Event, pipe):
        
        # For multiprocessing inheritance
        super(multiprocessing.Process, self).__init__()
        self.daemon = True
        self.rfQueue = queue 
        self.event = event
        self.pipe = pipe
        
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

    def run(self):
        
        while True:
            try:
                rfData = self.rfQueue.get(timeout=TIMEOUT)
            except Empty:
                logging.getLogger(__name__).debug(f"CID {self.cid} did not received data for {TIMEOUT} seconds, closing thread")
                return
            
            if rfData is None:
                logging.getLogger(__name__).debug(f"CID {self.cid} SIGTERM received, closing thread")
                break
            # else:
            #     logging.getLogger(__name__).debug(f"CID {self.cid} received RF Data {rfData}")
            
            # Load the data in the buffer
            self.buffer.shift(rfData)
            self.unprocessedSamples += len(rfData)

            # Wait for a full buffer, to be sure we can do acquisition at least
            if self.buffer.isFull():
                #logging.getLogger(__name__).debug(f"CID {self.cid} buffer full, processing enabled")
                self._processData()

            self.event.set()
            self.pipe[0].send(self)

        logging.getLogger(__name__).debug(f"CID {self.cid} Exiting run") 
        return

    # -------------------------------------------------------------------------

    def _processData(self):

        # Check channel state
        logging.getLogger(__name__).debug(f"CID {self.cid} is in {self.state} state")
        if self.state == ChannelState.IDLE:
            print(f"WARNING: Tracking channel {self.cid} is in IDLE.")
        elif self.state == ChannelState.ACQUIRING:
            buffer = self.buffer.getBuffer()
            self.acquisition.run(buffer[-self.dataRequiredAcquisition:])
            self.isAcquired = self.acquisition.isAcquired

            if self.isAcquired:
                logging.getLogger(__name__).debug(f"CID {self.cid} satellite G{self.svid} acquired")
                frequency, code = self.acquisition.getEstimation()
                self.tracking.setInitialValues(frequency)
                samplesRequired = self.tracking.getSamplesRequired()

                # Switching state to tracking for next loop
                self.switchState(ChannelState.TRACKING)

                # We take double the amount required to be sure one full code will fit
                self.currentSample = self.buffer.getBufferMaxSize() - 2 * samplesRequired + (code + 1)
                self.unprocessedSamples = self.buffer.getBufferMaxSize() - self.currentSample
            else:
                logging.getLogger(__name__).debug(f"CID {self.cid} satellite G{self.svid} not acquired, channel state switched to IDLE")
                self.switchState(ChannelState.IDLE)
            
        elif self.state == ChannelState.TRACKING:
            if self.isAcquired:
                # Reset the flag, otherwise we log acquisition each loop
                self.isAcquired = False
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

        logging.getLogger(__name__).debug(f"CID {self.cid} started with satellite G{self.svid}.")

        return

    # -------------------------------------------------------------------------

    def switchState(self, state:ChannelState):
        self.state = state
        return

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



