# -*- coding: utf-8 -*-
# ============================================================================
# Abstract class for channel definition.
# Author: Antoine GRENIER (TAU)
# Date: 2022.05.04
# References: 
# =============================================================================
# PACKAGES

import logging
import multiprocessing
import numpy as np
from typing import List
from abc import ABC, abstractmethod
from queue import Empty
from logging import Logger

from core.acquisition.acquisition_abstract import AcquisitionAbstract
from core.tracking.tracking_abstract import TrackingAbstract, TrackingFlags
from core.signal.gnsssignal import GNSSSignal
from core.decoding.message_abstract import NavigationMessageAbstract
from core.signal.rfsignal import RFSignal
from enum import Enum, unique

from core.utils.circularbuffer import CircularBuffer
import core.logger as logger

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
@unique
class ChannelCommunication(Enum):
    END_OF_PIPE        = 0
    CHANNEL_UPDATE     = 1
    ACQUISITION_UPDATE = 2
    TRACKING_UPDATE    = 3
    DECODING_UPDATE    = 4

    def __str__(self):
        return str(self.name)

# =============================================================================

class ChannelStatus():

    def __init__(self):

        self.cid = 0
        self.svid = 0
        self.state = ChannelState.IDLE
        self.trackingFlags = TrackingFlags.UNKNOWN
        self.week = 0
        self.tow = 0
        self.timeSinceTOW = 0
        
        return

# =============================================================================
class ChannelAbstract(ABC, multiprocessing.Process):
    cid    : int  # Channel ID
    svid   : int  # Satellite ID

    state                  : ChannelState
    trackingFlags          : TrackingFlags
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
        self.trackingFlags= TrackingFlags.UNKNOWN
        self.subframeFlags = []
        self.timeInSamples= timeInSamples
        self.samplesSinceFirstTOW = -1
        self.dbid = -1

        self.codeSinceLastTOW = 0
        self.timeSinceLastTOW = 0

        self.currentSample = 0
        self.unprocessedSamples = 0

        self.iPrompt = []

        self.isAcquired = False
        self.isTOWDecoded = False
        self.isEphemerisDecoded = False

        self.tow = np.nan
        self.week = np.nan

        # create logger with 'spam_application'
        logger = logging.getLogger(f'CID{self.cid}')
        logger.setLevel(logging.DEBUG)
        # create file handler which logs even debug messages
        fh = logging.FileHandler(f'./.results/log_CID{self.cid}.log')
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)

        self.logger = logger

        return
    
    # -------------------------------------------------------------------------

    def run(self):
        
        while True:
            try:
                rfData = self.rfQueue.get(timeout=TIMEOUT)
            except Empty:
                self.logger.debug(f"CID {self.cid} did not received data for {TIMEOUT} seconds, closing thread")
                return
            
            if rfData is None:
                self.logger.debug(f"CID {self.cid} SIGTERM received, closing thread")
                break
            # else:
            #    self.logger.debug(f"CID {self.cid} received RF Data {rfData}")
            
            # Load the data in the buffer
            self._updateBuffer(rfData)

            # Process the data according to the current channel state
            self._processData()

            # Signal the main thread that processing is done.
            self.event.set()

            self.send(ChannelCommunication.CHANNEL_UPDATE)
            self.send(ChannelCommunication.END_OF_PIPE)

        self.logger.debug(f"CID {self.cid} Exiting run") 

        return

    # -------------------------------------------------------------------------

    def _processData(self):

        # Check channel state
        #self.logger.debug(f"CID {self.cid} is in {self.state} state")

        if self.state == ChannelState.IDLE:
            print(f"WARNING: Tracking channel {self.cid} is in IDLE.")
            self.send({})

        elif self.state == ChannelState.ACQUIRING:
            self.doAcquisition()
            
        elif self.state == ChannelState.TRACKING:
            if self.isAcquired:
                self.isAcquired = False # Reset the flag, otherwise we log acquisition each loop
            # Track
            self.doTracking()

            # If decoding
            self.doDecoding()
        else:
            raise ValueError(f"Channel state {self.state} is not valid.")

        return

    # -------------------------------------------------------------------------

    def doAcquisition(self):

        if self.buffer.isFull():
            #logging.getLogger(__name__).debug(f"CID {self.cid} buffer full, processing enabled")

            buffer = self.buffer.getBuffer()
            self.acquisition.run(buffer[-self.dataRequiredAcquisition:])
            self.isAcquired = self.acquisition.isAcquired

            if self.isAcquired:
                self.logger.debug(f"CID {self.cid} satellite G{self.svid} acquired")
                frequency, code = self.acquisition.getEstimation()
                self.tracking.setInitialValues(frequency)
                samplesRequired = self.tracking.getSamplesRequired()

                # Switching state to tracking for next loop
                self.switchState(ChannelState.TRACKING)

                # We take double the amount required to be sure one full code will fit
                self.currentSample = self.buffer.getBufferMaxSize() - 2 * samplesRequired + (code + 1)
                self.unprocessedSamples = self.buffer.getBufferMaxSize() - self.currentSample

            else:
                self.logger.debug(f"CID {self.cid} satellite G{self.svid} not acquired, channel state switched to IDLE")
                self.switchState(ChannelState.IDLE)
        
            # Send to main program
            self.send(ChannelCommunication.ACQUISITION_UPDATE, self.acquisition.getDatabaseDict())

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

            # Send to main program
            self.send(ChannelCommunication.TRACKING_UPDATE, self.tracking.getDatabaseDict())

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

        if self.decoding.isNewSubframeFound:
            self.send(ChannelCommunication.DECODING_UPDATE, self.decoding.getDatabaseDict())
            self.decoding.isNewSubframeFound = False

        return

    # -------------------------------------------------------------------------

    def send(self, commType:ChannelCommunication, dictToSend: dict = None):
        """
        Send a packet to main program through a defined pipe.
        """
        if commType == ChannelCommunication.END_OF_PIPE:
            _packet = (commType)
        elif commType == ChannelCommunication.CHANNEL_UPDATE:
            _packet = (commType, self.state, self.trackingFlags, self.week, self.tow, self.getTimeSinceTOW())
        elif commType == ChannelCommunication.ACQUISITION_UPDATE \
            or commType == ChannelCommunication.TRACKING_UPDATE \
            or commType == ChannelCommunication.DECODING_UPDATE :

            dictToSend["unprocessed_samples"] = int(self.unprocessedSamples)
            _packet = (commType, dictToSend)
        else:
            raise ValueError(f"Channel communication {commType} is not valid.")
        
        self.pipe[0].send(_packet)

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

        self.logger.debug(f"CID {self.cid} started with satellite G{self.svid}.")

        return

    # -------------------------------------------------------------------------

    def switchState(self, state:ChannelState):
        self.state = state
        return

    # -------------------------------------------------------------------------

    def _updateBuffer(self, data:np.array):
        """
        Update buffer with new data, shift the previous data and update the 
        required variables.
        """

        self.buffer.shift(data)
        self.unprocessedSamples += len(data)

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



