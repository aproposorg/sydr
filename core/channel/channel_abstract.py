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
from enum import Enum, unique

from core.acquisition.acquisition_abstract import AcquisitionAbstract
from core.tracking.tracking_abstract import TrackingAbstract, TrackingFlags
from core.signal.gnsssignal import GNSSSignal
from core.signal.rfsignal import RFSignal
from core.decoding.message_abstract import NavigationMessageAbstract
from core.utils.circularbuffer import CircularBuffer
# =============================================================================
PROCESS_TIMEOUT = 60 # Seconds 
# =============================================================================
@unique
class ChannelState(Enum):
    """
    Channel state according to the defined state machine architecture.
    """
    OFF           = 0
    IDLE          = 1
    ACQUIRING     = 2
    TRACKING      = 3

    def __str__(self):
        return str(self.name)

# =============================================================================
@unique
class ChannelMessage(Enum):
    """
    Message sent by the channel to the main thread.
    """
    END_OF_PIPE        = 0
    CHANNEL_UPDATE     = 1
    ACQUISITION_UPDATE = 2
    TRACKING_UPDATE    = 3
    DECODING_UPDATE    = 4

    def __str__(self):
        return str(self.name)

# =============================================================================
class ChannelAbstract(ABC, multiprocessing.Process):

    # IDs
    cid  : int  # Channel ID
    svid : int  # Satellite ID

    # Signals
    gnssSignal : GNSSSignal # GNSS signal parameters 
    rfSignal   : RFSignal   # RF signal parameters

    # Status and flags
    state                  : ChannelState              # Current channel state 
    trackingFlags          : TrackingFlags             # Current tracking state (Not implemented yet)
    isAcquired             : bool                      # Flag set if satellite has been acquired
    isTOWDecoded           : bool                      # Flag set if TOW decoded (will be replaced with trackingFlags)
    isEphemerisDecoded     : bool                      # Flag set if ephemeris decoded

    # DSP classes
    acquisition            : AcquisitionAbstract       # Acquisition object
    tracking               : TrackingAbstract          # Tracking object
    decoding               : NavigationMessageAbstract # Decoding object

    # Keep track of data
    buffer                 : CircularBuffer            # Circular buffer, for limited data storage
    currentSample          : int                       # Store the current sample index
    unprocessedSamples     : int                       # Amount of samples not processed in the buffer

    # Keep track of time
    timeInSamples          : int                       # Number of samples received since the channel started, needed to synchronise the channels together
    tow                    : int                       # Curent Time of Week (TOW)
    codeSinceTOW           : int                       # Number of code since TOW decoded
    
    # Multiprocessing
    rfQueue                : multiprocessing.Queue     # Queue for reception of new data from main thread, pipe is not used as it has limited storage
    pipe                   : multiprocessing.Pipe      # Pipe for communication between channel and main thread
    daemon                 : bool                      # Start as a daemon process (see multiprocessing documentation)

    # Misc.
    dataRequiredAcquisition: int                       # Minimum data required for an acquistion
    dataRequiredTracking   : int                       # Minimim data required for a tracking epoch

    # =========================================================================

    @abstractmethod
    def __init__(self, cid:int, rfSignal:RFSignal, gnssSignal:GNSSSignal, timeInSamples:int, \
        queue:multiprocessing.Queue, pipe:multiprocessing.Pipe):
        """
        
        """
        
        # For multiprocessing inheritance
        super(multiprocessing.Process, self).__init__()

        # Initialisation 
        self.cid          = cid
        self.svid         = 0

        self.gnssSignal   = gnssSignal
        self.rfSignal     = rfSignal

        self.state         = ChannelState.IDLE
        self.trackingFlags = TrackingFlags.UNKNOWN

        self.timeInSamples = timeInSamples
        self.codeSinceTOW       = 0
        self.currentSample      = 0
        self.unprocessedSamples = 0

        self.isAcquired         = False
        self.isTOWDecoded       = False
        self.isEphemerisDecoded = False

        self.tow  = np.nan
        self.week = np.nan
        
        self.daemon  = True
        self.rfQueue = queue 
        self.pipe    = pipe

        # logger = logging.getLogger(f'CID{self.cid}')
        # logger.setLevel(logging.DEBUG)
        # # create file handler which logs even debug messages
        # fh = logging.FileHandler(f'./.results/log_CID{self.cid}.log')
        # fh.setLevel(logging.DEBUG)
        # logger.addHandler(fh)

        # self.logger = logger

        return
    
    # -------------------------------------------------------------------------

    def run(self):
        
        while True:
            try:
                rfData = self.rfQueue.get(timeout=PROCESS_TIMEOUT)
            except Empty:
                self.logger.debug(f"CID {self.cid} did not received data for {PROCESS_TIMEOUT} seconds, closing thread")
                return
            
            if rfData == "SIGTERM":
                logging.getLogger(__name__).debug(f"CID {self.cid} SIGTERM received, closing thread")
                break
            # else:
            #    self.logger.debug(f"CID {self.cid} received RF Data {rfData}")
            
            # Load the data in the buffer
            self._updateBuffer(rfData)

            # Process the data according to the current channel state
            self._processData()

            self.send(ChannelMessage.CHANNEL_UPDATE)
            self.send(ChannelMessage.END_OF_PIPE)

        logging.getLogger(__name__).debug(f"CID {self.cid} Exiting run") 

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
                logging.getLogger(__name__).debug(f"CID {self.cid} satellite G{self.svid} acquired")
                frequency, code = self.acquisition.getEstimation()
                self.tracking.setInitialValues(frequency)
                samplesRequired = self.tracking.getSamplesRequired()

                # Switching state to tracking for next loop
                self.state = ChannelState.TRACKING

                # We take double the amount required to be sure one full code will fit
                self.currentSample = self.buffer.getBufferMaxSize() - 2 * samplesRequired + (code + 1)
                self.unprocessedSamples = self.buffer.getBufferMaxSize() - self.currentSample

            else:
                logging.getLogger(__name__).debug(f"CID {self.cid} satellite G{self.svid} not acquired, channel state switched to IDLE")
                self.state = ChannelState.IDLE
        
            # Send to main program
            self.send(ChannelMessage.ACQUISITION_UPDATE, self.acquisition.getDatabaseDict())

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
                self.codeSinceTOW += 1

            # Update the index for samples
            self.currentSample = (self.currentSample + samplesRequired) % self.buffer.getBufferMaxSize()
            self.unprocessedSamples -= samplesRequired

            # Update for next loop
            samplesRequired = self.tracking.getSamplesRequired()
            buffer = self.buffer.getSlice(self.currentSample, samplesRequired)

            # Send to main program
            self.send(ChannelMessage.TRACKING_UPDATE, self.tracking.getDatabaseDict())

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

        if self.decoding.isTOWDecoded and np.isnan(self.tow):
            self.isTOWDecoded = True
            self.tow = self.decoding.tow
            self.codeSinceTOW = 0

        if self.decoding.isNewSubframeFound:
            self.send(ChannelMessage.DECODING_UPDATE, self.decoding.getDatabaseDict())
            self.decoding.isNewSubframeFound = False

        return

    # -------------------------------------------------------------------------

    def send(self, commType:ChannelMessage, dictToSend: dict = None):
        """
        Send a packet to main program through a defined pipe.
        """
        if commType == ChannelMessage.END_OF_PIPE:
            _packet = (commType)
        elif commType == ChannelMessage.CHANNEL_UPDATE:
            _packet = (commType, self.state, self.trackingFlags, self.week, self.tow, self.getTimeSinceTOW())
        elif commType == ChannelMessage.ACQUISITION_UPDATE \
            or commType == ChannelMessage.TRACKING_UPDATE \
            or commType == ChannelMessage.DECODING_UPDATE :

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
        timeSinceTOW += self.codeSinceTOW * self.gnssSignal.codeMs # Add number of code since TOW
        timeSinceTOW += self.unprocessedSamples / (self.rfSignal.samplingFrequency/1e3) # Add number of unprocessed samples 
        return timeSinceTOW

    # -------------------------------------------------------------------------

    def setSatellite(self, svid):
        # Update the configuration
        self.svid = svid

        # Update the methods
        self.acquisition.setSatellite(svid)
        self.tracking.setSatellite(svid)
        self.decoding.setSatellite(svid)

        # Set state to acquisition
        self.state = ChannelState.ACQUIRING

        logging.getLogger(__name__).debug(f"CID {self.cid} started with satellite G{self.svid}.")

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
        self.subframeFlags = []
        
        self.isTOWDecoded = False

        return

    def fromChannel(self, channel:ChannelAbstract):
        self.cid = channel.cid
        self.svid = channel.svid
        return self
