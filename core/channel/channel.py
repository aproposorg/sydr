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
from abc import ABC, abstractmethod
from queue import Empty
from enum import Enum, unique

from core.signal.rfsignal import RFSignal
from core.signal.gnsssignal import GenerateGPSGoldCode
from core.utils.circularbuffer import CircularBuffer
from core.dsp.tracking import TrackingFlags
from core.utils.constants import GPS_L1CA_CODE_MS
from core.utils.enumerations import GNSSSystems, GNSSSignalType

# =============================================================================
@unique
class ChannelState(Enum):
    """
    Enumeration class for channel state according to the defined state machine architecture.
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
    Enumeration class for message types sent by the channel to the main thread.
    """
    END_OF_PIPE        = 0
    CHANNEL_UPDATE     = 1
    ACQUISITION_UPDATE = 2
    TRACKING_UPDATE    = 3
    DECODING_UPDATE    = 4

    def __str__(self):
        return str(self.name)

# =============================================================================
class Channel(ABC, multiprocessing.Process):
    """
    Abstract class for Channel object definition.
    """

    TIMEOUT = 100000 # Seconds before event timeout

    configuration : dict # Configuration dictionnary

    # IDs
    channelID    : np.uint8       # Channel ID
    channelState : ChannelState   # Current channel state
    trackFlags   : TrackingFlags

    # RF signal
    rfSignal     : RFSignal       # RF signal parameters
    rfBuffer     : CircularBuffer # Circular buffer for limited data storage
    
    unprocessedSamples : int    # Number of samples waiting to be processed
    currentSample      : int    # Current sample in RF Buffer

    # Satellite and GNSS signal
    systemID     : GNSSSystems
    satelliteID  : np.uint8       # Satellite ID
    signalID     : GNSSSignalType
    
    # Channel - Manager communication
    resultQueue : multiprocessing.Queue  # Queue to place the results of the channel processing
    eventRun    : multiprocessing.Event  # Event to start the channel processing when set.
    eventDone   : multiprocessing.Event  # Event set when channel processing done.

    tow  : int
    week : int
    codeSinceTOW : int

    # =========================================================================

    @abstractmethod
    def __init__(self, cid:int, sharedBuffer:CircularBuffer, resultQueue:multiprocessing.Queue, rfSignal:RFSignal,
                 configuration:dict):
        """
        Abstract constructor for Channel class. 

        Args:
            cid (int): Channel ID.
            sharedBuffer (CircularBuffer): Circular buffer with the RF data.
            resultQueue (multiprocessing.Queue): Queue to place the results of the channel processing

        Returns:
            None
        
        Raises:
            None
        """
        
        # For multiprocessing inheritance
        super(multiprocessing.Process, self).__init__(name=f'CID{cid}', daemon=True)

        self.configuration = configuration

        # Initialisation 
        self.channelID = cid
        self.channelState = ChannelState.IDLE
        self.satelliteID = 0
        self.rfBuffer = sharedBuffer
        self.resultQueue = resultQueue
        self.eventRun = multiprocessing.Event()
        self.eventDone = multiprocessing.Event()
        self.currentSample = 0
        self.unprocessedSamples = 0
        self.rfSignal = rfSignal

        self.tow = 0
        self.week = 0
        self.codeSinceTOW = 0

        return
    
    # -------------------------------------------------------------------------

    def setSatellite(self, satelliteID:np.uint8):
        """
        Set the GNSS signal and satellite tracked by the channel.
        """
        self.satelliteID = satelliteID
        self.channelState = ChannelState.ACQUIRING

        return
    
    # -------------------------------------------------------------------------

    def run(self):
        """
        Main processing loop, hanlding new RF data and channel processing.

        Args:
            nbNewSamples (int) : Number of new samples added to the shared buffer at each run.

        Returns:
            None

        Raises:
            None

        """
        
        while True:
            # Wait for ChannelManager event signal
            timeoutFlag = self.eventRun.wait(timeout=self.TIMEOUT)
            self.eventRun.clear()
            
            if not timeoutFlag:
                logging.getLogger(__name__).debug(f"CID {self.channelID} timeout, exiting run.")
                self.eventDone.set()
                break
            
            # Update samples tracker
            self.unprocessedSamples += self.rfSignal.samplesPerMs
            self.rfBuffer.shiftIdxWrite(self.rfSignal.samplesPerMs) # Update our copy of buffer

            # Process the data according to the current channel state
            results = self._processHandler()

            # Add channel update 
            results.append(self.prepareChannelUpdate())

            # Send the results
            self.resultQueue.put(results)

            # Signal channel manager
            self.eventDone.set()
            
        
        return

    # -------------------------------------------------------------------------

    @abstractmethod
    def _processHandler(self):
        """
        Abstract method, handle the RF Data based on the current channel state.
        """
        return

    # -------------------------------------------------------------------------

    def send(self, commType:ChannelMessage, dictToSend:dict=None):
        """
        Send a packet to main program through a defined pipe.
        """
        if commType == ChannelMessage.CHANNEL_UPDATE:
            _packet = (commType, self.channelState, self.trackFlags, self.tow, self.getTimeSinceTOW())
        elif commType in \
            (ChannelMessage.ACQUISITION_UPDATE, ChannelMessage.TRACKING_UPDATE, ChannelMessage.DECODING_UPDATE):
            for results in dictToSend:

                results["unprocessed_samples"] = int(self.unprocessedSamples)
                _packet = (commType, dictToSend)
        else:
            raise ValueError(f"Channel communication {commType} is not valid.")
        
        self.communicationPipe[0].send(_packet)

        return

    # -------------------------------------------------------------------------

    def getTimeSinceTOW(self):
        """
        Time since the last TOW in milliseconds.
        """
        timeSinceTOW = 0
        timeSinceTOW += self.codeSinceTOW * GPS_L1CA_CODE_MS # Add number of code since TOW
        timeSinceTOW += self.unprocessedSamples / (self.rfSignal.samplingFrequency/1e3) # Add number of unprocessed samples 
        return timeSinceTOW
    
    # -------------------------------------------------------------------------

    def prepareResults(self):
        mdict = {
            "cid" : self.channelID
        }
        return mdict
    
    # -------------------------------------------------------------------------
    
    def prepareResultsAcquisition(self):
        """
        Prepare the acquisition result to be sent. 
        """
        mdict = self.prepareResults()
        mdict["type"] = ChannelMessage.ACQUISITION_UPDATE
        return mdict
    
    # -------------------------------------------------------------------------
    
    def prepareResultsTracking(self):
        """
        Prepare the tracking result to be sent. 
        """
        mdict = self.prepareResults()
        mdict["type"] = ChannelMessage.TRACKING_UPDATE
        return mdict
    
    # -------------------------------------------------------------------------
    
    def prepareResultsDecoding(self):
        """
        Prepare the decoding result to be sent. 
        """
        mdict = self.prepareResults()
        mdict["type"] = ChannelMessage.DECODING_UPDATE
        return mdict
    
    # -------------------------------------------------------------------------

    def prepareChannelUpdate(self):
        
        _packet = self.prepareResults()
        _packet['type'] = ChannelMessage.CHANNEL_UPDATE
        _packet['state'] = self.channelState
        _packet['tracking_flags'] = self.trackFlags
        _packet['tow'] = self.tow
        _packet['time_since_tow'] = self.getTimeSinceTOW() 
        
        return _packet
    
    # -------------------------------------------------------------------------

# =============================================================================

# TODO update variables

class ChannelStatus():

    def __init__(self, channelID:int, satelliteID:int):

        self.channelID = channelID
        self.satelliteID = satelliteID
        self.channelState = ChannelState.IDLE
        self.trackFlags = TrackingFlags.UNKNOWN
        self.week = 0
        self.tow = 0
        self.timeSinceTOW = 0
        self.subframeFlags = []
        
        self.isTOWDecoded = False

        return

    def fromChannel(self, channel:Channel):
        self.cid = channel.channelID
        self.svid = channel.satelliteID
        return self

# =============================================================================
# END OF FILE
