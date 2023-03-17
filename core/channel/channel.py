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

from core.signal.gnsssignal import GNSSSignal
from core.signal.rfsignal import RFSignal
from core.utils.circularbuffer import CircularBuffer
# =============================================================================
PROCESS_TIMEOUT = 60 # Seconds 
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

    TIMEOUT = 10 # Seconds before event timeout

    # IDs
    channelID    : np.uint8       # Channel ID
    channelState : ChannelState   # Current channel state

    # RF signal
    rfSignal     : RFSignal       # RF signal parameters
    rfBuffer     : CircularBuffer # Circular buffer for limited data storage
    
    nbUnprocessedSamples : int    # Number of samples waiting to be processed

    # Satellite and GNSS signal
    gnssSignal   : GNSSSignal     # GNSS signal parameters 
    satelliteID  : np.uint8       # Satellite ID
    
    # Channel - Manager communication
    resultQueue : multiprocessing.Queue  # Queue to place the results of the channel processing
    eventRun    : multiprocessing.Event  # Event to start the channel processing when set.
    eventDone   : multiprocessing.Event  # Event set when channel processing done.

    configuration : dict # Configuration dictionnary

    # =========================================================================

    @abstractmethod
    def __init__(self, cid:int, sharedBuffer:CircularBuffer, resultQueue:multiprocessing.Queue, configuration:dict):
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

        # Initialisation 
        self.channelID = cid
        self.channelState = ChannelState.IDLE
        self.satelliteID = 0
        self.rfBuffer = sharedBuffer
        self.resultQueue = resultQueue
        self.eventRun = multiprocessing.Event()
        self.eventDone = multiprocessing.Event()

        self.unprocessedSamples = 0

        self.configuration = configuration

        return

    # -------------------------------------------------------------------------

    def setSignalParameters(self, rfSignal:RFSignal, gnssSignal:GNSSSignal, satelliteID:np.uint8):
        """
        Set the GNSS signal and satellite tracked by the channel.
        """
        self.rfSignal = rfSignal
        self.satelliteID = satelliteID
        self.gnssSignal = gnssSignal
        self.channelState = ChannelState.ACQUIRING
        logging.getLogger(__name__).debug(f"CID {self.cid} initialised to satellite [G{self.svid}], signal [{gnssSignal.name}].")
        return
    
    # -------------------------------------------------------------------------

    def run(self, nbNewSamples:int):
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
            
            if timeoutFlag:
                logging.getLogger(__name__).debug(f"CID {self.cid} timeout, exiting run.") 
                break
            
            # Update samples tracker
            self.unprocessedSamples += nbNewSamples

            # Process the data according to the current channel state
            results = self._processHandler()

            # Send the results
            self.resultQueue.put(results)
        
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
        if commType == ChannelMessage.END_OF_PIPE:
            _packet = (commType)
        elif commType == ChannelMessage.CHANNEL_UPDATE:
            _packet = (commType, self.state, self.trackingFlags, self.tow, self.getTimeSinceTOW())
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
        timeSinceTOW += self.codeSinceTOW * self.gnssSignal.codeMs # Add number of code since TOW
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


# =============================================================================
# END OF FILE
