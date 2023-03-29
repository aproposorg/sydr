# -*- coding: utf-8 -*-
# ============================================================================
# Abstract class for channel definition.
# Author: Antoine GRENIER (TAU)
# Date: 2022.03.23
# References: 
# =============================================================================
# PACKAGES
import logging
import multiprocessing
import numpy as np
from abc import ABC, abstractmethod

from core.signal.rfsignal import RFSignal
from core.utils.circularbuffer import CircularBuffer
from core.utils.enumerations import TrackingFlags
from core.utils.enumerations import GNSSSystems, GNSSSignalType, ChannelState, ChannelMessage

# =====================================================================================================================

class Channel(ABC, multiprocessing.Process):
    """
    Abstract class for Channel object definition.
    """

    TIMEOUT = 100000 # Seconds before event timeout

    configuration : dict # Configuration dictionnary

    # IDs
    channelID    : np.uint8       # Channel ID
    channelState : ChannelState   # Current channel state
    trackFlags   : TrackingFlags  # Tracking flags handling the current status

    # RF signal
    rfSignal     : RFSignal       # RF signal parameters
    rfBuffer     : CircularBuffer # Circular buffer for limited data storage
    
    currentSample      : int    # Current sample in RF Buffer

    # Satellite and GNSS signal
    systemID     : GNSSSystems    # GNSS system ID
    satelliteID  : np.uint8       # Satellite ID
    signalID     : GNSSSignalType # GNSS signal ID
    
    # Channel - Manager communication
    resultQueue : multiprocessing.Queue  # Queue to place the results of the channel processing
    eventRun    : multiprocessing.Event  # Event to start the channel processing when set.
    eventDone   : multiprocessing.Event  # Event set when channel processing done.

    tow  : int
    week : int
    codeSinceTOW : int

    # -----------------------------------------------------------------------------------------------------------------

    @abstractmethod
    def __init__(self, cid:int, sharedBuffer:CircularBuffer, resultQueue:multiprocessing.Queue, rfSignal:RFSignal,
                 configuration:dict):
        """
        Abstract constructor for Channel class. 

        Args:
            cid (int): Channel ID.
            sharedBuffer (CircularBuffer): Circular buffer with the RF data.
            resultQueue (multiprocessing.Queue): Queue to place the results of the channel processing
            rfSignal (RFSignal): RFSignal object for RF configuration.
            configuration (dict): Configuration dictionnary for channel.

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
        self.rfSignal = rfSignal

        self.tow = 0
        self.week = 0
        self.codeSinceTOW = 0

        return
    
    # -----------------------------------------------------------------------------------------------------------------

    def setSatellite(self, satelliteID:int):
        """
        Set the GNSS signal and satellite tracked by the channel.

        Args:
            satelliteID (int): ID (PRN code) of the satellite.
        
        Returns:
            None
        
        Raises:
            None
        """
        self.satelliteID = satelliteID
        self.channelState = ChannelState.ACQUIRING

        return
    
    # -----------------------------------------------------------------------------------------------------------------

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
        self.unprocessedSamples = 0
        while True:
            # Wait for ChannelManager event signal
            timeoutFlag = self.eventRun.wait(timeout=self.TIMEOUT)
            self.eventRun.clear()
            
            if not timeoutFlag:
                logging.getLogger(__name__).debug(f"CID {self.channelID} timeout, exiting run.")
                self.eventDone.set()
                break
            
            # Update samples tracker
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

    # -----------------------------------------------------------------------------------------------------------------

    @abstractmethod
    def _processHandler(self):
        """
        Abstract method, handle the RF Data based on the current channel state.

        Args:
            None
        
        Returns:
            None

        Raises:
            None
        """
        return
    
    # -----------------------------------------------------------------------------------------------------------------

    def prepareResults(self):
        """
        Prepare the result packet sent by the channel. This method is suppose to be the basis of the other results 
        methods.

        Args:
            None
        
        Returns: 
            None

        Raises:
            None
        """

        # Create result dictionnary
        mdict = {
            "cid" : self.channelID
        }
        return mdict
    
# =====================================================================================================================

class ChannelStatus(ABC):
    """
    Abstract class for ChannelStatus handling.
    """

    def __init__(self, channelID:int, satelliteID:int):
        """
        Constructor for ChannelStatus class. 

        Args:
            channelID (int): Channel ID
            satellite (int): Satellite PRN code

        Returns:
            None

        Raises:
            None
        """

        self.channelID = channelID
        self.satelliteID = satelliteID
        self.channelState = ChannelState.IDLE
        self.trackFlags = TrackingFlags.UNKNOWN
        self.week = 0
        self.tow = 0
        self.timeSinceTOW = 0
        self.subframeFlags = []
        self.unprocessedSamples = 0
        
        self.isTOWDecoded = False

        return

# =====================================================================================================================
# END OF FILE
