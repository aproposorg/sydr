# -*- coding: utf-8 -*-
# ============================================================================
# Abstract class for channel definition.
# Author: Antoine GRENIER (TAU)
# Date: 2022.05.04
# References: 
# =============================================================================
# PACKAGES
import logging
import time
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
class Channel(ABC, multiprocessing.Process):
    """
    Abstract class for Channel object definition.
    """

    # IDs
    channelID    : np.uint8       # Channel ID
    channelState : ChannelState   # Current channel state

    # RF signal
    rfSignal     : RFSignal       # RF signal parameters
    rfBuffer     : CircularBuffer    # Circular buffer for limited data storage
    rfQueue      : multiprocessing.Queue  # Queue for reception of new data from main thread, pipe is not used as it has limited storage

    # Satellite and GNSS signal
    gnssSignal   : GNSSSignal     # GNSS signal parameters 
    satelliteID  : np.uint8       # Satellite ID
    
    # Channel - receiver communication
    communicationPipe : multiprocessing.Pipe  # Pipe for communication between channel and main thread

    # Multiprocessing
    daemon  : bool                      # Start as a daemon process (see multiprocessing documentation)

    # =========================================================================

    @abstractmethod
    def __init__(self, cid:int, pipe:multiprocessing.Pipe, multiprocessing=False):
        """
        Constructor.
        """
        
        # For multiprocessing inheritance
        if(multiprocessing):
            super(multiprocessing.Process, self).__init__()
            self.daemon  = True

        # Initialisation 
        self.channelID     = cid
        self.channelState  = ChannelState.IDLE

        # Communication
        self.pipe = pipe

        return
    
    # -------------------------------------------------------------------------

    def setGNSSSignalParameters(self, gnssSignal:GNSSSignal, satelliteID:np.uint8):
        self.satelliteID = satelliteID
        self.gnssSignal = gnssSignal
        self.channelState = ChannelState.ACQUIRING
        logging.getLogger(__name__).debug(f"CID {self.cid} initialised to satellite [G{self.svid}], signal [{gnssSignal.name}].")
        return
    
    # -------------------------------------------------------------------------

    def setRFSignalParameters(self, rfSignal:RFSignal, queue:multiprocessing.Queue):
        self.rfQueue  = queue 
        self.rfSignal = rfSignal
        logging.getLogger(__name__).debug(f"CID {self.cid} initialised with RF signal parameters.")
        return
    
    # -------------------------------------------------------------------------

    def run(self):
        """
        Main processing loop, handling the reception of new RF data and channel processing.
        """
        
        while True:
            try:
                rfData = self.rfQueue.get(timeout=PROCESS_TIMEOUT)
            except Empty:
                self.logger.debug(f"CID {self.cid} did not received data for {PROCESS_TIMEOUT} seconds, closing thread")
                return
            # TODO Add except if rfQueue not init
            
            if rfData == "SIGTERM":
                logging.getLogger(__name__).debug(f"CID {self.cid} SIGTERM received, closing thread")
                break
            # else:
            #    self.logger.debug(f"CID {self.cid} received RF Data {rfData}")
            
            # Load the data in the buffer
            self._updateBuffer(rfData)

            # Process the data according to the current channel state
            self._processHandler()

            self.send(ChannelMessage.CHANNEL_UPDATE)
            self.send(ChannelMessage.END_OF_PIPE)

        logging.getLogger(__name__).debug(f"CID {self.cid} Exiting run") 

        return

    # -------------------------------------------------------------------------

    def _processHandler(self):
        """
        Handle the RF Data based on the current channel state.
        """

        # Check channel state
        #self.logger.debug(f"CID {self.cid} is in {self.state} state")

        if self.state == ChannelState.IDLE:
            print(f"WARNING: Tracking channel {self.cid} is in IDLE.")
            self.send({})

        elif self.state == ChannelState.ACQUIRING:
            self.runAcquisition()
            
        elif self.state == ChannelState.TRACKING:
            if self.isAcquired:
                self.isAcquired = False # Reset the flag, otherwise we log acquisition each loop
            # Track
            self.runTracking()

            # If decoding
            self.doDecoding()
        else:
            raise ValueError(f"Channel state {self.state} is not valid.")

        return

    # -------------------------------------------------------------------------

    @abstractmethod
    def acquisition(self):
        """
        Abstract method, perform the acquisition operation on the current RF signal. 
        """
        return

    # -------------------------------------------------------------------------

    @abstractmethod
    def tracking(self):
        """
        Abstract method, perform the tracking operation on the current RF signal.  
        """
        return

    # -------------------------------------------------------------------------

    @abstractmethod
    def decoding(self):
        """
        Abstract method, perform the decoding operation on the current RF signal. 
        """
        return

    # -------------------------------------------------------------------------

    def send(self, commType:ChannelMessage, dictToSend:dict=None, processTimeNanos=0.0):
        """
        Send a packet to main program through a defined pipe.
        """
        if commType == ChannelMessage.END_OF_PIPE:
            _packet = (commType)
        elif commType == ChannelMessage.CHANNEL_UPDATE:
            _packet = (commType, self.state, self.trackingFlags, self.tow, self.getTimeSinceTOW())
        elif commType == ChannelMessage.ACQUISITION_UPDATE \
            or commType == ChannelMessage.TRACKING_UPDATE \
            or commType == ChannelMessage.DECODING_UPDATE :

            dictToSend["unprocessed_samples"] = int(self.unprocessedSamples)
            dictToSend["processTimeNanos"] = int(processTimeNanos)
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

    def _updateBuffer(self, data:np.array):
        """
        Update buffer with new data, shift the previous data and update the 
        required variables.
        """

        self.buffer.shift(data)
        self.unprocessedSamples += len(data)

        return

# =============================================================================
# END OF FILE
