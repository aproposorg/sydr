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
    DSP_UPDATE         = 2

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
        self.communicationPipe = pipe

        return
    
    # -------------------------------------------------------------------------

    def setGNSSSignalParameters(self, gnssSignal:GNSSSignal, satelliteID:np.uint8):
        """
        Set the GNSS signal and satellite tracked by the channel.
        """
        self.satelliteID = satelliteID
        self.gnssSignal = gnssSignal
        self.channelState = ChannelState.ACQUIRING
        logging.getLogger(__name__).debug(f"CID {self.cid} initialised to satellite [G{self.svid}], signal [{gnssSignal.name}].")
        return
    
    # -------------------------------------------------------------------------

    def setRFSignalParameters(self, rfSignal:RFSignal, queue:multiprocessing.Queue):
        """
        Set the parameters of the RF signals sent to the channel.
        """
        self.rfQueue  = queue 
        self.rfSignal = rfSignal
        logging.getLogger(__name__).debug(f"CID {self.cid} initialised with RF signal parameters.")
        return
    
    # -------------------------------------------------------------------------

    def setBufferSize(self, size):
        """
        Set the size of the RF data buffer.
        """
        self.rfBuffer = CircularBuffer(size)
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
            results = self._processHandler()

            # Send the results
            self.send(ChannelMessage.DSP_UPDATE, results)

            self.send(ChannelMessage.CHANNEL_UPDATE)
            self.send(ChannelMessage.END_OF_PIPE)

        logging.getLogger(__name__).debug(f"CID {self.cid} Exiting run") 

        return

    # -------------------------------------------------------------------------

    @abstractmethod
    def _processHandler(self):
        """
        Abstract method, handle the RF Data based on the current channel state.
        """
        return

    # -------------------------------------------------------------------------

    @abstractmethod
    def setAcquisition(self):
        """
        Abstract method, set the needed parameters for the acquisition operations
        """
        return

    # -------------------------------------------------------------------------

    @abstractmethod
    def runAcquisition(self):
        """
        Abstract method, perform the acquisition operation on the current RF signal. 
        """
        return

    # -------------------------------------------------------------------------

    @abstractmethod
    def setTracking(self):
        """
        Abstract method, set the needed parameters for the tracking operations
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

    def send(self, commType:ChannelMessage, dictToSend:dict=None):
        """
        Send a packet to main program through a defined pipe.
        """
        if commType == ChannelMessage.END_OF_PIPE:
            _packet = (commType)
        elif commType == ChannelMessage.CHANNEL_UPDATE:
            _packet = (commType, self.state, self.trackingFlags, self.tow, self.getTimeSinceTOW())
        elif commType == ChannelMessage.DSP_UPDATE:
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

    def _updateBuffer(self, data:np.array):
        """
        Update buffer with new data, shift the previous data and update the 
        required variables.
        """

        self.buffer.shift(data)
        self.unprocessedSamples += len(data)

        return
    
    # -------------------------------------------------------------------------
    
    def prepareResultsAcquisition(self):
        """
        Prepare the acquisition result to be sent. 
        """
        mdict = {
            "type" : "acquisition"
        }
        return mdict
    
    # -------------------------------------------------------------------------
    
    def prepareResultsTracking(self):
        """
        Prepare the acquisition result to be sent. 
        """
        mdict = {
            "type" : "tracking"
        }
        return mdict

# =============================================================================
# END OF FILE
