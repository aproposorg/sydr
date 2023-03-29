
import numpy as np
import logging
import multiprocessing
from multiprocessing import shared_memory
from queue import Empty

from core.channel.channel import Channel, ChannelState
from core.signal.rfsignal import RFSignal
from core.utils.circularbuffer import CircularBuffer

# =====================================================================================================================

class ChannelManager():
    """
    ChannelManager class to handle the channel processing and dissemination of RF data to channels.
    """

    TIMEOUT = 100

    channels : dict
    nbChannels : int

    # Memory management
    _sharedMemory : shared_memory.SharedMemory
    sharedBuffer : np.ndarray

    # Communication
    resultQueue : multiprocessing.Queue()

    # RF
    rfSignal : RFSignal

    def __init__(self, rfSignal:RFSignal) -> None:
        """
        Constructor for ChannelManager class. Some indication on the RF data needs to be provided for buffer memory
        allocation. 

        Args:
            rfSignal (RFSignal): Parameters of the RF Signal provided.

        Returns:
            None
        
        Raises:
            None
        """

        self.rfSignal = rfSignal

        self.channels = {}
        self.nbChannels = 0

        # Allocate shared memory 
        # Find the amount of space needed in shared storage
        # TODO Compute based on buffer size needed by channels
        buffersize = int(self.rfSignal.samplingFrequency * 1e-3 * 100)
        dtype = self.rfSignal.dtype
        nbytes = int(buffersize * np.dtype(dtype).itemsize)
        self._sharedMemory = shared_memory.SharedMemory(create=True, size=nbytes)
        self.sharedBuffer = CircularBuffer(buffersize, dtype, self._sharedMemory)

        # Communication
        self.resultQueue = multiprocessing.Queue()

        return

    # -----------------------------------------------------------------------------------------------------------------

    def addChannel(self, ChannelObject:type[Channel], configuration:dict, nbChannels=1):
        """
        Add channel(s) to the ChannelManager, providing the Channel type and the number of channels of that type to be
        added.

        Args:
            ChannelObject (type[Channel]): Channel class used to create the channel.
            configFilePath (str): Configuration file of the channel.
            nbChannels (int): Number of channel of that type to be created.

        Returns:
            None
        
        Raises:
            None
        """

        for i in range(nbChannels):
            channelID = self.nbChannels
            self.channels[channelID] = ChannelObject(channelID, self.sharedBuffer, self.resultQueue, 
                                                     self.rfSignal, configuration)
            self.nbChannels += 1
        
        return
    
    # -----------------------------------------------------------------------------------------------------------------

    def requestTracking(self, satelliteID:int):
        """
        Setup an available channel to track of a specific satellite.

        Args:
            satelliteID (int): Satellite PRN code.
        
        Returns:
            None

        Raises:
            None
        """
        
        # Loop through channels to find a free one
        # TODO handle case where there is no free channel
        channel : Channel
        success = False
        for channel in self.channels.values():
            if channel.channelState is ChannelState.IDLE:
                channel.setSatellite(satelliteID)
                channel.start()
                success = True
                break
        
        if success:
            logging.getLogger(__name__).debug(f"CID {channel.channelID} initialised to satellite [G{satelliteID}].")
        else:
            raise Warning(f"Could not find an IDLE channel for tracking satellite [G{satelliteID}].")
        
        return channel

    # -----------------------------------------------------------------------------------------------------------------

    def addNewRFData(self, data):
        """
        Add new RF signal data in the shared buffer area. 

        Args:
            data (np.array): New data added to the buffer.

        Returns:
            None
        
        Raises:
            None
        """
        self.sharedBuffer.shift(data)
        return
    
    # -----------------------------------------------------------------------------------------------------------------

    def run(self):
        """
        Run the channels based on the current RF data in the buffer. 

        Args:
            None

        Returns:
            results (list): Processing results from the channels.
        
        Raises:
            None
        """

        # Start the channels
        for chan in self.channels.values():
            chan.eventRun.set()
            #logging.getLogger(__name__).debug(f"CID {chan.channelID} processing started.")

        # Wait for channels to be done
        for chan in self.channels.values():
            chan.eventDone.wait()
            chan.eventDone.clear()
            #logging.getLogger(__name__).debug(f"CID {chan.channelID} processing finished.")

        # Process results
        _results = []
        # qsize is highlighted as "approximate count" in documentation, but in our case we have an event trigger 
        # before so the count should be exact.
        while self.resultQueue.qsize() > 0:
            try:
                packet = self.resultQueue.get(timeout=self.TIMEOUT)
            except Empty:
                break
            _results.append(packet)
        
        # Flatten list
        results = [element for sublist in _results for element in sublist]

        return results
    
    # -----------------------------------------------------------------------------------------------------------------
    
    def close(self):
        """
        Close the ChannelManager, kill the processes and free the memory allocations made for the buffer. 
        Should be called once at the end of the scenario when exiting the program.

        Args:
            None

        Returns:
            None
        
        Raises:
            None
        """

        self._sharedMemory.close()
        self._sharedMemory.unlink()

        return
    
    # -----------------------------------------------------------------------------------------------------------------

    def getChannel(self, channelID):
        """
        Return the channel provided a channelID.

        Args:
            channelID (int) : Channel ID number.

        Returns:
            channel (Channel): The channel with the requested Channel ID.

        Raises:
            ValueError: Channel ID does not exist.
        
        """
        if not (channelID in self.channels.keys()):
            raise ValueError("Channel ID does not exist.")

        return self.channels[channelID]

    # -----------------------------------------------------------------------------------------------------------------