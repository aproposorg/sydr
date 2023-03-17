
import numpy as np
import logging
import multiprocessing
from multiprocessing import shared_memory
from queue import Empty

from core.channel.channel import Channel, ChannelState
from core.signal.gnsssignal import GNSSSignal
from core.signal.rfsignal import RFSignal
from core.utils.circularbuffer import CircularBuffer


class ChannelManager():

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
        buffersize = self.rfSignal.samplingFrequency * 1e-3 * 100
        dtype = self.rfSignal.dtype
        nbytes = buffersize * np.dtype(dtype).itemsize
        self._sharedMemory = shared_memory.SharedMemory(create=True, size=nbytes)
        self.sharedBuffer = CircularBuffer(buffersize, dtype, self._sharedMemory)

        # RF queue
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
            self.channels[channelID] = ChannelObject(channelID, self.sharedBuffer, self.resultQueue, configuration)
            self.channels[channelID].start()
            self.nbChannels += 1
        
        return
    
    # -----------------------------------------------------------------------------------------------------------------

    def requestTracking(self, gnssSignal:GNSSSignal, satelliteID:int):
        """
        
        """
        
        # Loop through channels to find a free one
        channel : Channel
        for channel in self.channels:
            if channel.channelState is ChannelState.IDLE:
                channel.setSignalParameters(self.rfSignal, gnssSignal, satelliteID)
        
        return

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
        for chan in self.channels:
            chan.eventRun.set()

        # Wait for channels to be done
        for chan in self.channels:
            chan.eventDone.wait()

        # Process results
        results = []
        # qsize is highlighted as "approximate count" in documentation, but in our case we have an event trigger 
        # before so the count should be exact.
        while self.resultQueue.qsize() > 0:
            try:
                packet = self.resultQueue.get(timeout=self.timeout)
            except Empty:
                break
            results.append(packet)
                
        return results
    
    # -----------------------------------------------------------------------------------------------------------------
    
    def close(self):
        """
        Close the ChannelManager, kill the processes and free the memory allocations made for the buffer. 
        Should be called once at the end of the scenario when exiting the program.
        """
        self._sharedMemory.close()
        self._sharedMemory.unlink()

        for chan in self.channels:
            chan.terminate()

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