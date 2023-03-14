
import numpy as np
import logging
import multiprocessing
from multiprocessing import shared_memory
from queue import Empty

from core.channel.channel import Channel
from core.signal.gnsssignal import GNSSSignal
from core.signal.rfsignal import RFSignal
from core.utils.circularbuffer import CircularBuffer


class ChannelManager():

    channels : list
    nbChannels : int

    # Memory management
    _sharedMemory : shared_memory.SharedMemory
    sharedBuffer : np.ndarray

    # Communication
    resultQueue : multiprocessing.Queue()

    def __init__(self, buffersize:int, dtype:np.dtype) -> None:
        """
        Constructor for ChannelManager class. Some indication on the RF data needs to be provided for buffer memory
        allocation. 

        Args:
            buffersize (int): Size of the data buffer in number of elements.
            dtype (np.dtype): Type of data in buffer.

        Returns:
            None
        
        Raises:
            None
        """

        self.channels = []
        self.nbChannels = 0

        # Allocate shared memory 
        nbytes = buffersize * np.dtype(dtype).itemsize
        self._sharedMemory = shared_memory.SharedMemory(create=True, size=nbytes)
        self.sharedBuffer = CircularBuffer(buffersize, dtype, self._sharedMemory)

        # RF queue
        self.resultQueue = multiprocessing.Queue()

        return

    # -----------------------------------------------------------------------------------------------------------------

    def addChannel(self, ChannelObject:type[Channel], configFilePath:str, nbChannels=1):
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
            self.channels.append(ChannelObject(self.nbChannels, self.sharedBuffer, self.resultQueue))
            self.channels[-1].start()
            self.nbChannels += 1
        
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
            chan.event_run.set()

        # Wait for channels to be done
        for chan in self.channels:
            chan.event_done.wait()

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
        Close the ChannelManager and free the memory allocations made for the buffer. Should be called once at the end
        of the scenario when exiting the program.
        """
        self._sharedMemory.close()
        self._sharedMemory.unlink()
        return
    
    # -----------------------------------------------------------------------------------------------------------------