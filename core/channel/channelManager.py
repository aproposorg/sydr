
import numpy as np
import logging
import multiprocessing
from multiprocessing import shared_memory

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
    timeout : int

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
        self.sharedBuffer = np.ndarray(shape=(1, buffersize), dtype=dtype, buffer=self._sharedMemory.buf)

        # RF queue
        self.resultQueue = multiprocessing.Queue()

        return

    # -----------------------------------------------------------------------------------------------------------------

class ChannelManager():

    def __init__(self, size:int, dtype:np.int8, timeout:int) -> None:

        self.channels = []
        self.nbChannels = 0

        # Allocate shared memory 
        nbytes = size * np.dtype(dtype).itemsize
        self._sharedMemory = shared_memory.SharedMemory(create=True, size=nbytes)
        self.sharedBuffer = np.ndarray(shape=(1, size), dtype=dtype, buffer=self._sharedMemory.buf)

        # RF queue
        self.rfQueue = multiprocessing.Queue()
        self.receiverPipe = multiprocessing.Pipe()
        self.resultQueue = multiprocessing.Queue()
        self.timeout = timeout

        return
    
    def addChannel(self, ChannelObject, nbChannels=1):

        for i in range(nbChannels):
            events = (multiprocessing.Event(), multiprocessing.Event())
            self.channels.append(ChannelObject(self.nbChannels, self.sharedBuffer, self.resultQueue, events))
            self.channels[-1].start()
            self.nbChannels += 1
        return
    
    def addNewRFData(self, data):
        self.sharedBuffer[:] = data
        return
    
    def run(self):
        # Start the channels
        for chan in self.channels:
            chan.event_run.set()

        # Wait for channels to be done
        for chan in self.channels:
            chan.event_done.wait()

        # Process results
        results = []
        # qsize is highlighted as "approximate count" in documentation, but in our case we have an event trigger before.
        while self.resultQueue.qsize() > 0:
            try:
                packet = self.resultQueue.get(timeout=1)
            except Empty:
                break
            print(f"From manager: CID {packet['cid']}, {packet['result']}")
            results.append(packet)
                
        return results
    
    def close(self):
        self._sharedMemory.close()
        self._sharedMemory.unlink()
        return