
import multiprocessing
from multiprocessing import shared_memory
import numpy as np
from abc import ABC, abstractmethod
import logging
from queue import Empty

# =====================================================================================================================

class Channel(ABC, multiprocessing.Process):

    def __init__(self, cid, sharedBuffer, resultQueue):
        super(multiprocessing.Process, self).__init__(name=f'CID{cid}', daemon=True)
        self.cid = cid
        self.buffer = sharedBuffer
        self.resultQueue = resultQueue
        self.event_run = multiprocessing.Event()
        self.event_done = multiprocessing.Event()
        return
    
    @abstractmethod
    def run(self):
        self.event_run.wait()
        self.event_run.clear()
        return
    
# =====================================================================================================================

class ChannelGPSL1CA(Channel):

    def __init__(self, cid, sharedBuffer, resultQueue):
        super().__init__(cid, sharedBuffer, resultQueue)
        return
    
    def run(self):
        while 1:
            super().run()
            result = np.sum(self.buffer)
            packet = {}
            packet["cid"] = self.cid
            packet["result"] = result
            print(f"From channel: CID {packet['cid']}, {packet['result']}")
            self.resultQueue.put(packet)
            self.event_done.set()
        return 

# =====================================================================================================================

class ChannelManager():

    def __init__(self, size:int, dtype:np.int8, timeout:int) -> None:

        self.channels = []
        self.nbChannels = 0

        # Allocate shared memory 
        nbytes = size * np.dtype(dtype).itemsize
        self._sharedMemory = shared_memory.SharedMemory(create=True, size=nbytes)
        self.sharedBuffer = np.ndarray(shape=(1, size), dtype=dtype, buffer=self._sharedMemory.buf)

        # RF queue
        self.resultQueue = multiprocessing.Queue()
        self.timeout = timeout

        return
    
    def addChannel(self, ChannelObject, nbChannels=1):

        for i in range(nbChannels):
            self.channels.append(ChannelObject(self.nbChannels, self.sharedBuffer, self.resultQueue))
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
                packet = self.resultQueue.get(timeout=self.timeout)
            except Empty:
                break
            print(f"From manager: CID {packet['cid']}, {packet['result']}")
            results.append(packet)
                
        return results
    
    def close(self):
        self._sharedMemory.close()
        self._sharedMemory.unlink()
        return
    
# =====================================================================================================================

class Receiver():
    def __init__(self, nbChannels):
        self.nbChannels = nbChannels
        self.channelManager = ChannelManager(size=10, dtype=np.int8, timeout=1)
        self.channelManager.addChannel(ChannelGPSL1CA, nbChannels)
        return
    
    def run(self):

        data = np.arange(0, 100, 1)

        for i in range(10):
            self.channelManager.addNewRFData(data[i*10:i*10+10])
            results = self.channelManager.run()

            for packet in results:
                print(f"From receiver: CID {packet['cid']}, {packet['result']}")

        return
    
    def close(self):
        self.channelManager.close()
        return

if __name__ == '__main__':

    receiver = Receiver(4)
    receiver.run()
    receiver.close()

    # CURRENT POINT: FILL RUN IN RECEIVER AND TEST THE CODE