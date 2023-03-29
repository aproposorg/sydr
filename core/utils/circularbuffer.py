
import numpy as np
from multiprocessing import shared_memory
import multiprocessing

class CircularBuffer:
    """
    A circular buffer class, with possibility to add new ordered batch of data to a buffer without the need to shift and
    copy the elements to keep the order. 
    """

    buffer        : list
    idxWrite      : int
    idxRead       : int
    size          : int
    maxSize       : int
    full          : bool
    dtype         : np.dtype
    sharedMemory  : shared_memory.SharedMemory

    def __init__(self, size:int, dtype:np.dtype=complex, sharedMemory:shared_memory.SharedMemory=None):
        """
        Constructor for CircularBuffer class. 

        Args:
            size (int): Size of the data buffer in number of elements.
            dtype (np.dtype): (Optional) Type of data in buffer. Default value is np.float64.
            sharedMemory (SharedMemory): (Optional) Shared memory space to allocated the array. 

        Returns:
            None
        
        Raises:
            None
        
        """
        self.maxSize = size

        if sharedMemory == None:
            self.buffer = np.ndarray((1, self.maxSize), dtype=dtype)
        else:
            self.buffer = np.ndarray((1, self.maxSize), dtype=dtype, buffer=sharedMemory.buf)

        self.dtype = dtype
        self.sharedMemory = sharedMemory
        self.full = False
        self.idxWrite = 0
        self.size = 0

        return

    # -----------------------------------------------------------------------------------------------------------------

    def shift(self, data:np.array):
        """
        Add the new data in the circular buffer. Shift the the writting index accordingly. We assume that 
        the shift will always be a multiple from the buffer max size.

        Args:
            data (np.array): New data added to the buffer.

        Returns:
            None
        
        Raises:
            ValueError: Data shift need to be a multiple from the max buffer size.
        
        """

        shift = len(data)

        # Check if shift is multiple from buffer size
        if self.maxSize % shift != 0:
            raise ValueError("Data shift need to be a multiple from the max buffer size.")

        # Update buffer data
        self.buffer[:, self.idxWrite:self.idxWrite + shift] = data
        
        # Shift other variables
        self.shiftIdxWrite(shift)

        return
    
    # -----------------------------------------------------------------------------------------------------------------

    def shiftIdxWrite(self, shift:int):
        
        self.idxWrite += shift
        self.size = self.idxWrite

        # Update buffer size
        if self.full:
            self.idxWrite %= self.maxSize
        else:
            if self.idxWrite >= self.maxSize:
                self.full = True
                self.idxWrite %= self.maxSize
            if self.size > self.maxSize:
                self.size = self.maxSize
        
        return
    
    # -----------------------------------------------------------------------------------------------------------------
    
    def shiftIdxRead(self, shift:int):

        self.idxRead += shift
        self.idxRead %= self.maxSize

        return

    # -----------------------------------------------------------------------------------------------------------------

    def getSlice(self, idxStart:int=None, samplesRequired:int=0):
        """
        Get a data slice of the buffer, providing the starting index and the amount of samples needed.

        Args:
            idxStart (int) : Starting index of the returned array.
            samplesRequired (int): Size of the array returned. 

        Returns:
            None
        
        Raises:
            None
        """

        if idxStart is None:
            idxStart = self.idxRead

        idxStop = (idxStart + samplesRequired) % self.maxSize

        if idxStop < idxStart:
            return np.concatenate((self.buffer[:, idxStart:], self.buffer[:, :idxStop]), axis=1)
        else: 
            return self.buffer[:, idxStart:idxStop]
        
    # -----------------------------------------------------------------------------------------------------------------

    def getNbUnreadSamples(self, currentSample:int):
        """
        """
        
        if currentSample <= self.idxWrite:
            return self.idxWrite - currentSample
        else:
            return self.maxSize - currentSample + self.idxWrite

    # -----------------------------------------------------------------------------------------------------------------