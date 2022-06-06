

class CircularBuffer:
    """
    
    We assume that the shift will always be a multiple from the max size.
    """

    buffer        : list
    idxWrite      : int
    idxRead       : int
    bufferMaxSize : int
    bufferFull    : bool

    def __init__(self, size):
        self.bufferMaxSize = size
        self.buffer = [] * self.bufferMaxSize

        self.bufferFull = False

        self.idxWrite = 0
        self.idxRead = 0

        return

    def shift(self, data):
        """
        We assume that the shift will always be a multiple from the buffer max size.
        """

        shift = len(data)
        self.buffer[self.idxWrite:self.idxWrite + shift] = data
        self.idxWrite += shift

        # Update buffer size
        if self.bufferFull:
            self.idxWrite %= self.bufferMaxSize
        else:
            if self.idxWrite >= self.bufferMaxSize:
                self.bufferFull = True
                self.idxWrite %= self.bufferMaxSize

        return

    def getSlice(self, idxStart, samplesRequired):

        idxStop = (idxStart + samplesRequired) % self.bufferMaxSize

        if idxStop < idxStart:
            return self.buffer[idxStart:] + self.buffer[:idxStop]
        else: 
            return self.buffer[idxStart:idxStop]
        

        # if idxStart + samplesRequired <= self.bufferMaxSize:
        #     idxStop = idxStart + samplesRequired
        # else:
        #     idxStop = (idxStart + samplesRequired) % self.bufferMaxSize
        
        # if idxStop < idxStart:
        #     return self.buffer[idxStart:] + self.buffer[:idxStop]
        # else: 
        #     return self.buffer[idxStart:idxStop]

    def getBuffer(self):
        return self.buffer

    def getBufferMaxSize(self):
        return self.bufferMaxSize
    
    def isFull(self):
        return self.bufferFull