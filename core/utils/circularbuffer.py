

class CircularBuffer:
    """
    
    We assume that the shift will always be a multiple from the max size.
    """

    buffer        : list
    idxWrite      : int
    idxRead       : int
    size          : int
    maxSize       : int
    full          : bool

    def __init__(self, size):
        self.maxSize = size
        self.buffer = [] * self.maxSize

        self.full = False

        self.idxWrite = 0
        self.idxRead = 0

        self.size = 0

        return

    def shift(self, data):
        """
        We assume that the shift will always be a multiple from the buffer max size.
        """

        shift = len(data)
        self.buffer[self.idxWrite:self.idxWrite + shift] = data
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

    def getSlice(self, idxStart, samplesRequired):

        idxStop = (idxStart + samplesRequired) % self.maxSize

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
        return self.maxSize
    
    def isFull(self):
        return self.full