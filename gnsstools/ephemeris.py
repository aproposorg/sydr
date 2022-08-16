from abc import ABC, abstractclassmethod
import numpy as np

class Ephemeris(ABC):

    @abstractclassmethod
    def computePosition(self):
        pass

# =============================================================================

class BRDCEphemeris(Ephemeris):
    iode     : int
    iodc     : int
    toe      : float
    toc      : float
    tgd      : float
    af2      : float
    af1      : float
    af0      : float
    ecc      : float 
    sqrtA    : float
    crs      : float
    deltan   : float
    m0       : float
    cuc      : float
    cus      : float
    cic      : float
    omega0   : float
    cis      : float
    i0       : float
    crc      : float
    omega    : float
    omegaDot : float
    iDot     : float
    alpha0   : float
    ura      : float
    health   : float
    
    tow        : int
    weekNumber : int

    subframe1Flag : bool
    subframe2Flag : bool
    subframe3Flag : bool

    def __init__(self):
        self.subframe1Flag = False
        self.subframe2Flag = False
        self.subframe3Flag = False
        return

    def resetFlags(self):
        self.subframe1Flag = False
        self.subframe2Flag = False
        self.subframe3Flag = False

    def checkFlags(self):
        return self.subframe1Flag and self.subframe2Flag and self.subframe3Flag

    def computePosition(self):
        # TODO
        pass