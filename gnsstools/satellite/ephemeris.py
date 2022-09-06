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

        self.iode     = -1
        self.iodc     = -1
        self.toe      = -1.
        self.toc      = -1.
        self.tgd      = -1.
        self.af2      = -1.
        self.af1      = -1.
        self.af0      = -1.
        self.ecc      = -1. 
        self.sqrtA    = -1.
        self.crs      = -1.
        self.deltan   = -1.
        self.m0       = -1.
        self.cuc      = -1.
        self.cus      = -1.
        self.cic      = -1.
        self.omega0   = -1.
        self.cis      = -1.
        self.i0       = -1.
        self.crc      = -1.
        self.omega    = -1.
        self.omegaDot = -1.
        self.iDot     = -1.
        self.alpha0   = -1.
        self.ura      = -1.
        self.health   = -1.
        
        return

    def __eq__(self, other) -> bool:

        if self.iodc == other.iodc and self.iode == other.iode:
            return True
        else:
            return False 

    def resetFlags(self):
        self.subframe1Flag = False
        self.subframe2Flag = False
        self.subframe3Flag = False

    def checkFlags(self):
        return self.subframe1Flag and self.subframe2Flag and self.subframe3Flag

    def computePosition(self):
        # TODO
        pass