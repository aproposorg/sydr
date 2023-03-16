from enum import Enum, unique
import sqlite3

# =============================================================================

@unique
class GNSSSystems(Enum):
    GPS     = 1
    GLONASS = 2
    GALILEO = 3
    BEIDOU  = 4
    QZSS    = 5
    IRNSS   = 6
    SBAS    = 7
    UNKNOWN = 0

    def __str__(self):
        return str(self.name)

    def __conform__(self, protocol):
        """
        Definition of the object reperesentation.
        This is used for interaction with the database. 
        """
        if protocol is sqlite3.PrepareProtocol:
            return str(self.name)

# =============================================================================

@unique
class GNSSMeasurementType(Enum):
    PSEUDORANGE = 1
    PHASE       = 2
    DOPPLER     = 3
    SNR         = 4
    UNKNOWN     = 0

    def __str__(self):
        return str(self.name)

    def __conform__(self, protocol):
        """
        Definition of the object reperesentation.
        This is used for interaction with the database. 
        """
        if protocol is sqlite3.PrepareProtocol:
            return str(self.name)

# =============================================================================

@unique
class GNSSSignalType(Enum):
    GPS_L1_CA = 0
    
    def __str__(self):
        return str(self.name).replace("_", " ")
    
    def __conform__(self, protocol):
        """
        Definition of the object reperesentation.
        This is used for interaction with the database. 
        """
        if protocol is sqlite3.PrepareProtocol:
            return str(self.name)

# =============================================================================

@unique
class ReceiverState(Enum):
    OFF        = 0
    IDLE       = 1
    INIT       = 2
    NAVIGATION = 3

    def __str__(self):
        return str(self.name)


# =============================================================================