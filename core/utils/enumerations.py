# -*- coding: utf-8 -*-
# ============================================================================
# Abstract class for channel definition.
# Author: Antoine GRENIER (TAU)
# Date: 2022.03.23 (Updated: )
# References: 
# =============================================================================
# PACKAGES
from enum import Enum, unique, IntEnum
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

@unique
class ChannelState(Enum):
    """
    Enumeration class for channel state according to the defined state machine architecture.
    """
    OFF           = 0
    IDLE          = 1
    ACQUIRING     = 2
    TRACKING      = 3

    def __str__(self):
        return str(self.name)

# =============================================================================

@unique
class ChannelMessage(Enum):
    """
    Enumeration class for message types sent by the channel to the main thread.
    """
    END_OF_PIPE        = 0
    CHANNEL_UPDATE     = 1
    ACQUISITION_UPDATE = 2
    TRACKING_UPDATE    = 3
    DECODING_UPDATE    = 4

    def __str__(self):
        return str(self.name)

# =====================================================================================================================

@unique
class TrackingFlags(IntEnum):
    """
    Tracking flags to represent the current stage of tracking. They are to be intepreted in binary format, to allow 
    multiple state represesented in one decimal number. 
    Similar to states in https://developer.android.com/reference/android/location/GnssMeasurement 
    """

    UNKNOWN       = 0    # 0000 0000 No tracking
    CODE_LOCK     = 1    # 0000 0001 Code found (after first tracking?)
    BIT_SYNC      = 2    # 0000 0010 First bit identified 
    SUBFRAME_SYNC = 4    # 0000 0100 First subframe found
    TOW_DECODED   = 8    # 0000 1000 Time Of Week decoded from navigation message
    EPH_DECODED   = 16   # 0001 0000 Ephemeris from navigation message decoded
    TOW_KNOWN     = 32   # 0010 0000 Time Of Week known (retrieved from Assisted Data), to be set if TOW_DECODED set.
    EPH_KNOWN     = 64   # 0100 0000 Ephemeris known (retrieved from Assisted Data), to be set if EPH_DECODED set.
    FINE_LOCK     = 128  # 1000 0000 Fine tracking lock

    def __str__(self):
        return str(self.name)
    
# =============================================================================
