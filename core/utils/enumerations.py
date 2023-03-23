# -*- coding: utf-8 -*-
# ============================================================================
# Abstract class for channel definition.
# Author: Antoine GRENIER (TAU)
# Date: 2022.03.23 (Updated: )
# References: 
# =============================================================================
# PACKAGES
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
    
# =============================================================================
