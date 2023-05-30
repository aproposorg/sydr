import numpy as np
from sydr.channel.channel import Channel, ChannelState
from sydr.utils.coordinate import Coordinate
from sydr.utils.enumerations import GNSSMeasurementType, GNSSSignalType
from sydr.utils.time import Time

# =============================================================================

class GNSSmeasurements():

    time           : Time
    channel        : Channel
    positionID     : int
    enabled        : bool
    
    # measurements
    mtype          : GNSSMeasurementType 
    value          : float
    rawValue       : float
    residual       : float

    def __init__(self):
        return

# =============================================================================

class GNSSPosition():

    id           : int
    time         : Time
    timeSample   : int
    coordinate   : Coordinate
    clockError   : float
    measurements : list

    def __init__(self):
        self.id = -1
        self.time = Time()
        self.timeSample = -1
        self.coordinate = Coordinate()
        self.measurements = []
        return

# =============================================================================







