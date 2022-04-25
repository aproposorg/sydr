from enum import Enum, unique

@unique
class ChannelState(Enum):
    IDLE      = 0
    ACQUIRING = 1
    TRACKING  = 2
