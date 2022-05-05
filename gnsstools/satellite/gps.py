

from gnsstools.gnsssignal import SignalType
from gnsstools.satellite import Satellite


class GPSSatellite(Satellite):

    def __init__(self, svid):
        super().__init__(svid)

        self.signals = [SignalType.GPS_L1_CA]

        return