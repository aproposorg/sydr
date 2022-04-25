from abc import ABC, abstractmethod
from gnsstools.gnsssignal import GNSSSignal
from gnsstools.rfsignal import RFSignal
from gnsstools.utils import ChannelState
from gnsstools.acquisition import Acquisition_PCPS, AcquisitionAbstract
from gnsstools.tracking import Tracking_EPL, TrackingAbstract

class ChannelAbstract(ABC):

    @abstractmethod
    def __init__(self, cid:int, signalConfig:GNSSSignal, rfConfig:RFSignal):
        self.cid = cid
        self.rfConfig = rfConfig
        self.state = ChannelState.IDLE
        
        self.acquisition : AcquisitionAbstract
        self.tracking : TrackingAbstract

        return

    def run(self, rfData, numberOfms):
        # IDLE
        # Initialise the acquisition
        if self.state == ChannelState.IDLE:
            # Initialise acquisition
            self.state = ChannelState.ACQUIRING
        
        # ACQUIRING
        # Find coarse parameters of the signal
        if self.state == ChannelState.ACQUIRING:
            self.acquisition.run(rfData)

            if self.acquisition.isAcquired:
                frequency, code = self.acquisition.getEstimation()
                self.tracking.setInitialValues(frequency, code)
                self.state = ChannelState.TRACKING
        
        # TRACKING
        # Fine alignement of the signal replica  
        if self.state == ChannelState.TRACKING:
            self.tracking.run(rfData)

        # DECODING
        if self.tracking.preambuleFound:
            print("HOW decoding")

        if self.tracking.frameFound:
            print("Frame decoding")

        return

    def setSatellite(self, svid):
        # Update the configuration
        self.svid = svid

        # Update the methods
        self.acquisition.setSatellite(svid)
        self.tracking.setSatellite(svid)


class Channel(ChannelAbstract):
    def __init__(self, cid:int, signalConfig:GNSSSignal, rfConfig:RFSignal):
        super().__init__(cid, signalConfig, rfConfig)

        self.acquisition = Acquisition_PCPS(self.rfConfig, self.signalConfig)
        self.tracking    = Tracking_EPL(self.rfConfig, self.signalConfig)

        self.state = ChannelState.ACQUIRING

        return
    
    
        
        
        