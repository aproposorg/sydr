from abc import ABC, abstractmethod
from gnsstools.gnsssignal import GNSSSignal
from gnsstools.rfsignal import RFSignal
from gnsstools.utils import ChannelState
from gnsstools.acquisition import Acquisition_PCPS, AcquisitionAbstract
from gnsstools.tracking import Tracking_EPL, TrackingAbstract
import numpy as np

class ChannelAbstract(ABC):

    @abstractmethod
    def __init__(self, cid:int, signalConfig:GNSSSignal, rfConfig:RFSignal):
        self.cid = cid
        self.signalConfig = signalConfig
        self.rfConfig = rfConfig
        self.switchState(ChannelState.IDLE)
        
        self.acquisition : AcquisitionAbstract
        self.tracking : TrackingAbstract

        # Minimum data required for process in number of samples
        # Default is 1 ms
        self.dataRequiredAcquisition = int(1 * self.rfConfig.samplingFrequency * 1e-3)
        self.dataRequiredTracking = int(1 * self.rfConfig.samplingFrequency * 1e-3)

        # Buffer keeps in memory the last ms for each channel
        # Default is 10 ms for buffer
        # TODO Don't like a hardcoded default, need to find something
        self.bufferMaxSize = int(10 * self.rfConfig.samplingFrequency * 1e-3)
        self.buffer = np.empty(self.bufferMaxSize, dtype=np.complex128)
        self.buffer[:] = np.nan
        self.bufferSize = 0
        self.isBufferFull = False

        # save
        self.iPrompt = []
        self.qPrompt = []

        return

    def run(self, rfData, numberOfms=1):

        self.shiftBuffer(rfData, int(numberOfms*self.rfConfig.samplingFrequency*1e-3))
        if not self.isBufferFull:
            return

        # IDLE
        # Initialise the acquisition
        if self.state == ChannelState.IDLE:
            print(f"WARNING: Tracking channel {self.cid} is in IDLE.")
            return
        
        # ACQUIRING
        # Find coarse parameters of the signal
        if self.state == ChannelState.ACQUIRING:
            self.acquisition.run(self.buffer[-self.dataRequiredAcquisition:])

            if self.acquisition.isAcquired:
                frequency, code = self.acquisition.getEstimation()
                self.tracking.setInitialValues(frequency, code)
                self.switchState(ChannelState.TRACKING)
        
        # TRACKING
        # Fine alignement of the signal replica  
        if self.state == ChannelState.TRACKING:
            frequency, code = self.acquisition.getEstimation()
            self.tracking.run(self.buffer[-(self.dataRequiredTracking+code):-code])
            
            self.iPrompt.append(self.tracking.iPrompt)
            self.qPrompt.append(self.tracking.qPrompt)

        # # DECODING
        # if self.tracking.preambuleFound:
        #     print("HOW decoding")

        # if self.tracking.frameFound:
        #     print("Frame decoding")

        return

    def setSatellite(self, svid):
        # Update the configuration
        self.svid = svid

        # Update the methods
        self.acquisition.setSatellite(svid)
        self.tracking.setSatellite(svid)

        # Set state to acquisition
        self.switchState(ChannelState.ACQUIRING)

        return

    def switchState(self, newState):
        self.state = newState
        #self.updateMsRequired()
        return
    
    def getState(self):
        return self.state

    def shiftBuffer(self, data, shift:int):
        bufferShifted = np.empty_like(self.buffer)
        bufferShifted[self.bufferMaxSize-shift:] = data
        bufferShifted[:self.bufferMaxSize-shift] = self.buffer[shift:]
        self.buffer = bufferShifted
        
        # Update buffer size
        if not self.isBufferFull:
            self.bufferSize += shift
            if self.bufferSize >= self.bufferMaxSize:
                self.isBufferFull = True

        return


class Channel(ChannelAbstract):
    def __init__(self, cid:int, signalConfig:GNSSSignal, rfConfig:RFSignal):
        super().__init__(cid, signalConfig, rfConfig)

        self.acquisition = Acquisition_PCPS(self.rfConfig, self.signalConfig)
        self.tracking    = Tracking_EPL(self.rfConfig, self.signalConfig)

        self.dataRequiredAcquisition = int(self.signalConfig.codeMs * \
            self.acquisition.nonCohIntegration * \
            self.acquisition.cohIntegration * self.rfConfig.samplingFrequency * 1e-3)
        
        self.dataRequiredTracking = int(self.signalConfig.codeMs * self.rfConfig.samplingFrequency * 1e-3)

        # Buffer keeps in memory the last ms for each channel
        # Buffer size is based on the maximum amont of data required, most 
        # probably from acquisition.
        self.bufferMaxSize = np.max([self.dataRequiredTracking, self.dataRequiredAcquisition])
        self.buffer = np.empty(self.bufferMaxSize, dtype=np.complex128)
        self.buffer[:] = np.nan
        self.bufferSize = 0
        self.isBufferFull = False

        return
    
    
        
        
        