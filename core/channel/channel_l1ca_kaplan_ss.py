import numpy as np

from core.utils.enumerations import ChannelState
from core.channel.channel_l1ca_kaplan import ChannelL1CA_Kaplan
from core.dsp.acquisition import TwoCorrelationPeakComparison_SS, SerialSearch
from core.utils.constants import GPS_L1CA_CODE_SIZE_BITS, GPS_L1CA_CODE_FREQ

class ChannelL1CA_Kaplan_SS(ChannelL1CA_Kaplan):

    def runSignalSearch(self):
        samplesPerCode = round(self.rfSignal.samplingFrequency * GPS_L1CA_CODE_SIZE_BITS / GPS_L1CA_CODE_FREQ)
        frequencyBins = np.arange(-self.acq_dopplerRange, self.acq_dopplerRange+1, self.acq_dopplerSteps)
        correlationMap = np.zeros((len(frequencyBins), GPS_L1CA_CODE_SIZE_BITS))
        for idx in range(self.acq_nonCoherentIntegration):
            correlationMap += SerialSearch(
                rfdata = self.rfBuffer.getSlice(self.currentSample + idx * samplesPerCode, samplesPerCode),
                code = self.code[1:-1],
                dopplerRange=self.acq_dopplerRange,
                dopplerStep=self.acq_dopplerSteps,
                samplesPerCode=samplesPerCode,
                samplingFrequency=self.rfSignal.samplingFrequency)
            
        return correlationMap
    
    # -----------------------------------------------------------------------------------------------------------------
    
    def runPeakFinder(self, correlationMap):

        acqIndices, acqPeakRatio = TwoCorrelationPeakComparison_SS(correlationMap)

        return acqIndices, acqPeakRatio
    
    # -----------------------------------------------------------------------------------------------------------------
    
    def postAcquisitionUpdate(self, acqIndices):
        """
        """
        
        # Update variables
        dopplerShift = ((-self.acq_dopplerRange) + self.acq_dopplerSteps * acqIndices[0])
        self.carrierFrequency = -(self.rfSignal.interFrequency + dopplerShift)
        samplesPerCodeChip = self.rfSignal.samplingFrequency / GPS_L1CA_CODE_FREQ
        self.codeOffset = int(np.round(acqIndices[1] * samplesPerCodeChip))

        # Update index
        self.currentSample  = self.currentSample + self.acq_requiredSamples
        self.currentSample -= self.track_requiredSamples
        self.currentSample += self.codeOffset + 1
        
        # Switch channel state to tracking
        # TODO Test if succesful acquisition
        self.channelState = ChannelState.TRACKING

        return