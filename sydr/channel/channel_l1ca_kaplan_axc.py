import numpy as np

import evoapproxlib as eal

from sydr.utils.enumerations import ChannelState
from sydr.channel.channel_l1ca_kaplan import ChannelL1CA_Kaplan
from sydr.dsp.acquisition import TwoCorrelationPeakComparison_SS
from sydr.dsp.acquisition_axc import SerialSearch_AxC
from sydr.dsp.tracking import EPL, EPL_nonvector
from sydr.dsp.tracking_axc import EPL_AxC
from sydr.utils.constants import GPS_L1CA_CODE_SIZE_BITS, GPS_L1CA_CODE_FREQ, LNAV_MS_PER_BIT

axc_mult = eal.mul8s_1KV6.calc

class ChannelL1CA_Kaplan_AxC(ChannelL1CA_Kaplan):

    def runSignalSearch(self):
        samplesPerCode = round(self.rfSignal.samplingFrequency * GPS_L1CA_CODE_SIZE_BITS / GPS_L1CA_CODE_FREQ)
        frequencyBins = np.arange(-self.acq_dopplerRange, self.acq_dopplerRange+1, self.acq_dopplerSteps)
        correlationMap = np.zeros((len(frequencyBins), GPS_L1CA_CODE_SIZE_BITS))
        for idx in range(self.acq_nonCoherentIntegration):
            correlationMap += SerialSearch_AxC(
                rfdata = self.rfBuffer.getSlice(self.currentSample + idx * samplesPerCode, samplesPerCode),
                interFrequency = self.rfSignal.interFrequency,
                code = self.code[1:-1],
                dopplerRange=self.acq_dopplerRange,
                dopplerStep=self.acq_dopplerSteps,
                samplesPerCode=samplesPerCode,
                samplingFrequency=self.rfSignal.samplingFrequency, 
                axc_mult=axc_mult,
                n_bits=self.rfSignal.quantization)
            
        return correlationMap
    
    # # -----------------------------------------------------------------------------------------------------------------
    
    def runPeakFinder(self, correlationMap):

        acqIndices, acqPeakRatio = TwoCorrelationPeakComparison_SS(correlationMap)

        return acqIndices, acqPeakRatio
    
    # -----------------------------------------------------------------------------------------------------------------
    
    def postAcquisitionUpdate(self, acqIndices):
        """
        """
        
        # Update variables
        dopplerShift = ((-self.acq_dopplerRange) + self.acq_dopplerSteps * acqIndices[0])
        self.carrierFrequency = self.rfSignal.interFrequency + dopplerShift
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

    # -----------------------------------------------------------------------------------------------------------------

    def runCorrelators(self):
        """
        """

        self.correlatorsResults[:] = EPL_AxC(
            rfdata = self.rfBuffer.getSlice(self.currentSample, self.track_requiredSamples),
            code = self.code,
            samplingFrequency=self.rfSignal.samplingFrequency,
            carrierFrequency=self.carrierFrequency,
            remainingCarrier=self.remainingCarrier,
            remainingCode=self.remainingCode,
            codeStep=self.codeStep,
            correlatorsSpacing=self.track_correlatorsSpacing,
            axc_mult=axc_mult,
            n_bits=self.rfSignal.quantization)
        
        # self.correlatorsResults[:] = EPL_nonvector(
        #     rfData = self.rfBuffer.getSlice(self.currentSample, self.track_requiredSamples),
        #     code = self.code,
        #     samplingFrequency=self.rfSignal.samplingFrequency,
        #     carrierFrequency=self.carrierFrequency,
        #     remainingCarrier=self.remainingCarrier,
        #     remainingCode=self.remainingCode,
        #     codeStep=self.codeStep,
        #     correlatorsSpacing=self.track_correlatorsSpacing)
        
        # Check buffer index
        if self.correlatorsAccumCounter == LNAV_MS_PER_BIT:
            self.correlatorsAccumCounter = 0
            self.correlatorsAccum[:] = 0.0
        
        # Update accumulators
        self.correlatorsAccum += self.correlatorsResults[:]
        self.correlatorsAccumCounter += 1

        return