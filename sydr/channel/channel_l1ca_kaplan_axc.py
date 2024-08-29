import numpy as np

import evoapproxlib as eal

from sydr.utils.enumerations import ChannelState
from sydr.channel.channel_l1ca_kaplan import ChannelL1CA_Kaplan
from sydr.dsp.acquisition import TwoCorrelationPeakComparison_SS
from sydr.dsp.acquisition_axc import SerialSearch_AxC
from sydr.dsp.tracking import EPL, EPL_nonvector
from sydr.dsp.tracking_axc import EPL_AxC
from sydr.utils.constants import GPS_L1CA_CODE_SIZE_BITS, GPS_L1CA_CODE_FREQ, LNAV_MS_PER_BIT

EAL_MULTIPLIERS_8BIT_SIGNED = {
    '1KV6' : eal.mul8s_1KV6.calc,
    '1KVM' : eal.mul8s_1KVM.calc,
    '1KXF' : eal.mul8s_1KXF.calc,
    '1L12' : eal.mul8s_1L12.calc,
    '1KV8' : eal.mul8s_1KV8.calc,
    '1KV9' : eal.mul8s_1KV9.calc,
    '1KVP' : eal.mul8s_1KVP.calc,
    '1KVQ' : eal.mul8s_1KVQ.calc,
    '1KX5' : eal.mul8s_1KX5.calc,
    '1KVA' : eal.mul8s_1KVA.calc
}

class ChannelL1CA_Kaplan_AxC(ChannelL1CA_Kaplan):

    def setAcquisition(self, configuration:dict):
        """
        """

        super().setAcquisition(configuration)

        self.acq_axc_mult = EAL_MULTIPLIERS_8BIT_SIGNED[configuration['axc_mult']]

        return
    
    def setTracking(self, configuration:dict):
        """
        """

        super().setTracking(configuration)

        self.track_axc_mult = EAL_MULTIPLIERS_8BIT_SIGNED[configuration['axc_mult']]

        return

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
                axc_mult=self.acq_axc_mult,
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
            axc_mult=self.track_axc_mult,
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