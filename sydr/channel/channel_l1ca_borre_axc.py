import numpy as np

import evoapproxlib as eal

from sydr.utils.enumerations import ChannelState
from sydr.channel.channel_l1ca_borre import ChannelL1CA
from sydr.dsp.acquisition import TwoCorrelationPeakComparison_SS
from sydr.dsp.acquisition_axc import SerialSearch_AxC
from sydr.dsp.tracking import EPL, EPL_nonvector
from sydr.dsp.tracking_axc import EPL_AxC
from sydr.utils.constants import GPS_L1CA_CODE_SIZE_BITS, GPS_L1CA_CODE_FREQ, LNAV_MS_PER_BIT
from sydr.dsp.tracking import EPL, DLL_NNEML, PLL_costa, LoopFiltersCoefficients, BorreLoopFilter, EPL_nonvector, EPL_circular
from sydr.dsp.lockindicator import CN0_NWPR, CN0_Beaulieu
from sydr.utils.enumerations import GNSSSystems, GNSSSignalType, TrackingFlags

axc_mult = eal.mul8s_1KV6.calc

class ChannelL1CA_Borre_AxC(ChannelL1CA):

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
    
    # -----------------------------------------------------------------------------------------------------------------
    
    def runPeakFinder(self, correlationMap):

        acqIndices, acqPeakRatio = TwoCorrelationPeakComparison_SS(correlationMap)

        return acqIndices, acqPeakRatio
    
    # -----------------------------------------------------------------------------------------------------------------
    
    def postAcquisitionUpdate(self, acqIndices):
        """
        """
        
        # Update variables
        dopplerShift = (-self.acq_dopplerRange) + self.acq_dopplerSteps * acqIndices[0]
        self.carrierFrequency = self.rfSignal.interFrequency - dopplerShift
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

    def runTracking(self):
        """
        Perform the tracking operations, using the EPL method.

        Args:
            None
        
        Returns: 
            None

        Raises:
            None
        """

        # Check if sufficient data in buffer
        if self.rfBuffer.getNbUnreadSamples(self.currentSample) < self.track_requiredSamples:
            return
        
        normalisedPower = np.nan

        # Correlators
        # correlatorResults = EPL(rfData = self.rfBuffer.getSlice(self.currentSample, self.track_requiredSamples),
        #                         code = self.code,
        #                         samplingFrequency=self.rfSignal.samplingFrequency,
        #                         carrierFrequency=self.carrierFrequency,
        #                         remainingCarrier=self.NCO_remainingCarrier,
        #                         remainingCode=self.NCO_remainingCode,
        #                         codeStep=self.codeStep,
        #                         correlatorsSpacing=self.track_correlatorsSpacing)
        
        # Non-vector
        # correlatorResults = EPL_nonvector(
        #                         rfData = self.rfBuffer.getSlice(self.currentSample, self.track_requiredSamples),
        #                         code = self.code,
        #                         samplingFrequency=self.rfSignal.samplingFrequency,
        #                         carrierFrequency=self.carrierFrequency,
        #                         remainingCarrier=self.NCO_remainingCarrier,
        #                         remainingCode=self.NCO_remainingCode,
        #                         codeStep=self.codeStep,
        #                         correlatorsSpacing=self.track_correlatorsSpacing)

        # Circular correlation
        # correlatorResults = EPL_circular(rfData = self.rfBuffer.getSlice(self.currentSample, self.track_requiredSamples),
        #                         code = self.code[1:-1],
        #                         samplingFrequency=self.rfSignal.samplingFrequency,
        #                         carrierFrequency=self.carrierFrequency,
        #                         remainingCarrier=self.NCO_remainingCarrier,
        #                         remainingCode=self.NCO_remainingCode,
        #                         codeStep=self.codeStep,
        #                         correlatorsSpacing=self.track_correlatorsSpacing,
        #                         codeOffset=self.codeOffset)


        correlatorResults = EPL_AxC(rfdata = self.rfBuffer.getSlice(self.currentSample, self.track_requiredSamples),
                                code = self.code,
                                samplingFrequency=self.rfSignal.samplingFrequency,
                                carrierFrequency=self.carrierFrequency,
                                remainingCarrier=self.NCO_remainingCarrier,
                                remainingCode=self.NCO_remainingCode,
                                codeStep=self.codeStep,
                                correlatorsSpacing=self.track_correlatorsSpacing,
                                axc_mult=axc_mult,
                                n_bits=self.rfSignal.quantization)

        # Compute remaining carrier phase
        self.NCO_remainingCarrier -= self.carrierFrequency * 2.0 * np.pi * self.track_requiredSamples / self.rfSignal.samplingFrequency
        self.NCO_remainingCarrier %= (2*np.pi)

        self.correlatorsBuffer[self.nbPrompt, :] = correlatorResults[:]
        self.iPrompt_sum += correlatorResults[2]
        self.qPrompt_sum += correlatorResults[3]
        self.iPrompt_sum2 += correlatorResults[2]
        self.qPrompt_sum2 += correlatorResults[3]
        self.nbPrompt += 1

        # Check coherent integration
        iEarly  = correlatorResults[0]
        qEarly  = correlatorResults[1]
        iPrompt = correlatorResults[2]
        qPrompt = correlatorResults[3]
        iLate   = correlatorResults[4]
        qLate   = correlatorResults[5]
        
        # Delay Lock Loop 
        codeError = DLL_NNEML(iEarly=iEarly, qEarly=qEarly, iLate=iLate, qLate=qLate)
        # Loop Filter
        self.NCO_code = BorreLoopFilter(codeError, self.NCO_codeError, self.track_dll_tau1, 
                                            self.track_dll_tau2, 
                                            self.track_dll_pdi)
        self.NCO_codeError = codeError
            
        # Phase Lock Loop
        phaseError = PLL_costa(iPrompt=iPrompt, qPrompt=qPrompt)
        # Loop Filter
        self.NCO_carrier = BorreLoopFilter(phaseError, self.NCO_carrierError, self.track_pll_tau1, 
                                            self.track_pll_tau2, 
                                            self.track_pll_pdi)
        self.NCO_carrierError = phaseError

        # Check if bit sync
        iPrompt = correlatorResults[2]
        qPrompt = correlatorResults[3]
        if not (self.trackFlags & TrackingFlags.BIT_SYNC):
            # if not bit sync yet, check if there is a bit inversion
            if (self.trackFlags & TrackingFlags.CODE_LOCK) \
                and (self.codeCounter > self.MIN_CONVERGENCE_TIME)\
                and np.sign(self.iPrompt) != np.sign(iPrompt):
                    self.trackFlags |= TrackingFlags.BIT_SYNC
                    self.resetPrompt()
        else:
            # Compute Normalised Power ratio (for CN0 estimation)
            if self.nbPrompt == LNAV_MS_PER_BIT:
                # Compute normalised power
                # normalisedPower = CN0_NWPR(self.iPrompt_sum, self.qPrompt_sum, self.iPrompt_sum2, self.qPrompt_sum2)
                normalisedPower = 0.0
                
        # CN0
        self.cn0_PdPnRatio += (iPrompt**2 + qPrompt**2) / (abs(iPrompt) - abs(qPrompt)) ** 2
        self.cn0_counter += 1
        if self.cn0_counter == LNAV_MS_PER_BIT:
            self.cn0 = CN0_Beaulieu(self.cn0_PdPnRatio, self.cn0_counter, self.cn0_counter*1e-3, self.cn0)
            self.cn0_PdPnRatio = 0.0
            self.cn0_counter = 0

        # Update some variables
        # TODO Check if tracking was succesful an update the flags
        self.trackFlags |= TrackingFlags.CODE_LOCK
        self.iPrompt = iPrompt
        self.qPrompt = qPrompt
        self.codeCounter += 1 # TODO What if we have skip some tracking? need to update the codeCounter accordingly
        self.codeSinceTOW += 1
        self.codeFrequency -= self.NCO_code
        self.carrierFrequency += self.NCO_carrier
        self.NCO_remainingCode += self.track_requiredSamples * self.codeStep - GPS_L1CA_CODE_SIZE_BITS
        self.codeStep = self.codeFrequency / self.rfSignal.samplingFrequency

        # Update index
        self.currentSample = (self.currentSample + self.track_requiredSamples) % self.rfBuffer.maxSize
        self.track_requiredSamples = int(np.ceil((GPS_L1CA_CODE_SIZE_BITS - self.NCO_remainingCode) / self.codeStep))

        # Results sent back to the receiver
        results = self.prepareResultsTracking()
        results["i_early"]           = correlatorResults[0]
        results["q_early"]           = correlatorResults[1]
        results["i_prompt"]          = correlatorResults[2]
        results["q_prompt"]          = correlatorResults[3]
        results["i_late"]            = correlatorResults[4]
        results["q_late"]            = correlatorResults[5]
        results["dll"]               = self.NCO_code
        results["pll"]               = self.NCO_carrier
        results["fll"]               = self.fll
        results["carrier_frequency"] = self.carrierFrequency
        results["code_frequency"]    = self.codeFrequency       
        results["cn0"]               = normalisedPower
        results["pll_lock"]          = 0.0
        results["fll_lock"]          = 0.0
        results["lock_state"]        = 0
        results["carrier_frequency_error"] = self.NCO_carrierError
        results["code_frequency_error"]    = self.NCO_codeError  

        return results