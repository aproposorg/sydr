# -*- coding: utf-8 -*-
# ============================================================================
# Implementation of Channel for GPS L1 C/A signals
# Author: Antoine GRENIER (TAU)
# Date: 2022.03.23
# References: 
# =============================================================================
# PACKAGES
import multiprocessing
import numpy as np
import logging

from core.channel.channel_l1ca import ChannelL1CA
from core.dsp.tracking import EPL, DLL_NNEML, PLL_costa, LoopFiltersCoefficients, BorreLoopFilter, FLL_ATAN2, FLL_ATAN
from core.dsp.tracking import secondOrferDLF, FLLassistedPLL_2ndOrder, FLLassistedPLL_3rdOrder
from core.dsp.lockindicator import PLL_Lock_Borre, FLL_Lock_Borre, CN0_Beaulieu
from core.utils.circularbuffer import CircularBuffer
from core.signal.rfsignal import RFSignal
from core.utils.enumerations import GNSSSystems, GNSSSignalType, TrackingFlags, LoopLockState
from core.utils.constants import TWO_PI
from core.utils.constants import GPS_L1CA_CODE_FREQ, GPS_L1CA_CODE_SIZE_BITS, GPS_L1CA_CODE_MS
from core.utils.constants import W0_BANDWIDTH_1, W0_BANDWIDTH_2, W0_BANDWIDTH_3, W0_SCALE_A2, W0_SCALE_A3, W0_SCALE_B3
from core.utils.constants import LNAV_MS_PER_BIT, LNAV_SUBFRAME_SIZE, LNAV_WORD_SIZE

# =====================================================================================================================

class ChannelL1CA_Kaplan(ChannelL1CA):

    def __init__(self, cid: int, sharedBuffer:CircularBuffer, resultQueue: multiprocessing.Queue, 
                 rfSignal:RFSignal, configuration: dict):
        super().__init__(cid, sharedBuffer, resultQueue, rfSignal, configuration)

        self.cn0_PdPnRatio = 0.0
        self.cn0 = 0.0

        self.loopLockState = LoopLockState.PULL_IN

        self.iPromptPrev = 0.0
        self.qPromptPrev = 0.0

        return
    
    # -----------------------------------------------------------------------------------------------------------------
    
    def setTracking(self, configuration:dict):

        # self.track_coherentIntegration = int(configuration['coherent_integration'])
        # # Coherent integration of 1 ms is the same as no coherent integration.
        # # We put it to 1 to avoid null numbers in future computations.
        # if self.track_coherentIntegration == 0:
        #     self.track_coherentIntegration = 1
        # self.coherentIntegration = 1
        # self.fineTrackThreshold = int(configuration['fine_track_treshold'])

        # Correlators
        wide = float(configuration['correlator_epl_wide'])
        narrow = float(configuration['correlator_epl_narrow'])
        self.dll_epl_wide   = [-wide, 0.0, wide]
        self.dll_epl_narrow = [-narrow, 0.0, narrow]
        self.track_correlatorsSpacing = self.dll_epl_wide
        
        self.IDX_I_EARLY  = 0
        self.IDX_Q_EARLY  = 1
        self.IDX_I_PROMPT = 2
        self.IDX_Q_PROMPT = 3
        self.IDX_I_LATE   = 4
        self.IDX_Q_LATE   = 5

        # DLL
        self.track_dll_tau1, self.track_dll_tau2 = LoopFiltersCoefficients(
            loopNoiseBandwidth=float(configuration['dll_noise_bandwidth']),
            dampingRatio=float(configuration['dll_damping_ratio']),
            loopGain=float(configuration['dll_loop_gain']))
        self.track_dll_pdi = float(configuration['dll_pdi'])
        self.dllLockThreshold = float(configuration['dll_threshold'])
        # self.dll_bandwidth_wide = float(configuration['dll_bandwidth_wide'])
        # self.dll_bandwidth_narrow = float(configuration['dll_bandwidth_narrow'])

        # FLL
        self.fll_bandwidth_pullin = float(configuration['fll_bandwidth_pullin'])
        self.fll_bandwidth_wide   = float(configuration['fll_bandwidth_wide'])
        self.fll_bandwidth_narrow = float(configuration['fll_bandwidth_narrow'])
        self.fll_threshold_wide   = float(configuration['fll_threshold_wide'])
        self.fll_threshold_narrow = float(configuration['fll_threshold_narrow'])

        # PLL
        self.pll_bandwidth_wide   = float(configuration['pll_bandwidth_wide'])
        self.pll_bandwidth_narrow = float(configuration['pll_bandwidth_narrow'])
        self.pll_threshold_wide   = float(configuration['pll_threshold_wide'])
        self.pll_threshold_narrow = float(configuration['pll_threshold_narrow'])

        # Initialise
        self.dllDiscrim = 0.0
        self.pllDiscrim = 0.0
        self.fllDiscrim = 0.0
        self.fllBandwidth = self.fll_bandwidth_pullin
        self.pllBandwidth = self.pll_bandwidth_wide
        self.dllLockIndicator = 0.0
        self.fllLockIndicator = 0.0
        self.pllLockIndicator = 0.0

        self.correlatorsResults = np.squeeze(np.empty((1, len(self.track_correlatorsSpacing)*2)))
        self.correlatorsAccum = np.squeeze(np.empty((1, len(self.track_correlatorsSpacing)*2)))
        self.correlatorsAccum [:] = 0.0
        self.correlatorsAccumCounter = 0

        self.codeStep = GPS_L1CA_CODE_FREQ / self.rfSignal.samplingFrequency
        self.track_requiredSamples = int(np.ceil((GPS_L1CA_CODE_SIZE_BITS - self.NCO_remainingCode) / self.codeStep))
        self.trackFlags = TrackingFlags.UNKNOWN

        self.lastLockStateSwitch = 0
        
        # TODO To be removed
        self.maxSizeCorrelatorBuffer = LNAV_MS_PER_BIT
        self.correlatorsBuffer = np.empty((self.maxSizeCorrelatorBuffer, len(self.track_correlatorsSpacing)*2))
        self.correlatorsBuffer[:, :] = 0.0

        return
    
    # -----------------------------------------------------------------------------------------------------------------

    def runTracking(self):
        """
        """

        # Compute correlators
        self.runCorrelators()

        # Compute discriminators
        dllDiscrim, fllDiscrim, pllDiscrim = self.runDiscriminators()

        # Compute carrier frequency loop filter
        carrierFrequencyError = self.runCarrierFrequencyFilter(fllDiscrim=fllDiscrim, pllDiscrim=pllDiscrim, 
                                                               integrationTime=1e-3)
        
        # Compute code frequency loop filter
        codeFrequencyError = self.runCodeFrequencyFilter(dllDiscrim=dllDiscrim)

        # Compute the lock loop indicators
        self.runLoopIndicators()

        # Update the NCO and other things
        self.postTrackingUpdate(dllDiscrim, fllDiscrim, pllDiscrim, carrierFrequencyError, codeFrequencyError)

        # Update the lock states for next loop
        self.trackingStateUpdate()

        # Prepare result package
        results = self.prepareResultsTracking()
        
        return results
    
    # -----------------------------------------------------------------------------------------------------------------

    def runCorrelators(self):

        self.correlatorsResults[:] = EPL(rfData = self.rfBuffer.getSlice(self.currentSample, self.track_requiredSamples),
                                        code = self.code,
                                        samplingFrequency=self.rfSignal.samplingFrequency,
                                        carrierFrequency=self.carrierFrequency,
                                        remainingCarrier=self.NCO_remainingCarrier,
                                        remainingCode=self.NCO_remainingCode,
                                        codeStep=self.codeStep,
                                        correlatorsSpacing=self.track_correlatorsSpacing)
            
        # Update accumulators
        self.correlatorsAccum += self.correlatorsResults[:]
        self.correlatorsAccumCounter += 1

        # For decoding part
        self.nbPrompt = self.correlatorsAccumCounter
        self.iPrompt_sum = self.correlatorsAccum[self.IDX_I_PROMPT] 

        return

    # -----------------------------------------------------------------------------------------------------------------

    def runDiscriminators(self):

        # Compute discriminators
        dllDiscrim = 0.0
        fllDiscrim = 0.0
        pllDiscrim = 0.0
        if self.loopLockState == LoopLockState.PULL_IN:
            dllDiscrim = self.runCodeDiscriminator(self.correlatorsResults)
            # No PLL during pull-in state
            if self.correlatorsAccumCounter > 1:
                fllDiscrim = self.runFrequencyDiscriminator(self.correlatorsResults)
        # elif self.loopLockState == LoopLockState.FINE_TRACK \
        #     and self.track_coherentIntegration > 1 \
        #     and self.trackFlags & TrackingFlags.BIT_SYNC:
        #     # Coherent integration
        #     if self.correlatorsAccumCounter % self.track_coherentIntegration != 0:
        #         return self.dllDiscrim, self.fllDiscrim, self.pllDiscrim
            
        #     # No FLL during pull-in state (complex as it depends on the coherent integration time)
        #     dllDiscrim = self.runCodeDiscriminator(self.correlatorsAccum)
        #     pllDiscrim = self.runPhaseDiscriminator(self.correlatorsAccum)
        else:
            dllDiscrim = self.runCodeDiscriminator(self.correlatorsResults)
            fllDiscrim = self.runFrequencyDiscriminator(self.correlatorsResults)
            pllDiscrim = self.runPhaseDiscriminator(self.correlatorsResults)

        return dllDiscrim, fllDiscrim, pllDiscrim

    # -----------------------------------------------------------------------------------------------------------------

    def runCarrierFrequencyFilter(self, fllDiscrim=0.0, pllDiscrim=0.0, integrationTime=1e-3):

        carrierFrequencyError, self.fll_vel_memory = FLLassistedPLL_2ndOrder(
                    pllDiscrim, fllDiscrim, w0f = self.fllBandwidth / W0_BANDWIDTH_1, 
                    w0p = self.pllBandwidth / W0_BANDWIDTH_2,
                    a2 = W0_SCALE_A2, integrationTime=integrationTime, 
                    velMemory=self.fll_vel_memory)

        return carrierFrequencyError
    
    # -----------------------------------------------------------------------------------------------------------------

    def runCodeFrequencyFilter(self, dllDiscrim:float):

        codeFrequencyError  = BorreLoopFilter(dllDiscrim, self.dllDiscrim, self.track_dll_tau1, 
                                              self.track_dll_tau2, self.track_dll_pdi)

        return codeFrequencyError
    
    # -----------------------------------------------------------------------------------------------------------------

    def runLoopIndicators(self):

        # FLL and PLL lock indicators
        if self.codeCounter == 0:
            return
        
        iprompt = self.correlatorsResults[self.IDX_I_PROMPT]
        qprompt = self.correlatorsResults[self.IDX_Q_PROMPT]
        
        self.fllLockIndicator = FLL_Lock_Borre(iprompt=iprompt, qprompt=qprompt, 
                                               iprompt_prev=self.iPromptPrev, qprompt_prev=self.qPromptPrev,
                                                fll_lock_prev=self.fllLockIndicator, alpha=0.01)
        
        if self.loopLockState > LoopLockState.PULL_IN:
            self.pllLockIndicator = PLL_Lock_Borre(iprompt=iprompt, qprompt=qprompt,  
                                                    pll_lock_prev=self.pllLockIndicator, alpha=0.01)
        
        # CN0
        self.cn0_PdPnRatio += (iprompt**2 + qprompt**2) / (abs(iprompt) - abs(qprompt)) ** 2
        if self.correlatorsAccumCounter == LNAV_MS_PER_BIT:
            self.cn0 = CN0_Beaulieu(self.cn0_PdPnRatio, self.correlatorsAccumCounter, self.correlatorsAccumCounter * 1e-3, self.cn0)
            self.cn0_PdPnRatio = 0.0
        
        self.dllLockIndicator = self.cn0

        return

    # -----------------------------------------------------------------------------------------------------------------

    def postTrackingUpdate(self, dllDiscrim, fllDiscrim, pllDiscrim, carrierFrequencyError, codeFrequencyError):
        """
        """
        # Update counters
        self.codeCounter  += 1 # TODO What if we have skip some tracking? need to update the codeCounter accordingly
        self.codeSinceTOW += 1

        # Update discriminators and loop results
        self.dllDiscrim = dllDiscrim
        self.fllDiscrim = fllDiscrim
        self.pllDiscrim = pllDiscrim
        self.carrierFrequencyError = carrierFrequencyError
        self.codeFrequencyError    = codeFrequencyError

        # Update Numerically Controlled Oscilator (NCO)
        self.NCO_remainingCarrier -= self.carrierFrequency * TWO_PI * self.track_requiredSamples / self.rfSignal.samplingFrequency
        self.NCO_remainingCarrier %= TWO_PI
        self.codeFrequency        -= self.codeFrequencyError
        self.carrierFrequency     += self.carrierFrequencyError
        self.NCO_remainingCode    += self.track_requiredSamples * self.codeStep - GPS_L1CA_CODE_SIZE_BITS
        self.codeStep              = self.codeFrequency / self.rfSignal.samplingFrequency

        # Update sample reading index
        self.currentSample = (self.currentSample + self.track_requiredSamples) % self.rfBuffer.maxSize
        self.track_requiredSamples = int(np.ceil((GPS_L1CA_CODE_SIZE_BITS - self.NCO_remainingCode) / self.codeStep))

        # Check buffers
        if self.correlatorsAccumCounter == LNAV_MS_PER_BIT:
            self.correlatorsAccum [:] = 0
            self.correlatorsAccumCounter = 0

        return
    
    # -----------------------------------------------------------------------------------------------------------------

    def trackingStateUpdate(self):
        """
        """
        
        # Update the lock states for next loop

        # Check if code lock
        if self.dllLockIndicator > self.dllLockThreshold and not (self.trackFlags & TrackingFlags.CODE_LOCK):
            self.trackFlags |= TrackingFlags.CODE_LOCK
            logging.getLogger(__name__).debug(f"CID {self.channelID} tracking reached {TrackingFlags.CODE_LOCK}.")
        elif self.dllLockIndicator < self.dllLockThreshold and (self.trackFlags & TrackingFlags.CODE_LOCK):
            self.trackFlags ^= TrackingFlags.CODE_LOCK

        # Check if bit sync
        if (self.trackFlags & TrackingFlags.CODE_LOCK) and not (self.trackFlags & TrackingFlags.BIT_SYNC):
            if np.sign(self.iPromptPrev) != np.sign(self.correlatorsResults[self.IDX_I_PROMPT]):
                self.trackFlags |= TrackingFlags.BIT_SYNC
                self.correlatorsAccum[:] = 0
                self.correlatorsAccumCounter = 0
                logging.getLogger(__name__).info(f"CID {self.channelID} tracking reached {TrackingFlags.BIT_SYNC}.")
        # Update prompt memory
        self.iPromptPrev = self.correlatorsResults[self.IDX_I_PROMPT]
        self.qPromptPrev = self.correlatorsResults[self.IDX_Q_PROMPT]

        # # Switch to fine tracking 
        # if self.loopLockState != LoopLockState.FINE_TRACK \
        #     and self.loopLockState == LoopLockState.NARROW_TRACK \
        #     and self.lastLockStateSwitch > self.fineTrackThreshold:

        #     self.loopLockState = LoopLockState.FINE_TRACK
        #     self.coherentIntegration = self.track_coherentIntegration

        # Switch to narrow tracking
        if self.loopLockState != LoopLockState.NARROW_TRACK \
            and self.fllLockIndicator >= self.fll_threshold_narrow \
            and self.pllLockIndicator >= self.pll_threshold_narrow:

            self.loopLockState = LoopLockState.NARROW_TRACK
            self.fllBandwidth = self.fll_bandwidth_narrow
            self.pllBandwidth = self.pll_bandwidth_narrow
            self.track_correlatorsSpacing = self.dll_epl_narrow
        
        # Switch to wide tracking
        elif self.loopLockState != LoopLockState.WIDE_TRACK \
            and self.fllLockIndicator >= self.fll_threshold_wide \
            and self.fllLockIndicator < self.fll_threshold_narrow:
            
            self.loopLockState = LoopLockState.WIDE_TRACK
            self.fllBandwidth = self.fll_bandwidth_wide
            self.pllBandwidth = self.pll_bandwidth_wide
            self.track_correlatorsSpacing = self.dll_epl_wide
        
        # Switch to pull-in
        elif self.loopLockState != LoopLockState.PULL_IN \
            and self.fllLockIndicator <= self.fll_threshold_wide:
            
            self.loopLockState = LoopLockState.PULL_IN
            self.fllBandwidth = self.fll_bandwidth_pullin
            self.pllBandwidth = 0.0
            self.track_correlatorsSpacing = self.dll_epl_wide 
        
        else:
            # Return here if no change, this is to update variables below if it switch and avoid copy-paste
            return
        
        self.lastLockStateSwitch = self.codeCounter
        logging.getLogger(__name__).debug(f"CID {self.channelID} tracking switched to {self.loopLockState}.")

        return
    
    # -----------------------------------------------------------------------------------------------------------------

    def runFrequencyDiscriminator(self, correlatorResults):
        """
        """
        discrim = FLL_ATAN(iPrompt=correlatorResults[self.IDX_I_PROMPT], iPromptPrev=self.iPromptPrev, 
                           qPrompt=correlatorResults[self.IDX_Q_PROMPT], qPromptPrev=self.qPromptPrev, 
                           deltaT=1e-3)
        return discrim

    # -----------------------------------------------------------------------------------------------------------------

    def runPhaseDiscriminator(self, correlatorResults):
        """
        """
        discrim = PLL_costa(iPrompt=correlatorResults[self.IDX_I_PROMPT], 
                            qPrompt=correlatorResults[self.IDX_Q_PROMPT])
        return discrim

    # -----------------------------------------------------------------------------------------------------------------

    def runCodeDiscriminator(self, correlatorResults):
        """
        """
        discrim = DLL_NNEML(iEarly=correlatorResults[self.IDX_I_EARLY], 
                            qEarly=correlatorResults[self.IDX_Q_EARLY],
                            iLate=correlatorResults[self.IDX_I_LATE], 
                            qLate=correlatorResults[self.IDX_Q_LATE])
        return discrim

    # -----------------------------------------------------------------------------------------------------------------

    def runCarrierFrequencyFilter(self, fllDiscrim=0.0, pllDiscrim=0.0, integrationTime=1e-3):

        carrierFrequencyError, self.fll_vel_memory = FLLassistedPLL_2ndOrder(
                    pllDiscrim, fllDiscrim, w0f = self.fllBandwidth / W0_BANDWIDTH_1, 
                    w0p = self.pllBandwidth / W0_BANDWIDTH_2,
                    a2 = W0_SCALE_A2, integrationTime=integrationTime, 
                    velMemory=self.fll_vel_memory)

        return carrierFrequencyError
    
    # -----------------------------------------------------------------------------------------------------------------

    def runCodeFrequencyFilter(self, dllDiscrim:float):

        codeFrequencyError  = BorreLoopFilter(dllDiscrim, self.dllDiscrim, self.track_dll_tau1, 
                                              self.track_dll_tau2, self.track_dll_pdi)

        return codeFrequencyError
        
    # -----------------------------------------------------------------------------------------------------------------

    def prepareResultsTracking(self):
        """
        """
        results = super().prepareResultsTracking()
        results["i_early"]                 = self.correlatorsResults[0]
        results["q_early"]                 = self.correlatorsResults[1]
        results["i_prompt"]                = self.correlatorsResults[2]
        results["q_prompt"]                = self.correlatorsResults[3]
        results["i_late"]                  = self.correlatorsResults[4]
        results["q_late"]                  = self.correlatorsResults[5]
        results["carrier_frequency"]       = self.carrierFrequency
        results["code_frequency"]          = self.codeFrequency   
        results["carrier_frequency_error"] = self.carrierFrequencyError
        results["code_frequency_error"]    = self.codeFrequencyError     
        results["cn0"]                     = self.cn0
        results["pll_lock"]                = self.pllLockIndicator
        results["fll_lock"]                = self.fllLockIndicator
        results["dll"]                     = self.dllDiscrim
        results["pll"]                     = self.pllDiscrim
        results["fll"]                     = self.fllDiscrim
        results["lock_state"]              = self.loopLockState

        return results

    # -----------------------------------------------------------------------------------------------------------------