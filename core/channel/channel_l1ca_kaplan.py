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
from core.dsp.tracking import EPL, DLL_NNEML, PLL_costa, LoopFiltersCoefficients, BorreLoopFilter, FLL_ATAN2
from core.dsp.tracking import secondOrferDLF, FLLassistedPLL_2ndOrder, FLLassistedPLL_3rdOrder
from core.dsp.cn0 import CN0_Beaulieu
from core.dsp.lockindicator import PLL_Lock_Borre, FLL_Lock_Borre
from core.utils.circularbuffer import CircularBuffer
from core.signal.rfsignal import RFSignal
from core.utils.enumerations import GNSSSystems, GNSSSignalType, TrackingFlags
from core.utils.constants import GPS_L1CA_CODE_FREQ, GPS_L1CA_CODE_SIZE_BITS, GPS_L1CA_CODE_MS
from core.utils.constants import W0_BANDWIDTH_SCALE, W0_SCALE_A2, W0_SCALE_A3, W0_SCALE_B3
from core.utils.constants import LNAV_MS_PER_BIT, LNAV_SUBFRAME_SIZE, LNAV_WORD_SIZE

# =====================================================================================================================

class ChannelL1CA_Kaplan(ChannelL1CA):

    def __init__(self, cid: int, sharedBuffer:CircularBuffer, resultQueue: multiprocessing.Queue, 
                 rfSignal:RFSignal, configuration: dict):
        super().__init__(cid, sharedBuffer, resultQueue, rfSignal, configuration)

        self.dll_prev = 0.0
        self.pll_prev = 0.0
        self.fll_prev = 0.0

        self.cn0_PdPnRatio = 0.0
        self.cn0 = 0.0

        self.pll_lock = 0.0
        self.fll_lock = 0.0
        self.pll_lock_iprompt_sum = 0.0
        self.pll_lock_qprompt_sum = 0.0

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
        correlatorResults = EPL(rfData = self.rfBuffer.getSlice(self.currentSample, self.track_requiredSamples),
                                code = self.code,
                                samplingFrequency=self.rfSignal.samplingFrequency,
                                carrierFrequency=self.carrierFrequency,
                                remainingCarrier=self.NCO_remainingCarrier,
                                remainingCode=self.NCO_remainingCode,
                                codeStep=self.codeStep,
                                correlatorsSpacing=self.track_correlatorsSpacing)

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
        if self.track_coherentIntegration == 0:
            runLoopDiscrimators = True
        else:
            runLoopDiscrimators = False
        
        if self.track_coherentIntegration > 0 \
            and self.nbPrompt % self.track_coherentIntegration == 0 \
            and self.trackFlags & TrackingFlags.BIT_SYNC:
            iEarly  = np.sum(self.correlatorsBuffer[self.nbPrompt-self.track_coherentIntegration : self.nbPrompt, 
                                            self.IDX_I_EARLY])
            qEarly  = np.sum(self.correlatorsBuffer[self.nbPrompt-self.track_coherentIntegration : self.nbPrompt, 
                                            self.IDX_Q_EARLY])
            iPrompt = np.sum(self.correlatorsBuffer[self.nbPrompt-self.track_coherentIntegration : self.nbPrompt, 
                                            self.IDX_I_PROMPT])
            qPrompt = np.sum(self.correlatorsBuffer[self.nbPrompt-self.track_coherentIntegration : self.nbPrompt, 
                                            self.IDX_Q_PROMPT])
            iLate   = np.sum(self.correlatorsBuffer[self.nbPrompt-self.track_coherentIntegration : self.nbPrompt, 
                                            self.IDX_I_LATE])
            qLate   = np.sum(self.correlatorsBuffer[self.nbPrompt-self.track_coherentIntegration : self.nbPrompt, 
                                            self.IDX_Q_LATE])
            runLoopDiscrimators = True
        else:
            iEarly  = correlatorResults[0]
            qEarly  = correlatorResults[1]
            iPrompt = correlatorResults[2]
            qPrompt = correlatorResults[3]
            iLate   = correlatorResults[4]
            qLate   = correlatorResults[5]
        
        dll = 0.0
        pll = 0.0
        fll = 0.0
        codeFrequencyError = 0.0
        carrierFrequencyError = 0.0
        if runLoopDiscrimators or self.codeCounter < self.MIN_CONVERGENCE_TIME:
            # Delay Lock Loop 
            dll = DLL_NNEML(iEarly=iEarly, qEarly=qEarly, iLate=iLate, qLate=qLate)
            # Loop Filter
            codeFrequencyError = BorreLoopFilter(dll, self.dll_prev, self.track_dll_tau1, 
                                             self.track_dll_tau2, 
                                             self.track_dll_pdi * self.track_coherentIntegration)
            
            # Phase Lock Loop
            pll = PLL_costa(iPrompt=iPrompt, qPrompt=qPrompt)

            if self.codeCounter > 1: # and self.codeCounter % 2 == 0:
                # Frequency Lock Loop
                fll = FLL_ATAN2(iPrompt, qPrompt, self.iPrompt, self.qPrompt, 1e-3)

                # self.fll, self.fll_vel_memory = secondOrferDLF(
                #     input=frequencyError, w0=self.fll_noise_bandwidth/W0_BANDWIDTH_SCALE, a2=W0_SCALE_A2, 
                #     integrationTime=1e-3, memory=self.fll_vel_memory)
                
                # carrierFrequencyError, self.fll_vel_memory = FLLassistedPLL_2ndOrder(
                #     pll, fll, w0f = self.fll_noise_bandwidth / W0_BANDWIDTH_SCALE, 
                #     w0p = self.pll_noise_bandwidth / W0_BANDWIDTH_SCALE,
                #     a2 = W0_SCALE_A2, integrationTime=1e-3 * self.track_coherentIntegration, 
                #     velMemory=self.fll_vel_memory)
                
                # self.fll, self.fll_vel_memory, self.fll_acc_memory = FLLassistedPLL_3rdOrder(
                #     phaseError, frequencyError, w0f=self.fll_noise_bandwidth / W0_BANDWIDTH_SCALE, 
                #     w0p=self.pll_noise_bandwidth / W0_BANDWIDTH_SCALE, 
                #     a2=W0_SCALE_A2, a3=W0_SCALE_A3, b3=W0_SCALE_B3,
                #     integrationTime=1e-3, velMemory=self.fll_vel_memory, accMemory=self.fll_acc_memory)

            # Loop Filter
            carrierFrequencyError = BorreLoopFilter(pll, self.pll_prev, self.track_pll_tau1, 
                                                    self.track_pll_tau2, self.track_pll_pdi)

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
            self.cn0_PdPnRatio += (iPrompt**2 + qPrompt**2) / (abs(iPrompt) - abs(qPrompt)) ** 2
            if self.nbPrompt == LNAV_MS_PER_BIT:
                self.cn0 = CN0_Beaulieu(self.cn0_PdPnRatio, self.nbPrompt, self.nbPrompt * 1e-3, self.cn0)
                self.cn0_PdPnRatio = 0.0

        # Lock indicators
        if self.codeCounter > 0:
            self.fll_lock = FLL_Lock_Borre(iPrompt, self.iPrompt, qPrompt, self.qPrompt, self.fll_lock)
            self.pll_lock = PLL_Lock_Borre(np.sum(self.correlatorsBuffer[:, self.IDX_I_PROMPT]),
                                           np.sum(self.correlatorsBuffer[:, self.IDX_Q_PROMPT]), self.pll_lock)

        # Update some variables
        # TODO Check if tracking was succesful an update the flags
        self.trackFlags |= TrackingFlags.CODE_LOCK
        self.iPrompt = iPrompt
        self.qPrompt = qPrompt
        self.codeCounter += 1 # TODO What if we have skip some tracking? need to update the codeCounter accordingly
        self.codeSinceTOW += 1
        self.codeFrequency -= codeFrequencyError
        self.carrierFrequency += carrierFrequencyError
        self.NCO_remainingCode += self.track_requiredSamples * self.codeStep - GPS_L1CA_CODE_SIZE_BITS
        self.codeStep = self.codeFrequency / self.rfSignal.samplingFrequency
        self.dll_prev = dll
        self.pll_prev = pll
        self.fll_prev = fll

        # Update index
        self.currentSample = (self.currentSample + self.track_requiredSamples) % self.rfBuffer.maxSize
        self.track_requiredSamples = int(np.ceil((GPS_L1CA_CODE_SIZE_BITS - self.NCO_remainingCode) / self.codeStep))

        # Results sent back to the receiver
        results = self.prepareResultsTracking()
        results["i_early"]                 = correlatorResults[0]
        results["q_early"]                 = correlatorResults[1]
        results["i_prompt"]                = correlatorResults[2]
        results["q_prompt"]                = correlatorResults[3]
        results["i_late"]                  = correlatorResults[4]
        results["q_late"]                  = correlatorResults[5]
        results["dll"]                     = dll
        results["pll"]                     = pll
        results["fll"]                     = fll
        results["carrier_frequency"]       = self.carrierFrequency
        results["code_frequency"]          = self.codeFrequency   
        results["carrier_frequency_error"] = carrierFrequencyError
        results["code_frequency_error"]    = codeFrequencyError     
        results["cn0"]                     = self.cn0
        results["pll_lock"]                = self.pll_lock
        results["fll_lock"]                = self.fll_lock

        return results