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

from sydr.channel.channel import Channel
from sydr.dsp.acquisition import PCPS, PCPS_padded, TwoCorrelationPeakComparison, SerialSearch, TwoCorrelationPeakComparison_SS
from sydr.dsp.tracking import EPL, DLL_NNEML, PLL_costa, LoopFiltersCoefficients, BorreLoopFilter, FLL_ATAN
from sydr.dsp.tracking import FLLassistedPLL_2ndOrder
from sydr.dsp.decoding import Prompt2Bit, LNAV_CheckPreambule, LNAV_DecodeTOW
from sydr.dsp.lockindicator import PLL_Lock_Borre, FLL_Lock_Borre, CN0_Beaulieu, CN0_NWPR
from sydr.utils.circularbuffer import CircularBuffer
from sydr.signal.rfsignal import RFSignal
from sydr.signal.gnsssignal import UpsampleCode, GenerateGPSGoldCode
from sydr.utils.enumerations import TrackingFlags, LoopLockState, ChannelState, ChannelMessage, GNSSSystems, GNSSSignalType
from sydr.utils.constants import TWO_PI
from sydr.utils.constants import GPS_L1CA_CODE_FREQ, GPS_L1CA_CODE_SIZE_BITS, GPS_L1CA_CODE_MS
from sydr.utils.constants import W0_BANDWIDTH_1, W0_BANDWIDTH_2, W0_BANDWIDTH_3, W0_SCALE_A2, W0_SCALE_A3, W0_SCALE_B3
from sydr.utils.constants import LNAV_MS_PER_BIT, LNAV_SUBFRAME_SIZE, LNAV_WORD_SIZE
import sydr.utils.benchmark as benchmark

# =====================================================================================================================

class ChannelL1CA_Kaplan(Channel):

    def __init__(self, cid: int, sharedBuffer:CircularBuffer, resultQueue: multiprocessing.Queue, 
                 rfSignal:RFSignal, configuration: dict):
        super().__init__(cid, sharedBuffer, resultQueue, rfSignal, configuration)

        # Initialisation from configuration
        self.setAcquisition(configuration['ACQUISITION'])
        self.setTracking(configuration['TRACKING'])
        self.setDecoding()

        self.carrierFrequency = 0.0

        return
    
    # =================================================================================================================

    def _processHandler(self):
        """
        Handle the RF Data based on the current channel state. Basic acquisition -> tracking -> decoding stages.
        See documentations for the complete machine state.

        Args:
            None
        
        Returns: 
            None

        Raises:
            None
        """
        super()._processHandler()

        # Check channel state
        #self.logger.debug(f"CID {self.cid} is in {self.state} state")

        _results = []
        if self.channelState == ChannelState.IDLE:
            raise Warning(f"Tracking channel {self.channelID} is in IDLE.")
        elif self.channelState == ChannelState.ACQUIRING:
            _results.append(self.runAcquisition())
        elif self.channelState == ChannelState.TRACKING:
            _results.append(self.runTracking())
            _results.append(self.runDecoding())
        else:
            raise ValueError(f"Channel state {self.state} is not valid.")
        
        # Remove None objects
        results = [i for i in _results if i is not None]

        return results
    
    # -----------------------------------------------------------------------------------------------------------------

    def setSatellite(self, satelliteID:np.uint8):
        """
        Set the GNSS signal and satellite tracked by the channel.

        Args:
            satelliteID (int): ID (PRN code) of the satellite.
        
        Returns:
            None
        
        Raises:
            None
        """
        super().setSatellite(satelliteID)

        # Set GNSS system
        self.systemID = GNSSSystems.GPS
        self.signalID = GNSSSignalType.GPS_L1_CA
        
        # Get the satellite PRN code
        code = GenerateGPSGoldCode(satelliteID)

        # Code saved include previous and post code bit for correlation purposes
        self.code = np.r_[code[-1], code, code[0]]

        self.codeFrequency = GPS_L1CA_CODE_FREQ

        return

    # -----------------------------------------------------------------------------------------------------------------

    def getTimeSinceTOW(self):
        """
        Return current time since the last TOW in milliseconds.

        Args:
            None
        
        Returns: 
            None

        Raises:
            None
        """
        
        timeSinceTOW = 0
        timeSinceTOW += self.codeSinceTOW * GPS_L1CA_CODE_MS # Add number of code since TOW
        timeSinceTOW += self.rfBuffer.getNbUnreadSamples(self.currentSample) / (self.rfSignal.samplingFrequency/1e3) # Add number of unprocessed samples

        return timeSinceTOW

    # =================================================================================================================
    # ACQUISITION METHODS

    def setAcquisition(self, configuration:dict):
        """
        """

        self.acq_dopplerRange           = float(configuration['doppler_range'])
        self.acq_dopplerSteps           = float(configuration['doppler_steps'])
        self.acq_coherentIntegration    = int  (configuration['coherent_integration'])
        self.acq_nonCoherentIntegration = int  (configuration['non_coherent_integration'])
        self.acq_threshold              = float(configuration['threshold'])

        self.acq_requiredSamples = int(self.rfSignal.samplingFrequency * 1e-3 * \
            self.acq_nonCoherentIntegration * self.acq_coherentIntegration)

        return
    
    # -----------------------------------------------------------------------------------------------------------------

    def runAcquisition(self):
        """
        """

        # Check if sufficient data in buffer
        if self.rfBuffer.getNbUnreadSamples(self.currentSample) < self.acq_requiredSamples:
            return

        # Perform signal search
        correlationMap = self.runSignalSearch()

        # Compute acquisition quality metric
        acqIndices, acqPeakRatio = self.runPeakFinder(correlationMap)

        # Updates necessary variables
        self.postAcquisitionUpdate(acqIndices, acqPeakRatio)

        # Prepare results
        results = self.prepareResultsAcquisition(correlationMap, acqIndices, acqPeakRatio)

        return results
    
    # -----------------------------------------------------------------------------------------------------------------
    
    @benchmark.timeit
    def runSignalSearch(self):
        """
        """

        # Prepare necessary variables
        code = UpsampleCode(self.code[1:-1], self.rfSignal.samplingFrequency)

        # # Zero padding
        # code = np.pad(code, (0, int(pow(2, np.ceil(np.log(len(code))/np.log(2)))) - len(code)), mode='constant') 
        # codeFFT = np.conj(np.fft.fft(code))
        
        samplesPerCode = round(self.rfSignal.samplingFrequency * GPS_L1CA_CODE_SIZE_BITS / GPS_L1CA_CODE_FREQ)

        # correlationMap = PCPS(
        #     rfData = self.rfBuffer.getSlice(self.currentSample, self.acq_requiredSamples), 
        #     interFrequency = self.rfSignal.interFrequency,
        #     samplingFrequency = self.rfSignal.samplingFrequency,
        #     code=self.code[1:-1],
        #     dopplerRange=self.acq_dopplerRange,
        #     dopplerStep=self.acq_dopplerSteps,
        #     samplesPerCode=samplesPerCode, 
        #     coherentIntegration=self.acq_coherentIntegration,
        #     nonCoherentIntegration=self.acq_nonCoherentIntegration)
        
        # Padded acquisition
        # TODO do another channel layout for this instead
        correlationMap = PCPS(
            rfData = self.rfBuffer.getSlice(self.currentSample, self.acq_requiredSamples), 
            interFrequency = self.rfSignal.interFrequency,
            samplingFrequency = self.rfSignal.samplingFrequency,
            code=self.code[1:-1],
            dopplerRange=self.acq_dopplerRange,
            dopplerStep=self.acq_dopplerSteps,
            samplesPerCode=samplesPerCode)

        return correlationMap
    
    # -----------------------------------------------------------------------------------------------------------------
    
    @benchmark.timeit
    def runPeakFinder(self, correlationMap):

        samplesPerCode = round(self.rfSignal.samplingFrequency * GPS_L1CA_CODE_SIZE_BITS / GPS_L1CA_CODE_FREQ)
        samplesPerCodeChip = round(self.rfSignal.samplingFrequency / GPS_L1CA_CODE_FREQ)

        acqIndices, acqPeakRatio = TwoCorrelationPeakComparison(
                                                        correlationMap=correlationMap,
                                                        samplesPerCode=samplesPerCode,
                                                        samplesPerCodeChip=samplesPerCodeChip)

        return acqIndices, acqPeakRatio
    
    # -----------------------------------------------------------------------------------------------------------------
    
    def postAcquisitionUpdate(self, acqIndices, acqPeakRatio):
        """
        """
        
        # Update variables
        dopplerShift = -self.acq_dopplerRange + self.acq_dopplerSteps * acqIndices[0]
        self.codeOffset = int(np.round(acqIndices[1]))
        self.carrierFrequency = self.rfSignal.interFrequency + dopplerShift

        # Update index
        self.currentSample  = self.currentSample + self.acq_requiredSamples
        self.currentSample -= self.track_requiredSamples
        self.currentSample += self.codeOffset + 1
        
        # Switch channel state to tracking
        # TODO Test if succesful acquisition
        logging.getLogger(__name__).debug(f"CID {self.channelID} successful acquisition [code: {self.codeOffset:.1f}, doppler: {self.carrierFrequency:.1f}, metric: {acqPeakRatio:.1f}")
        self.channelState = ChannelState.TRACKING

        return
    
    # -----------------------------------------------------------------------------------------------------------------
    
    def prepareResultsAcquisition(self, correlationMap, acqIndices, acqPeakRatio):
        
        results = super().prepareResults()

        results["type"] = ChannelMessage.ACQUISITION_UPDATE
        results["carrierFrequency"]  = self.carrierFrequency
        results["codeOffset"]        = self.codeOffset
        results["frequency_idx"]     = acqIndices[0]
        results["code_idx"]          = acqIndices[1]
        results["correlation_map"]   = correlationMap
        results["peak_ratio"]        = acqPeakRatio

        return results

    # =================================================================================================================
    # TRACKING METHODS

    def setTracking(self, configuration:dict):
        """
        """

        # Correlators parameters
        wide = float(configuration['correlator_epl_wide'])
        narrow = float(configuration['correlator_epl_narrow'])
        self.dll_epl_wide   = [-wide, 0.0, wide]
        self.dll_epl_narrow = [-narrow, 0.0, narrow]
        self.track_correlatorsSpacing = self.dll_epl_wide

        self.correlatorsResults = np.squeeze(np.empty((1, len(self.track_correlatorsSpacing)*2)))
        self.correlatorsAccum = np.squeeze(np.empty((1, len(self.track_correlatorsSpacing)*2)))
        self.correlatorsAccum [:] = 0.0
        self.correlatorsAccumCounter = 0

        self.iPromptPrev = 0.0
        self.qPromptPrev = 0.0

        self.IDX_I_EARLY  = 0
        self.IDX_Q_EARLY  = 1
        self.IDX_I_PROMPT = 2
        self.IDX_Q_PROMPT = 3
        self.IDX_I_LATE   = 4
        self.IDX_Q_LATE   = 5

        # self.maxSizeCorrelatorBuffer = LNAV_MS_PER_BIT
        # self.correlatorsBuffer = np.empty((self.maxSizeCorrelatorBuffer, len(self.track_correlatorsSpacing)*2))
        # self.correlatorsBuffer[:, :] = 0.0
        # self.correlatorsBufferIndex = 0

        # DLL
        self.track_dll_tau1, self.track_dll_tau2 = LoopFiltersCoefficients(
            loopNoiseBandwidth=float(configuration['dll_noise_bandwidth']),
            dampingRatio=float(configuration['dll_damping_ratio']),
            loopGain=float(configuration['dll_loop_gain']))
        self.track_dll_pdi = float(configuration['dll_pdi'])
        self.dllLockThreshold = float(configuration['dll_threshold'])
        self.cn0_PdPnRatio = 0.0
        self.cn0 = 0.0
        self.iPromptSum  = 0.0
        self.qPromptSum  = 0.0
        self.iPromptSum2 = 0.0
        self.qPromptSum2 = 0.0

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
        self.fll_vel_memory = 0.0

        # Tracking flags and state
        self.timeSinceLastState = 0
        self.loopLockState = LoopLockState.PULL_IN
        self.trackFlags = TrackingFlags.UNKNOWN

        # Misc
        self.remainingCode = 0.0
        self.remainingCarrier = 0.0
        self.codeStep = GPS_L1CA_CODE_FREQ / self.rfSignal.samplingFrequency
        self.track_requiredSamples = int(np.ceil((GPS_L1CA_CODE_SIZE_BITS - self.remainingCode) / self.codeStep))

        self.codeCounter = 0

        return
    
    # -----------------------------------------------------------------------------------------------------------------

    def runTracking(self):
        """
        """

        # Check if sufficient data in buffer
        if self.rfBuffer.getNbUnreadSamples(self.currentSample) < self.track_requiredSamples:
            return

        # Compute correlators
        self.runCorrelators()

        # Compute discriminators
        dllDiscrim, fllDiscrim, pllDiscrim = self.runDiscriminators()

        # Compute carrier frequency loop filter
        carrierFrequencyError = self.runCarrierFrequencyFilter(fllDiscrim=fllDiscrim, pllDiscrim=pllDiscrim)
        
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

    @benchmark.timeit
    def runCorrelators(self):
        """
        """

        self.correlatorsResults[:] = EPL(
            rfData = self.rfBuffer.getSlice(self.currentSample, self.track_requiredSamples),
            code = self.code,
            samplingFrequency=self.rfSignal.samplingFrequency,
            carrierFrequency=self.carrierFrequency,
            remainingCarrier=self.remainingCarrier,
            remainingCode=self.remainingCode,
            codeStep=self.codeStep,
            correlatorsSpacing=self.track_correlatorsSpacing)
        
        # Check buffer index
        if self.correlatorsAccumCounter == LNAV_MS_PER_BIT:
            self.correlatorsAccumCounter = 0
            self.correlatorsAccum[:] = 0.0
        
        # Update accumulators
        self.correlatorsAccum += self.correlatorsResults[:]
        self.correlatorsAccumCounter += 1

        return

    # -----------------------------------------------------------------------------------------------------------------

    @benchmark.timeit
    def runDiscriminators(self):

        # Compute discriminators
        fllDiscrim = 0.0
        pllDiscrim = 0.0
        dllDiscrim = 0.0

        # # Check if coherent tracking enabled
        # if self.coherentTrackEnabled \
        #     and self.track_coherentIntegration > 1 \
        #     and (self.correlatorsBufferIndex + 1) % self.track_coherentIntegration == 0:
        #     pllDiscrim = self.runPhaseDiscriminator(self.correlatorsAccum)
        #     dllDiscrim = self.runCodeDiscriminator(self.correlatorsAccum)
        #     return dllDiscrim, fllDiscrim, pllDiscrim
        
        if self.loopLockState == LoopLockState.PULL_IN:
            # No PLL during pull-in state
            if self.codeCounter > 1:
                fllDiscrim = self.runFrequencyDiscriminator(self.correlatorsResults)
            dllDiscrim = self.runCodeDiscriminator(self.correlatorsResults)
        else:
            fllDiscrim = self.runFrequencyDiscriminator(self.correlatorsResults)
            pllDiscrim = self.runPhaseDiscriminator(self.correlatorsResults)
            dllDiscrim = self.runCodeDiscriminator(self.correlatorsResults)

        return dllDiscrim, fllDiscrim, pllDiscrim

    # -----------------------------------------------------------------------------------------------------------------

    @benchmark.timeit
    def runCarrierFrequencyFilter(self, fllDiscrim=0.0, pllDiscrim=0.0, coherentIntegration=1):

        # if self.coherentTrackEnabled:
        #     coherentIntegration = self.track_coherentIntegration
        # else:
        #     coherentIntegration = 1

        carrierFrequencyError, self.fll_vel_memory = FLLassistedPLL_2ndOrder(
                    pllDiscrim, fllDiscrim, w0f = self.fllBandwidth / W0_BANDWIDTH_1, 
                    w0p = self.pllBandwidth / W0_BANDWIDTH_2,
                    a2 = W0_SCALE_A2, integrationTime=coherentIntegration * 1e-3, 
                    velMemory=self.fll_vel_memory)

        return carrierFrequencyError
    
    # -----------------------------------------------------------------------------------------------------------------

    @benchmark.timeit
    def runCodeFrequencyFilter(self, dllDiscrim:float, coherentIntegration=1):

        # if self.coherentTrackEnabled:
        #     coherentIntegration = self.track_coherentIntegration
        # else:
        #     coherentIntegration = 1

        codeFrequencyError  = BorreLoopFilter(dllDiscrim, self.dllDiscrim, self.track_dll_tau1, 
                                              self.track_dll_tau2, self.track_dll_pdi * coherentIntegration)

        return codeFrequencyError
    
    # -----------------------------------------------------------------------------------------------------------------

    @benchmark.timeit
    def runLoopIndicators(self):

        # FLL and PLL lock indicators
        if self.codeCounter == 0:
            return
        
        iprompt = self.correlatorsResults[self.IDX_I_PROMPT]
        qprompt = self.correlatorsResults[self.IDX_Q_PROMPT]
        
        self.fllLockIndicator = FLL_Lock_Borre(iprompt=iprompt, qprompt=qprompt, 
                                               iprompt_prev=self.iPromptPrev, qprompt_prev=self.qPromptPrev,
                                                fll_lock_prev=self.fllLockIndicator, alpha=0.005)
        
        if self.loopLockState > LoopLockState.PULL_IN:
            self.pllLockIndicator = PLL_Lock_Borre(iprompt=iprompt, qprompt=qprompt,  
                                                    pll_lock_prev=self.pllLockIndicator, alpha=0.005)
        
        # CN0
        self.cn0_PdPnRatio += (iprompt**2 + qprompt**2) / (abs(iprompt) - abs(qprompt)) ** 2
        self.iPromptSum  += abs(iprompt)
        self.qPromptSum  += abs(qprompt)
        self.iPromptSum2 += iprompt**2
        self.qPromptSum2 += qprompt**2
        if self.correlatorsAccumCounter == LNAV_MS_PER_BIT:
            self.cn0 = CN0_Beaulieu(self.cn0_PdPnRatio, 
                                    self.correlatorsAccumCounter, 
                                    self.correlatorsAccumCounter * 1e-3, self.cn0)
            # self.cn0 = CN0_NWPR(self.iPromptSum, self.qPromptSum, self.iPromptSum2, self.qPromptSum2,
            #                     self.correlatorsAccumCounter, 1e-3)
            self.cn0_PdPnRatio = 0.0
            self.iPromptSum  = 0.0
            self.qPromptSum  = 0.0
            self.iPromptSum2 = 0.0
            self.qPromptSum2 = 0.0
        
        self.dllLockIndicator = self.cn0

        return

    # -----------------------------------------------------------------------------------------------------------------

    def postTrackingUpdate(self, dllDiscrim, fllDiscrim, pllDiscrim, carrierFrequencyError, codeFrequencyError):
        """
        """
        # Update counters
        self.codeCounter  += 1 # TODO What if we have skip some tracking? need to update the codeCounter accordingly
        self.codeSinceTOW += 1

        #logging.getLogger(__name__).debug(f"CID {self.channelID} codeSinceTOW {self.codeSinceTOW}.")

        # Update discriminators and loop results
        self.dllDiscrim = dllDiscrim
        self.fllDiscrim = fllDiscrim
        self.pllDiscrim = pllDiscrim
        self.carrierFrequencyError = carrierFrequencyError
        self.codeFrequencyError    = codeFrequencyError

        # Update Numerically Controlled Oscilator (NCO)
        self.remainingCarrier -= self.carrierFrequency * TWO_PI * self.track_requiredSamples / self.rfSignal.samplingFrequency
        self.remainingCarrier %= TWO_PI
        self.codeFrequency    -= self.codeFrequencyError
        self.carrierFrequency += self.carrierFrequencyError
        self.remainingCode    += self.track_requiredSamples * self.codeStep - GPS_L1CA_CODE_SIZE_BITS
        self.codeStep          = self.codeFrequency / self.rfSignal.samplingFrequency

        # Update sample reading index
        self.currentSample = (self.currentSample + self.track_requiredSamples) % self.rfBuffer.maxSize
        self.track_requiredSamples = int(np.ceil((GPS_L1CA_CODE_SIZE_BITS - self.remainingCode) / self.codeStep))

        return
    
    # -----------------------------------------------------------------------------------------------------------------

    def trackingStateUpdate(self):
        """
        """
        
        # Update the lock states for next loop

        # Check if code lock
        if self.loopLockState != LoopLockState.PULL_IN \
            and self.dllLockIndicator > self.dllLockThreshold \
            and not (self.trackFlags & TrackingFlags.CODE_LOCK):
            self.trackFlags |= TrackingFlags.CODE_LOCK
            logging.getLogger(__name__).debug(f"CID {self.channelID} tracking in {TrackingFlags.CODE_LOCK}.")
        elif self.dllLockIndicator < self.dllLockThreshold and (self.trackFlags & TrackingFlags.CODE_LOCK):
            self.trackFlags ^= TrackingFlags.CODE_LOCK
            logging.getLogger(__name__).debug(f"CID {self.channelID} tracking not in {TrackingFlags.CODE_LOCK}.")

        # Check if bit sync
        if (self.trackFlags & TrackingFlags.CODE_LOCK) and not (self.trackFlags & TrackingFlags.BIT_SYNC):
            if np.sign(self.iPromptPrev) != np.sign(self.correlatorsResults[self.IDX_I_PROMPT]):
                self.trackFlags |= TrackingFlags.BIT_SYNC
                self.correlatorsAccum[:] = self.correlatorsResults[:]
                self.correlatorsAccumCounter = 1
                self.cn0_PdPnRatio = 0.0
                self.iPromptSum  = 0.0
                self.qPromptSum  = 0.0
                self.iPromptSum2 = 0.0
                self.qPromptSum2 = 0.0
                logging.getLogger(__name__).info(f"CID {self.channelID} tracking in {TrackingFlags.BIT_SYNC}.")
        # Update prompt memory
        self.iPromptPrev = self.correlatorsResults[self.IDX_I_PROMPT]
        self.qPromptPrev = self.correlatorsResults[self.IDX_Q_PROMPT]

        # # Enable coherent tracking
        # if not self.coherentTrackEnabled \
        #     and self.loopLockState != LoopLockState.PULL_IN \
        #     and self.timeSinceLastState > self.timeInStateThreshold \
        #     and (self.trackFlags & TrackingFlags.BIT_SYNC):
        #     self.coherentTrackEnabled = True
        #     logging.getLogger(__name__).debug(f"CID {self.channelID} coherent tracking enabled.")
        # elif self.coherentTrackEnabled \
        #     and self.loopLockState == LoopLockState.PULL_IN:
        #     self.coherentTrackEnabled = False
        #     logging.getLogger(__name__).debug(f"CID {self.channelID} coherent tracking disabled.")

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
            self.timeSinceLastState += 1
            return
        
        # In case there was changes
        self.timeSinceLastState = 0
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

    def prepareResultsTracking(self):
        """
        """
        results = super().prepareResults()
        results['type']                    = ChannelMessage.TRACKING_UPDATE
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

    # =================================================================================================================
    # DECODING METHODS

    def setDecoding(self):
        """
        """

        self.navPromptSum = 0.0
        self.navPromptSumCounter = 0

        self.navBitBufferSize = LNAV_SUBFRAME_SIZE + 2 * LNAV_WORD_SIZE + 2
        self.navBitsBuffer = np.squeeze(np.empty((1, self.navBitBufferSize), dtype=int))
        self.navBitsCounter = 0

        self.preambuleFound = False
        self.subframeFlags  = [False, False, False, False, False]

        self.tow = 0
        self.subframeID = 0
        self.subframeBits = []
        
        return
    
    # -----------------------------------------------------------------------------------------------------------------

    @benchmark.timeit
    def runDecoding(self):
        
        # Decode new bit if possible
        success = self.decodeBit()
        if not success:
            # Nothing to do if no new bit available
            return

        # Decode new suframe
        success = self.decodeSubframe()
        if not success:
            return
        
        # Update necessary variables
        success = self.postDecodingUpdate()
        if not success:
            return

        # Prepare results
        results = self.prepareResultsDecoding()

        return results
    
    # -----------------------------------------------------------------------------------------------------------------

    def decodeBit(self):
        """
        """

        # Check if bit sync achieved during tracking
        if not (self.trackFlags & TrackingFlags.BIT_SYNC):
            # No bit sync yet, nothing to do
            self.navPromptSum = 0.0
            self.navPromptSumCounter = 0
            return False
        
        # Sum prompt correlator results
        self.navPromptSum += self.correlatorsResults[self.IDX_I_PROMPT]
        self.navPromptSumCounter += 1

        # Check if new bit to decode
        if not (self.navPromptSumCounter == LNAV_MS_PER_BIT):
            # No new bit, nothing to do
            return False
        
        # Convert prompt correlator to bits
        self.navBitsBuffer[self.navBitsCounter] = Prompt2Bit(self.navPromptSum)
        self.navBitsCounter += 1
        self.navPromptSum = 0.0
        self.navPromptSumCounter = 0

        return True
    
    # -----------------------------------------------------------------------------------------------------------------

    def decodeSubframe(self):
        """
        """

        # Check if minimum number of bits decoded
        # Need at least the preambule bits plus the previous 2 bit to perform checks plus the 2 words afterwards.
        minBits = 2 + 2 * LNAV_WORD_SIZE
        if self.navBitsCounter < minBits:
            return False
        
        # Check if first subframe found
        if not(self.trackFlags & TrackingFlags.SUBFRAME_SYNC):
            idx = self.navBitsCounter - minBits
            #print(f"{idx} {self.navBitsBuffer[idx+2:idx+2+8]}")
            if not LNAV_CheckPreambule(self.navBitsBuffer[idx:idx + minBits]):
                # Preambule not found
                if self.navBitsCounter == self.navBitBufferSize:
                    # shift bit values, could be done with circular buffer but it should only be performed until the
                    # subframe has been found
                    _navBitsBuffer = np.empty_like(self.navBitsBuffer   )
                    _navBitsBuffer[:-1] = self.navBitsBuffer[1:]
                    self.navBitsBuffer = _navBitsBuffer
                    self.navBitsCounter -= 1
                return False
            
            # Check if preambule was found before and the new one is the next subframe
            if self.preambuleFound and idx == LNAV_SUBFRAME_SIZE:
                # Update tracking flags 
                self.trackFlags |= TrackingFlags.SUBFRAME_SYNC # OR logic
                logging.getLogger(__name__).debug(f"CID {self.channelID} subframe sync.")
            else:
                # Flush previous bits
                _newbuffer = np.empty_like(self.navBitsBuffer)
                _newbuffer[minBits:] = 0
                _newbuffer[:minBits] = self.navBitsBuffer[idx:idx + 2*LNAV_WORD_SIZE + 2]
                self.navBitsBuffer = _newbuffer
                self.navBitsCounter = minBits
                self.preambuleFound = True
            
        # Check if complete subframe can be decoded
        if self.navBitsCounter < self.navBitBufferSize:
            return False
        
        # Check if the beginning of the new subframe match the preambule
        idx = self.navBitsCounter - minBits
        if not LNAV_CheckPreambule(self.navBitsBuffer[idx:idx + 2*LNAV_WORD_SIZE+2]):
            # Preambule not found, reset counter and flag
            self.navBitsCounter = 0
            self.trackFlags ^= TrackingFlags.SUBFRAME_SYNC # XOR logic
            return False
        
        # Decode only essential in subframe
        self.tow, self.subframeID, self.subframeBits = LNAV_DecodeTOW(
            self.navBitsBuffer[2:2+LNAV_SUBFRAME_SIZE], 
            self.navBitsBuffer[1])
        
        # Flush decoded bits
        _newbuffer = np.empty_like(self.navBitsBuffer)
        _newbuffer[:minBits] = self.navBitsBuffer[idx:idx + minBits]
        self.navBitsBuffer = _newbuffer
        self.navBitsCounter = minBits

        # Align TOW to current bits
        # TODO check if update should be done after
        self.tow += self.navBitsCounter * LNAV_MS_PER_BIT * 1e-3
        
        return True
    
    # -----------------------------------------------------------------------------------------------------------------

    def postDecodingUpdate(self):
        """
        """

        # Reset code loop counter
        self.codeSinceTOW = 0

        try:
            self.subframeFlags[self.subframeID-1] = True
            # Update tracking flags
            # TODO Add success check?
            self.trackFlags |= TrackingFlags.TOW_DECODED 
            self.trackFlags |= TrackingFlags.TOW_KNOWN
        except IndexError:
            self.trackFlags ^= TrackingFlags.TOW_DECODED 
            self.trackFlags ^= TrackingFlags.TOW_KNOWN
            logging.getLogger(__name__).warning(f"CID {self.channelID} Error in subframe ID decoding.")
            return False

        if not (self.trackFlags & TrackingFlags.EPH_DECODED) and all(self.subframeFlags[0:3]):
            self.trackFlags |= TrackingFlags.EPH_DECODED 
            self.trackFlags |= TrackingFlags.EPH_KNOWN

        logging.getLogger(__name__).debug(f"CID {self.channelID} subframe {self.subframeID} decoded "+\
                                          f"(TOW: {int(self.tow)}).")

        return True
    
    # -----------------------------------------------------------------------------------------------------------------

    def prepareResultsDecoding(self):
        """
        """

        results = super().prepareResults()
        results["type"] = ChannelMessage.DECODING_UPDATE
        results["subframe_id"] = self.subframeID
        results["tow"] = int(self.tow)
        results["bits"] = self.subframeBits

        return results
    
    # -----------------------------------------------------------------------------------------------------------------
