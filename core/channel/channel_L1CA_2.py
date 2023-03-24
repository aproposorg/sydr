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

from core.channel.channel import Channel, ChannelState, ChannelMessage, ChannelStatus
from core.dsp.acquisition import PCPS, TwoCorrelationPeakComparison
from core.dsp.tracking import EPL, DLL_NNEML, PLL_costa, LoopFiltersCoefficients, TrackingFlags
from core.dsp.decoding import Prompt2Bit, LNAV_CheckPreambule, LNAV_DecodeTOW, MessageType
from core.dsp.cn0 import NWPR
from core.utils.constants import LNAV_MS_PER_BIT, LNAV_SUBFRAME_SIZE, LNAV_WORD_SIZE, GPS_L1CA_CODE_FREQ, GPS_L1CA_CODE_SIZE_BITS
from core.utils.circularbuffer import CircularBuffer
from core.signal.rfsignal import RFSignal
from core.signal.gnsssignal import UpsampleCode
from core.signal.gnsssignal import GenerateGPSGoldCode
from core.utils.enumerations import GNSSSystems, GNSSSignalType

# =====================================================================================================================

class ChannelL1CA(Channel):

    MIN_CONVERGENCE_TIME = 0 # Number of millisecond given to tracking filters convergence, before checking bit sync

    codeOffset       : int
    codeFrequency    : float
    carrierFrequency : float
    initialFrequency : float
    navBitsBuffer    : np.array
    navBitsSamples   : np.array
    
    sampleCounter    : int
    codeCounter      : int
    
    correlatorsBuffer       : np.array
    sizeCorrelatorsBuffer   : int
    maxSizeCorrelatorBuffer : int

    # Acquisition
    acq_dopplerRange           : tuple
    acq_dopplerSteps           : np.uint16
    acq_coherentIntegration    : np.uint8
    acq_nonCoherentIntegration : np.uint8
    acq_threshold              : float
    acq_requiredSamples        : int

    # Tracking
    track_correlatorsSpacing : tuple
    track_dll_tau1           : float
    track_dll_tau2           : float
    track_dll_pdi            : float
    track_pll_tau1           : float
    track_pll_tau2           : float
    track_pll_pdi            : float
    track_requiredSamples    : int
    track_iPrompt_sum        : float
    track_qPrompt_sum        : float
    track_nbPrompt           : int

    # Decoding
    navBitBufferSize : int
    navBitsCounter   : int
    subframeFlags    : list
    tow              : int
    preambuleFound   : bool

    # -----------------------------------------------------------------------------------------------------------------

    def __init__(self, cid:int, sharedBuffer:CircularBuffer, resultQueue:multiprocessing.Queue, rfSignal:RFSignal,
                 configuration:dict):
        """
        Constructor for ChannelL1CA class. 

        Args:
            cid (int): Channel ID.
            sharedBuffer (CircularBuffer): Circular buffer with the RF data.
            resultQueue (multiprocessing.Queue): Queue to place the results of the channel processing
            rfSignal (RFSignal): RFSignal object for RF configuration.
            configuration (dict): Configuration dictionnary for channel.

        Returns:
            None
        
        Raises:
            None
        """
        
        # Super init
        super().__init__(cid, sharedBuffer, resultQueue, rfSignal, configuration)

        # Initialisation
        self.codeOffset       = 0
        self.codeFrequency    = 0.0
        self.carrierFrequency = 0.0
        self.initialFrequency = 0.0

        self.NCO_code             = 0.0
        self.NCO_codeError        = 0.0
        self.NCO_remainingCode    = 0.0
        self.NCO_carrier          = 0.0
        self.NCO_carrierError     = 0.0
        self.NCO_remainingCarrier = 0.0

        # Save one bit length of prompt values
        self.iPromptAvg = 0.0
        self.qPromptAvg = 0.0

        self.sampleCounter = 0
        self.codeCounter = 0

        self.navBitBufferSize = LNAV_SUBFRAME_SIZE + 2 * LNAV_WORD_SIZE + 2
        self.navBitsBuffer = np.squeeze(np.empty((1, self.navBitBufferSize), dtype=int))
        self.navBitsCounter = 0
        self.subframeFlags = [False, False, False, False, False]
        self.tow = 0
        self.preambuleFound = False

        # Initialisation from configuration
        self.setAcquisition(configuration['ACQUISITION'])
        self.setTracking(configuration['TRACKING'])

        return
    
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

        return
    
    # -----------------------------------------------------------------------------------------------------------------

    def setAcquisition(self, configuration:dict):
        """
        Set parameters for acquisition operations.

        Args:
            configuration (dict): Configuration dictionnary.
        
        Returns:
            None
        
        Raises:
            None
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

    def setTracking(self, configuration:dict):
        """
        Set parameters for tracking operations.

        Args:
            configuration (dict): Configuration dictionnary.
        
        Returns:
            None
        
        Raises:
            None
        """

        self.track_correlatorsSpacing = [float(configuration['correlator_early']),
                                         float(configuration['correlator_prompt']),
                                         float(configuration['correlator_late'])]
        
        self.IDX_I_EARLY  = 0
        self.IDX_Q_EARLY  = 1
        self.IDX_I_PROMPT = 2
        self.IDX_Q_PROMPT = 3
        self.IDX_I_LATE   = 4
        self.IDX_Q_LATE   = 5

        self.track_dll_tau1, self.track_dll_tau2 = LoopFiltersCoefficients(
            loopNoiseBandwidth=float(configuration['dll_noise_bandwidth']),
            dampingRatio=float(configuration['dll_dumping_ratio']),
            loopGain=float(configuration['dll_loop_gain']))
        self.track_pll_tau1, self.track_pll_tau2 = LoopFiltersCoefficients(
            loopNoiseBandwidth=float(configuration['pll_noise_bandwidth']),
            dampingRatio=float(configuration['pll_dumping_ratio']),
            loopGain=float(configuration['pll_loop_gain']))
        self.track_dll_pdi = float(configuration['dll_pdi'])
        self.track_pll_pdi = float(configuration['pll_pdi'])
        
        self.codeStep = GPS_L1CA_CODE_FREQ / self.rfSignal.samplingFrequency
        self.track_requiredSamples = int(np.ceil((GPS_L1CA_CODE_SIZE_BITS - self.NCO_remainingCode) / self.codeStep))

        self.trackFlags = TrackingFlags.UNKNOWN

        self.maxSizeCorrelatorBuffer = LNAV_MS_PER_BIT
        self.correlatorsBuffer = np.empty((self.maxSizeCorrelatorBuffer, len(self.track_correlatorsSpacing)*2))
        self.correlatorsBuffer[:, :] = 0.0
        self.nbPrompt = 0

        return
    
    # -----------------------------------------------------------------------------------------------------------------

    def runAcquisition(self):
        """
        Perform the acquisition operations, using the PCPS method.

        Args:
            None
        
        Returns: 
            None

        Raises:
            None
        """
            
        # Check if sufficient data in buffer
        if self.rfBuffer.getNbUnreadSamples(self.currentSample) < self.acq_requiredSamples:
            return
        
        code = UpsampleCode(self.code[1:-1], self.rfSignal.samplingFrequency)
        codeFFT = np.conj(np.fft.fft(code))
        samplesPerCode = round(self.rfSignal.samplingFrequency * GPS_L1CA_CODE_SIZE_BITS / GPS_L1CA_CODE_FREQ)
        samplesPerCodeChip = round(self.rfSignal.samplingFrequency / GPS_L1CA_CODE_FREQ)

        # Parallel Code Phase Search method (PCPS)
        correlationMap = PCPS(rfData = self.rfBuffer.getSlice(self.currentSample, self.acq_requiredSamples), 
                              interFrequency = self.rfSignal.interFrequency,
                              samplingFrequency = self.rfSignal.samplingFrequency,
                              codeFFT=codeFFT,
                              dopplerRange=self.acq_dopplerRange,
                              dopplerStep=self.acq_dopplerSteps,
                              samplesPerCode=samplesPerCode, 
                              coherentIntegration=self.acq_coherentIntegration,
                              nonCoherentIntegration=self.acq_nonCoherentIntegration)
        
        indices, peakRatio = TwoCorrelationPeakComparison(correlationMap=correlationMap,
                                                          samplesPerCode=samplesPerCode,
                                                          samplesPerCodeChip=samplesPerCodeChip)

        # Update variables
        dopplerShift = -((-self.acq_dopplerRange) + self.acq_dopplerSteps * indices[0])
        self.codeOffset = int(np.round(indices[1]))
        self.carrierFrequency = self.rfSignal.interFrequency + dopplerShift
        self.initialFrequency = self.rfSignal.interFrequency + dopplerShift
        # TODO Can we merge the two variables inside the loop?

        # Update index
        #self.currentSample = (self.currentSample + self.track_requiredSamples) - self.codeOffset
        self.currentSample = self.rfBuffer.size - 2 * self.track_requiredSamples + (self.codeOffset + 1)
        
        # Switch channel state to tracking
        # TODO Test if succesful acquisition
        self.channelState = ChannelState.TRACKING

        # Results sent back to the receiver
        results = self.prepareResultsAcquisition()
        results["carrierFrequency"]  = self.carrierFrequency
        results["codeOffset"]        = self.codeOffset
        results["frequency_idx"]     = indices[0]
        results["code_idx"]          = indices[1]
        results["correlation_map"]   = correlationMap
        results["peakRatio"]         = peakRatio

        #logging.getLogger(__name__).debug(f"svid={self.svid}, freq={self.estimatedDoppler: .1f}, code={self.estimatedCode:.1f}, threshold={self.acquisitionMetric:.2f}")

        return results
    
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
        
        # Delay Lock Loop 
        self.NCO_code, self.NCO_codeError = DLL_NNEML(
            iEarly=correlatorResults[0], qEarly=correlatorResults[1], 
            iLate=correlatorResults[4], qLate=correlatorResults[5],
            NCO_code=self.NCO_code, NCO_codeError=self.NCO_codeError, 
            tau1=self.track_dll_tau1, tau2=self.track_dll_tau2, pdi=self.track_dll_pdi)
        # Update NCO code frequency
        self.codeFrequency = GPS_L1CA_CODE_FREQ - self.NCO_code
        
        # Phase Lock Loop
        self.NCO_carrier, self.NCO_carrierError = PLL_costa(
            iPrompt=correlatorResults[2], qPrompt=correlatorResults[3],
            NCO_carrier=self.NCO_carrier, NCO_carrierError=self.NCO_carrierError,
            tau1=self.track_pll_tau1, tau2=self.track_pll_tau2, pdi=self.track_pll_pdi)
        # Update NCO carrier frequency
        self.carrierFrequency = self.initialFrequency + self.NCO_carrier

        # Check if bit sync
        iPrompt = correlatorResults[2]
        if not (self.trackFlags & TrackingFlags.BIT_SYNC):
            # if not bit sync yet, check if there is a bit inversion
            if (self.trackFlags & TrackingFlags.CODE_LOCK) \
                and (self.codeCounter > self.MIN_CONVERGENCE_TIME)\
                and np.sign(self.correlatorsBuffer[self.nbPrompt-1, self.IDX_I_PROMPT]) != np.sign(iPrompt):
                    self.trackFlags |= TrackingFlags.BIT_SYNC
                    self.nbPrompt = 0
        else:
            # Compute Normalised Power ratio (for CN0 estimation)
            if self.nbPrompt == LNAV_MS_PER_BIT:
                # Compute normalised power
                normalisedPower = NWPR(
                    iPrompt = np.squeeze(self.correlatorsBuffer[:, self.IDX_I_PROMPT]), 
                    qPrompt = np.squeeze(self.correlatorsBuffer[:, self.IDX_Q_PROMPT]))

        # TODO Check if tracking was succesful an update the flags
        self.trackFlags |= TrackingFlags.CODE_LOCK
        self.correlatorsBuffer[self.nbPrompt, :] = correlatorResults[:]
        self.nbPrompt += 1

        # Update some variables
        self.sampleCounter += self.track_requiredSamples
        self.codeCounter += 1 # TODO What if we have skip some tracking? need to update the codeCounter accordingly
        self.codeSinceTOW += 1
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
        results["carrier_frequency"] = self.carrierFrequency
        results["code_frequency"]    = self.codeFrequency       
        results["normalised_power"]  = normalisedPower

        return results
    
    # -----------------------------------------------------------------------------------------------------------------

    def runDecoding(self):
        """
        Perform the decoding operations.

        Args:
            None
        
        Returns: 
            None

        Raises:
            None
        """
        
        # return # TODO For debugging purposes, should be debugged later
        
        # Check if time to decode the bit
        if not (self.trackFlags & TrackingFlags.BIT_SYNC):
            # No bit sync yet, nothing to do
            return
        
        # Check if new bit to decode
        if not (self.nbPrompt == LNAV_MS_PER_BIT):
            # No new bit, nothing to do
            return
        
        # Convert prompt correlator to bits
        self.navBitsBuffer[self.navBitsCounter] = Prompt2Bit(np.mean(self.correlatorsBuffer[:, self.IDX_I_PROMPT])) # TODO Check if this function work
        self.navBitsCounter += 1

        # Check if minimum number of bits decoded
        # Need at least the preambule bits plus the previous 2 bit to perform checks plus the 2 words afterwards.
        minBits = 2 + 2 * LNAV_WORD_SIZE
        if self.navBitsCounter < minBits:
            return 
        
        # Check if first subframe found
        if not(self.trackFlags & TrackingFlags.SUBFRAME_SYNC):
            idx = self.navBitsCounter - minBits
            print(f"{idx} {self.navBitsBuffer[idx+2:idx+2+8]}")
            if not LNAV_CheckPreambule(self.navBitsBuffer[idx-2:idx + 2*LNAV_WORD_SIZE]):
                # Preambule not found
                if self.navBitsCounter == self.navBitBufferSize:
                    # shift bit values, could be done with circular buffer but it should only be performed until the
                    # subframe has been found
                    _navBitsBuffer = np.empty_like(self.navBitsBuffer)
                    _navBitsBuffer[:-1] = self.navBitsBuffer[1:]
                    self.navBitsBuffer = _navBitsBuffer
                    self.navBitsCounter -= 1
                return
            
            logging.getLogger(__name__).debug(f"CID {self.channelID} preambule found.")
            
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
            return
        
        # Check if the beginning of the new subframe match the preambule
        idx = self.navBitsCounter - minBits
        if not LNAV_CheckPreambule(self.navBitsBuffer[idx:idx + 2*LNAV_WORD_SIZE+2]):
            # Preambule not found, reset counter and flag
            self.navBitsCounter = 0
            self.trackFlags ^= TrackingFlags.SUBFRAME_SYNC # XOR logic
            return
        
        # Decode only essential in subframe
        self.tow, subframeID, subframeBits = LNAV_DecodeTOW(self.navBitsBuffer[2:2+LNAV_SUBFRAME_SIZE], self.navBitsBuffer[1])
        self.subframeFlags[subframeID-1] = True
        
        # Update tracking flags
        # TODO Add success check?
        self.trackFlags |= TrackingFlags.TOW_DECODED 
        self.trackFlags |= TrackingFlags.TOW_KNOWN

        if not (self.trackFlags & TrackingFlags.EPH_DECODED) and all(self.subframeFlags[0:3]):
            self.trackFlags |= TrackingFlags.EPH_DECODED 
            self.trackFlags |= TrackingFlags.EPH_KNOWN

        # Results sent back to the receiver
        results = self.prepareResultsDecoding()
        results["type"] = ChannelMessage.DECODING_UPDATE
        results["subframe_id"] = subframeID
        results["tow"] = self.tow
        results["bits"] = subframeBits

        # Flush decoded bits
        _newbuffer = np.empty_like(self.navBitsBuffer)
        _newbuffer[:minBits] = self.navBitsBuffer[idx:idx + minBits]
        self.navBitsBuffer = _newbuffer
        self.navBitsCounter = minBits

        return results

    # -----------------------------------------------------------------------------------------------------------------

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

        results = []
        if self.channelState == ChannelState.IDLE:
            raise Warning(f"Tracking channel {self.channelID} is in IDLE.")

        elif self.channelState == ChannelState.ACQUIRING:
            self.addResult(results, self.runAcquisition())
        elif self.channelState == ChannelState.TRACKING:
            self.addResult(results, self.runTracking())
            self.addResult(results, self.runDecoding())
        else:
            raise ValueError(f"Channel state {self.state} is not valid.")
        
        # Check correlator history
        if self.nbPrompt == self.maxSizeCorrelatorBuffer:
            self.nbPrompt = 0

        return results
    
    # -----------------------------------------------------------------------------------------------------------------

    def addResult(self, resultsList, result):
        """
        Check if result is not None.

        Args:
            resultsList (list): List of results.
            result (dict): Result object to test.
        
        Returns: 
            None

        Raises:
            None
        """

        if result is not None:
            resultsList.append(result)
        
        return resultsList
    
    # -----------------------------------------------------------------------------------------------------------------
    
    def prepareResultsAcquisition(self):
        """
        Prepare the acquisition result to be sent. 

        Args:
            None
        
        Returns: 
            None

        Raises:
            None
        """

        mdict = super().prepareResults()
        mdict["type"] = ChannelMessage.ACQUISITION_UPDATE

        return mdict
    
    # -----------------------------------------------------------------------------------------------------------------
    
    def prepareResultsTracking(self):
        """
        Prepare the tracking result to be sent. 
        
        Args:
            None
        
        Returns: 
            None

        Raises:
            None
        """

        mdict = super().prepareResults()
        mdict["type"] = ChannelMessage.TRACKING_UPDATE

        return mdict
    
    # -----------------------------------------------------------------------------------------------------------------
    
    def prepareResultsDecoding(self):
        """
        Prepare the decoding result to be sent. 
        
        Args:
            None
        
        Returns: 
            None

        Raises:
            None
        """

        mdict = super().prepareResults()
        mdict["type"] = ChannelMessage.DECODING_UPDATE

        return mdict

# =====================================================================================================================

class ChannelStatusL1CA(ChannelStatus):
    """
    class for ChannelStatusL1CA handling.
    """
    def __init__(self, channelID:int, satelliteID:int):
        """
        Constructor for ChannelStatusL1CA class. 

        Args:
            channelID (int): Channel ID
            satellite (int): Satellite PRN code
            
        Returns:
            None

        Raises:
            None
        """
        super().__init__(channelID, satelliteID)
        self.subframeFlags = [False, False, False, False, False]

# =====================================================================================================================
    