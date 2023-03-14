
import multiprocessing
import numpy as np

from core.channel.channel import Channel, ChannelState, ChannelMessage
from core.dsp.acquisition import PCPS, TwoCorrelationPeakComparison
from core.dsp.tracking import EPL, DLL_NNEML, PLL_costa, LoopFiltersCoefficients, TrackingFlags
from core.dsp.decoding import Prompt2Bit, LNAV_CheckPreambule, LNAV_DecodeSubframe, MessageType
from core.utils.constants import LNAV_MS_PER_BIT, LNAV_SUBFRAME_SIZE, LNAV_WORD_SIZE
from core.utils.circularbuffer import CircularBuffer
from core.satellite.ephemeris import BRDCEphemeris

class ChannelL1CA(Channel):

    codeOffset       : int
    codeFrequency    : float
    carrierFrequency : float
    initialFrequency : float
    iPrompt          : np.array
    qPrompt          : np.array
    navBitsBuffer    : np.array
    navBitsSamples   : np.array
    ephemeris        : BRDCEphemeris
    trackFlags       : TrackingFlags
    
    sampleCounter    : int
    codeCounter      : int
    navBitsCounter   : int

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
    lnav_preambuleBits : np.array
    lnav_msInBit       : int
    lnav_subframeBits  : int
    lnav_wordBits      : int
    lnav_iPrompt_sum   : float
    lnav_nbPrompt      : int

    def __init__(self, cid:int, pipe:multiprocessing.Pipe, multiprocessing=False):
        
        # Super init
        super().__init__(cid, pipe, multiprocessing=multiprocessing)

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

        self.iPrompt = 0.0
        self.qPrompt = 0.0
        self.nbPrompt = 0.0

        self.sampleCounter = 0
        self.codeCounter = 0
        self.navBitsCounter = 0

        return
    
    # -------------------------------------------------------------------------

    def setAcquisition(self, configuration:dict):
        super().setAcquisition()

        self.acq_dopplerRange           = float(configuration['dopplerRange'])
        self.acq_dopplerSteps           = float(configuration['dopplerStep'])
        self.acq_coherentIntegration    = int(configuration['coherentIntegration'])
        self.acq_nonCoherentIntegration = int(configuration['nonCoherentIntegration'])
        self.acq_threshold              = float(configuration['threshold'])

        self.acq_requiredSamples = int(self.gnssSignal.codeMs * self.rfSignal.samplingFrequency * 1e-3 * \
            self.acq_nonCoherentIntegration * self.acq_coherentIntegration)

        return
    
    # -------------------------------------------------------------------------

    def runAcquisition(self):
        """
        Perform the acquisition process with the current RF data.
        """
        super().runAcquisition()
            
        # Check if sufficient data in buffer
        if self.rfBuffer.size < self.acq_requiredSamples:
            return
            
        codeFFT = np.conj(np.fft.fft(self.code))
        samplesPerCode = round(self.rfSignal.samplingFrequency * self.gnssSignal.codeBits / self.gnssSignal.codeFrequency)
        samplesPerCodeChip = round(self.rfSignal.samplingFrequency / self.gnssSignal.codeFrequency)

        # Parallel Code Phase Search method (PCPS)
        correlationMap = PCPS(rfData = self.rfbuffer.buffer[-self.acq_requiredSamples:], 
                              interFrequency = self.rfSignal.interFrequency,
                              samplingFrequency = self.rfSignal.samplingFrequency,
                              codeFFT=codeFFT,
                              dopplerRange=self.acq_dopplerRange,
                              dopplerSteps=self.acq_dopplerSteps,
                              samplesPerCode=samplesPerCode, 
                              coherentIntegration=self.acq_coherentIntegration,
                              nonCoherentIntegration=self.acq_nonCoherentIntegration)
        
        indices, peakRatio = TwoCorrelationPeakComparison(correlationMap=correlationMap,
                                                          samplesPerCode=samplesPerCode,
                                                          samplesPerCodeChip=samplesPerCodeChip)
        
        dopplerShift = self.acq_dopplerRange[0]  + self.acq_dopplerSteps * indices[0] 
        self.codeOffset = int(np.round(indices[1]))
        self.carrierFrequency = self.rfSignal.interFrequency + dopplerShift
        self.initialFrequency = self.rfSignal.interFrequency + dopplerShift
        # TODO Can we merge the two variables inside the loop?

        # Results sent back to the receiver
        results = super().prepareResultsAcquisition()
        results["carrierFrequency"]  = self.carrierFrequency
        results["codeOffset"]        = self.codeOffset
        results["frequency_idx"]     = indices[0]
        results["code_idx"]          = indices[1]
        results["correlation_map"]   = correlationMap
        results["peakRatio"]         = peakRatio

        #logging.getLogger(__name__).debug(f"svid={self.svid}, freq={self.estimatedDoppler: .1f}, code={self.estimatedCode:.1f}, threshold={self.acquisitionMetric:.2f}")

        return results
    
    # -------------------------------------------------------------------------

    def setTracking(self, configuration):

        self.track_correlatorsSpacing = configuration['correlatorsSpacing']

        self.track_dll_tau1, self.track_dll_tau2 = LoopFiltersCoefficients(
            loopNoiseBandwidth=float(configuration['dll_NoiseBandwidth']),
            dampingRatio=float(configuration['dll_DampingRatio']),
            loopGain=float(configuration['dll_LoopGain']))
        self.track_pll_tau1, self.track_pll_tau2 = LoopFiltersCoefficients(
            loopNoiseBandwidth=float(configuration['pll_NoiseBandwidth']),
            dampingRatio=float(configuration['pll_DampingRatio']),
            loopGain=float(configuration['pll_LoopGain']))
        self.track_dll_pdi = float(configuration['dll_pdi'])
        self.track_pll_pdi = float(configuration['pll_pdi'])
        
        self.codeStep = self.gnssSignal.codeFrequency / self.rfSignal.samplingFrequency
        self.samplesRequired = int(np.ceil((self.gnssSignal.codeBits - self.NCO_remainingCode) / self.codeStep))

        self.track_requiredSamples = int(self.gnssSignal.codeMs * self.rfSignal.samplingFrequency * 1e-3)

        self.trackFlags = TrackingFlags.UNKNOWN

        return
    
    # -------------------------------------------------------------------------

    def runTracking(self):
        super().runTracking()

        # Check if sufficient data in buffer
        if self.rfBuffer.size < self.track_requiredSamples:
            return

        # Correlators
        correlatorResults = EPL(rfData = self.rfbuffer.buffer[-self.track_requiredSamples:],
                                code = self.code,
                                samplingFrequency=self.rfSignal.samplingFrequency,
                                carrierFrequency=self.gnssSignal.carrierFrequency,
                                remainingCarrier=self.NCO_remainingCarrier,
                                remainingCode=self.NCO_remainingCode,
                                codeStep=self.codeStep,
                                correlatorsSpacing=self.correlatorsSpacing)
        iPrompt_new = correlatorResults[2]
        qPrompt_new = correlatorResults[3]
        
        # Delay Lock Loop 
        self.NCO_code, self.NCO_codeError = DLL_NNEML(
            iEarly=correlatorResults[0], qEarly=correlatorResults[1], 
            iLate=correlatorResults[4], qLate=correlatorResults[5],
            NCOcode=self.NCO_code, NCOcodeError=self.NCO_codeError, 
            tau1=self.dll_tau1, tau2=self.dll_tau2, pdi=self.dll_pdi)
        # Update NCO code frequency
        self.codeFrequency = self.gnssSignal.codeFrequency - self.NCO_code
        
        # Phase Lock Loop
        self.NCO_carrier, self.NCO_carrierError = PLL_costa(
            iPrompt=correlatorResults[2], qPrompt=correlatorResults[3],
            NCO_carrier=self.NCO_carrier, NCO_carrierError=self.NCO_carrierError,
            tau1=self.pll_tau1, tau2=self.pll_tau2, pdi=self.pll_pdi)
        # Update NCO carrier frequency
        self.carrierFrequency = self.initialFrequency + self.NCO_carrier

        # Check if bit sync
        if not (self.trackFlags & TrackingFlags.BIT_SYNC):
            # if not bit sync yet, check if there is a bit inversion
            if self.trackFlags & TrackingFlags.CODE_LOCK:
                if np.sign(self.iPrompt) != np.sign(iPrompt_new):
                    self.trackFlags |= TrackingFlags.BIT_SYNC

        # TODO Check if tracking was succesful an update the flags
        # TODO Add more flags
        self.trackFlags |= TrackingFlags.CODE_LOCK

        # Update some variables
        # TODO Previously linespace, check if correct
        nbSamples = len(self.rfBuffer.buffer) # TODO Should be the same as samplesRequired no?
        self.sampleCounter += nbSamples
        self.codeCounter += 1 # TODO What if we have skip some tracking? need to update the codeCounter accordingly
        self.NCO_remainingCode += nbSamples * self.codeStep - self.gnssSignal.codeBits
        self.codeStep = self.codeFrequency / self.rfSignal.samplingFrequency
        self.samplesRequired = int(np.ceil((self.gnssSignal.codeBits - self.NCO_remainingCode) / self.codeStep))
        
        # Results sent back to the receiver
        results = super().prepareResultsTracking()
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

        return results
    
    # -------------------------------------------------------------------------
    
    def setDecoding(self):
        super().setDecoding()

        self.navBitBufferSize = LNAV_SUBFRAME_SIZE + 2 * LNAV_WORD_SIZE + 2
        self.navBitsBuffer = np.empty((1, self.navBitBufferSize))
        self.navBitsCounter = 0
        self.ephemeris = BRDCEphemeris()

        return
    
    # -------------------------------------------------------------------------

    def runDecoding(self):
        
        # Check if time to decode the bit
        if not (self.trackFlags & TrackingFlags.BIT_SYNC):
            # No bit sync yet, nothing to do
            return
        
        # Check if new bit to decode
        if not (self.lnav_nbPrompt == LNAV_MS_PER_BIT):
            # No new bit, nothing to do
            return
        
        # Convert prompt correlator to bits
        self.navBitsCounter += 1
        self.navBitsBuffer[self.navBitsCounter-1] = Prompt2Bit(self.lnav_iPrompt_sum) # TODO Check if this function work

        # Check if minimum number of bits decoded
        # Need at least the preambule bits plus the previous 2 bit to perform checks plus the 2 words afterwards.
        minBits = 2 + 2 * LNAV_WORD_SIZE
        if self.navBitsCounter < minBits:
            return 
        
        # Check if first subframe found
        if not(self.trackFlags & TrackingFlags.SUBFRAME_SYNC):
            idx = self.navBitsCounter - minBits
            if not LNAV_CheckPreambule(self.navBitsBuffer[idx-2:idx + 2*LNAV_WORD_SIZE]):
                # Preambule not found
                return
            
            # Flush previous bits
            _newbuffer = np.empty_like(self.navBitsBuffer)
            _newbuffer[:minBits] = self.navBitsBuffer[idx-2:idx + 2*LNAV_WORD_SIZE]
            self.navBitsBuffer = _newbuffer
            self.navBitsCounter = minBits
    	    
            # Update tracking flags 
            self.trackFlags |= TrackingFlags.SUBFRAME_SYNC # OR logic
        
        # Check if complete subframe can be decoded
        if self.navBitsCounter < self.navBitBufferSize:
            return
        
        # Check if the beginning of the new subframe match the preambule
        idx = self.navBitsCounter - minBits
        if not LNAV_CheckPreambule(self.navBitsBuffer[idx-2:idx + 2*LNAV_WORD_SIZE]):
            # Preambule not found, reset counter and flag
            self.navBitsCounter = 0
            self.trackFlags ^= TrackingFlags.SUBFRAME_SYNC # XOR logic
            return
        
        # Decode only essential in subframe
        self.tow, subframeID = LNAV_DecodeSubframe(self.navBitsBuffer[2:], self.navBitsBuffer[1])
        
        # Update tracking flags
        # TODO Add success check?
        self.trackFlags |= (TrackingFlags.TOW_DECODED & TrackingFlags.TOW_KNOWN)

        if not (self.trackFlags & TrackingFlags.EPH_DECODED) and self.ephemeris.checkFlags():
            self.trackFlags |= (TrackingFlags.EPH_DECODED & TrackingFlags.EPH_KNOWN)

        # Results sent back to the receiver
        results = super().prepareResultsDecoding()
        results["type"] = MessageType.GPS_LNAV
        results["subframe_id"] = subframeID
        results["tow"] = self.tow
        results["bits"] = self.navBitBufferSize

        # Flush decoded bits
        _newbuffer = np.empty_like(self.navBitsBuffer)
        _newbuffer[:minBits] = self.navBitsBuffer[idx-2:idx + 2*LNAV_WORD_SIZE]
        self.navBitsBuffer = _newbuffer
        self.navBitsCounter = minBits

        return

    # -------------------------------------------------------------------------

    def _processHandler(self):
        """
        Handle the RF Data based on the current channel state. 
        Basic acquisition -> tracking -> decoding stages. See documentations 
        for the complete machine state.
        """
        super()._processHandler()

        # Check channel state
        #self.logger.debug(f"CID {self.cid} is in {self.state} state")

        results = []
        if self.state == ChannelState.IDLE:
            print(f"WARNING: Tracking channel {self.cid} is in IDLE.")
            self.send({})

        elif self.state == ChannelState.ACQUIRING:
            results.append(self.runAcquisition())
            
        elif self.state == ChannelState.TRACKING:
            if self.isAcquired:
                self.isAcquired = False # Reset the flag, otherwise we log acquisition each loop
            # Track
            results.append(self.tracking())
            results.append(self.decoding())
        else:
            raise ValueError(f"Channel state {self.state} is not valid.")

        return results
    
    # -------------------------------------------------------------------------

    