
import multiprocessing
import numpy as np

from core.channel.channel import Channel, ChannelState, ChannelMessage
from core.dsp.acquisition import PCPS, TwoCorrelationPeakComparison
from core.dsp.tracking import EPL, DLL_NNEML, PLL_costa, LoopFiltersCoefficients

class ChannelL1CA(Channel):

    codeOffset : int
    codeFrequency : float
    carrierFrequency : float
    initialFrequency : float

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

        return
    
    # -------------------------------------------------------------------------

    def setAcquisition(self, configuration:dict):
        super().setAcquisition()

        self.acq_dopplerRange           = configuration['dopplerRange']
        self.acq_dopplerSteps           = configuration['dopplerStep']
        self.acq_coherentIntegration    = configuration['coherentIntegration']
        self.acq_nonCoherentIntegration = configuration['nonCoherentIntegration']
        self.acq_threshold              = configuration['threshold']

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
            loopNoiseBandwidth=configuration['dll_NoiseBandwidth'],
            dampingRatio=configuration['dll_DampingRatio'],
            loopGain=configuration['dll_LoopGain'])
        self.track_pll_tau1, self.track_pll_tau2 = LoopFiltersCoefficients(
            loopNoiseBandwidth=configuration['pll_NoiseBandwidth'],
            dampingRatio=configuration['pll_DampingRatio'],
            loopGain=configuration['pll_LoopGain'])
        self.track_dll_pdi = configuration['dll_pdi']
        self.track_pll_pdi = configuration['pll_pdi']
        
        self.codeStep = self.gnssSignal.codeFrequency / self.rfSignal.samplingFrequency
        self.samplesRequired = int(np.ceil((self.gnssSignal.codeBits - self.NCO_remainingCode) / self.codeStep))

        self.track_requiredSamples = int(self.gnssSignal.codeMs * self.rfSignal.samplingFrequency * 1e-3)

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

        # Update some variables
        # TODO Previously linespace, check if correct
        nbSamples = len(self.rfBuffer.buffer)
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

            # If decoding
            results.append(self.decoding())
        else:
            raise ValueError(f"Channel state {self.state} is not valid.")

        return results
    
    # -------------------------------------------------------------------------

    