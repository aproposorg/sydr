import configparser
import numpy as np
import matplotlib.pyplot as plt

from gnsstools.gnsssignal import GNSSSignal
from gnsstools.rffile import RFFile
from gnsstools.acquisition import Acquisition

class Tracking:

    def __init__(self, configfile, acquisition:Acquisition):
        config = configparser.ConfigParser()
        config.read(configfile)
        
        self.pdiCode    = config.getfloat('TRACKING', 'pdi_code')
        self.pdiCarrier = config.getfloat('TRACKING', 'pdi_carrier')

        # Load tracking parameters
        self.dllCorrelatorSpacing = config.getfloat('TRACKING', 'dll_correlator_spacing')
        self.dllDumpingRatio      = config.getfloat('TRACKING', 'dll_dumping_ratio')
        self.dllNoiseBandwidth    = config.getfloat('TRACKING', 'dll_noise_bandwidth')
        self.dllLoopGain          = config.getfloat('TRACKING', 'dll_loop_gain')

        self.pllDumpingRatio      = config.getfloat('TRACKING', 'pll_dumping_ratio')
        self.pllNoiseBandwidth    = config.getfloat('TRACKING', 'pll_noise_bandwidth')
        self.pllLoopGain          = config.getfloat('TRACKING', 'pll_loop_gain')
        
        self.acquisition = acquisition
        self.prn         = self.acquisition.prn
        self.signal      = self.acquisition.signal
        self.init_freq   = self.acquisition.coarseFreq
        self.init_code   = self.acquisition.coarseCode

        self.codeFrequency    = []
        self.carrierFrequency = []
        self.codeError        = []
        self.codeNCO          = []
        self.carrierError     = []
        self.carrierNCO       = []
        self.iEarly  = []
        self.qEarly  = []
        self.iPrompt = []
        self.qPrompt = []
        self.iLate   = []
        self.qLate   = []   
        
        # Initialise
        self.dllTau1, self.dllTau2 = self.getLoopCoefficients(self.dllNoiseBandwidth, \
            self.dllDumpingRatio, self.dllLoopGain)
        self.pllTau1, self.pllTau2 = self.getLoopCoefficients(self.pllNoiseBandwidth, \
            self.pllDumpingRatio, self.pllLoopGain)

        return
    
    def getLoopCoefficients(self, loopNoiseBandwidth, dumpingRatio, loopGain):
        """
        Compute loop coeficients for PLL and DLL loops. From code of [Borre, 2007].

        Parameters
        ----------
        loopNoiseBandwidth : float
            Loop Noise Bandwith parameter
        dumpingRatio : float
            Dumping Ratio parameter, a.k.a. zeta
        loopGain
            Loop Gain parameter
        
        Returns
        -------
        tau1, tau2 : float, float
            Loop filter coeficients
        """

        Wn = loopNoiseBandwidth * 8.0 * dumpingRatio / \
            (4.0 * dumpingRatio**2 +1)
        
        tau1 = loopGain / Wn**2
        tau2 = 2.0 * dumpingRatio / Wn

        return tau1, tau2

    def track(self, signal_file:RFFile, ms_to_process):
        """
        Track signal based on acquisition results. Largely inspired by [Borre, 2017]
        and the Python implementation from perrysou (Github).
        """
        self.msProcessed = ms_to_process

        # Generate CA
        caCode = self.signal.getCode(self.prn) # Could be stored in acquisition

        # Extend a bit the code with last and first value
        caCode = np.r_[caCode[-1], caCode, caCode[0]]

        # Initialize
        carrierFrequency = self.init_freq
        codeFrequency    = self.signal.code_freq
        remCarrierPhase  = 0.0 # Keep the remaining part of the carrier phase for next iteration
        remCodePhase     = 0.0 # Keep the remaining part of the code phase for next iteration
        codeNCO          = 0.0
        carrierNCO       = 0.0
        oldCodeError     = 0.0
        oldCarrierError  = 0.0

        # Start tracking
        for code_counter in range(ms_to_process):
            # -----------------------------------------------------------------
            # Read signal from file
            codePhaseStep = codeFrequency / signal_file.samp_freq
            chunck = int(np.ceil((self.signal.code_bit - remCodePhase) / codePhaseStep))
            
            if code_counter == 0:
                skip = self.init_code+1
                #skip = 11628
                rawSignal = signal_file.readFileByValues(nb_values=chunck, skip=skip, keep_open=True)
            else:
                rawSignal = signal_file.readFileByValues(nb_values=chunck, keep_open=True)

            # Check if there is enough data in file
            if len(rawSignal) < chunck:
                raise EOFError("EOF encountered earlier than expected in file.")
            
            # -----------------------------------------------------------------
            # Generate the code replica for correlators
            ## Early code 
            idx = np.ceil(np.linspace(remCodePhase - self.dllCorrelatorSpacing, \
                chunck * codePhaseStep + remCodePhase - self.dllCorrelatorSpacing, \
                chunck, endpoint=False)).astype(int)
            
            earlyCode = caCode[idx]

            ## Late code
            idx = np.ceil(np.linspace(remCodePhase + self.dllCorrelatorSpacing, \
                chunck * codePhaseStep + remCodePhase + self.dllCorrelatorSpacing, \
                chunck, endpoint=False)).astype(int)

            lateCode = caCode[idx]

            ## Prompt code
            # We don't apply ceil nor int here because we need the float value
            # of the last index to compute the remaining phase
            idx = np.linspace(remCodePhase, \
                chunck * codePhaseStep + remCodePhase, \
                chunck, endpoint=False)
            
            promptCode = caCode[np.ceil(idx).astype(int)]

            # Update the remain code phase variable
            remCodePhase = idx[chunck-1] + codePhaseStep - 1023.0

            # -----------------------------------------------------------------
            # Generate carrier replica and mix to remove frequency shift
            # We use (chunck+1) and not (chunck) because we want one more to
            # estimate the remaining of the carrier phase
            time = np.arange(0, chunck+1) / signal_file.samp_freq
            #temp = carrierFrequency * 2.0 * np.pi * time + remCarrierPhase
            temp = (carrierFrequency * 2.0 * np.pi * time) + remCarrierPhase

            remCarrierPhase = temp[chunck] % (2 * np.pi)
            
            carrierSignal = np.exp(1j * temp[:chunck]) * rawSignal
            iSignal = np.imag(carrierSignal)
            qSignal = np.real(carrierSignal)
            #iSignal = np.sin(temp[:chunck]) * rawSignal # In-phase
            #qSignal = np.cos(temp[:chunck]) * rawSignal # Quadraphase

            # -----------------------------------------------------------------
            # Correlators update
            iEarly  = np.sum(earlyCode  * iSignal)
            qEarly  = np.sum(earlyCode  * qSignal)
            iPrompt = np.sum(promptCode * iSignal)
            qPrompt = np.sum(promptCode * qSignal)
            iLate   = np.sum(lateCode   * iSignal)
            qLate   = np.sum(lateCode   * qSignal)
            
            # -----------------------------------------------------------------
            # DLL
            codeError = (np.sqrt(iEarly**2 + qEarly**2) - np.sqrt(iLate**2 + qLate**2)) / \
                (np.sqrt(iEarly**2 + qEarly**2) + np.sqrt(iLate**2 + qLate**2))
            
            # Update NCO code
            codeNCO += self.dllTau2 / self.dllTau1 * (codeError - oldCodeError) \
                + codeError * (self.pdiCode / self.dllTau1)
            
            oldCodeError = codeError

            codeFrequency = self.signal.code_freq - codeNCO

            # -----------------------------------------------------------------
            # PLL
            carrierError = np.arctan(qPrompt / iPrompt) / 2.0 / np.pi

            # Update NCO carrier
            carrierNCO += self.pllTau2 / self.pllTau1 * (carrierError - oldCarrierError) \
                + carrierError * (self.pdiCarrier / self.pllTau1)

            oldCarrierError = carrierError

            carrierFrequency = self.init_freq + carrierNCO

            # -----------------------------------------------------------------
            # FLL
            # TODO

            # Save variables
            self.codeFrequency.append(codeFrequency)
            self.carrierFrequency.append(carrierFrequency)
            self.codeError.append(codeError)
            self.codeNCO.append(codeNCO)
            self.carrierError.append(carrierError)
            self.carrierNCO .append(carrierNCO)
            self.iEarly.append(iEarly)
            self.qEarly.append(qEarly)
            self.iPrompt.append(iPrompt)
            self.qPrompt.append(qPrompt)
            self.iLate.append(iLate)
            self.qLate.append(qLate)

        # TODO Should pre-allocate all the arrays and put this in numpy arrays right away
        self.iPrompt = np.array(self.iPrompt)

        signal_file.closeFile()

        return