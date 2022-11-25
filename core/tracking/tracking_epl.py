# -*- coding: utf-8 -*-
# ============================================================================
# Class for tracking using Early-Prompt-Late method.
# Author: Antoine GRENIER (TAU)
# Date: 2022.05.04
# References: 
# =============================================================================
# PACKAGES
import configparser
import logging
import numpy as np
from core.signal.gnsssignal import GNSSSignal
from core.signal.rfsignal import RFSignal
from core.tracking.tracking_abstract import TrackingAbstract
# =============================================================================
class Tracking(TrackingAbstract):
    """
    TODO
    """

    def __init__(self, rfSignal:RFSignal, gnssSignal:GNSSSignal):
        """
        TODO
        """
        super().__init__(rfSignal, gnssSignal)

        config = self.gnssSignal.config

        self.pdiCode    = config.getfloat('TRACKING', 'pdi_code')
        self.pdiCarrier = config.getfloat('TRACKING', 'pdi_carrier')
        
        correlatorNumber = config.getint  ('TRACKING', 'correlator_number')
        self.correlatorSpacing = []
        for i in range(correlatorNumber):
            self.correlatorSpacing.append(config.getfloat('TRACKING', f'correlator_{i}'))
        self.correlatorPrompt = config.getint('TRACKING', f'correlator_prompt')

        self.dllDumpingRatio   = config.getfloat('TRACKING', 'dll_dumping_ratio')
        self.dllNoiseBandwidth = config.getfloat('TRACKING', 'dll_noise_bandwidth')
        self.dllLoopGain       = config.getfloat('TRACKING', 'dll_loop_gain')

        self.pllDumpingRatio   = config.getfloat('TRACKING', 'pll_dumping_ratio')
        self.pllNoiseBandwidth = config.getfloat('TRACKING', 'pll_noise_bandwidth')
        self.pllLoopGain       = config.getfloat('TRACKING', 'pll_loop_gain')

        # Initialise class variables
        self.remCodePhase     = 0.0
        self.remCarrierPhase  = 0.0
        self.codePhaseStep    = 0.0
        self.codeFrequency    = 0.0
        self.codeNCO          = 0.0
        self.codeError        = 0.0
        self.initialFrequency = 0.0
        self.carrierFrequency = 0.0
        self.carrierNCO       = 0.0
        self.carrierError     = 0.0

        self.code    = []
        self.iSignal = []
        self.qSignal = []

        self.dllTau1, self.dllTau2 = self.getLoopCoefficients(self.dllNoiseBandwidth, \
            self.dllDumpingRatio, self.dllLoopGain)
        self.pllTau1, self.pllTau2 = self.getLoopCoefficients(self.pllNoiseBandwidth, \
            self.pllDumpingRatio, self.pllLoopGain)

        self.codeFrequency   = self.gnssSignal.codeFrequency
        self.codePhaseStep   = self.gnssSignal.codeFrequency / self.rfSignal.samplingFrequency
        self.samplesRequired = int(np.ceil((self.gnssSignal.codeBits - self.remCodePhase) / self.codePhaseStep))
        
        self.correlatorResults = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.pll = 0.0
        self.dll = 0.0

        self.time = np.arange(0, self.samplesRequired+2) / self.rfSignal.samplingFrequency

        return
    
    # -------------------------------------------------------------------------

    def run(self, rfData):
        """
        TODO
        """

        replica = self.generateReplica()
        
        carrierSignal = replica * rfData
        self.iSignal = np.real(carrierSignal)
        self.qSignal = np.imag(carrierSignal)

        # Build correlators (Early-Prompt-Late)
        iEarly , qEarly  = self.getCorrelator(self.correlatorSpacing[0])
        iPrompt, qPrompt = self.getCorrelator(self.correlatorSpacing[1])
        iLate  , qLate   = self.getCorrelator(self.correlatorSpacing[2])

        self.correlatorResults = [iEarly, qEarly, iPrompt, qPrompt, iLate, qLate]
        
        # Delay Lock Loop (DLL)
        self.delayLockLoop(iEarly, qEarly, iLate, qLate)
        
        # Phase Lock Loop (PLL)
        self.phaseLockLoop(iPrompt, qPrompt)

        # Get remaining phase
        idx = np.linspace(self.remCodePhase, self.samplesRequired * self.codePhaseStep + self.remCodePhase, \
                          self.samplesRequired, endpoint=False)
        self.remCodePhase = idx[self.samplesRequired-1] + self.codePhaseStep - self.gnssSignal.codeBits

        self.codePhaseStep = self.codeFrequency / self.rfSignal.samplingFrequency
        self.samplesRequired = int(np.ceil((self.gnssSignal.codeBits - self.remCodePhase) / self.codePhaseStep))
        
        # it will create a log of lines in the logfile if uncommented
        logging.getLogger(__name__).debug(f"svid={self.svid}, iprompt={iPrompt: 10.2f}, qprompt={iPrompt: 10.2f}, DLL={self.dll: 5.3f}, PLL={self.dll: 5.3f}")

        return

    # -------------------------------------------------------------------------
    
    def delayLockLoop(self, iEarly, qEarly, iLate, qLate):
        """
        TODO
        """

        newCodeError = (np.sqrt(iEarly**2 + qEarly**2) - np.sqrt(iLate**2 + qLate**2)) / \
                    (np.sqrt(iEarly**2 + qEarly**2) + np.sqrt(iLate**2 + qLate**2))
            
        # Update NCO code
        self.codeNCO += self.dllTau2 / self.dllTau1 * (newCodeError - self.codeError)
        self.codeNCO += self.pdiCode / self.dllTau1 * newCodeError
        
        self.codeError = newCodeError
        self.codeFrequency = self.gnssSignal.codeFrequency - self.codeNCO
        self.dll = self.codeNCO

        return

    # -------------------------------------------------------------------------

    def phaseLockLoop(self, iPrompt, qPrompt):
        """
        TODO
        """

        newCarrierError = np.arctan(qPrompt / iPrompt) / 2.0 / np.pi

        # Update NCO frequency
        self.carrierNCO += self.pllTau2 / self.pllTau1 * (newCarrierError - self.carrierError)
        self.carrierNCO += self.pdiCarrier / self.pllTau1 * newCarrierError

        self.carrierError = newCarrierError
        self.carrierFrequency = self.initialFrequency + self.carrierNCO
        self.pll = self.carrierNCO

        return

    # -------------------------------------------------------------------------
    
    def getLoopCoefficients(self, loopNoiseBandwidth, dumpingRatio, loopGain):
        """
        Compute loop coeficients for PLL and DLL loops. From code of [Borre, 2007].

        Args:
            loopNoiseBandwidth (float): Loop Noise Bandwith parameter
            dumpingRatio (float): Dumping Ratio parameter, a.k.a. zeta
            loopGain (float): Loop Gain parameter
        
        Returns
            tau1 (float): Loop filter coeficient (1st)
            tau2 (float): Loop filter coeficient (2nd)

        Raises:
            None
        """

        Wn = loopNoiseBandwidth * 8.0 * dumpingRatio / \
            (4.0 * dumpingRatio**2 +1)
        
        tau1 = loopGain / Wn**2
        tau2 = 2.0 * dumpingRatio / Wn

        return tau1, tau2

    # -------------------------------------------------------------------------

    def getPrompt(self):
        return self.correlatorResults[2], self.correlatorResults[3]

    # -------------------------------------------------------------------------

    def getDatabaseDict(self):
        """
        Contains the information to be save in the database in the form of a 
        dictionnary. The key is the column name.

        Returns:
            mdict (Dict): Information to be saved.

        """
        mdict = super().getDatabaseDict()
        mdict["i_early"]           = self.correlatorResults[0],
        mdict["q_early"]           = self.correlatorResults[1],
        mdict["i_prompt"]          = self.correlatorResults[2],
        mdict["q_prompt"]          = self.correlatorResults[3],
        mdict["i_late"]            = self.correlatorResults[4],
        mdict["q_late"]            = self.correlatorResults[5],
        mdict["dll"]               = self.dll,
        mdict["pll"]               = self.pll,
        mdict["carrier_frequency"] = self.carrierFrequency,
        mdict["code_frequency"]    = self.codeFrequency

        return mdict
    
    # -------------------------------------------------------------------------
    # END OF CLASS


