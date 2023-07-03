# -*- coding: utf-8 -*-
# ============================================================================
# Implementation of Channel for GPS L1 C/A signals with circular correlation
# Author: Antoine GRENIER (TAU)
# Date: 2023.07.03
# References: 
# =============================================================================
# PACKAGES
import multiprocessing
import numpy as np
import logging

from sydr.channel.channel_l1ca_kaplan import ChannelL1CA_Kaplan
from sydr.dsp.tracking import EPL_circular
from sydr.dsp.acquisition import PCPS
from sydr.signal.gnsssignal import UpsampleCode, GenerateGPSGoldCode
from sydr.utils.enumerations import GNSSSystems, GNSSSignalType, ChannelState
from sydr.utils.constants import GPS_L1CA_CODE_FREQ, GPS_L1CA_CODE_SIZE_BITS
from sydr.utils.constants import LNAV_MS_PER_BIT

# =====================================================================================================================

class ChannelL1CA(ChannelL1CA_Kaplan):

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
        self.code = GenerateGPSGoldCode(satelliteID)

        self.codeFrequency = GPS_L1CA_CODE_FREQ

        return

    # -----------------------------------------------------------------------------------------------------------------

    def runSignalSearch(self):
        """
        """

        # Prepare necessary variables
        samplesPerCode = round(self.rfSignal.samplingFrequency * GPS_L1CA_CODE_SIZE_BITS / GPS_L1CA_CODE_FREQ)

        correlationMap = PCPS(
            rfData = self.rfBuffer.getSlice(self.currentSample, self.acq_requiredSamples), 
            interFrequency = self.rfSignal.interFrequency,
            samplingFrequency = self.rfSignal.samplingFrequency,
            code=self.code,
            dopplerRange=self.acq_dopplerRange,
            dopplerStep=self.acq_dopplerSteps,
            samplesPerCode=samplesPerCode, 
            coherentIntegration=self.acq_coherentIntegration,
            nonCoherentIntegration=self.acq_nonCoherentIntegration)

        return correlationMap

# -----------------------------------------------------------------------------------------------------------------

    def postAcquisitionUpdate(self, acqIndices):
        """
        """
        
        # Update variables
        dopplerShift = -self.acq_dopplerRange + self.acq_dopplerSteps * acqIndices[0]
        self.codeOffset = int(np.round(acqIndices[1]))
        self.carrierFrequency = self.rfSignal.interFrequency + dopplerShift

        # Update index
        self.currentSample = self.currentSample + self.acq_requiredSamples
        
        # Switch channel state to tracking
        # TODO Test if succesful acquisition
        self.channelState = ChannelState.TRACKING

        return

# -----------------------------------------------------------------------------------------------------------------
    
    def runCorrelators(self):
        """
        """
        # Prepare the code
        self.correlatorsResults[:] = EPL_circular(
            rfData = self.rfBuffer.getSlice(self.currentSample, self.track_requiredSamples),
            code = self.code,
            samplingFrequency=self.rfSignal.samplingFrequency,
            carrierFrequency=self.carrierFrequency,
            remainingCarrier=self.remainingCarrier,
            remainingCode=self.remainingCode,
            codeStep=self.codeStep,
            correlatorsSpacing=self.track_correlatorsSpacing,
            codeOffset=self.codeOffset)
        
        # Check buffer index
        if self.correlatorsAccumCounter == LNAV_MS_PER_BIT:
            self.correlatorsAccumCounter = 0
            self.correlatorsAccum[:] = 0.0
        
        # Update accumulators
        self.correlatorsAccum += self.correlatorsResults[:]
        self.correlatorsAccumCounter += 1

        return