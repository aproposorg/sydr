# -*- coding: utf-8 -*-
# ============================================================================
# Abstract class for tracking process.
# Author: Antoine GRENIER (TAU)
# Date: 2022.05.04
# References: 
# =============================================================================
# PACKAGES
import numpy as np
import configparser
from gnsstools.channel.abstract import ChannelState
from gnsstools.channel.channel_default import Channel
from gnsstools.gnsssignal import GNSSSignal
from gnsstools.measurements import DSPmeasurements
from gnsstools.satellite import Satellite
# =============================================================================
class Receiver():

    def __init__(self, receiverConfigFile, signalConfig:GNSSSignal):

        self.signalConfig = signalConfig
        
        config = configparser.ConfigParser()
        config.read(receiverConfigFile)

        self.name        = config.get   ('DEFAULT', 'name')
        self.nbChannels  = config.getint('DEFAULT', 'nb_channels')
        self.msToProcess = config.getint('DEFAULT', 'ms_to_process')

        self.channels = []

        return

    # -------------------------------------------------------------------------
    
    def run(self, rfConfig, satelliteList):

        # Initialise the channels
        for idx in range(self.nbChannels):
            self.channels.append(Channel(idx, self.signalConfig, rfConfig))

        # Initialise satellite structure
        self.satelliteDict = {}
        for svid in satelliteList:
            self.satelliteDict[svid] = Satellite(svid)
            self.satelliteDict[svid].dspMeasurements.append(DSPmeasurements(self.signalConfig.signalType))

        # Loop through the file contents
        # TODO Load by chunck of data instead of ms per ms
        msInSamples = int(rfConfig.samplingFrequency * 1e-3)
        rfData = np.empty(msInSamples)
        for msProcessed in range(self.msToProcess):
            rfData = rfConfig.readFile(timeLength=1, keep_open=True)

            if rfData.size < msInSamples:
                raise EOFError("EOF encountered earlier than expected in file.")

            # Run channels
            for chan in self.channels:
                if chan.getState() == ChannelState.IDLE:
                    # Give a new satellite from list
                    svid = satelliteList.pop(0)

                    # Set the satellite parameters in the channel
                    chan.setSatellite(svid)
                chan.run(rfData)

            # Handle results
            for chan in self.channels:
                state = chan.getState()
                dsp   = self.satelliteDict[svid].dspMeasurements[-1] # For cleaner code
                if state == ChannelState.IDLE:
                    # Signal was not aquired
                    self.satelliteDict[svid].dspMeasurements.append(DSPmeasurements())
                    pass
                elif state == ChannelState.ACQUIRING:
                    # Buffer not full (most probably)
                    pass
                elif state == ChannelState.ACQUIRED:
                    # Signal was acquired
                    frequency, code, correlationMap = chan.getAcquisitionEstimation()
                    dsp.estimatedFrequency = frequency
                    dsp.estimatedCode = code
                    dsp.correlationMap = correlationMap
                    pass
                elif state == ChannelState.TRACKING:
                    # Signal is being tracked
                    carrier, code, iPrompt, qPrompt, dll, pll = chan.getTrackingEstimation()
                    dsp.carrierFrequency.append(carrier)
                    dsp.codeFrequency.append(code)
                    dsp.iPrompt.append(iPrompt)
                    dsp.qPrompt.append(qPrompt)
                    dsp.dll.append(dll)
                    dsp.pll.append(pll)
                else:
                    raise ValueError(f"State {state} in channel {chan.cid} is not a valid state.")
            
            msProcessed += 1
        return

    # END OF CLASS


