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
from gnsstools.channel.channel_default import Channel
from gnsstools.utils import ChannelState
# =============================================================================
class Receiver():

    def __init__(self, receiverConfigFile, signalConfig):

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
                    chan.setSatellite(satelliteList.pop(0))
                chan.run(rfData)
            
            msProcessed += 1
        return

    # END OF CLASS


