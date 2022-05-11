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
import pickle
from gnsstools.channel.abstract import ChannelState
from gnsstools.channel.channel_default import Channel
from gnsstools.gnsssignal import GNSSSignal, SignalType
from gnsstools.satellite import Satellite
# =============================================================================
class Receiver():

    def __init__(self, receiverConfigFile, gnssSignal:GNSSSignal):

        self.gnssSignal = gnssSignal
        
        config = configparser.ConfigParser()
        config.read(receiverConfigFile)

        self.name        = config.get   ('DEFAULT', 'name')
        self.nbChannels  = config.getint('DEFAULT', 'nb_channels')
        self.msToProcess = config.getint('DEFAULT', 'ms_to_process')

        self.channels = []

        return

    # -------------------------------------------------------------------------
    
    def run(self, rfConfig, satelliteList):

        # TEMPORARY solution TODO change
        signalDict = {}
        signalDict[SignalType.GPS_L1_CA] = self.gnssSignal

        # Initialise the channels
        for idx in range(min(self.nbChannels, len(satelliteList))):
            self.channels.append(Channel(idx, self.gnssSignal, rfConfig))

        # Initialise satellite structure
        self.satelliteDict = {}
        for svid in satelliteList:
            self.satelliteDict[svid] = Satellite(svid, signalDict)

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
                if chan.getState() == ChannelState.OFF:
                    continue
                elif chan.getState() == ChannelState.IDLE:
                    # Give a new satellite from list
                    if not satelliteList:
                        # List empty, disable channel
                        chan.switchState(ChannelState.OFF)
                        continue
                    svid = satelliteList.pop(0)
                    # Set the satellite parameters in the channel
                    chan.setSatellite(svid)
                chan.run(rfData)

            # Handle results
            self.processResults(msProcessed)
            
            msProcessed += 1
        return

    # -------------------------------------------------------------------------

    def processResults(self, msProcessed):
        for chan in self.channels:
            svid = chan.svid
            signal = chan.gnssSignal.signalType
            state = chan.getState()
            dsp   = self.satelliteDict[svid].dspEpochs[signal] # For cleaner code
            if chan.getState() == ChannelState.OFF:
                continue
            elif state == ChannelState.IDLE:
                # Signal was not aquired
                dsp.addAcquisition(msProcessed, state, chan.acquisition)
                pass
            elif state == ChannelState.ACQUIRING:
                # Buffer not full (most probably)
                pass
            elif state == ChannelState.ACQUIRED:
                # Signal was acquired
                dsp.addAcquisition(msProcessed, state, chan.acquisition)
                pass
            elif state == ChannelState.TRACKING:
                # Signal is being tracked
                dsp.addTracking(msProcessed, state, chan.tracking)
            else:
                raise ValueError(f"State {state} in channel {chan.cid} is not a valid state.")
        return

    def saveSatellites(self, outfile):
        with open(outfile , 'wb') as f:
            pickle.dump(self.satelliteDict, f, pickle.HIGHEST_PROTOCOL)
        return


    # END OF CLASS


