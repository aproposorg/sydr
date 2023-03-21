# -*- coding: utf-8 -*-
# =====================================================================================================================
# Abstract class for tracking process.
# Author: Antoine GRENIER (TAU)
# Date: 2023.03.15
# References: 
# =====================================================================================================================
# PACKAGES

import configparser
import logging

from core.receiver.receiver import Receiver
from core.utils.enumerations import ReceiverState
from core.channel.channel_L1CA_2 import ChannelL1CA, ChannelStatusL1CA
from core.channel.channel import ChannelMessage
from core.enlightengui import EnlightenGUI
from core.satellite.satellite import Satellite, GNSSSystems
from core.utils.time import Time

# =====================================================================================================================

class ReceiverGPSL1CA(Receiver):
    """
    Implementation of receiver for GPS L1 C/A signals. 
    """
    
    configuration : dict

    satelliteDict : dict

    channelsStatus : dict

    nextMeasurementTime : Time

    def __init__(self, configuration:dict, overwrite=True, gui:EnlightenGUI=None):
        """
        Constructor for ReceiverGPSL1CA class.

        Args:
            None

        Returns:
            None
        
        Raises:
            None
        
        """
        super().__init__(configuration, overwrite, gui)

        # Set satellites to track
        self.prnList = list(map(int, self.configuration.get('SATELLITES', 'include_prn').split(',')))

        # Add channels in channel manager
        channelConfig = configparser.ConfigParser()
        channelConfig.read(self.configuration['CHANNELS']['gps_l1ca'])
        self.channelManager.addChannel(ChannelL1CA, channelConfig, len(self.prnList))

        # Set satellites to track
        self.satelliteDict = {}
        self.channelsStatus = {}
        for prn in self.prnList:
            channel = self.channelManager.requestTracking(prn)
            self.addChannelDatabase(channel)
            self.channelsStatus[channel.channelID] = ChannelStatusL1CA(channel.channelID, prn)
            self.satelliteDict[prn] = Satellite(GNSSSystems.GPS, prn)

        self.nextMeasurementTime = Time()

        # Initialise GUI
        self.gui.createReceiverGUI(self)
        self.gui.updateMainStatus(stage=f'Processing {self.name}', status='RUNNING')

        return

    # -----------------------------------------------------------------------------------------------------------------

    def run(self):
        """
        Start the processing.

        Args:
            None

        Returns:
            None
        
        Raises:
            None
        """
        super().run()

        return
    
    # -----------------------------------------------------------------------------------------------------------------

    def _processChannelResults(self, results:list):
        super()._processChannelResults(results)

        for packet in results:
            channel : ChannelL1CA
            channel = self.channelManager.getChannel(packet['cid'])
            if packet['type'] == ChannelMessage.DECODING_UPDATE:
                satellite : Satellite
                satellite = self.satelliteDict[channel.satelliteID]
                satellite.addSubframe(packet['subframe_id'], packet['bits'])
                continue
            elif packet['type'] == ChannelMessage.CHANNEL_UPDATE:
                self.channelsStatus[channel.channelID].state         = packet['state']
                self.channelsStatus[channel.channelID].trackingFlags = packet['tracking_flags']
                self.channelsStatus[channel.channelID].tow           = packet['tow']
                self.channelsStatus[channel.channelID].timeSinceTOW  = packet['time_since_tow']
            elif packet['type'] in (ChannelMessage.ACQUISITION_UPDATE, ChannelMessage.TRACKING_UPDATE):
                continue
            else:
                raise ValueError(
                    f"Unknown channel message '{packet['type']}' received from channel {channel.channelID}.")

        return
    
    # -----------------------------------------------------------------------------------------------------------------

    def computeGNSSMeasurements(self):
        """
        
        """
        super().computeGNSSMeasurements()

        # # Compute measurements based on receiver time
        # if self.clock.isInitialised or (self.clock. < self.nextMeasurementTime):
        #     return
        
        # # TODO
        
        return


