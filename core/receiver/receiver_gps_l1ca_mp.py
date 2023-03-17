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
from core.channel.channel_L1CA_2 import ChannelL1CA
from core.channel.channel import ChannelMessage

# =====================================================================================================================

class ReceiverGPSL1CA(Receiver):
    """
    Implementation of receiver for GPS L1 C/A signals. 
    """
    
    configuration : dict

    def __init__(self, configFilePath:str):
        """
        Constructor for ReceiverGPSL1CA class.

        Args:
            None

        Returns:
            None
        
        Raises:
            None
        
        """
        super().__init__()

        # Set satellites to track
        self.prnList = list(map(int, config.get('SATELLITES', 'include_prn').split(',')))

        # Add channels in channel manager
        config = configparser.ConfigParser(self.configuration['CHANNELS']['gps_l1ca'])
        self.channelManager.addChannel(ChannelL1CA, config, len(self.prnList))

        # Set satellites to track
        for prn in self.prnList:
            self.channelManager.requestTracking()

        return

    # -----------------------------------------------------------------------------------------------------------------

    def run(self, satellitesList):
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
        super()._processChannelResults()

        for packet in results:
            channel : ChannelL1CA
            channel = self.channelManager.getChannel(packet['cid'])
            if packet['type'] == ChannelMessage.DECODING_UPDATE:
                satellite = self.satelliteDict[chan.svid]
                satellite.addSubframe(channelPacket[1]['subframe_id'], channelPacket[1]['bits'])
                self.channelsStatus[chan.cid].subframeFlags[channelPacket[1]['subframe_id']-1] = True
                self.addDecodingDatabase(chan.cid, channelPacket[1])
                continue
            elif packet['type'] == ChannelMessage.CHANNEL_UPDATE:
                self.channelsStatus[chan.cid].state            = channelPacket[1]
                self.channelsStatus[chan.cid].trackingFlags    = channelPacket[2]
                self.channelsStatus[chan.cid].week             = channelPacket[3]
                self.channelsStatus[chan.cid].tow              = channelPacket[4]
                self.channelsStatus[chan.cid].timeSinceTOW     = channelPacket[5]
                if not np.isnan(self.channelsStatus[chan.cid].tow):
                    self.channelsStatus[chan.cid].isTOWDecoded = True
            


        return
    
    # -----------------------------------------------------------------------------------------------------------------




