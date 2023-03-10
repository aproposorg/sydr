
import numpy as np
import logging
import multiprocessing

from core.channel.channel import Channel
from core.signal.gnsssignal import GNSSSignal
from core.signal.rfsignal import RFSignal
from core.utils.circularbuffer import CircularBuffer

class ChannelManager():

    rfSignal : RFSignal  # Only used to save the properties of the RF signal.
    nbChannels : int
    nbChannelsMax : int

    channels : list
    channelsStatus : list

    channelsConfigDict : dict

    # Communication
    communicationPipe : multiprocessing.Pipe
    rfQueue : multiprocessing.Queue

    # -----------------------------------------------------------------------------------------------------------------

    def __init__(self, rfSignal:RFSignal, nbChannelsMax:int):
        self.rfSignal = rfSignal 
        self.nbChannels = 0
        self.nbChannelsMax = nbChannelsMax
        
        # Communication with receiver
        # The pipe is for "light" communication between the receiver and the manager, it has two connection (in, out).
        # The queue is only for receiving the RF data from receiver to manager, as they don't fit in a Pipe object.
        self.communicationPipe = multiprocessing.Pipe()  
        self.rfQueue = multiprocessing.Queue() 

        return
    
    # -----------------------------------------------------------------------------------------------------------------

    def setChannelConfig(self, channelTypeName:str, channel:Channel):
        """
        Set the channel object to be used and link it to a custom name.

        Args:
            channelTypeName (str): Custom name to link the channel object.
            channel (Channel): Channel object.
    
        Returns:
            None

        Raises:
            None

        """

        self.channelsConfigDict[channelTypeName] = channel

        return

    
    # -----------------------------------------------------------------------------------------------------------------
    
    def addChannel(self, channel:Channel):
        """
        Add a new channel to channel manager.

        Args:
            channel (Channel): a Channel child class. 
    
        Returns:
            None

        Raises:
            None

        """
        # Check current amount of channels
        if self.nbChannels >= self.nbChannelsMax:
            msg = f"Maximum number of channels exceeded, no channel created."
            logging.getLogger(__name__).warning(msg)
            raise Warning(msg)

        # Add channel to list
        self.channels.append(channel)

        return
    
    # -----------------------------------------------------------------------------------------------------------------

    

    # -----------------------------------------------------------------------------------------------------------------

