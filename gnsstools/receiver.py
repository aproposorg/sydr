
import numpy as np
import configparser

import matplotlib.pyplot as plt


from gnsstools.channel import Channel
from gnsstools.rffile import RFFile
from gnsstools.utils import ChannelState


class Receiver():

    READ_CHUNK = 1000 # in ms
    MS_CHUNCK = 2 # in ms

    def __init__(self, receiverConfigFile, signalConfig):
        
        config = configparser.ConfigParser()
        config.read(receiverConfigFile)

        self.name        = config.get   ('DEFAULT', 'name')
        self.nbChannels  = config.getint('DEFAULT', 'nb_channels')
        self.msToProcess = config.getint('DEFAULT', 'ms_to_process')

        # Signal received
        self.gpsL1CAEnabled = config.getboolean('SIGNAL', 'GPS_L1_CA_enabled')

        self.signalConfig = signalConfig

        return
    
    def run(self, rfConfig, satelliteList):

        # Variables
        minMsRequired = 0

        # Initialise the channels
        self.channels = []
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

            if msProcessed == 30000:
                print("here")
            
            msProcessed += 1
        return