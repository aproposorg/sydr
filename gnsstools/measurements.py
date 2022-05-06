import numpy as np
from gnsstools.channel.abstract import ChannelState
from gnsstools.gnsssignal import GNSSSignal, SignalType

class DSPmeasurements():

    signalType: SignalType
    time: list
    channelState: list

    # Acquisition 
    estimatedFrequency: float
    estimatedCode: float
    acquisitionMetric : float

    # Tracking
    iPrompt: list
    qPrompt: list
    dll: list
    pll: list
    codeFrequency: list
    carrierFrequency: list

    def __init__(self, signalConfig:GNSSSignal):
        self.signalConfig = signalConfig
        self.time = []
        self.channelState = []

        self.iPrompt = []
        self.qPrompt = []
        self.dll = []
        self.pll = []
        self.codeFrequency = [] 
        self.carrierFrequency = []

        return


class GNSSmeasurements():

    signalType: SignalType
    time: list
    channelState: ChannelState

    pseudorange: list
    doppler: list 
    cn0: list
    phase: list

    def __init__(self):
        return