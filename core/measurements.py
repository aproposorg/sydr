import numpy as np
from core.acquisition.acquisition_pcps import Acquisition
from core.channel.channel_abstract import ChannelAbstract, ChannelState
from core.channel.channel_l1ca import ChannelL1CA
from core.signal.gnsssignal import SignalType

class DSPmeasurement():
    idx : int
    sample  : float
    state : ChannelState

class AcquisitionMeasurement(DSPmeasurement):
    correlationMap   : np.array
    idxDoppler       : int
    idxCode          : int
    dopplerFrequency : float
    codeShift        : float
    acquisitionMetric: float

class TrackingMeasurement(DSPmeasurement):
    dopplerFrequency : float
    codeFrequency    : float 
    correlators      : np.array
    iPrompt          : float
    qPrompt          : float
    dll              : float
    pll              : float
    fll              : float


class DSPEpochs():
    satelliteID     : int
    signalID        : SignalType
    time            : list
    samples         : list
    state           : list
    dspMeasurements : list

    def __init__(self, satelliteID, signalID:SignalType):
        self.satelliteID = satelliteID
        self.signalID = signalID
        self.time = []
        self.state = []
        self.dspMeasurements = []
        self.samples = []
        self.dspMeasurementCounter = 0
        return

    def addAcquisition(self, time, samples, channel:ChannelAbstract):
        self.time.append(time)
        self.state.append(channel.state)

        acquisition = channel.acquisition
        
        dsp = AcquisitionMeasurement()
        dsp.idx = self.dspMeasurementCounter 
        dsp.sample = samples - channel.unprocessedSamples
        dsp.state = channel.state
        dsp.correlationMap = acquisition.correlationMap
        dsp.idxDoppler = acquisition.idxEstimatedFrequency
        dsp.idxCode = acquisition.idxEstimatedCode
        dsp.codeShift = acquisition.estimatedCode
        dsp.dopplerFrequency = acquisition.estimatedDoppler
        dsp.acquisitionMetric = acquisition.acquisitionMetric

        self.dspMeasurements.append(dsp)
        self.samples.append(dsp.sample)

        self.dspMeasurementCounter += 1

    def addTracking(self, time, samples, channel:ChannelAbstract):
        
        self.time.append(time)
        self.state.append(channel.state)

        tracking = channel.tracking
        
        dsp = TrackingMeasurement()
        dsp.idx = self.dspMeasurementCounter 
        dsp.sample = samples - channel.unprocessedSamples
        dsp.state = channel.state
        dsp.dopplerFrequency = tracking.carrierFrequency
        dsp.codeFrequency = tracking.codeFrequency
        dsp.correlators = tracking.correlatorResults
        dsp.iPrompt, dsp.qPrompt = tracking.getPrompt()
        dsp.dll = tracking.dll
        dsp.pll = tracking.pll
        dsp.fll = np.nan #TODO add FLL in tracking

        self.dspMeasurements.append(dsp)
        self.samples.append(dsp.sample)

        self.dspMeasurementCounter += 1

        return

    def getLastMeasurement(self):
        return self.dspMeasurements[-1]


# =============================================================================

class GNSSmeasurements():

    time        : float
    pseudorange : float
    doppler     : float 
    cn0         : float
    phase       : float

class GNSSEpochs():
    satelliteID     : int
    signalID        : SignalType
    time            : list
    state           : list
    gnssMeasurements : GNSSmeasurements

    def __init__(self, satelliteID, signalID:SignalType):
        self.satelliteID = satelliteID
        self.signalID = signalID
        self.time = []
        self.state = []
        self.dspMeasurements = []
        return



