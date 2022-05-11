import numpy as np
from gnsstools.acquisition.acquisition_pcps import Acquisition
from gnsstools.channel.abstract import ChannelConfig, ChannelState
from gnsstools.gnsssignal import GNSSSignal, SignalType
from gnsstools.tracking.tracking_epl import Tracking

class DSPmeasurement():
    time  : float
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
    state           : list
    dspMeasurements : list

    def __init__(self, satelliteID, signalID:SignalType):
        self.satelliteID = satelliteID
        self.signalID = signalID
        self.time = []
        self.state = []
        self.dspMeasurements = []
        return

    def addAcquisition(self, time, state:ChannelState, acquisition:Acquisition):
        self.time.append(time)
        self.state.append(state)
        
        dsp = AcquisitionMeasurement()
        dsp.time = time
        dsp.state = state
        dsp.correlationMap = acquisition.correlationMap
        dsp.idxDoppler = acquisition.idxEstimatedFrequency
        dsp.idxCode = acquisition.idxEstimatedCode
        dsp.codeShift = acquisition.estimatedCode
        dsp.dopplerFrequency = acquisition.estimatedDoppler
        dsp.acquisitionMetric = acquisition.acquisitionMetric

        self.dspMeasurements.append(dsp)

    def addTracking(self, time, state:ChannelState, tracking:Tracking):

        self.time.append(time)
        self.state.append(state)
        
        dsp = TrackingMeasurement()
        dsp.time = time
        dsp.state = state
        dsp.dopplerFrequency = tracking.carrierFrequency
        dsp.codeFrequency = tracking.codeFrequency
        dsp.correlators = tracking.correlatorResults
        dsp.iPrompt = tracking.correlatorResults[2*tracking.correlatorPrompt]
        dsp.qPrompt = tracking.correlatorResults[2*tracking.correlatorPrompt+1]
        dsp.dll = tracking.dll
        dsp.pll = tracking.pll
        dsp.fll = np.nan #TODO add FLL in tracking

        self.dspMeasurements.append(dsp)

        return


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



