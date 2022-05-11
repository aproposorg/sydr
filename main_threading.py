
from gnsstools.analysis import Analysis
from gnsstools.gnsssignal import GNSSSignal, SignalType
from gnsstools.receiver import Receiver
from gnsstools.rfsignal import RFSignal
from gnsstools.visualisation import Visualisation

# Files 
receiverConfigFile = './config/receiver.ini'
rfConfigFile       = './config/rf.ini'

rfSignal = RFSignal(rfConfigFile)

gnssSignals = {}
gnssSignals[SignalType.GPS_L1_CA] = GNSSSignal('./config/signals/GPS_L1_CA.ini', SignalType.GPS_L1_CA)

receiver = Receiver(receiverConfigFile, gnssSignals[SignalType.GPS_L1_CA])

# receiver.run(rfSignal, [2])

# receiver.saveSatellites('./_results/dump_satellites.pkl')

# # Analysis
# correlationMapEnabled = True
# analysis = Analysis(rfconfig)
# analysis.acquisition(receiver.satelliteDict, correlationMapEnabled)
# analysis.tracking(receiver.satelliteDict)

visual = Visualisation(rfSignal, gnssSignals)
visual.importSatellites('./_results/dump_satellites.pkl')
visual.run(SignalType.GPS_L1_CA)