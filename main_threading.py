
from gnsstools.analysis import Analysis
from gnsstools.gnsssignal import GNSSSignal, SignalType
from gnsstools.receiver import Receiver
from gnsstools.rfsignal import RFSignal
from gnsstools.visualisation import Visualisation

import pickle

# Files 
receiverConfigFile = './config/receiver.ini'
rfConfigFile       = './config/rf.ini'
benchmarkConfigFile = './config/benchmark.ini'

rfSignal = RFSignal(rfConfigFile)

gnssSignals = {}
gnssSignals[SignalType.GPS_L1_CA] = GNSSSignal('./config/signals/GPS_L1_CA.ini', SignalType.GPS_L1_CA)

receiver = Receiver(receiverConfigFile, gnssSignals[SignalType.GPS_L1_CA], rfSignal)

receiver.run([2,3,4,6,9])
receiver.saveSatellites('./_results/dump_satellites.pkl')

# with open('./_results/dump_receiver.pkl' , 'wb') as f:
#     pickle.dump(receiver, f, pickle.HIGHEST_PROTOCOL)

#receiver.loadSatellites('./_results/dump_satellites.pkl')
#receiver.computeGNSSMeasurements(292310000)

# # Analysis
# correlationMapEnabled = True
# analysis = Analysis(rfconfig)
# analysis.acquisition(receiver.satelliteDict, correlationMapEnabled)
# analysis.tracking(receiver.satelliteDict)

visual = Visualisation(benchmarkConfigFile, rfSignal, gnssSignals)
visual.importSatellites('./_results/dump_satellites.pkl')
visual.receiver = receiver
visual.run(SignalType.GPS_L1_CA)