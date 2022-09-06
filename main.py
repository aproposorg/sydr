
from core.signal.gnsssignal import GNSSSignal, SignalType
from core.receiver.receiver_gps_l1 import ReceiverGPSL1CA
from core.signal.rfsignal import RFSignal
from core.analysis.visualisation import Visualisation

# Files 
receiverConfigFile = './config/receiver.ini'
rfConfigFile       = './config/rf.ini'
benchmarkConfigFile = './config/benchmark.ini'

rfSignal = RFSignal(rfConfigFile)

gnssSignals = {}
gnssSignals[SignalType.GPS_L1_CA] = GNSSSignal('./config/signals/GPS_L1_CA.ini', SignalType.GPS_L1_CA)

receiver = ReceiverGPSL1CA(receiverConfigFile, gnssSignals[SignalType.GPS_L1_CA], rfSignal)

# Run the processing
receiver.run([2,3,4,6,9])
receiver.saveSatellites('./.results/dump_satellites.pkl')

# Extract visuals
visual = Visualisation(benchmarkConfigFile, rfSignal, gnssSignals)
visual.importSatellites('./.results/dump_satellites.pkl')
visual.receiver = receiver
visual.run(SignalType.GPS_L1_CA)