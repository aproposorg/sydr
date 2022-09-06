
from gnsstools.signal.gnsssignal import GNSSSignal, SignalType
from gnsstools.receiver import Receiver
from gnsstools.signal.rfsignal import RFSignal
from gnsstools.analysis.visualisation import Visualisation

# Files 
receiverConfigFile = './config/receiver.ini'
rfConfigFile       = './config/rf.ini'
benchmarkConfigFile = './config/benchmark.ini'

rfSignal = RFSignal(rfConfigFile)

gnssSignals = {}
gnssSignals[SignalType.GPS_L1_CA] = GNSSSignal('./config/signals/GPS_L1_CA.ini', SignalType.GPS_L1_CA)

receiver = Receiver(receiverConfigFile, gnssSignals[SignalType.GPS_L1_CA], rfSignal)

# Run the processing
receiver.run([2,3,4,6,9])
receiver.saveSatellites('./_results/dump_satellites.pkl')

# Extract visuals
visual = Visualisation(benchmarkConfigFile, rfSignal, gnssSignals)
visual.importSatellites('./_results/dump_satellites.pkl')
visual.receiver = receiver
visual.run(SignalType.GPS_L1_CA)