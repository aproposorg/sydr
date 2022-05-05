
from gnsstools.analysis import Analysis
from gnsstools.gnsssignal import GNSSSignal, SignalType
from gnsstools.receiver import Receiver
from gnsstools.rfsignal import RFSignal

# Files 
receiverConfigFile = './config/receiver.ini'
rfConfigFile       = './config/rf.ini'

rffile = RFSignal(rfConfigFile)

signalConfig = {}
signalConfig[SignalType.GPS_L1_CA] = GNSSSignal('./config/signals/GPS_L1_CA.ini', SignalType.GPS_L1_CA)

receiver = Receiver(receiverConfigFile, signalConfig[SignalType.GPS_L1_CA])

receiver.run(rffile, [2])

# Analysis
analysis = Analysis()
analysis.tracking(receiver.satelliteDict)