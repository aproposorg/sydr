
from gnsstools.analysis import Analysis
from gnsstools.gnsssignal import GNSSSignal, SignalType
from gnsstools.receiver import Receiver
from gnsstools.rfsignal import RFSignal

# Files 
receiverConfigFile = './config/receiver.ini'
rfConfigFile       = './config/rf.ini'

rfconfig = RFSignal(rfConfigFile)

signalConfig = {}
signalConfig[SignalType.GPS_L1_CA] = GNSSSignal('./config/signals/GPS_L1_CA.ini', SignalType.GPS_L1_CA)

receiver = Receiver(receiverConfigFile, signalConfig[SignalType.GPS_L1_CA])

receiver.run(rfconfig, list(range(1,33)))

# Analysis
correlationMapEnabled = True
analysis = Analysis(rfconfig)
analysis.acquisition(receiver.satelliteDict, correlationMapEnabled)
analysis.tracking(receiver.satelliteDict)