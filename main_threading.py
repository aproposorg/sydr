
from gnsstools.gnsssignal import GNSSSignal, SignalType
from gnsstools.receiver import Receiver
from gnsstools.rffile import RFFile

# Files 
receiverConfigFile = './config/receiver.ini'
rfConfigFile       = './config/rf.ini'

rffile = RFFile(rfConfigFile)

signalConfig = {}
signalConfig[SignalType.GPS_L1_CA] = GNSSSignal('./config/signals/GPS_L1_CA.ini', SignalType.GPS_L1_CA)

receiver = Receiver(receiverConfigFile, signalConfig[SignalType.GPS_L1_CA])

receiver.run(rffile, [2,3,4,6,9,29,31])

