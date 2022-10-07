
from core.analysis.visualisationV2 import VisualisationV2
from core.record.database import DatabaseHandler
from core.signal.gnsssignal import GNSSSignal
from core.receiver.receiver_gps_l1 import ReceiverGPSL1CA
from core.signal.rfsignal import RFSignal
from core.analysis.visualisation import Visualisation
from core.utils.enumerations import GNSSSignalType

# Files 
receiverConfigFile = './config/receiver.ini'
rfConfigFile       = './config/rf.ini' 
benchmarkConfigFile = './config/benchmark.ini'

rfSignal = RFSignal(rfConfigFile)

gnssSignals = {}
gnssSignals[GNSSSignalType.GPS_L1_CA] = GNSSSignal('./config/signals/GPS_L1_CA.ini', GNSSSignalType.GPS_L1_CA)

receiver = ReceiverGPSL1CA(receiverConfigFile, rfSignal)

# Setup database
receiver.database = DatabaseHandler(f".results/REC1.db", overwrite=True)
receiver.database.importRinexNav('/mnt/c/Users/vmangr/Documents/Datasets/2021_11_30-TAU_Roof_Antenna_Tallysman/data/BRDC00IGS_R_20213340000_01D_MN.rnx')

# Run the processing
receiver.run([2,3,4,6,9,29])
receiver.saveSatellites('./.results/dump_satellites.pkl')

# # Extract visuals
# visual = Visualisation(benchmarkConfigFile, rfSignal, gnssSignals)
# visual.importSatellites('./.results/dump_satellites.pkl')
# visual.receiver = receiver
# visual.run(GNSSSignalType.GPS_L1_CA)

visual = VisualisationV2(benchmarkConfigFile, rfSignal)
visual.setDatabase(DatabaseHandler(f".results/REC1.db"))
visual.setConfig(receiverConfigFile)
visual.run()
