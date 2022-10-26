
from core.analysis.visualisationV2 import VisualisationV2
from core.record.database import DatabaseHandler
from core.receiver.receiver_gps_l1 import ReceiverGPSL1CA
from core.signal.rfsignal import RFSignal
import core.logger as logger

# Files 
receiverConfigFile  = './config/receiver.ini'
rfConfigFile        = './config/rf.ini' 
benchmarkConfigFile = './config/benchmark.ini'
loggingConfigFile   = './config/logging.ini'

# Filepath 
databasePath = ".results/REC1.db"
rinexNav = '/mnt/c/Users/vmangr/Documents/Datasets/2021_11_30-TAU_Roof_Antenna_Tallysman/data/BRDC00IGS_R_20213340000_01D_MN.rnx'

# Initialisation
logger.configureLogger(loggingConfigFile)
rfSignal = RFSignal(rfConfigFile)
receiver = ReceiverGPSL1CA(receiverConfigFile, rfSignal)

# Setup database
receiver.database = DatabaseHandler(databasePath, overwrite=True)
receiver.database.importRinexNav(rinexNav)

# Run the processing
receiver.run([2,3,4,6,9])

# Extract visuals
visual = VisualisationV2(benchmarkConfigFile, rfSignal)
visual.setDatabase(DatabaseHandler(f".results/REC1.db"))
visual.setConfig(receiverConfigFile)
visual.run()
