
import enlighten

from core.analysis.visualisationV2 import VisualisationV2
from core.record.database import DatabaseHandler
from core.receiver.receiver_gps_l1 import ReceiverGPSL1CA
from core.signal.rfsignal import RFSignal
import core.logger as logger

def main():
    """ 
    Main function
    """

    # Files 
    receiverConfigFile  = './config/receiver.ini'
    rfConfigFile        = './config/rf.ini' 
    benchmarkConfigFile = './config/benchmark.ini'
    loggingConfigFile   = './config/logging.ini'
    databasePath        = ".results/REC1.db"
    rinexNav            = '/mnt/c/Users/vmangr/Documents/Datasets/2021_11_30-TAU_Roof_Antenna_Tallysman/data/BRDC00IGS_R_20213340000_01D_MN.rnx'

    # GUI
    manager = enlighten.get_manager()
    status_format = ' {program}{fill}Stage: {stage}{fill} Status {status} '
    status_bar = manager.status_bar(status_format=status_format,
                                    color='bold_slategray',
                                    program='SYDR',
                                    stage='IDLE',
                                    status='RUNNING',
                                    position=1)

    # Initialisation
    logger.configureLogger(name=__name__, filepath=loggingConfigFile)
    status_bar.update(stage='Initialize', status='RUNNING')
    
    rfSignal = RFSignal(rfConfigFile)
    receiver = ReceiverGPSL1CA(receiverConfigFile, rfSignal)

    # Setup database
    receiver.database = DatabaseHandler(databasePath, overwrite=True)
    receiver.database.importRinexNav(rinexNav)

    # Run the processing
    status_bar.update(stage=f'Processing {receiver.name}', status='RUNNING')
    receiver.run(manager)
    status_bar.update(stage=f'Processing {receiver.name}', status='DONE')

    # Extract visuals
    visual = VisualisationV2(benchmarkConfigFile, rfSignal)
    visual.setDatabase(DatabaseHandler(f".results/REC1.db"))
    visual.setConfig(receiverConfigFile)
    status_bar.update(stage=f'Create report', status='RUNNING')
    visual.run()
    status_bar.update(status='DONE')

    status_bar.update(stage=f'PROCESSING COMPLETED', status='DONE')

    return

if __name__ == "__main__":
    main()
