
import configparser

from core.enlightengui import EnlightenGUI
from core.receiver.receiver_gps_l1ca_mp import ReceiverGPSL1CA
from core.analysis.visualisationV2 import VisualisationV2

import core.logger as logger

def main():
    """ 
    Main function
    """

    # Configuration
    receiverConfigFile = './config/receiver.ini'
    receiverConfig  = configparser.ConfigParser()
    receiverConfig.read(receiverConfigFile)

    # Create GUI
    gui = EnlightenGUI()
    gui.updateMainStatus(stage='Initialize', status='RUNNING')

    # Create logger
    logger.configureLogger(name=__name__, filepath='./config/logging.ini')

    # Create receiver
    receiver = ReceiverGPSL1CA(receiverConfig, overwrite=True, gui=gui)

    # Run receiver
    receiver.run()

    # Create report
    benchmarkConfigFile = './config/benchmark.ini'
    visual = VisualisationV2(benchmarkConfigFile, receiver.rfSignal)
    visual.setDatabase(receiver.database)
    visual.setConfig(receiverConfigFile)
    gui.updateMainStatus(stage='Create report', status='RUNNING')
    visual.run()

    # Closing receiver to free database and memory
    receiver.close()

    gui.updateMainStatus(stage=f'PROCESSING COMPLETED', status='DONE')

    return

if __name__ == "__main__":
    main()
