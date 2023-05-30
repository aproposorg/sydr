
import configparser

from sydr.enlightengui import EnlightenGUI
from sydr.receiver.receiver_gps_l1ca import ReceiverGPSL1CA
from sydr.io.visualisation import Visualisation

import sydr.logger as logger

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

    # Closing receiver to free database and memory
    receiver.close()

    # Create report
    gui.updateMainStatus(stage='Create report', status='RUNNING')
    visual = Visualisation(receiverConfig)
    visual.run()    

    gui.updateMainStatus(stage=f'PROCESSING COMPLETED', status='DONE')

    return

if __name__ == "__main__":
    main()
