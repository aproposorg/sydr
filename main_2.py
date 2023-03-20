
import configparser

from core.enlightengui import EnlightenGUI
from core.receiver.receiver_gps_l1ca_mp import ReceiverGPSL1CA

import core.logger as logger

def main():
    """ 
    Main function
    """

    # Configuration
    receiverConfig  = configparser.ConfigParser()
    receiverConfig.read('./config/receiver.ini')

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
    # TODO

    gui.updateMainStatus(stage=f'PROCESSING COMPLETED', status='DONE')

    return

if __name__ == "__main__":
    main()
