
from abc import ABC, abstractmethod
import numpy as np
import configparser
import logging
import time

from core.utils.enumerations import ReceiverState
from core.signal.rfsignal import RFSignal
from core.channel.channelManager import ChannelManager
from core.record.database import DatabaseHandler
from core.utils.clock import Clock
from core.measurements import GNSSPosition
from core.enlightengui import EnlightenGUI
from core.channel.channel import Channel, ChannelMessage

# =============================================================================

class Receiver(ABC):

    name : str
    receiverState : ReceiverState
    
    clock : Clock
    position : GNSSPosition

    # Processing
    msToProcess : int

    # I/O
    outfolder : str
    rfSignal : RFSignal
    gui : EnlightenGUI

    def __init__(self, configFilePath:str, overwrite=True, gui:EnlightenGUI=None):
        configuration = configparser.ConfigParser(configFilePath)
        
        # Load configuration
        self.name = configuration['DEFAULT']['name']
        self.msToProcess = configuration['DEFAULT']['ms_to_process']
        self.outfolder = configuration['DEFAULT']['outfolder']

        self.rfSignal = RFSignal(configuration['RFSIGNAL']['filepath'])
        
        self.database = DatabaseHandler(f"{self.outfolder}/{self.name}.db", overwrite)

        # Initialise
        self.receiverState = ReceiverState.IDLE
        self.clock = Clock()
        self.position = GNSSPosition()

        # Find the amount of space needed in shared storage
        # TODO Compute based on buffer size needed by channels
        buffersize = self.rfSignal.samplingFrequency * 1e-3 * 100
        dtype = self.rfSignal.dtype
        self.channelManager = ChannelManager(buffersize, dtype)

        # Create GUI
        if not (gui is None):
            self.gui = gui
            self.gui.createReceiverGUI(self)

        return

    # -------------------------------------------------------------------------

    @abstractmethod
    def run(self):
        logging.getLogger(__name__).info(f"Processing in receiver {self.name} started.")
        self.receiverState = ReceiverState.INIT

        # Read the I/Q file
        msPerLoop = 1
        for msProcessed in range(self.msToProcess):
            # Get the new RF data
            ChannelManager.addNewRFData(self.rfSignal.getMilliseconds(nbMilliseconds=msPerLoop))

            # Update clock
            self.clock.addTime(msPerLoop * 1e-3)

            # Process the data
            results = self.channelManager.run()

            # Process the results
            self._processChannelResults(results)

        return
    
    # -------------------------------------------------------------------------

    @abstractmethod
    def _processChannelResults(self, results):
        """
        Abstract method to process the results from channels.
        """
        self._updateDatabaseFromChannels(results)
        return
    
    # -------------------------------------------------------------------------
    
    def _updateDatabaseFromChannels(self, results:list):
        """
        Update the database based on the received results. Database column and contents will be updated according to
        the dictionnary strucutre returned by the channel results. 

        Args:
            results (list) : Results from channels.

        Returns: 
            None
        
        Raises:
            None

        """
        for packet in results:
            channel : Channel
            channel = self.channelManager.getChannel(packet['cid'])

            if packet['type'] == ChannelMessage.ACQUISITION_UPDATE:
                self.addAcquisitionDatabase(packet)
            elif packet['type'] == ChannelMessage.TRACKING_UPDATE:
                self.database.addData("tracking", packet)
                logging.getLogger(__name__).debug(
                    f"Logged tracking results in database for CID {channel.channelID} (G{channel.satelliteID})")
            
            elif packet['type'] == ChannelMessage.DECODING_UPDATE:
                self.database.addData("decoding", packet)
                logging.getLogger(__name__).debug(
                    f"Logged tracking results in database for CID {channel.channelID} (G{channel.satelliteID})")
            
            elif packet['type'] == ChannelMessage.CHANNEL_UPDATE:
                self.channelsStatus[chan.cid].state            = channelPacket[1]
                self.channelsStatus[chan.cid].trackingFlags    = channelPacket[2]
                self.channelsStatus[chan.cid].week             = channelPacket[3]
                self.channelsStatus[chan.cid].tow              = channelPacket[4]
                self.channelsStatus[chan.cid].timeSinceTOW     = channelPacket[5]
                if not np.isnan(self.channelsStatus[chan.cid].tow):
                    self.channelsStatus[chan.cid].isTOWDecoded = True

        return

    # -------------------------------------------------------------------------

    def addAcquisitionDatabase(self, result:dict):
        """
        """

        channel : Channel
        channel = self.channelManager.getChannel(result['cid'])

        # Add the mandatory values
        result["channel_id"]   = channel.channelID
        result["time"]         = time.time()
        result["time_sample"]  = float(self.sampleCounter - results["unprocessed_samples"])

        self.database.addData("acquisition", results)

        logging.getLogger(__name__).debug(
            f"Logged acquisition results in database for CID {channel.channelID} (G{channel.satelliteID})")

        return

    # -------------------------------------------------------------------------

    def addTrackingDatabase(self, channelID:int, results:dict):

        # Add the mandatory values
        results["channel_id"]   = channelID
        results["time"]         = time.time()
        results["time_sample"]  = float(self.sampleCounter - results["unprocessed_samples"])

        self.database.addData("tracking", results)

        return

    # -------------------------------------------------------------------------

    def addDecodingDatabase(self, channelID:int, results:dict):

        # Add the mandatory values
        results["channel_id"]   = channelID
        results["time"]         = time.time()
        results["time_sample"]  = float(self.sampleCounter - results["unprocessed_samples"])

        self.database.addData("decoding", results)

        return
    
    # -------------------------------------------------------------------------
    
    def updateGUI(self):
        self.gui.updateReceiverGUI(self)
        return
