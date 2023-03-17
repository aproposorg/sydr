
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
            if packet['type'] == ChannelMessage.ACQUISITION_UPDATE:
                self.addAcquisitionDatabase(packet)
            elif packet['type'] == ChannelMessage.TRACKING_UPDATE:
                self.addTrackingDatabase(packet)
            elif packet['type'] == ChannelMessage.DECODING_UPDATE:
                self.addDecodingDatabase(packet)

        return

    # -------------------------------------------------------------------------

    def addAcquisitionDatabase(self, result:dict):
        """
        Add acquisition result to database. 
        This method should be supercharge in case of a custom database saving.

        Args:
            result (dict) : Acquisition result from channel.

        Returns: 
            None
        
        Raises:
            None
        
        """

        channel : Channel
        channel = self.channelManager.getChannel(result['cid'])

        # Add the mandatory values
        result["channel_id"]   = channel.channelID
        result["time"]         = time.time()
        result["time_sample"]  = float(self.sampleCounter - result["unprocessed_samples"])

        self.database.addData("acquisition", result)

        logging.getLogger(__name__).debug(
            f"Logged acquisition results in database for CID {channel.channelID} (G{channel.satelliteID})")

        return

    # -------------------------------------------------------------------------

    def addTrackingDatabase(self, result:dict):
        """
        Add tracking result to database.
        This method should be supercharge in case of a custom database saving.

        Args:
            result (dict) : Tracking result from channel.

        Returns: 
            None
        
        Raises:
            None
        
        """

        channel : Channel
        channel = self.channelManager.getChannel(result['cid'])

        # Add the mandatory values
        result["channel_id"]   = channel.channelID
        result["time"]         = time.time()
        result["time_sample"]  = float(self.sampleCounter - result["unprocessed_samples"])

        self.database.addData("tracking", result)

        logging.getLogger(__name__).debug(
            f"Logged tracking results in database for CID {channel.channelID} (G{channel.satelliteID})")

        return

    # -------------------------------------------------------------------------

    def addDecodingDatabase(self, result:dict):
        """
        Add decoding result to database.
        This method should be supercharge in case of a custom database saving.

        Args:
            result (dict) : Decoding result from channel.

        Returns: 
            None
        
        Raises:
            None
        
        """

        channel : Channel
        channel = self.channelManager.getChannel(result['cid'])

        # Add the mandatory values
        result["channel_id"]   = channel.channelID
        result["time"]         = time.time()
        result["time_sample"]  = float(self.sampleCounter - result["unprocessed_samples"])

        self.database.addData("decoding", result)

        logging.getLogger(__name__).debug(
            f"Logged decoding results in database for CID {channel.channelID} (G{channel.satelliteID})")

        return
    
    # -------------------------------------------------------------------------
    
    def updateGUI(self):
        self.gui.updateReceiverGUI(self)
        return
