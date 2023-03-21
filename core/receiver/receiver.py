
from abc import ABC, abstractmethod
import numpy as np
import configparser
import logging
import time

from core.utils.enumerations import ReceiverState
from core.signal.rfsignal import RFSignal
from core.channel.channelManager import ChannelManager
from core.record.database import DatabaseHandler
from core.utils.time import Clock
from core.measurements import GNSSPosition
from core.utils.coordinate import Coordinate
from core.enlightengui import EnlightenGUI
from core.channel.channel import Channel, ChannelMessage

# =============================================================================

class Receiver(ABC):
    
    configuration : dict

    name : str
    receiverState : ReceiverState
    
    clock : Clock
    coordinate : Coordinate

    # Processing
    msToProcess : int

    # I/O
    outfolder : str
    rfSignal : RFSignal
    gui : EnlightenGUI

    def __init__(self, configuration:dict, overwrite=True, gui:EnlightenGUI=None):
        self.configuration = configuration
        
        # Load configuration
        self.name        = str(configuration['DEFAULT']['name'])
        self.msToProcess = int(configuration['DEFAULT']['ms_to_process'])
        self.outfolder   = str(configuration['DEFAULT']['outfolder'])

        self.rfSignal = RFSignal(configuration['RFSIGNAL'])
        
        self.database = DatabaseHandler(f"{self.outfolder}/{self.name}.db", overwrite)

        # Initialise
        self.receiverState = ReceiverState.IDLE
        self.clock = Clock()
        self.coordinate = Coordinate()
        self.channelManager = ChannelManager(self.rfSignal)

        # Create GUI
        if not (gui is None):
            self.gui = gui

        return

    # -------------------------------------------------------------------------

    @abstractmethod
    def run(self):
        logging.getLogger(__name__).info(f"Processing in receiver {self.name} started.")
        self.receiverState = ReceiverState.INIT

        # Read the I/Q file
        msPerLoop = 1
        for idx in range(self.msToProcess):
            # Get the new RF data
            data = self.rfSignal.getMilliseconds(nbMilliseconds=msPerLoop)
            self.channelManager.addNewRFData(data)

            # Update clock
            self.clock.addTime(msPerLoop * 1e-3)

            # Process the data
            results = self.channelManager.run()

            # Wait for processing to finish 
            # timeoutFlag = self.channelManager.eventDone.wait(timeout=300)
            # if not timeoutFlag:
            #     logging.getLogger(__name__).warning(f"Channel manager timeout, exiting run.")
            #     return
            # self.channelManager.eventDone.clear()

            # Process the results
            self._processChannelResults(results)

            # Compute measurements and position
            self.computeGNSSMeasurements()

            # Update GUI
            self._updateGUI()

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

    @abstractmethod
    def computeGNSSMeasurements(self):
        return

    # -------------------------------------------------------------------------
    
    def _updateDatabaseFromChannels(self, results:list):
        """
        Update the database based on the received results. Database column and contents will be updated according to
        the dictionnary strucutre returned by the channel results. The update is not commited on the database.

        Args:
            results (list) : Results from channels.

        Returns: 
            None
        
        Raises:
            None

        """
        for packet in results:
            if packet == None:
                continue
            elif packet['type'] == ChannelMessage.ACQUISITION_UPDATE:
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
        #result["time_sample"]  = float(self.sampleCounter - result["unprocessed_samples"])

        self.database.addData("acquisition", result)

        # logging.getLogger(__name__).debug(
        #     f"Logged acquisition results in database for CID {channel.channelID} (G{channel.satelliteID})")

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
        #result["time_sample"]  = float(self.sampleCounter - result["unprocessed_samples"])

        self.database.addData("tracking", result)

        # logging.getLogger(__name__).debug(
        #     f"Logged tracking results in database for CID {channel.channelID} (G{channel.satelliteID})")

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
        #result["time_sample"]  = float(self.sampleCounter - result["unprocessed_samples"])

        self.database.addData("decoding", result)

        # logging.getLogger(__name__).debug(
        #     f"Logged decoding results in database for CID {channel.channelID} (G{channel.satelliteID})")

        return
    
    # -------------------------------------------------------------------------
    
    def _updateGUI(self):
        self.gui.updateReceiverGUI(self)
        return
    
    # -------------------------------------------------------------------------

    def close(self):
        self.channelManager.close()
        self.database.close()
        self.gui.updateMainStatus(stage=f'Processing {self.name}', status='DONE')
