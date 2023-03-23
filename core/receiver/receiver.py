
from abc import ABC, abstractmethod
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
    """
    Abstract class for Receiver object definition
    """
    
    configuration : dict

    name : str
    receiverState : ReceiverState
    
    clock : Clock
    samplesCounter : int
    coordinate : Coordinate

    # Processing
    msToProcess : int

    # I/O
    outfolder : str
    rfSignal : RFSignal
    gui : EnlightenGUI

    def __init__(self, configuration:dict, overwrite=True, gui:EnlightenGUI=None):
        """
        Constructor for Receiver class.

        Args: 
            configuration (dict): Configuration dictionnary.
            overwrite (bool): Boolean to overwrite previous database results.
            gui (EnlightenGUI): GUI object.

        Returns:
            None
        
        Raises:
            None
        """
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
        self.samplesCounter = 0

        # Create GUI
        if not (gui is None):
            self.gui = gui

        return

    # -------------------------------------------------------------------------
    
    def run(self):
        """
        Run receiver processing.

        Args:
            None
        
        Returns:
            None
        
        Raises:
            None
        """
        
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
            self.samplesCounter += msPerLoop * self.rfSignal.samplesPerMs

            # Process the data
            results = self.channelManager.run()

            # Process the results
            self._processChannelResults(results)

            # Compute measurements and position
            self.computeGNSSMeasurements()

            # Update GUI
            self._updateGUI()

        # Commit database
        self.database.commit()

        return
    
    # -------------------------------------------------------------------------

    @abstractmethod
    def _processChannelResults(self, results:list):
        """
        Abstract method to process the channels' results.

        Args:
            results (list): List of results (dictionnary).

        Returns:
            None

        Raises:
            None
        """

        self._updateDatabaseFromChannels(results)
        
        return
    
    # -------------------------------------------------------------------------

    @abstractmethod
    def computeGNSSMeasurements(self):
        """
        Abstract method to process the channels' results to produce GNSS measurements.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """
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
        result["time_sample"]  = self.samplesCounter

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
        result["time_sample"]  = self.samplesCounter

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
        result["time_sample"]  = self.samplesCounter
        #result["time_sample"]  = float(self.sampleCounter - result["unprocessed_samples"])

        self.database.addData("decoding", result)

        # logging.getLogger(__name__).debug(
        #     f"Logged decoding results in database for CID {channel.channelID} (G{channel.satelliteID})")

        return
    
    # -------------------------------------------------------------------------

    def addChannelDatabase(self, channel:Channel):
        """
        Save channel parameters to database.

        Args:
            channel (Channel): Channel object.

        Returns:
            None
        
        Raises:
            None
        """
        
        mdict = {
            "id"           : channel.channelID,
            "system"       : channel.systemID,
            "satellite_id" : channel.satelliteID,
            "signal"       : channel.signalID,
            "start_time"   : time.time(),
            "start_sample" : self.samplesCounter
        }

        self.database.addData("channel", mdict)

        return

    # -------------------------------------------------------------------------
    
    def _updateGUI(self):
        """
        Update the GUI with current status.

        Args:
            None
        
        Returns:
            None
        
        Raises:
            None
        """
        self.gui.updateReceiverGUI(self)
        return
    
    # -------------------------------------------------------------------------

    def close(self):
        self.channelManager.close()
        self.database.close()
        self.gui.updateMainStatus(stage=f'Processing {self.name}', status='DONE')
