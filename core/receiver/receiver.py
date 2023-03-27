
from abc import ABC, abstractmethod
import logging
import time
import math
import numpy as np

from core.utils.enumerations import ReceiverState, ChannelMessage, TrackingFlags, GNSSMeasurementType
from core.utils.constants import AVG_TRAVEL_TIME_MS, SPEED_OF_LIGHT
from core.signal.rfsignal import RFSignal
from core.channel.channelManager import ChannelManager
from core.record.database import DatabaseHandler
from core.utils.time import Clock, Time
from core.measurements import GNSSPosition
from core.utils.coordinate import Coordinate
from core.enlightengui import EnlightenGUI
from core.channel.channel import Channel, ChannelStatus
from core.satellite.satellite import Satellite
from core.measurements import GNSSmeasurements
from core.navigation.lse import LeastSquareEstimation

# =============================================================================

class Receiver(ABC):
    """
    Abstract class for Receiver object definition
    """
    
    configuration : dict

    name : str
    receiverState : ReceiverState
    channelsStatus : dict
    
    clock : Clock
    position : GNSSPosition
    samplesCounter : int
    coordinate : Coordinate
    satelliteDict : dict

    # Processing
    msToProcess : int
    nextMeasurementTime : Time
    measurementFrequency : float

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

        self.measurementFrequency = float(configuration['MEASUREMENTS']['frequency'])
        self.measurementsEnabled = {}
        self.measurementsEnabled[GNSSMeasurementType.PSEUDORANGE] = bool(configuration['MEASUREMENTS']['pseudorange'])
        self.measurementsEnabled[GNSSMeasurementType.DOPPLER]     = bool(configuration['MEASUREMENTS']['doppler'])

        # Initialise
        self.receiverState = ReceiverState.IDLE
        self.clock = Clock()
        self.position = GNSSPosition()
        self.coordinate = Coordinate()
        self.channelManager = ChannelManager(self.rfSignal)
        self.samplesCounter = 0
        self.channelsStatus = {}
        self.satelliteDict = {}
        self.navigation = LeastSquareEstimation()

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

    def computeReceiverPosition(self, week, time, measurements):
        
        nbMeasurements = len(measurements)
        G = np.zeros((nbMeasurements, 4))
        W = np.zeros((nbMeasurements, nbMeasurements))
        Ql = np.zeros((nbMeasurements, nbMeasurements))
        y = np.zeros(nbMeasurements)
        self.navigation.setState(self.approxPosition, 0.0)
        for i in range(10):
            x = self.navigation.x

            if np.linalg.norm(self.navigation.dX) < 1e-6:
                break
            # Make matrices
            idx = 0
            for meas in measurements:
                if not meas.enabled:
                    continue
                if meas.mtype == GNSSMeasurementType.PSEUDORANGE:
                    travelTime = meas.value / SPEED_OF_LIGHT
    	            
                    transmitTime = time - travelTime
                    satellite = self.satelliteDict[meas.channel.svid]
                    satpos, satclock = satellite.computePosition(transmitTime)
                    satpos = self.correctEarthRotation(travelTime, np.transpose(satpos))
                    
                    # Geometric range
                    p = np.sqrt((x[0] - satpos[0])**2 + (x[1] - satpos[1])**2 + (x[2] - satpos[2])**2)

                    # Observation vector
                    y[idx] = meas.value - p - x[3]

                    # Design matrix
                    G[idx, 0] = (x[0] - satpos[0]) / p
                    G[idx, 1] = (x[1] - satpos[1]) / p
                    G[idx, 2] = (x[2] - satpos[2]) / p
                    G[idx, 3] = 1

                    # Weight matrix
                    # TODO Implement sigma for each measurements
                    _SIGMA_PSEUDORANGE = 1.0
                    Ql[idx, idx] = _SIGMA_PSEUDORANGE
                else:
                    # TODO Adapt to other measurement types
                    continue

                idx += 1 
            
            # Least Squares
            self.navigation.G = G
            self.navigation.y = y
            self.navigation.W = W
            self.navigation.Ql = Ql

            sucess = self.navigation.compute()
        
        # Correct after minimisation
        idx = 0
        for meas in measurements:
            meas.residual = self.navigation.v[idx]
            if meas.mtype == GNSSMeasurementType.PSEUDORANGE:
                meas.value -= self.navigation.x[3]
            else:
                # TODO Adapt to other measurement types
                pass
            if meas.enabled:
                self.receiverPosition.measurements.append(meas)
            
            logging.getLogger(__name__).debug(f"CID {meas.channel.cid} SVID {meas.channel.svid:02d} {meas.mtype:11} {meas.value:13.4f} (residual: {meas.residual: .4f}, enabled: {meas.enabled})")

            idx += 1

        if sucess:
            state = self.navigation.x
            statePrecision = self.navigation.getStatePrecision()
            self.receiverPosition.time = Time.fromGPSTime(week, time)
            self.receiverPosition.coordinate.setCoordinates(state[0], state[1], state[2])
            self.receiverPosition.coordinate.setPrecision(statePrecision[0], statePrecision[1], statePrecision[2])
            self.receiverPosition.clockError = self.navigation.x[3]
            self.receiverPosition.id += 1

            self.receiverClock.absoluteTime.applyCorrection(-self.receiverPosition.clockError / SPEED_OF_LIGHT)
            self.addPositionDatabase(self.receiverPosition, measurements)
            self.measurementTimeList.append(time)
        
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
    
    def addPositionDatabase(self, position:GNSSPosition, measurements):
        
        posdict = {}

        # Position
        posdict["id"]            = position.id
        posdict["time"]          = time.time()
        posdict["time_sample"]   = self.samplesCounter
        posdict["time_receiver"] = position.time.datetime
        posdict["x"]             = position.coordinate.x
        posdict["y"]             = position.coordinate.y
        posdict["z"]             = position.coordinate.z
        posdict["clock"]         = position.clockError
        # posdict["sigma_x"]       = positon.sigma_x
        # posdict["sigma_y"]       = positon.sigma_y
        # posdict["sigma_z"]       = positon.sigma_z
        # posdict["sigma_clock"]   = positon.sigma_clock
        # posdict["gdop"]          = positon.gdop
        # posdict["pdop"]          = positon.pdop
        # posdict["hdop"]          = positon.hdop

        self.database.addData("position", posdict)

        # Measurements
        for meas in measurements:
            meas : GNSSmeasurements
            measdict = {}
            measdict["channel_id"]   = meas.channel.channelID
            measdict["time"]         = time.time()
            measdict["time_sample"]  = self.samplesCounter
            measdict["position_id"]  = position.id
            measdict["enabled"]      = meas.enabled
            measdict["type"]         = meas.mtype
            measdict["value"]        = meas.value
            measdict["raw_value"]    = meas.rawValue
            measdict["residual"]     = meas.residual

            self.database.addData("measurement", measdict)

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
