# -*- coding: utf-8 -*-
# ============================================================================
# Abstract class for tracking process.
# Author: Antoine GRENIER (TAU)
# Date: 2022.05.04
# References: 
# =============================================================================
# PACKAGES
import logging
import pickle
from datetime import datetime
import math
import time
import numpy as np
import configparser
from enlighten import Manager
from termcolor import colored
import multiprocessing

from core.channel.channel_abstract import ChannelAbstract, ChannelState, ChannelCommunication
from core.channel.channel_l1ca import ChannelL1CA
from core.measurements import GNSSPosition, GNSSmeasurements
from core.navigation.lse import LeastSquareEstimation
from core.receiver.receiver_abstract import ReceiverAbstract, ReceiverState
from core.record.database import DatabaseHandler
from core.signal.gnsssignal import GNSSSignal
from core.utils.clock import Clock
from core.utils.coordinate import Coordinate
from core.utils.enumerations import GNSSMeasurementType, GNSSSignalType, GNSSSystems
from core.signal.rfsignal import RFSignal
from core.satellite.satellite import Satellite
from core.utils.constants import AVG_TRAVEL_TIME_MS, EARTH_ROTATION_RATE, SPEED_OF_LIGHT
from core.utils.time import Time

#{sf1} {sf2} {sf3} {sf4} {sf5}
CHANNEL_BAR_FORMAT = u'{desc} |{prn}| [{state:^10} {tow} SUBFRAMES:{sf1}{sf2}{sf3}{sf4}{sf5}] {percentage:3.0f}%|{bar}| ' + \
                     u'{count:{len_total}d}/{total:d} ' + \
                     u'[{elapsed}<{eta}, {rate:.2f}{unit_pad}{unit}/s]'

RECEIVER_BAR_FORMAT = u'{desc}{desc_pad}[{state:^10}] {percentage:3.0f}%|{bar}| ' + \
                      u'{count:{len_total}d}/{total:d} ' + \
                      u'[{elapsed}<{eta}, {rate:.2f}{unit_pad}{unit}/s]'

RECEIVER_STATUS_FORMAT = u'{receiver} {fill} ' + \
                         u'X: {x:12.4f} (\u03C3: {sx: .4f}) ' + \
                         u'Y: {y:12.4f} (\u03C3: {sy: .4f}) ' + \
                         u'Z: {z:12.4f} (\u03C3: {sy: .4f}) ' + \
                         u'{fill}{datetime} (GPS Time: {gpstime})'


# =============================================================================

class ReceiverGPSL1CA(ReceiverAbstract):
    """
    Implementation of receiver for GPS L1 C/A signals. 
    """

    name        : str # Name of the receiver (for display purposes)
    nbChannels  : int # Total number of "physical" channels
    msToProcess : int # Number of millisecond to be processed

    database : DatabaseHandler

    def __init__(self, configFilePath, rfSignal:RFSignal):
        """
        Class constructor.

        Args: 
            configFilePath (str): Path to receiver '.ini' file. 
            rfSignal (RFSignal) : RF signal object with raw data.
        """
        super().__init__()
        
        config = configparser.ConfigParser()
        config.read(configFilePath)

        # DEFAULT
        self.name        = config.get   ('DEFAULT', 'name')
        self.nbChannels  = config.getint('DEFAULT', 'nb_channels')
        self.msToProcess = config.getint('DEFAULT', 'ms_to_process')
        self.outfolder   = config.get   ('DEFAULT', 'outfolder')
        
        self.approxPosition = [\
            config.getfloat('DEFAULT', 'approx_pos_x'),\
            config.getfloat('DEFAULT', 'approx_pos_y'),
            config.getfloat('DEFAULT', 'approx_pos_z')]
        
        # Navigation technique
        # TODO Add selection between LSE, Kalman, etc...
        self.navigation = LeastSquareEstimation()

        # TODO Only work for one mono-frequency receiver
        if config.getboolean('SIGNAL', 'GPS_L1_CA_enabled', fallback=False):
            self.gnssSignal  = GNSSSignal(config.get('SIGNAL', 'GPS_L1_CA_path'), GNSSSignalType.GPS_L1_CA)
        
        self.isBRDCEphemerisAssited = config.getboolean('AGNSS', 'broadcast_ephemeris')
        self.isClockAssisted        = config.getboolean('AGNSS', 'clock_assited')
        self.clockAssistedValue     = datetime.strptime(config.get('AGNSS', 'clock_assisted_value'), ("%Y-%m-%d %H:%M:%S.%f"))

        # SATELLITES
        self.satelliteList = list(map(int, config.get('SATELLITES', 'include_prn').split(',')))

        self.satelliteDict = {}
        for svid in self.satelliteList:
            self.satelliteDict[svid] = Satellite(GNSSSystems.GPS, svid)

        # MEASUREMENTS
        # GNSS measurements used and produced
        self.measurementsEnabled = {}
        self.measurementsEnabled[GNSSMeasurementType.PSEUDORANGE] = config.getboolean('MEASUREMENTS', 'pseudorange')
        self.measurementsEnabled[GNSSMeasurementType.DOPPLER]     = config.getboolean('MEASUREMENTS', 'doppler')

        # Channel parameters
        self.channels = []
        self.channelCounter = 0

        self.isClockInitialised = False
        self.sampleCounter = 0
        self.receiverClock = Clock()
        if self.isClockAssisted:
            self.receiverClock.absoluteTime.setDatetime(self.clockAssistedValue)
        self.receiverClockError = []
        self.measurementTimeList = []
        self.receiverPosition = GNSSPosition()

        self.measurementFrequency = 10  # In Hz
        self.measurementPeriod = 1 / self.measurementFrequency
        self.nextMeasurementTime = Time()

        self.rfSignal = rfSignal

        self.receiverState = ReceiverState.IDLE

        logging.getLogger(__name__).info(f"Receiver {self.name} initialized.")
        return

    # -------------------------------------------------------------------------
    
    def run(self, manager:Manager):
        """
        Start the processing.
        """
        logging.getLogger(__name__).info(f"Processing in receiver {self.name} started.")

        self.receiverState = ReceiverState.INIT

        # Create GUI interface
        clock = self.receiverClock.absoluteTime
        statusBar   = manager.status_bar(status_format=RECEIVER_STATUS_FORMAT,
                                    color='bold_white_on_steelblue4',
                                    receiver=self.name,
                                    datetime=str(clock.datetime)[:-3],
                                    gpstime=f"{clock.gpsTime.week_number} {clock.gpsTime.seconds + clock.gpsTime.femtoseconds/1e15:.3f}",
                                    x = self.receiverPosition.coordinate.x,
                                    y = self.receiverPosition.coordinate.y,
                                    z = self.receiverPosition.coordinate.z,
                                    sx = self.receiverPosition.coordinate.xPrecison,
                                    sy = self.receiverPosition.coordinate.yPrecison,
                                    sz = self.receiverPosition.coordinate.zPrecison)
        
        progressBar = manager.counter(total=self.msToProcess, desc=f'Processing', unit='ms', color='springgreen3', \
            min_delta=0.5, state=f"{self.receiverState}", bar_format=RECEIVER_BAR_FORMAT)
        
        # Initialisation 
        self._initChannels(manager)

        # Loop through the file contents
        # TODO Load by chunck of data instead of ms per ms
        msPerLoop = 1
        samplesInMs = int(msPerLoop * self.rfSignal.samplingFrequency * 1e-3)
        rfData = np.empty(samplesInMs)
        for msProcessed in range(self.msToProcess):
            # Read in file
            rfData = self.rfSignal.readFile(timeLength=msPerLoop, keep_open=True)
            self.sampleCounter += samplesInMs
            self.receiverClock.addTime(msPerLoop * 1e-3)

            if rfData.size < samplesInMs:
                logging.getLogger(__name__).error("EOF encountered earlier than expected in I/Q file.")
                raise EOFError("EOF encountered earlier than expected in I/Q file.")

            # Run the channels
            self._runChannels(rfData)

            # Handle channels results
            self._processChannels(msProcessed, self.sampleCounter)

            # Compute measurements based on receiver time
            if not self.receiverClock.isInitialised:
                # First time we run it to estimate receiver clock error
                self.computeGNSSMeasurements()
            elif self.receiverClock.absoluteTime >= self.nextMeasurementTime:
                # For measurement frequency of 1 Hz, that's every round second.
                self.computeGNSSMeasurements()

            # Update the database
            if int(self.sampleCounter % 1e7) == 0:
                self.updateDatabase()

            msProcessed += 1

            # Update GUI
            progressBar.update(state=f"{self.receiverState}")
            statusBar.update(datetime=str(clock.datetime)[:-3], 
                gpstime=f"{clock.gpsTime.week_number} {clock.gpsTime.seconds + clock.gpsTime.femtoseconds/1e15:.3f}",
                x = self.receiverPosition.coordinate.x, sx = self.receiverPosition.coordinate.xPrecison,
                y = self.receiverPosition.coordinate.y, sy = self.receiverPosition.coordinate.yPrecison,
                z = self.receiverPosition.coordinate.z, sz = self.receiverPosition.coordinate.zPrecison)
        
        # Update the last measurements
        self.updateDatabase()

        # Close all the channel threads
        for chan in self.channels:
            chan.rfQueue.put("SIGTERM")
        for chan in self.channels:
            chan.join()
        
        return

    # -------------------------------------------------------------------------

    def _initChannels(self, manager:Manager):

        # Initialise the channels
        self.channelsPB = []
        for idx in range(min(self.nbChannels, len(self.satelliteList))):
            
            # Create object for communication between processes
            _queue = multiprocessing.Queue() # The queue is for giving data to the channels
            _event = multiprocessing.Event() # The event is for informing the main process that the data has been processed
            _pipe  = multiprocessing.Pipe()  # The pipe is to send back the results from the threads. Pipe is a tuple of two connection (in, out)

            # Create channel
            self.channels.append(ChannelL1CA(idx, self.gnssSignal, self.rfSignal, 0, _queue, _event, _pipe))
            
            # Create GUI
            _tow = colored(" TOW ", 'white', 'on_red')
            _sf1 = colored("1", 'white', 'on_red')
            _sf2 = colored("2", 'white', 'on_red')
            _sf3 = colored("3", 'white', 'on_red')
            _sf4 = colored("4", 'white', 'on_red')
            _sf5 = colored("5", 'white', 'on_red')
            counter = manager.counter(total=self.msToProcess, desc=f"    Channel {idx}", \
                leave=False, unit='ms', color='lightseagreen', min_delta=0.5,
                state=f"{self.channels[idx].state}", bar_format=CHANNEL_BAR_FORMAT, prn='G00', \
                tow=_tow, sf1=_sf1, sf2=_sf2, sf3=_sf3, sf4=_sf4, sf5=_sf5)
            self.channelsPB.append(counter)

        return

    # -------------------------------------------------------------------------

    def _runChannels(self, rfData):
        for chan in self.channels:
            if chan.state == ChannelState.OFF:
                continue
            elif chan.state == ChannelState.IDLE:
                # Give a new satellite from list
                if not self.satelliteList:
                    # List empty, disable channel
                    chan.switchState(ChannelState.OFF)
                    continue
                svid = self.satelliteList.pop(0)
                # Set the satellite parameters in the channel
                chan.setSatellite(svid, self.channelCounter)

                # State the channel
                chan.start()

                self.channelCounter += 1
                self.addChannelDatabase(chan)
                self.updateDatabase() 
            
            # Run channels
            chan.rfQueue.put(rfData)

            # GUI
            _tow = colored(f" TOW: {chan.tow:6.0f}", 'white', 'on_green' if chan.isTOWDecoded else 'on_red')
            _sf1 = colored("1", 'white', 'on_green' if chan.decoding.subframes[1] else 'on_red')
            _sf2 = colored("2", 'white', 'on_green' if chan.decoding.subframes[2] else 'on_red')
            _sf3 = colored("3", 'white', 'on_green' if chan.decoding.subframes[3] else 'on_red')
            _sf4 = colored("4", 'white', 'on_green' if chan.decoding.subframes[4] else 'on_red')
            _sf5 = colored("5", 'white', 'on_green' if chan.decoding.subframes[5] else 'on_red')
            self.channelsPB[chan.cid].update(state=f"{chan.state}", prn=f'G{chan.svid:02d}', tow=_tow,\
                sf1=_sf1, sf2=_sf2, sf3=_sf3, sf4=_sf4, sf5=_sf5)
        return

    # -------------------------------------------------------------------------

    def _processChannels(self, msProcessed, sampleCounter):
        for chan in self.channels:
            # Wait for thread signal that it has finished his processing
            chan.event.wait()
            chan.event.clear()
            
            while True:
                channelPacket = chan.pipe[1].recv()
                if channelPacket == ChannelCommunication.END_OF_PIPE:
                    break
                commType = channelPacket[0]
                if commType == ChannelCommunication.ACQUISITION_UPDATE:
                    self.addAcquisitionDatabase(chan.cid, channelPacket[1])
                    continue
                elif commType == ChannelCommunication.TRACKING_UPDATE:
                    self.addTrackingDatabase(chan.cid, channelPacket[1])
                    continue
                elif commType == ChannelCommunication.DECODING_UPDATE:
                    self.addDecodingDatabase(chan.cid, channelPacket[1])
                    continue
                
                satellite = self.satelliteDict[chan.svid]
                if chan.state == ChannelState.OFF:
                    continue
                elif chan.state == ChannelState.IDLE:
                    # Signal was not aquired
                    #satellite.addDSPMeasurement(msProcessed, sampleCounter, channelResults)
                    logging.getLogger(__name__).info(f"CID {chan.cid} could not acquire satellite G{chan.svid}.")    
                    pass
                elif chan.state == ChannelState.ACQUIRING:
                    # Nothing to do, we just pass
                    pass
                elif chan.state == ChannelState.TRACKING:
                    # Check if just switched from acquiring 
                    if chan.isAcquired:
                        self.addAcquisitionDatabase(chan)
                        #satellite.addDSPMeasurement(msProcessed, sampleCounter, channelResults)
                        logging.getLogger(__name__).info(f"CID {chan.cid} found satellite G{chan.svid}, tracking started.")
                        pass

                    # Process tracking measurements
                    self.addTrackingDatabase(chan)

                    # Signal is being tracked
                    #satellite.addDSPMeasurement(msProcessed, sampleCounter, channelResults)

                    # Add navigation message
                    # TODO This pass a reference, maybe be it should be done only at init? 
                    satellite.addNavMessage(chan.decoding)

                    if chan.decoding.isNewSubframeFound:
                        self.addDecodingDatabase(chan)
                        chan.decoding.isNewSubframeFound = False

                    # Check if ephemeris decoded
                    if chan.isEphemerisDecoded:
                        # Check if already saved
                        if chan.decoding.ephemeris != satellite.getLastBRDCEphemeris():
                            satellite.addBRDCEphemeris(chan.decoding.ephemeris)
                 
            else:
                logging.getLogger(__name__).error(f"State {chan.state} in channel {chan.cid} is not a valid state.")
                raise ValueError(f"State {chan.state} in channel {chan.cid} is not a valid state.")
        return
    
    # -------------------------------------------------------------------------

    def computeGNSSMeasurements(self):
        
        # Check if channels ready for measurements
        prnList = []
        selectedChannels = []
        for chan in self.channels:
            if chan.isTOWDecoded and (chan.isEphemerisDecoded or self.isBRDCEphemerisAssited):
                prnList.append(chan.svid)
                selectedChannels.append(chan)

        # In case multi-frequency/Channel tracking, we want at least 5 unique satellites.
        # This is to avoid rank deficiency in matrices.
        prnList = set(prnList) 
        
        if len(prnList) < 5:
            # Not enough satellites
            self.receiverState = ReceiverState.INIT
            return
        self.receiverState = ReceiverState.NAVIGATION

        # Find earliest signal
        maxTOW = -1
        for chan in selectedChannels:
            # This assumes all the channels were given the same number of samples to process
            if maxTOW < chan.getTimeSinceTOW():
                maxTOW = chan.getTimeSinceTOW()
                earliestChannel = chan
        
        # Received time
        if not self.receiverClock.isInitialised:
            if self.isClockAssisted:
                week = self.receiverClock.absoluteTime.getGPSWeek()
            else:
                week = earliestChannel.week
            receivedTime = earliestChannel.tow + AVG_TRAVEL_TIME_MS / 1e3
            self.receiverClock.absoluteTime.setGPSTime(week, receivedTime)
            self.receiverClock.isInitialised = True
            self.nextMeasurementTime.setGPSTime(week, math.ceil(receivedTime))
            tow = earliestChannel.tow
        else:
            # Compute the residual time to have a "round" received time
            week =  self.receiverClock.absoluteTime.getGPSWeek()
            timeResidual = (self.receiverClock.absoluteTime - self.nextMeasurementTime).total_seconds()
            receivedTime = self.receiverClock.absoluteTime.getGPSSeconds() - timeResidual
            self.nextMeasurementTime.setGPSTime(self.receiverClock.absoluteTime.getGPSWeek(), receivedTime + self.measurementPeriod)
            tow = earliestChannel.tow + earliestChannel.getTimeSinceTOW() / 1e3 - timeResidual
        
        idx = 0
        gnssMeasurementsList = []
        for chan in selectedChannels:
            satellite = self.satelliteDict[chan.svid]
            if self.isBRDCEphemerisAssited:
                satellite.addBRDCEphemeris(self.database.fetchBRDC(self.receiverClock.absoluteTime, satellite.system, satellite.svid))

            # Compute the time of transmission
            relativeTime = (earliestChannel.getTimeSinceTOW() - chan.getTimeSinceTOW()) * 1e-3
            transmitTime = tow - relativeTime

            # Compute pseudoranges
            # TODO add remaining phase
            pseudoranges = (receivedTime - transmitTime) * SPEED_OF_LIGHT

            # Compute satellite positions and clock errors
            satellitePosition, satelliteClock = satellite.computePosition(transmitTime)

            # Apply corrections
            # TODO Ionosphere, troposhere ...
            correctedPseudoranges  = pseudoranges
            correctedPseudoranges += satelliteClock * SPEED_OF_LIGHT # Satellite clock error
            correctedPseudoranges += satellite.getTGD() * SPEED_OF_LIGHT    # Total Group Delay (TODO this is frequency dependant)

            # Pseudorange
            if self.measurementsEnabled[GNSSMeasurementType.PSEUDORANGE]:
                gnssMeasurements = GNSSmeasurements()
                gnssMeasurements.channel  = chan
                gnssMeasurements.time     = Time.fromGPSTime(week, receivedTime)
                gnssMeasurements.mtype    = GNSSMeasurementType.PSEUDORANGE
                gnssMeasurements.value    = correctedPseudoranges
                gnssMeasurements.rawValue = pseudoranges
                gnssMeasurements.residual = 0.0
                gnssMeasurements.enabled  = True
                gnssMeasurementsList.append(gnssMeasurements)

            # Doppler
            if self.measurementsEnabled[GNSSMeasurementType.DOPPLER]:
                gnssMeasurements = GNSSmeasurements()
                gnssMeasurements.channel  = chan
                gnssMeasurements.time     = Time.fromGPSTime(week, receivedTime)
                gnssMeasurements.mtype    = GNSSMeasurementType.DOPPLER
                gnssMeasurements.value    = chan.tracking.carrierFrequency
                gnssMeasurements.rawValue = 0.0
                gnssMeasurements.residual = 0.0
                gnssMeasurements.enabled  = False
                gnssMeasurementsList.append(gnssMeasurements)
            
            idx += 1
        
        # Compute position and measurements
        self.computeReceiverPosition(receivedTime, gnssMeasurementsList)

        # Update the receiver position
        state = self.navigation.x
        statePrecision = self.navigation.getStatePrecision()
        self.receiverPosition.time = Time.fromGPSTime(week, receivedTime)
        self.receiverPosition.coordinate.setCoordinates(state[0], state[1], state[2])
        self.receiverPosition.coordinate.setPrecision(statePrecision[0], statePrecision[1], statePrecision[2])
        self.receiverPosition.clockError = self.navigation.x[3]
        self.receiverPosition.id += 1

        # Update receiver clock
        self.receiverClock.absoluteTime.applyCorrection(-self.receiverPosition.clockError / SPEED_OF_LIGHT)

        self.addPositionDatabase(self.receiverPosition, gnssMeasurementsList)
        self.measurementTimeList.append(receivedTime)
        
        coord = self.receiverPosition.coordinate
        logging.getLogger(__name__).info(f"New measurements computed (Receiver time: {receivedTime:.3f})")
        logging.getLogger(__name__).debug(f"Position=({coord.x:12.4f} {coord.y:12.4f} {coord.z:12.4f}), precision=({coord.xPrecison:8.4f} {coord.yPrecison:8.4f} {coord.zPrecison:8.4f})")
        logging.getLogger(__name__).debug(f"Clock error={self.receiverPosition.clockError: 12.4f}")
        logging.getLogger(__name__).debug(f"Receiver clock : ({self.receiverClock.absoluteTime.gpsTime.week_number} {self.receiverClock.absoluteTime.gpsTime.seconds} {self.receiverClock.absoluteTime.gpsTime.femtoseconds})")

        return

    # -------------------------------------------------------------------------

    def computeReceiverPosition(self, time, measurements):
        
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

            self.navigation.compute()

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
            
            logging.getLogger(__name__).debug(f"CID {meas.channel.cid} {meas.mtype:11} {meas.value:13.4f} (residual: {meas.residual: .4f}, enabled: {meas.enabled})")

            idx += 1

        return
    
    # -------------------------------------------------------------------------

    def updateDatabase(self):
        """
        TODO
        """

        # Update channel

        # Commit
        self.database.commit()

        return 

    # -------------------------------------------------------------------------

    def addDSPToDatabase(self, channelID:int, results:dict):

        # Add the mandatory values
        results["channel_id"]   = channelID
        results["time"]         = time.time()
        results["time_sample"]  = float(self.sampleCounter - results["unprocessedSamples"])

        self.database.addData("acquisition", results)

        return

    # -------------------------------------------------------------------------

    def addChannelDatabase(self, channel:ChannelAbstract):
        
        mdict = {
            "id"           : self.channelCounter,
            "physical_id"  : channel.cid, 
            "system"       : channel.gnssSignal.getSystem(),
            "satellite_id" : channel.svid,
            "signal"       : channel.gnssSignal.signalType,
            "start_time"   : time.time(),
            "start_sample" : self.sampleCounter
        }

        self.database.addData("channel", mdict)

        return

    # -------------------------------------------------------------------------

    def addAcquisitionDatabase(self, channelID:int, results:dict):

        # Add the mandatory values
        results["channel_id"]   = channelID
        results["time"]         = time.time()
        results["time_sample"]  = float(self.sampleCounter - results["unprocessed_samples"])

        self.database.addData("acquisition", results)
        return

    # -------------------------------------------------------------------------

    def addTrackingDatabase(self, channelID:int, results:dict):

        # Add the mandatory values
        results["channel_id"]   = channelID
        results["time"]         = time.time()
        results["time_sample"]  = float(self.sampleCounter - results["unprocessedSamples"])

        self.database.addData("tracking", results)

        return

    # -------------------------------------------------------------------------

    def addDecodingDatabase(self, channelID:int, results:dict):

        # Add the mandatory values
        results["channel_id"]   = channelID
        results["time"]         = time.time()
        results["time_sample"]  = float(self.sampleCounter - results["unprocessedSamples"])

        self.database.addData("decoding", results)

        return

    # -------------------------------------------------------------------------

    def addPositionDatabase(self, position:GNSSPosition, measurements):
        
        posdict = {}

        # Position
        posdict["id"]            = position.id
        posdict["time"]          = time.time()
        posdict["time_sample"]   = self.sampleCounter
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
            measdict = {}
            measdict["channel_id"]   = meas.channel.dbid
            measdict["time"]         = time.time()
            measdict["time_sample"]  = self.sampleCounter
            measdict["position_id"]  = position.id
            measdict["enabled"]      = meas.enabled
            measdict["type"]         = meas.mtype
            measdict["value"]        = meas.value
            measdict["raw_value"]    = meas.rawValue
            measdict["residual"]     = meas.residual

            self.database.addData("measurement", measdict)

        return

    # -------------------------------------------------------------------------

    def saveSatellites(self, outfile):
        with open(outfile , 'wb') as f:
            pickle.dump(self.satelliteDict, f, pickle.HIGHEST_PROTOCOL)
        return

    # -------------------------------------------------------------------------



