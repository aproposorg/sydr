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

from core.channel.channel_abstract import ChannelAbstract, ChannelState, ChannelMessage, ChannelStatus
from core.channel.channel_l1ca import ChannelL1CA, ChannelStatusL1CA
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
                         u'Z: {z:12.4f} (\u03C3: {sz: .4f}) ' + \
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
        self.channelsStatus = {}
        self.channelCounter = 0

        self.isClockInitialised = False
        self.sampleCounter = 0
        self.receiverClock = Clock()
        if self.isClockAssisted:
            self.receiverClock.absoluteTime.setDatetime(self.clockAssistedValue)
        self.receiverClockError = []
        self.measurementTimeList = []
        self.receiverPosition = GNSSPosition()

        self.measurementFrequency = 1  # In Hz
        self.measurementPeriod = 1 / self.measurementFrequency
        self.nextMeasurementTime = Time()

        self.rfSignal = rfSignal

        self.receiverState = ReceiverState.IDLE

        self.deactivatedSatellites = []

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

        logging.getLogger(__name__).info(f"Processing in receiver {self.name} ended.")
        
        return

    # -------------------------------------------------------------------------

    def _initChannels(self, manager:Manager):

        # Initialise the channels

        channelManager = 


        self.channelsPB = []
        for idx in range(min(self.nbChannels, len(self.satelliteList))):
            
            # Create object for communication between processes
            _queue = multiprocessing.Queue() # The queue is for giving data to the channels
            _pipe  = multiprocessing.Pipe()  # The pipe is to send back the results from the threads. Pipe is a tuple of two connection (in, out)

            # Create channel
            self.channels.append(ChannelL1CA(idx, self.gnssSignal, self.rfSignal, 0, _queue, _pipe))
            self.channelsStatus[idx] = ChannelStatusL1CA().fromChannel(self.channels[-1])
            self.channels[-1].hasPreviousMeasurement = False

            # Create GUI
            _tow = colored(" TOW ", 'white', 'on_red')
            _sf1 = colored("1", 'white', 'on_red')
            _sf2 = colored("2", 'white', 'on_red')
            _sf3 = colored("3", 'white', 'on_red')
            _sf4 = colored("4", 'white', 'on_red')
            _sf5 = colored("5", 'white', 'on_red')
            counter = manager.counter(total=self.msToProcess, desc=f"    Channel {idx}", \
                leave=False, unit='ms', color='lightseagreen', min_delta=0.5,
                state=f"{self.channelsStatus[idx].state}", bar_format=CHANNEL_BAR_FORMAT, prn='G00', \
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
                chan.setSatellite(svid)
                self.channelsStatus[chan.cid].svid = svid

                # State the channel
                chan.start()

                self.channelCounter += 1
                self.addChannelDatabase(chan)
                self.updateDatabase() 
            
            # Run channels
            chan.rfQueue.put(rfData)

            self.updateChannelGUI(self.channelsStatus[chan.cid])
        return

    # -------------------------------------------------------------------------

    def _processChannels(self, msProcessed, sampleCounter):
        for chan in self.channels:
            while True:
                channelPacket = chan.pipe[1].recv()
                if channelPacket == ChannelMessage.END_OF_PIPE:
                    break
                commType = channelPacket[0]
                if commType == ChannelMessage.ACQUISITION_UPDATE:
                    logging.getLogger(__name__).info(f"CID {chan.cid} found satellite G{chan.svid}, tracking started.")
                    self.addAcquisitionDatabase(chan.cid, channelPacket[1])
                    continue
                elif commType == ChannelMessage.TRACKING_UPDATE:
                    self.addTrackingDatabase(chan.cid, channelPacket[1])
                    continue
                elif commType == ChannelMessage.DECODING_UPDATE:
                    satellite = self.satelliteDict[chan.svid]
                    satellite.addSubframe(channelPacket[1]['subframe_id'], channelPacket[1]['bits'])
                    self.channelsStatus[chan.cid].subframeFlags[channelPacket[1]['subframe_id']-1] = True
                    self.addDecodingDatabase(chan.cid, channelPacket[1])
                    continue
                elif commType == ChannelMessage.CHANNEL_UPDATE:
                    self.channelsStatus[chan.cid].state            = channelPacket[1]
                    self.channelsStatus[chan.cid].trackingFlags    = channelPacket[2]
                    self.channelsStatus[chan.cid].week             = channelPacket[3]
                    self.channelsStatus[chan.cid].tow              = channelPacket[4]
                    self.channelsStatus[chan.cid].timeSinceTOW     = channelPacket[5]
                    if not np.isnan(self.channelsStatus[chan.cid].tow):
                        self.channelsStatus[chan.cid].isTOWDecoded = True
                    #logging.getLogger(__name__).debug(f"CID {chan.cid} update : {channelPacket[1]} {channelPacket[2]} {channelPacket[3]} {channelPacket[4]} {channelPacket[5]}")  

        return
    
    # -------------------------------------------------------------------------

    def computeGNSSMeasurements(self):
        
        # Check if channels ready for measurements
        prnList = []
        selectedChannels = []
        for chan in self.channelsStatus.values():
            satellite = self.satelliteDict[chan.svid]
            if chan.isTOWDecoded \
                and (satellite.isEphemerisDecoded or self.isBRDCEphemerisAssited):
                prnList.append(chan.svid)
                selectedChannels.append(chan)
        


        # In case multi-frequency/Channel tracking, we want at least 5 unique satellites.
        # This is to avoid rank deficiency in matrices.
        prnList = set(prnList) 
        
        if len(prnList) < len(self.channels):
            # Not enough satellites
            self.receiverState = ReceiverState.INIT
            return
        self.receiverState = ReceiverState.NAVIGATION

        # Find earliest signal
        maxTOW = -1
        for chan in selectedChannels:
            # This assumes all the channels were given the same number of samples to process
            if maxTOW < chan.timeSinceTOW:
                maxTOW = chan.timeSinceTOW
                earliestChannel = chan
        
        logging.getLogger(__name__).debug(f"SVID {earliestChannel.svid}, max TOW {maxTOW}")
        
        # Received time
        if not self.receiverClock.isInitialised:
            if self.isClockAssisted:
                week = self.receiverClock.absoluteTime.getGPSWeek()
            else:
                week = earliestChannel.week
            
            tow = earliestChannel.tow + earliestChannel.timeSinceTOW / 1e3
            
            receivedTime = tow + AVG_TRAVEL_TIME_MS / 1e3
            self.receiverClock.absoluteTime.setGPSTime(week, receivedTime)
            self.receiverClock.isInitialised = True
            self.nextMeasurementTime.setGPSTime(week, math.ceil(receivedTime))

        else:
            # Compute the residual time to have a "round" received time
            week =  self.receiverClock.absoluteTime.getGPSWeek()
            timeResidual = (self.receiverClock.absoluteTime - self.nextMeasurementTime).total_seconds()
            receivedTime = self.receiverClock.absoluteTime.getGPSSeconds() - timeResidual
            self.nextMeasurementTime.setGPSTime(self.receiverClock.absoluteTime.getGPSWeek(), receivedTime + self.measurementPeriod)
            
            tow = earliestChannel.tow + earliestChannel.timeSinceTOW / 1e3 - timeResidual

            logging.getLogger(__name__).debug(f"Week {week}, time residual {timeResidual}, received time {receivedTime}, tow {tow}")
        
        idx = 0
        gnssMeasurementsList = []
        for chan in selectedChannels:
            isPseudorangeValid = True
            satellite = self.satelliteDict[chan.svid]
            if self.isBRDCEphemerisAssited:
                satellite.lastBRDCEphemeris = self.database.fetchBRDC(self.receiverClock.absoluteTime, satellite.system, satellite.svid)
                #satellite.addBRDCEphemeris(self.database.fetchBRDC(self.receiverClock.absoluteTime, satellite.system, satellite.svid))

            # Compute the time of transmission
            relativeTime = (earliestChannel.timeSinceTOW - chan.timeSinceTOW) * 1e-3
            transmitTime = tow - relativeTime

            # Compute pseudoranges
            # TODO add remaining phase
            pseudoranges = (receivedTime - transmitTime) * SPEED_OF_LIGHT

            # Compute satellite positions and clock errorsW
            satellitePosition, satelliteClock = satellite.computePosition(transmitTime)

            # Apply corrections
            # TODO Ionosphere, troposhere ...
            correctedPseudoranges  = pseudoranges
            correctedPseudoranges += satelliteClock * SPEED_OF_LIGHT # Satellite clock error
            correctedPseudoranges += satellite.getTGD() * SPEED_OF_LIGHT  # Total Group Delay (TODO this is frequency dependant)

            logging.getLogger(__name__).debug(f"SVID {chan.svid}, timeSinceLastTOW {chan.timeSinceTOW}, relativeTime {relativeTime}, transmitTime {transmitTime}, pseudoranges {pseudoranges}, correctedPseudoranges {correctedPseudoranges}")
            #logging.getLogger(__name__).debug(f"SVID {chan.svid}, IODE {satellite.lastBRDCEphemeris.iode}, satellitePosition {satellitePosition}, satelliteClock {satelliteClock}")

            # Check if measurement looks correct
            # if not chan.hasPreviousMeasurement:
            #     chan.prevRelativeTime = relativeTime
            #     chan.hasPreviousMeasurement = True
            # else:
            #     chan.prevRelativeTime = chan.relativeTime
            # chan.relativeTime = relativeTime

            # if np.abs(chan.relativeTime - chan.prevRelativeTime) > 10e-5:
            #     logging.getLogger(__name__).warning(f"CID {chan.cid} SVID {chan.svid} Large difference ({np.abs(chan.relativeTime - chan.prevRelativeTime)}) in TOW compare to previous epoch, measurement discarded.")
            #     isPseudorangeValid = False
            #     self.deactivatedSatellites.append(chan.svid)

            # if chan.svid in self.deactivatedSatellites:
            #     isPseudorangeValid = False
            
            # Pseudorange
            if self.measurementsEnabled[GNSSMeasurementType.PSEUDORANGE]:
                gnssMeasurements = GNSSmeasurements()
                gnssMeasurements.channel  = chan
                gnssMeasurements.time     = Time.fromGPSTime(week, receivedTime)
                gnssMeasurements.mtype    = GNSSMeasurementType.PSEUDORANGE
                gnssMeasurements.value    = correctedPseudoranges
                gnssMeasurements.rawValue = pseudoranges
                gnssMeasurements.residual = 0.0
                gnssMeasurements.enabled  = isPseudorangeValid
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
        self.computeReceiverPosition(week, receivedTime, gnssMeasurementsList)
        
        coord = self.receiverPosition.coordinate
        logging.getLogger(__name__).info(f"New measurements computed (Receiver time: {receivedTime:.3f})")
        logging.getLogger(__name__).debug(f"Position=({coord.x:12.4f} {coord.y:12.4f} {coord.z:12.4f}), precision=({coord.xPrecison:8.4f} {coord.yPrecison:8.4f} {coord.zPrecison:8.4f})")
        logging.getLogger(__name__).debug(f"Clock error={self.receiverPosition.clockError: 12.4f}")
        logging.getLogger(__name__).debug(f"Receiver clock : ({self.receiverClock.absoluteTime.gpsTime.week_number} {self.receiverClock.absoluteTime.gpsTime.seconds} {self.receiverClock.absoluteTime.gpsTime.femtoseconds})")
        logging.getLogger(__name__).debug(f"-------------------------------")
        
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
            measdict["channel_id"]   = meas.channel.cid
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

    # =========================================================================
    # GUI

    def updateChannelGUI(self, chan:ChannelStatus):
        _tow = colored(f" TOW: {chan.tow:6.0f}", 'white', 'on_green' if chan.isTOWDecoded else 'on_red')
        _sf1 = colored("1", 'white', 'on_green' if chan.subframeFlags[0] else 'on_red')
        _sf2 = colored("2", 'white', 'on_green' if chan.subframeFlags[1] else 'on_red')
        _sf3 = colored("3", 'white', 'on_green' if chan.subframeFlags[2] else 'on_red')
        _sf4 = colored("4", 'white', 'on_green' if chan.subframeFlags[3] else 'on_red')
        _sf5 = colored("5", 'white', 'on_green' if chan.subframeFlags[4] else 'on_red')
        self.channelsPB[chan.cid].update(state=f"{chan.state}", prn=f'G{chan.svid:02d}', tow=_tow,\
            sf1=_sf1, sf2=_sf2, sf3=_sf3, sf4=_sf4, sf5=_sf5)
        

        return




