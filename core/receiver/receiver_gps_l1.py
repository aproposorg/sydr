# -*- coding: utf-8 -*-
# ============================================================================
# Abstract class for tracking process.
# Author: Antoine GRENIER (TAU)
# Date: 2022.05.04
# References: 
# =============================================================================
# PACKAGES
import copy
import logging
import pickle
from datetime import datetime
import math
import time
import numpy as np
import configparser

from core.channel.channel_abstract import ChannelAbstract, ChannelState
from core.channel.channel_l1ca import ChannelL1CA
from core.measurements import GNSSPosition, GNSSmeasurements
from core.navigation.lse import LeastSquareEstimation
from core.receiver.receiver_abstract import ReceiverAbstract
from core.record.database import DatabaseHandler
from core.signal.gnsssignal import GNSSSignal
from core.utils.clock import Clock
from core.utils.coordinate import Coordinate
from core.utils.enumerations import GNSSMeasurementType, GNSSSignalType, GNSSSystems
from core.signal.rfsignal import RFSignal
from core.satellite.satellite import Satellite
from core.utils.constants import AVG_TRAVEL_TIME_MS, EARTH_ROTATION_RATE, SPEED_OF_LIGHT
from core.utils.time import Time

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

        # MEASUREMENTS
        # GNSS measurements used and produced
        self.measurementsEnabled = {}
        self.measurementsEnabled[GNSSMeasurementType.PSEUDORANGE] = config.getboolean('MEASUREMENTS', 'pseudorange')
        self.measurementsEnabled[GNSSMeasurementType.DOPPLER]     = config.getboolean('MEASUREMENTS', 'doppler')

        # Channel parameters
        self.channels = []
        self.channelsStates = [ChannelState.IDLE] *  self.nbChannels
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

        return

    # -------------------------------------------------------------------------
    
    def run(self, satelliteList):
        """
        Start the processing.

        Args:
            satelliteList (list): List of satellite to look for

        """
        self.satelliteList = satelliteList

        # Initialise the channels
        for idx in range(min(self.nbChannels, len(satelliteList))):
            self.channels.append(ChannelL1CA(idx, self.gnssSignal, self.rfSignal, 0))
        
        # Initialise satellite structure
        self.satelliteDict = {}
        for svid in satelliteList:
            self.satelliteDict[svid] = Satellite(GNSSSystems.GPS, svid)

        # Loop through the file contents
        # TODO Load by chunck of data instead of ms per ms
        msPerLoop = 1
        samplesInMs = int(msPerLoop * self.rfSignal.samplingFrequency * 1e-3)
        rfData = np.empty(samplesInMs)
        for msProcessed in range(self.msToProcess):
            rfData = self.rfSignal.readFile(timeLength=msPerLoop, keep_open=True)
            self.sampleCounter += samplesInMs
            self.receiverClock.addTime(msPerLoop * 1e-3)

            if rfData.size < samplesInMs:
                raise EOFError("EOF encountered earlier than expected in file.")

            # Run the channels
            self.runChannels(rfData)

            # Handle channels results
            self.processChannels(msProcessed, self.sampleCounter)

            # Compute measurements based on receiver time
            # For measurement frequency of 1 Hz, that's every round second.
            if not self.receiverClock.isInitialised:
                # First time we run it to estimate receiver clock error
                self.computeGNSSMeasurements()
            elif self.receiverClock.absoluteTime >= self.nextMeasurementTime:
                self.computeGNSSMeasurements()

            # Update the database
            if int(self.sampleCounter % 1e7) == 0:
                self.updateDatabase()

            msProcessed += 1

        # Update the last measurements
        self.updateDatabase()
        
        return

    # -------------------------------------------------------------------------

    def runChannels(self, rfData):
        for chan in self.channels:
            if chan.getState() == ChannelState.OFF:
                continue
            elif chan.getState() == ChannelState.IDLE:
                # Give a new satellite from list
                if not self.satelliteList:
                    # List empty, disable channel
                    chan.switchState(ChannelState.OFF)
                    continue
                svid = self.satelliteList.pop(0)
                # Set the satellite parameters in the channel
                chan.setSatellite(svid, self.channelCounter)

                self.addChannelDatabase(chan)
                logging.getLogger(__name__).info(f"Channel {chan.cid} started with satellite G{svid}.")

                self.channelCounter += 1

                self.updateDatabase()     
                
            chan.run(rfData)
        
        return

    # -------------------------------------------------------------------------

    def processChannels(self, msProcessed, sampleCounter):
        for chan in self.channels:
            svid    = chan.svid
            state   = chan.getState()
            satellite = self.satelliteDict[svid]
            if state == ChannelState.OFF:
                continue
            elif state == ChannelState.IDLE:
                # Signal was not aquired
                satellite.addDSPMeasurement(msProcessed, sampleCounter, chan)
                print(f"Channel {chan.cid} could not acquire satellite G{svid}.")    
                pass
            elif state == ChannelState.ACQUIRING:
                if chan.isAcquired:
                    self.addAcquisitionDatabase(chan)
                    satellite.addDSPMeasurement(msProcessed, sampleCounter, chan)
                    chan.switchState(ChannelState.TRACKING)
                    logging.getLogger(__name__).info(f"Channel {chan.cid} found satellite G{svid}, tracking started.")
                    #print(f"Channel {chan.cid} found satellite G{svid}, tracking started.")    
                else:
                    # Buffer not full (most probably)
                    pass
            elif state == ChannelState.TRACKING:
                self.addTrackingDatabase(chan)

                # Signal is being tracked
                satellite.addDSPMeasurement(msProcessed, sampleCounter, chan)

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
                raise ValueError(f"State {state} in channel {chan.cid} is not a valid state.")
            
            self.channelsStates[chan.cid] = state

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
            return

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
        self.receiverPosition.time = Time.fromGPSTime(week, receivedTime)
        self.receiverPosition.coordinate.setCoordinates(self.navigation.x[0], self.navigation.x[1], self.navigation.x[2])
        self.receiverPosition.clockError = self.navigation.x[3]
        self.receiverPosition.id += 1

        # Update receiver clock
        self.receiverClock.absoluteTime.applyCorrection(-self.receiverPosition.clockError / SPEED_OF_LIGHT)

        self.addPositionDatabase(self.receiverPosition, gnssMeasurementsList)
        self.measurementTimeList.append(receivedTime)
        
        logging.getLogger(__name__).info(f"New measurements computed (Receiver time: {receivedTime:.3f})")
        logging.getLogger(__name__).debug(f"Position       : ({self.receiverPosition.coordinate.x:12.4f} {self.receiverPosition.coordinate.y:12.4f} {self.receiverPosition.coordinate.z:12.4f})")
        logging.getLogger(__name__).debug(f"Clock error    : {self.receiverPosition.clockError:12.4f}")
        logging.getLogger(__name__).debug(f"Receiver clock : ({self.receiverClock.absoluteTime.gpsTime.week_number} {receivedTime} {self.receiverClock.absoluteTime.gpsTime.femtoseconds})")

        return

    # -------------------------------------------------------------------------

    def computeReceiverPosition(self, time, measurements):
        
        nbMeasurements = len(measurements)
        G = np.zeros((nbMeasurements, 4))
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
                else:
                    # TODO Adapt to other measurement types
                    continue

                idx += 1 
            
            # Least Squares
            self.navigation.setDesignMatrix(G)
            self.navigation.setObservationVector(y)
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

    def addAcquisitionDatabase(self, channel:ChannelAbstract):
        mdict = channel.acquisition.getDatabaseDict()

        # Add the mandatory values
        mdict["channel_id"]   = channel.dbid
        mdict["time"]         = time.time()
        mdict["time_sample"]  = float(self.sampleCounter - channel.unprocessedSamples)

        self.database.addData("acquisition", mdict)
        return

    # -------------------------------------------------------------------------

    def addTrackingDatabase(self, channel:ChannelAbstract):
        
        mdict = channel.tracking.getDatabaseDict()

        # Add the mandatory values
        mdict["channel_id"]   = channel.dbid
        mdict["time"]         = time.time()
        mdict["time_sample"]  = float(self.sampleCounter - channel.unprocessedSamples)

        self.database.addData("tracking", mdict)

        return

    # -------------------------------------------------------------------------

    def addDecodingDatabase(self, channel:ChannelAbstract):
        
        mdict = channel.decoding.getDatabaseDict()

        # Add the mandatory values
        mdict["channel_id"]   = channel.dbid
        mdict["time"]         = time.time()
        mdict["time_sample"]  = float(self.sampleCounter - channel.unprocessedSamples)

        self.database.addData("decoding", mdict)

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



