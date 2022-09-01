# -*- coding: utf-8 -*-
# ============================================================================
# Abstract class for tracking process.
# Author: Antoine GRENIER (TAU)
# Date: 2022.05.04
# References: 
# =============================================================================
# PACKAGES
import math
import numpy as np
import configparser
import pickle
from gnsstools.channel.abstract import ChannelState
from gnsstools.channel.channel_default import Channel
from gnsstools.gnsssignal import GNSSSignal, SignalType
from gnsstools.measurements import DSPEpochs
from gnsstools.message.abstract import NavMessageType
from gnsstools.rfsignal import RFSignal
from gnsstools.satellite import Satellite
from gnsstools.constants import AVG_TRAVEL_TIME_MS, EARTH_ROTATION_RATE, SPEED_OF_LIGHT
# =============================================================================
class Receiver():

    def __init__(self, receiverConfigFile, gnssSignal:GNSSSignal, rfSignal:RFSignal):

        self.gnssSignal = gnssSignal
        
        config = configparser.ConfigParser()
        config.read(receiverConfigFile)

        self.name        = config.get   ('DEFAULT', 'name')
        self.nbChannels  = config.getint('DEFAULT', 'nb_channels')
        self.msToProcess = config.getint('DEFAULT', 'ms_to_process')

        self.channels = []
        self.channelsStates = [ChannelState.IDLE] *  self.nbChannels

        self.isClockInitialised = False

        self.receiverClock =  0.0
        self.receiverClockError = []
        self.measurementTimeList = []
        self.receiverPosition = []

        self.measurementFrequency = 2  # In Hz
        self.measurementPeriod = 1 / self.measurementFrequency

        self.rfSignal = rfSignal

        return

    # -------------------------------------------------------------------------
    
    def run(self, satelliteList):

        # TEMPORARY solution TODO change
        signalDict = {}
        signalDict[SignalType.GPS_L1_CA] = self.gnssSignal
        self.satelliteList = satelliteList

        # Initialise the channels
        for idx in range(min(self.nbChannels, len(satelliteList))):
            self.channels.append(Channel(idx, self.gnssSignal, self.rfSignal, 0))
        
        # Initialise satellite structure
        self.satelliteDict = {}
        for svid in satelliteList:
            self.satelliteDict[svid] = Satellite(svid, signalDict)

        # Loop through the file contents
        # TODO Load by chunck of data instead of ms per ms
        msPerLoop = 1
        samplesInMs = int(msPerLoop * self.rfSignal.samplingFrequency * 1e-3)
        rfData = np.empty(samplesInMs)
        self.sampleCounter = 0
        for msProcessed in range(self.msToProcess):
            rfData = self.rfSignal.readFile(timeLength=msPerLoop, keep_open=True)
            self.sampleCounter += samplesInMs
            self.receiverClock += msPerLoop * 1e-3

            if rfData.size < samplesInMs:
                raise EOFError("EOF encountered earlier than expected in file.")

            # Run the channels
            self.runChannels(rfData)

            # Handle channels results
            self.processChannels(msProcessed, self.sampleCounter)

            # Compute measurements based on receiver time
            # For measurement frequency of 1 Hz, that's every round second.
            if not self.isClockInitialised:
                # First time we run it to estimate receiver clock error
                self.computeGNSSMeasurements(receivedTime = 0)
            elif self.receiverClock >= self.nextMeasurementTime:
                self.computeGNSSMeasurements(receivedTime=self.receiverClock)

            msProcessed += 1
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
                chan.setSatellite(svid)
                print(f"Channel {chan.cid} started with satellite G{svid}.")            
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
                # Buffer not full (most probably)
                if chan.isAcquired:
                    satellite.addDSPMeasurement(msProcessed, sampleCounter, chan)
                    chan.switchState(ChannelState.TRACKING)
                    print(f"Channel {chan.cid} found satellite G{svid}, tracking started.")    
                else:
                    # Buffer not full (most probably)
                    pass
            elif state == ChannelState.TRACKING:
                # Signal is being tracked
                satellite.addDSPMeasurement(msProcessed, sampleCounter, chan)

                # Add navigation message
                # TODO This pass a reference, maybe be it should be done only at init? 
                satellite.addNavMessage(chan.decoding)

                # DEBUG (TO BE REMOVED)
                if not chan.isEphemerisDecoded:
                    f = open(f"ephemeris_{chan.svid}.pkl", 'rb')
                    chan.decoding.ephemeris = pickle.load(f)
                    chan.decoding.isEphemerisDecoded = True
                    chan.isEphemerisDecoded = True

                # Check if ephemeris decoded
                if chan.isEphemerisDecoded:
                    # Check if already saved
                    if chan.decoding.ephemeris != satellite.getLastBRDCEphemeris():
                        satellite.addBRDCEphemeris(chan.decoding.ephemeris)

                        # # DEBUG (TO BE REMOVED)
                        # f = open(f"ephemeris_{chan.svid}.pkl", 'wb')
                        # pickle.dump(chan.decoding.ephemeris, f, pickle.HIGHEST_PROTOCOL)
                 
            else:
                raise ValueError(f"State {state} in channel {chan.cid} is not a valid state.")
            
            self.channelsStates[chan.cid] = state
        return

    
    # -------------------------------------------------------------------------

    def computeGNSSMeasurements(self, receivedTime=0):

        samplingFrequency = self.rfSignal.samplingFrequency

        # Check if channels ready for measurements
        prnList = []
        selectedChannels = []
        for chan in self.channels:
            if chan.isTOWDecoded and chan.isEphemerisDecoded:
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
        if not self.isClockInitialised:
            receivedTime = earliestChannel.tow + AVG_TRAVEL_TIME_MS / 1e3
            self.receiverClock = receivedTime
            self.isClockInitialised = True
            self.nextMeasurementTime = math.ceil(receivedTime)
            tow = earliestChannel.tow
        else:
            # Compute the residual time to have a "round" received time
            timeResidual = receivedTime - self.nextMeasurementTime
            receivedTime = self.receiverClock - timeResidual
            self.nextMeasurementTime = receivedTime + self.measurementPeriod
            tow = earliestChannel.tow + earliestChannel.getTimeSinceTOW() / 1e3 - timeResidual
        
        satellitesPositions = np.zeros((len(prnList), 3))
        satellitesClocks = np.zeros(len(prnList))
        pseudoranges = np.zeros(len(prnList))
        correctedPseudoranges = np.zeros(len(prnList))
        idx = 0
        for chan in selectedChannels:
            satellite = self.satelliteDict[chan.svid]

            # Compute the time of transmission
            relativeTime = (earliestChannel.getTimeSinceTOW() - chan.getTimeSinceTOW()) * 1e-3
            transmitTime = tow - relativeTime

            # Compute pseudoranges
            # TODO add remaining phase
            pseudoranges[idx] = (receivedTime - transmitTime) * SPEED_OF_LIGHT

            # Compute satellite positions and clock errors
            satellitesPositions[idx, :], satellitesClocks[idx] = satellite.computePosition(transmitTime)

            # Apply corrections
            # TODO Ionosphere, troposhere ...
            correctedPseudoranges[idx] = pseudoranges[idx]
            correctedPseudoranges[idx] += satellitesClocks[idx] * SPEED_OF_LIGHT # Satellite clock error
            correctedPseudoranges[idx] -= satellite.getTGD() * SPEED_OF_LIGHT    # Total Group Delay (TODO this is frequency dependant)
            
            idx += 1
        
        self.computeReceiverPosition(correctedPseudoranges, satellitesPositions)
        correctedPseudoranges -= self.receiverClockError[-1]
        self.receiverClock -= self.receiverClockError[-1] / SPEED_OF_LIGHT

        self.measurementTimeList.append(receivedTime)

        # print("---")
        # print(self.receiverPosition)
        # print(self.receiverClockError)
        # print(self.receiverClock)


        return

    # -------------------------------------------------------------------------
        
    def getStartSample(self, gnssSignal:GNSSSignal):
        
        startSampleList = []
        for prn, satellite in self.satelliteDict.items():
            startSampleList.append(satellite.navMessage[gnssSignal].getSampleSubframe())
        
        # Find the latest signal
        latestSample = np.max(startSampleList)

        # Find in the other signals what is the closest sample compare to the lastest one
        startSampleIdx = {}
        for prn, satellite in self.satelliteDict.items():
            diff = satellite.tracking.absoluteSample - latestSample
            idx = np.argsort(np.abs(diff))
            startSampleIdx[prn] = np.min(idx[:2])
        
        return startSampleIdx, latestSample

    # -------------------------------------------------------------------------

    def computeReceiverPosition(self, pseudoranges, satpos):
        nbMeasurements = len(pseudoranges)
        G = np.zeros((nbMeasurements, 4))
        y = np.zeros(nbMeasurements)
        dX = np.zeros(4)
        dX[:4] = [1.0, 1.0, 1.0, 1.0]
        #x = np.zeros(4)
        #x = np.array([2794767.59, 1236088.19, 5579632.92, 0])
        x = np.array([2793000.0, 1235000.0, 5578000.0, 0])
        v = []
        for i in range(10):
            if np.linalg.norm(dX) < 1e-6:
                break
            # Make matrices
            for idx in range(nbMeasurements):
                if idx == 0:
                    _satpos = satpos[idx, :]
                else:
                    p = np.sqrt((x[0] - satpos[idx, 0])**2 + (x[1] - satpos[idx, 1])**2 + (x[2] - satpos[idx, 2])**2)
                    travelTime = p / SPEED_OF_LIGHT
                    _satpos = self.correctEarthRotation(travelTime, np.transpose(satpos[idx, :]))
                
                p = np.sqrt((x[0] - _satpos[0])**2 + (x[1] - _satpos[1])**2 + (x[2] - _satpos[2])**2)

                G[idx, 0] = (x[0] - satpos[idx, 0]) / p
                G[idx, 1] = (x[1] - satpos[idx, 1]) / p
                G[idx, 2] = (x[2] - satpos[idx, 2]) / p
                G[idx, 3] = 1

                y[idx] = pseudoranges[idx] - p - x[3]
            
            # Least Squares
            N = np.transpose(G).dot(G)
            C = np.transpose(G).dot(y)
            dX = np.linalg.inv(N).dot(C)
            x = x + dX # Update solution
            v = G.dot(dX) - y
            #print(v)
            #print(dX)
        
        self.receiverPosition.append(x[0:3])
        self.receiverClockError.append(x[3])

        return

    # -------------------------------------------------------------------------

    @staticmethod
    def correctEarthRotation(traveltime, X_sat):
        """
        E_R_CORR  Returns rotated satellite ECEF coordinates due to Earth
        rotation during signal travel time

        X_sat_rot = e_r_corr(traveltime, X_sat);

          Inputs:
              travelTime  - signal travel time
              X_sat       - satellite's ECEF coordinates

          Outputs:
              X_sat_rot   - rotated satellite's coordinates (ECEF)

        Written by Kai Borre
        Copyright (c) by Kai Borre
        """

        # --- Find rotation angle --------------------------------------------------
        omegatau = EARTH_ROTATION_RATE * traveltime

        # --- Make a rotation matrix -----------------------------------------------
        R3 = np.array([[np.cos(omegatau), np.sin(omegatau), 0.0],
                       [-np.sin(omegatau), np.cos(omegatau), 0.0],
                       [0.0, 0.0, 1.0]])

        # --- Do the rotation ------------------------------------------------------
        X_sat_rot = R3.dot(X_sat)
        
        return X_sat_rot

    # -------------------------------------------------------------------------

    def saveSatellites(self, outfile):
        with open(outfile , 'wb') as f:
            pickle.dump(self.satelliteDict, f, pickle.HIGHEST_PROTOCOL)
        return

    def loadSatellites(self, infile):
        with open(infile, 'rb') as f:
                self.satelliteDict = pickle.load(f)
        return 

    def saveChannels(self, outfile):
        with open(outfile , 'wb') as f:
            pickle.dump(self.channels, f, pickle.HIGHEST_PROTOCOL)
        return

    def loadChannels(self, infile):
        with open(infile, 'rb') as f:
                self.channels = pickle.load(f)
        return 



    # END OF CLASS


# -------------------------------------------------------------------------

    def computeGNSSMeasurements_(self, sampleCounter):

        # Check if satellite ready for measurement
        prnList = []
        for prn, satellite in self.satelliteDict.items():
            if not satellite.isSatelliteReady():
                continue
            prnList.append(prn)
        
        if len(prnList) < 5:
            # Not enough satellites
            return
        
        # sampleCounter = 292310000

        # Retrieve transmitted time
        samplingFrequency = self.rfSignal.samplingFrequency
        samplesPerCode = self.gnssSignal.getSamplesPerCode(samplingFrequency)
        transmittedTime = np.zeros(len(prnList))
        satellitesPositions = np.zeros((len(prnList), 3))
        satellitesClocks = np.zeros(len(prnList))
        idx = 0
        for prn in prnList:
            satellite = self.satelliteDict[prn]
            navMessage = satellite.navMessage[self.gnssSignal.signalType]
            lastDSPMeasurement = satellite.dspEpochs[self.gnssSignal.signalType].getLastMeasurement()

            # Get the last TOW
            tow = navMessage.getTow()

            # Get sample index of subframe
            sampleSubframe = navMessage.getSampleTOW()

            print(sampleSubframe)

            # Find difference between last TOW and last code tracked
            diffTowCode = lastDSPMeasurement.sample - sampleSubframe

            # Find difference between current sample and last code tracked
            diffCurrentCode = sampleCounter - lastDSPMeasurement.sample

            # Find the code phase
            # codePhase = diffCurrentCode / samplesPerCode

            # Build transmitted time 
            transmittedTime[idx] = tow + (diffTowCode + diffCurrentCode) / samplingFrequency

            satellitesPositions[idx, :], satellitesClocks[idx] = satellite.computePosition(transmittedTime[idx])

            idx += 1 
        
        # Estimated received time
        if not self.isClockInitialised:
            # The largest value is the one that was received first
            # We add an average travel time to obtain a more realistic initial value
            self.receiverClock = np.min(transmittedTime) + (AVG_TRAVEL_TIME_MS/1e3)
        else:
            self.receiverClock += 1 / self.measurementFrequency
        
        # Compute pseudoranges
        pseudoranges = (self.receiverClock - transmittedTime) * SPEED_OF_LIGHT

        # Correct pseudoranges
        idx = 0
        for prn in prnList:
            satellite = self.satelliteDict[prn]
            pseudoranges += (satellitesClocks[idx] + satellite.getTGD()) * SPEED_OF_LIGHT
            idx += 1

        # Compute receiver position
        self.computeReceiverPosition(pseudoranges, satellitesPositions)

        return
