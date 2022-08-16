# -*- coding: utf-8 -*-
# ============================================================================
# Abstract class for tracking process.
# Author: Antoine GRENIER (TAU)
# Date: 2022.05.04
# References: 
# =============================================================================
# PACKAGES
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
        self.receiverClockError = 0.0
        self.lastMeasurementTime = 0.0
        self.receiverPosition = []

        self.measurementFrequency = 1  # In Hz

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
            self.channels.append(Channel(idx, self.gnssSignal, self.rfSignal))

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
            self.receiverClock += msPerLoop

            if rfData.size < samplesInMs:
                raise EOFError("EOF encountered earlier than expected in file.")

            # Run the channels
            self.runChannels(rfData)

            # Handle channels results
            self.processChannels(msProcessed, self.sampleCounter)

            # if (self.sampleCounter == 292310000):
            #     return

            # # Compute measurements based on receiver time
            # # For measurement frequency of 1 Hz, that's every round second.
            # if not self.isClockInitialised:
            #     # First time we run it to estimate receiver clock error
            #     self.computeGNSSMeasurements()
            # elif int(self.receiverClock % int(1/self.measurementFrequency*1e3)) == 0:
            #     self.computeGNSSMeasurements()

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

                # if satellite.isTOWDecoded and not self.isClockInitialised:
                #     # Receiver clock initialised on the first TOW decoded
                #     self.isClockInitialised = True
                #     self.receiverClock = satellite.tow
                #     self.timeSinceTow = 
            else:
                raise ValueError(f"State {state} in channel {chan.cid} is not a valid state.")
            
            self.channelsStates[chan.cid] = state
        return

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

    # -------------------------------------------------------------------------

    def computeGNSSMeasurements(self):

        samplingFrequency = self.rfSignal.samplingFrequency

        # Check if satellite ready for measurement
        prnList = []
        for prn, satellite in self.satelliteDict.items():
            if not satellite.isSatelliteReady():
                continue
            prnList.append(prn)
        
        if len(prnList) < 5:
            # Not enough satellites
            return

        # Find earliest signal
        latestTowSample = 0
        earliestTowSample = np.inf
        for prn in prnList:
            navMessage = self.satelliteDict[prn].navMessage[self.gnssSignal.signalType]
            if earliestTowSample > navMessage.getSampleSubframe():
                earliestTowSample = navMessage.getSampleSubframe()
                tow = self.satelliteDict[prn].tow
        
        if not self.isClockInitialised:
            self.receiverClock = tow
            self.isClockInitialised = True
        
        satellitesPositions = np.zeros((len(prnList), 3))
        satellitesClocks = np.zeros(len(prnList))
        pseudoranges = np.zeros(len(prnList))
        correctedPseudoranges = np.zeros(len(prnList))
        idx = 0
        for prn in prnList:
            satellite = self.satelliteDict[prn]
            dspEpochs = dspEpochs[self.gnssSignal.signalType]
            navMessage = satellite.navMessage[self.gnssSignal.signalType]

            # # Rewind to be align with the earliest TOW sample
            # idxCode = navMessage.getCodeSubframe()
            # sample = dspEpochs.dspMeasurements[idxCode].sample 
            # while sample < earliestTowSample:
            #     idxCode -= 1
            #     sample = dspEpochs.dspMeasurements[idxCode].sample 

            # Compute the relative difference between satellites
            relativeTime = (navMessage.getSampleSubframe() - earliestTowSample) / samplingFrequency
            transmitTime = tow - (AVG_TRAVEL_TIME_MS * 1e3 + relativeTime)

            pseudoranges[idx] = (transmitTime - self.receiverClock) * SPEED_OF_LIGHT

            # Compute the time of transmission
            relativeTime = (navMessage.getSampleSubframe() - earliestTowSample) / samplingFrequency
            transmitTime = satellite.tow + timeSinceTOW - relativeTime

            # Compute satellite positions and clock errors
            satellitesPositions[idx, :], satellitesClocks[idx] = satellite.computePosition(transmitTime)

            # Compute pseudoranges
            # TODO add remaining phase
            pseudoranges[idx] = (self.receiverClock - transmitTime) * SPEED_OF_LIGHT

            # Apply corrections
            # TODO Ionosphere, troposhere ...
            correctedPseudoranges[idx] = pseudoranges
            correctedPseudoranges[idx] -= satellitesClocks[idx] * SPEED_OF_LIGHT # Satellite clock error
            correctedPseudoranges[idx] -= satellite.getTDG() * SPEED_OF_LIGHT    # Total Group Delay (TODO this is frequency dependant)
            
            idx += 1
        
        self.computeReceiverPosition(correctedPseudoranges, satellitesPositions)

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
        A = np.zeros((nbMeasurements, 4))
        B = np.zeros(nbMeasurements)
        x = np.zeros(4)
        #x = np.array([2794767.59, 1236088.19, 5579632.92, 0])
        v = []
        for i in range(10):
            # Make matrices
            for idx in range(nbMeasurements):
                if idx == 0:
                    _satpos = satpos[idx, :]
                else:
                    p = np.sqrt((x[0] - satpos[idx, 0])**2 + (x[1] - satpos[idx, 1])**2 + (x[2] - satpos[idx, 2])**2)
                    travelTime = p / SPEED_OF_LIGHT
                    _satpos = self.correctEarthRotation(travelTime, np.transpose(satpos[idx, :]))
                
                p = np.sqrt((x[0] - _satpos[0])**2 + (x[1] - _satpos[1])**2 + (x[2] - _satpos[2])**2)

                A[idx, 0] = (satpos[idx, 0] - x[0]) / p
                A[idx, 1] = (satpos[idx, 1] - x[1]) / p
                A[idx, 2] = (satpos[idx, 2] - x[2]) / p
                A[idx, 3] = 1

                B[idx] = pseudoranges[idx] - p #- x[3]
            
            # Least Squares
            N = np.transpose(A).dot(A)
            _x = np.linalg.inv(N).dot(np.transpose(A)).dot(B)
            x = x - _x # Update solution
            v = A.dot(_x) - B
        
        self.receiverPosition.append(x[0:3])
        self.receiverClockError = x[3]

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



    # END OF CLASS


