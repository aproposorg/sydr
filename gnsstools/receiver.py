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
from gnsstools.message.abstract import NavMessageType
from gnsstools.satellite import Satellite
from gnsstools.constants import AVG_TRAVEL_TIME_MS, SPEED_OF_LIGHT
# =============================================================================
class Receiver():

    def __init__(self, receiverConfigFile, gnssSignal:GNSSSignal):

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

        self.measurementFrequency = 1  # In Hz

        return

    # -------------------------------------------------------------------------
    
    def run(self, rfSignal, satelliteList):

        self.rfSignal = rfSignal

        # TEMPORARY solution TODO change
        signalDict = {}
        signalDict[SignalType.GPS_L1_CA] = self.gnssSignal
        self.satelliteList = satelliteList

        # Initialise the channels
        for idx in range(min(self.nbChannels, len(satelliteList))):
            self.channels.append(Channel(idx, self.gnssSignal, rfSignal))

        # Initialise satellite structure
        self.satelliteDict = {}
        for svid in satelliteList:
            self.satelliteDict[svid] = Satellite(svid, signalDict)

        # Loop through the file contents
        # TODO Load by chunck of data instead of ms per ms
        samplesInMs = int(rfSignal.samplingFrequency * 1e-3)
        rfData = np.empty(samplesInMs)
        sampleCounter = 0
        for msProcessed in range(self.msToProcess):
            rfData = rfSignal.readFile(timeLength=1, keep_open=True)
            sampleCounter += samplesInMs

            if rfData.size < samplesInMs:
                raise EOFError("EOF encountered earlier than expected in file.")

            # Run the channels
            self.runChannels(rfData)

            # Handle channels results
            self.processChannels(msProcessed, sampleCounter)

            # Compute measurements
            #self.computeGNSSMeasurements(sampleCounter, self.gnssSignal)

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

    def processChannels(self, msProcessed, samplesProcessed):
        for chan in self.channels:
            svid    = chan.svid
            state   = chan.getState()
            satellite = self.satelliteDict[svid]
            if state == ChannelState.OFF:
                continue
            elif state == ChannelState.IDLE:
                # Signal was not aquired
                satellite.addDSPMeasurement(msProcessed, samplesProcessed, chan)
                print(f"Channel {chan.cid} could not acquire satellite G{svid}.")    
                pass
            elif state == ChannelState.ACQUIRING:
                # Buffer not full (most probably)
                if chan.isAcquired:
                    satellite.addDSPMeasurement(msProcessed, samplesProcessed, chan)
                    chan.switchState(ChannelState.TRACKING)
                    print(f"Channel {chan.cid} found satellite G{svid}, tracking started.")    
                else:
                    # Buffer not full (most probably)
                    pass
            elif state == ChannelState.TRACKING:
                # Signal is being tracked
                satellite.addDSPMeasurement(msProcessed, samplesProcessed, chan)
            else:
                raise ValueError(f"State {state} in channel {chan.cid} is not a valid state.")
            
            self.channelsStates[chan.cid] = state
        return

    # -------------------------------------------------------------------------

    def computeGNSSMeasurements(self, sampleCounter):

        # Check if satellite ready for measurement
        prnList = []
        for prn, satellite in self.satelliteDict.items():
            if not satellite.isSatelliteReady():
                continue
            prnList.append(prn)
        
        if len(prnList) < 5:
            # Not enough satellites
            return

        # Retrieve transmitted time
        samplingFrequency = self.rfSignal.samplingFrequency
        samplesPerCode = self.gnssSignal.getSamplesPerCode(samplingFrequency)
        transmittedTime = np.zeros(len(prnList))
        satellitesPositions = np.zeros((len(prnList), 3))
        idx = 0
        for prn in prnList:
            satellite = self.satelliteDict[prn]
            navMessage = satellite.navMessage[self.gnssSignal.signalType]
            lastDSPMeasurement = satellite.dspEpochs[self.gnssSignal.signalType].getLastMeasurement()

            # Get the last TOW
            tow = navMessage.getTow()

            # Get sample index of subframe
            sampleSuframe = navMessage.getSampleSubframe()

            # Find difference between last TOW and last code tracked
            diffTowCode = lastDSPMeasurement.sample - sampleSuframe

            # Find difference between current sample and last code tracked
            diffCurrentCode = sampleCounter - lastDSPMeasurement.sample

            # Find the code phase
            # codePhase = diffCurrentCode / samplesPerCode

            # Build transmitted time 
            transmittedTime[idx] = tow + (diffTowCode + diffCurrentCode) * samplingFrequency

            # Correct the time
            transmittedTime += satellite.getTGD()  # Total Group Delay

            satellitesPositions[idx, :], satclk = satellite.computePosition(transmittedTime[idx])
            transmittedTime += satclk

            idx += 1 
        
        # Estimated received time
        if not self.isClockInitialised:
            # The largest value is the one that was received first
            # We add an average travel time to obtain a more realistic initial value
            self.receiverClock = np.max(transmittedTime) + (AVG_TRAVEL_TIME_MS/1e3)
        else:
            self.receiverClock += 1 / self.measurementFrequency
        
        # Compute pseudoranges
        pseudoranges = (self.receiverClock - transmittedTime) * SPEED_OF_LIGHT

        # Compute receiver position
        self.computeReceiverPosition(pseudoranges, satellitesPositions)

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

                B[idx] = pseudoranges[idx] - p - x[3]
            
            # Least Squares
            N = np.transpose(A).dot(A)
            _x = np.linalg.inv(N).dot(np.transpose(A)).dot(B)
            x = x - _x # Update solution
            v = A.dot(_x) - B
        
        self.receiverPosition.append(x[0:3])
        self.receiverClockError.append(x[3])

        return

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


