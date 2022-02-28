
import configparser
import numpy as np

import gnsstools.constants as constants

class Navigation:

    def __init__(self, configfile):

        config = configparser.ConfigParser()
        config.read(configfile)
    	
        self.configfile = configfile
        self.samplingFrequency    = config.getfloat('DEFAULT', 'samp_freq')
        self.msToProcess          = config.getint  ('DEFAULT', 'ms_to_process')
        self.measurementFrequency = config.getint  ('DEFAULT', 'measurement_frequency')
        
        self.referenceReceiverPosition = np.array([
            config.getfloat('DEFAULT', 'receiver_position_lat'), \
            config.getfloat('DEFAULT', 'receiver_position_lon'), \
            config.getfloat('DEFAULT', 'receiver_position_hgt')])

        self.receiverPosition = []
        self.receiverClockError = []

        return

    def computePseudoranges(self, satelliteDict):
        # Compute pseudoranges
        subFrameStart = []
        for prn, satellite in satelliteDict.items():
            subFrameStart.append(satellite.decoding.firstSubFrame)
        subFrameStart = np.array(subFrameStart)
        
        # Generate number of measurements to generate
        solutionPeriod = 1e3 * (1/self.measurementFrequency)
        nbMeasurements = int(np.fix(self.msToProcess - subFrameStart.max()) / solutionPeriod)
        
        # Generate each pseudoranges
        for measIdx in range(nbMeasurements):
            timestep = solutionPeriod * measIdx # in ms
            
            ## Start by creating the "coarse" pseudorange
            # In reality, "coarse" is redundant here, this why these measurements
            # are called "Pseudo" ranges.
            travelTime = []
            prnlist    = []
            satellitesPositions = []
            satellitesClocksCorrections = []
            for prn, satellite in satelliteDict.items():
                # Finding travel time
                samplesPerCode = satellite.tracking.signal.getSamplesPerCode(self.samplingFrequency)
                msOfSignal = int(satellite.decoding.firstSubFrame + timestep)
                travelTime.append(satellite.tracking.absoluteSample[msOfSignal] / samplesPerCode)
                prnlist.append(prn)

                # Compute satellite position
                time = satellite.decoding.TOW + (timestep/1e3)
                satpos, satclk = satellite.computePosition(time)
                satellitesPositions.append(satpos)
                satellitesClocksCorrections.append(satclk)
            
            travelTime = np.array(travelTime)
            satellitesPositions = np.array(satellitesPositions)
            satellitesClocksCorrections = np.array(satellitesClocksCorrections)
            
            minimum    = np.floor(travelTime.min())
            travelTime = travelTime - minimum + constants.AVG_TRAVEL_TIME_MS
            
            coarsePseudoranges  = travelTime * constants.SPEED_OF_LIGHT / 1000
            coarsePseudoranges += satellitesClocksCorrections * constants.SPEED_OF_LIGHT

            # Compute the receiver position
            self.computeReceiverPosition(coarsePseudoranges, satellitesPositions)

            # Correct pseudoranges based on receiver error
            pseudoranges = coarsePseudoranges + self.receiverClockError[0]

            i = 0
            for prn in prnlist:
                satelliteDict[prn].pseudoranges.append(pseudoranges[i])
                satelliteDict[prn].coarsePseudoranges.append(coarsePseudoranges[i])
                i += 1
        
        return

    def computeReceiverPosition(self, pseudoranges, satpos):
        
        nbMeasurements = len(pseudoranges)
        A = np.zeros((nbMeasurements, 4))
        B = np.zeros(nbMeasurements)
        x = np.zeros(4)
        #x = np.array([2794767.59, 1236088.19, 5579632.92, 0])

        for i in range(10):
            # Make matrices
            for idx in range(nbMeasurements):
                p = np.sqrt((x[0] - satpos[idx, 0])**2 + (x[1] - satpos[idx, 1])**2 + (x[2] - satpos[idx, 2])**2)
                travelTime = p / constants.SPEED_OF_LIGHT
                _satpos = self.correctEarthRotation(travelTime, np.transpose(satpos[idx, :]))
                
                A[idx, 0] = -(_satpos[0] - x[0]) / p
                A[idx, 1] = -(_satpos[1] - x[1]) / p
                A[idx, 2] = -(_satpos[2] - x[2]) / p
                A[idx, 3] = 1

                B[idx] = pseudoranges[idx] - p - x[3]
            
            # Least Squares
            N = np.transpose(A).dot(A)
            _x = np.linalg.inv(N).dot(np.transpose(A)).dot(B)
            x = x + _x # Update solution
        
        self.receiverPosition.append(x[0:3])
        self.receiverClockError.append(x[3])

        return

    @staticmethod
    def correctEarthRotation(traveltime, X_sat, *args, **kwargs):
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

        Omegae_dot = 7.292115147e-05

        # --- Find rotation angle --------------------------------------------------
        omegatau = Omegae_dot * traveltime

        # --- Make a rotation matrix -----------------------------------------------
        R3 = np.array([[np.cos(omegatau), np.sin(omegatau), 0.0],
                       [-np.sin(omegatau), np.cos(omegatau), 0.0],
                       [0.0, 0.0, 1.0]])

        # --- Do the rotation ------------------------------------------------------
        X_sat_rot = R3.dot(X_sat)
        
        return X_sat_rot