
import configparser
from http.client import REQUESTED_RANGE_NOT_SATISFIABLE
import numpy as np
import pymap3d as pm

import sydr.constants as constants

class Navigation:

    def __init__(self, configfile):

        config = configparser.ConfigParser()
        config.read(configfile)
    	
        self.configfile = configfile
        self.msToProcess          = config.getint  ('DEFAULT', 'ms_to_process')
        self.measurementFrequency = config.getint  ('DEFAULT', 'measurement_frequency')
        self.samplingFrequency    = config.getfloat('RF_FILE', 'samp_freq')
        
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

        # Find start samples
        startSampleIdx, latestSample = self.getStartSample(satelliteDict)
        
        # Generate each pseudoranges
        for measIdx in range(nbMeasurements):
            samplesPerMs = self.samplingFrequency / 1e3
            targetSampleNumber = int(latestSample + (solutionPeriod * measIdx) * samplesPerMs)
            
            # Start by creating the pseudorange
            n = len(satelliteDict)
            prnlist                     = np.zeros(n)
            satellitesPositions         = np.zeros((n, 3))
            satellitesClocksCorrections = np.zeros(n)
            transmittedTime             = np.zeros(n)
            satellitesTDG               = np.zeros(n)
            tropoCorrections            = np.zeros(n)
            codePhase                   = np.zeros(n)
            doppler                     = np.zeros(n)
            i = 0
            for prn, satellite in satelliteDict.items():
                if not satellite.decodingEnabled:
                    continue

                # Find what are the samples around our targeted sample number
                absoluteSample = satellite.tracking.absoluteSample
                carrierFrequency = satellite.tracking.carrierFrequency
                carrierFrequencyRef = satellite.tracking.signal.carrierFreq
                codePhase_all = satellite.tracking.codePhase
                code_ms = satellite.tracking.signal.code_ms
                code_freq = satellite.tracking.signal.code_freq
                sampleNumber = absoluteSample[startSampleIdx[prn]]
                idx = startSampleIdx[prn]
                while sampleNumber < targetSampleNumber:
                    idx = int(idx + code_ms)
                    sampleNumber = absoluteSample[idx]
                idx_min = idx - code_ms
                idx_max = idx
            
                # Compute transmition time
                tow = satellite.decoding.TOW
                epoch = idx_min - satellite.decoding.firstSubFrame
                phase = (targetSampleNumber - absoluteSample[idx_min]) \
                    / (absoluteSample[idx_max] - absoluteSample[idx_min])
                codeDiff = codePhase_all[idx_max] - codePhase_all[idx_min]

                transmittedTime[i] = tow + (epoch + code_ms * phase)/1e3
                codePhase[i] = (codePhase_all[idx_min] + phase * codeDiff) / code_freq
                doppler[i]   = carrierFrequency[idx_min] * constants.SPEED_OF_LIGHT / carrierFrequencyRef
                
                # Compute satellite position
                satpos, satclk = satellite.computePosition(transmittedTime[i])

                # Compute tropospheric corrections 
                # Check if there is already an estimate of the receiver position
                # if self.receiverPosition:
                #     # TODO Height should be from Mean Sea Level instead of ellipsoid
                #     coord = self.receiverPosition[-1]
                #     llh = pm.ecef2geodetic(coord[0], coord[1], coord[2], ell=None, deg=True)
                #     doy = satellite.decoding.doy
                #     [az, elev, r] = pm.ecef2aer(satpos[0], satpos[1], satpos[2], llh[0], llh[1], llh[2]) 
                #     tropoCorrections[i] = self.troposphericCorrection(doy[1], llh, elev)

                satellitesTDG[i]               = satellite.getTGD()
                satellitesPositions[i,:]       = satpos
                satellitesClocksCorrections[i] = satclk
                prnlist[i]                     = prn
                i += 1

            # Estimate reception time
            if measIdx == 0:
                receivedTime = np.max(transmittedTime) + constants.AVG_TRAVEL_TIME_MS/1e3
            else:
                receivedTime += (solutionPeriod * measIdx)/1e3

            receivedTime = np.max(transmittedTime) + constants.AVG_TRAVEL_TIME_MS/1e3

            # Compute pseudorange
            pseudoranges = (receivedTime - transmittedTime - codePhase) * constants.SPEED_OF_LIGHT
            
            # Satellite clock error
            pseudoranges += satellitesClocksCorrections * constants.SPEED_OF_LIGHT

            # Total Group Delay (TGD) error, which is frequency dependent 
            # TODO Should adapt to different frequencies (see ESA GNSS book)
            pseudoranges +=  satellitesTDG * constants.SPEED_OF_LIGHT

            # Tropospheric error
            # TODO Not working yet
            # pseudoranges += tropoCorrections

            # Ionospheric error
            # TODO

            # Compute receiver position
            self.computeReceiverPosition(pseudoranges, satellitesPositions)

            # Correct pseudoranges based on receiver error
            #pseudoranges += self.receiverClockError[0]

            i = 0
            for prn in prnlist:
                satelliteDict[prn].measurementsTOW.append(satellite.decoding.TOW + receivedTime)
                satelliteDict[prn].pseudoranges.append(pseudoranges[i])
                satelliteDict[prn].coarsePseudoranges.append(pseudoranges[i])
                satelliteDict[prn].doppler.append(doppler[i])
                i += 1
        
        return

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
                    travelTime = p / constants.SPEED_OF_LIGHT
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
        omegatau = constants.EARTH_ROTATION_RATE * traveltime

        # --- Make a rotation matrix -----------------------------------------------
        R3 = np.array([[np.cos(omegatau), np.sin(omegatau), 0.0],
                       [-np.sin(omegatau), np.cos(omegatau), 0.0],
                       [0.0, 0.0, 1.0]])

        # --- Do the rotation ------------------------------------------------------
        X_sat_rot = R3.dot(X_sat)
        
        return X_sat_rot

    def getStartSample(self, satelliteDict):
        
        startSampleList = []
        for prn, satellite in satelliteDict.items():
            startSampleList.append(satellite.tracking.absoluteSample[satellite.decoding.firstSubFrame])
        
        # Find the latest signal
        latestSample = np.max(startSampleList)

        # Find in the other signals what is the closest sample compare to the lastest one
        startSampleIdx = {}
        for prn, satellite in satelliteDict.items():
            diff = satellite.tracking.absoluteSample - latestSample
            idx = np.argsort(np.abs(diff))
            startSampleIdx[prn] = np.min(idx[:2])
        
        return startSampleIdx, latestSample

    def troposphericCorrection(self, doy, receiverPosition, satelliteElevation):
        """
        Compute troposheric corrections based on the model from [Collins, 1999].
        References: 
        [Collins, 1999] Collins, J., 1999. Assessment and Development of a Tropospheric
                        Delay Model for Aircraft Users of the Global Positioning
                        System. MScE thesis, University of New Brunswick, Fredericton, New
                        Brunswick, Canada.
        [ESA, 2013]     European Space Agency. GNSS Data Processing Volume I: Fundamentals
                        and Algorithms.
        """
        
        # Obliquity factor (valid for elevation > 5 degrees)
        m = 1.001 / np.sqrt(0.002001 + np.sin(np.deg2rad(satelliteElevation))**2)

        # Compute standard values 
        # Pressure
        p = self.getMeteoParameter(receiverPosition[0], doy, \
                                   constants.TROPO_METEO_AVG_P0, constants.TROPO_METEO_VAR_P0)
        # Temperature
        t = self.getMeteoParameter(receiverPosition[0], doy, \
                                   constants.TROPO_METEO_AVG_T0, constants.TROPO_METEO_VAR_T0)
        # Water vapor 
        e = self.getMeteoParameter(receiverPosition[0], doy, \
                                   constants.TROPO_METEO_AVG_E0, constants.TROPO_METEO_VAR_E0)
        # Temperature lapse rate
        b = self.getMeteoParameter(receiverPosition[0], doy, \
                                   constants.TROPO_METEO_AVG_B0, constants.TROPO_METEO_VAR_B0)
        # Water vapor lapse rate
        l = self.getMeteoParameter(receiverPosition[0], doy, \
                                   constants.TROPO_METEO_AVG_L0, constants.TROPO_METEO_VAR_L0)

        tDry = 10e-6 * constants.TROPO_K1 * constants.TROPO_R * p / constants.TROPO_G_M
        tDry *= (1 - b * receiverPosition[2] / t)**(constants.TROPO_G / (constants.TROPO_R * b))
        tWet = 10e-6 * constants.TROPO_K2 * constants.TROPO_R / \
            ((l+1) * constants.TROPO_G_M - b * constants.TROPO_R) * (e/t)
        tWet *= (1 - b * receiverPosition[2] / t)**((l+1) * constants.TROPO_G / (constants.TROPO_R * b)-1)
        
        correction = m * (tDry + tWet)

        return correction
    
    @staticmethod
    def getMeteoParameter(receiverLatitude, doy, avgValues, varValues):
        
        paramAvg = np.interp(receiverLatitude, constants.TROPO_METEO_AVG_LAT, avgValues)
        paramVar = np.interp(receiverLatitude, constants.TROPO_METEO_VAR_LAT, varValues)

        if receiverLatitude > 0:
            doy_min = 28
        else:
            doy_min = 211

        param = paramAvg - paramVar * np.cos(2 * np.pi * (doy - doy_min) / 365.25)
        
        return param

    # def ionosphericCorrection(self, receiverPosition, satelliteAER, tow):
    #     """
    #     Compute ionospheric corrections based on the model from [Klobuchar, 1987].
    #     References: 
    #     [Klobuchar, 1987]   Klobuchar, J., 1987. Ionospheric Time-Delay Algorithms
    #                         for Single-Frequency GPS Users. IEEE Transactions on
    #                         Aerospace and Electronic Systems AES-23(3), pp. 325{331.
        
    #     [ESA, 2013]         European Space Agency. GNSS Data Processing Volume I: Fundamentals
    #                         and Algorithms.
    #     """
    #     satelliteAER[0:2]      = np.deg2rad(satelliteAER[0:2])
    #     receiverPosition[0:2]  = np.deg2rad(receiverPosition[0:2])
    #     earthAngle  = np.pi / 2 - satelliteAER[1]
    #     earthAngle -= np.arcsin(constants.EARTH_RADIUS / (constants.EARTH_RADIUS + satelliteAER[2]) \
    #                             * np.cos(satelliteAER[1]))
        
    #     ippLatitude = np.arcsin(np.sin(receiverPosition[0]) * np.cos(earthAngle) \
    #                           + np.cos(receiverPosition[0]) * np.sin(earthAngle) * np.cos(satelliteAER[0]))

    #     ippLongitude = receiverPosition[1] + (earthAngle * np.sin(satelliteAER[0])) / np.cos(ippLatitude)

    #     magLatitude = np.arcsin(np.sin(ippLatitude) * np.sin(constants.IONO_MAG_LAT) 
    #                 + np.cos(ippLatitude) * np.cos(constants.IONO_MAG_LAT) 
    #                 * np.cos(ippLongitude - constants.IONO_MAG_LON))
        
    #     t = 43200 * ippLongitude / np.pi + (tow % constants.SECONDS_PER_DAY)

    #     for n in range(3):


    #     return correction

