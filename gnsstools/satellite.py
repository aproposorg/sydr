from abc import ABC
from asyncore import read
from sys import flags
from typing import Dict
from xmlrpc.client import Boolean
import numpy as np
from gnsstools.channel.abstract import ChannelState

import gnsstools.constants as constants
from gnsstools.ephemeris import Ephemeris
from gnsstools.gnsssignal import GNSSSignal, SignalType
from gnsstools.measurements import DSPEpochs, DSPmeasurement
from gnsstools.message.abstract import NavMessageType
from gnsstools.message.lnav import LNAV


class Satellite(ABC):

    satelliteID : int
    dspEpochs   : Dict[SignalType, DSPmeasurement]
    gnssEpochs  : dict
    ephemeris   : Ephemeris
    navMessage  : dict

    # Flags
    isAcquired         : bool
    isFineTracking     : bool
    isTOWDecoded       : bool
    isEphemerisDecoded : bool

    lastPosition : np.array

    def __init__(self, svid, signals):

        self.satelliteID = svid
        self.dspEpochs   = {}
        self.gnssEpochs  = {}
        self.navMessage  = {}
        for sig in signals:
            self.dspEpochs[sig] = DSPEpochs(svid, sig)
            self.navMessage[sig] = self.selectNavMessage(sig)
            self.navMessage[sig].setSatellite(svid)

        self.isTOWDecoded = False
        self.isEphemerisDecoded = False
        
        return

    # -------------------------------------------------------------------------

    def getEphemeris(self):
        return self.ephemeris

    # -------------------------------------------------------------------------

    def selectNavMessage(self, sig):
        if sig == SignalType.GPS_L1_CA:
            return LNAV()
        else:
            raise ValueError("Incorrect signal type.")

    # -------------------------------------------------------------------------

    def isSatelliteReady(self):
        
        ready = True

        # Check if TOW found
        ready = ready and self.isTOWDecoded

        # Check if ephemeris available
        ready = ready and self.isEphemerisDecoded

        # Check if satellite flag as healthy 
        # TODO

        return ready

    # =========================================================================

    def computePosition(self, time):
        """
        Compute the satellite position based on ephemeris data.
        Inputs
            time : integer
            The time where the satellite position in computed in GPS seconds
            of week.
        Outputs
            satellitePosition : numpy.array(3)
            Satellite position in ECEF 
        """        
        eph = self.ephemeris

        # Compute difference between current time and orbit reference time
        # Check for week rollover at the same time
        dt = self.timeCheck(time - eph.toc)

        # Find the satellite clock correction and apply
        satClkCorr = (eph.af2 * dt + eph.af1) * dt + eph.af0
        time -= satClkCorr

        # Orbit computations
        tk = self.timeCheck(time - eph.toe)
        a  = eph.sqrtA * eph.sqrtA
        n0 = np.sqrt(constants.EARTH_GM / a ** 3)
        n  = n0 + eph.deltan
        
        ## Eccentricity computation
        M = eph.m0 + n * tk
        M = np.remainder(M + 2 * constants.PI, 2 * constants.PI)
        E = M
        for i in range(10):
            E_old = E
            E = M + eph.ecc * np.sin(E)
            dE = np.remainder(E - E_old, 2 * constants.PI)
            if abs(dE) < 1e-12:
                break
        E = np.remainder(E + 2 * constants.PI, 2 * constants.PI)
        
        dtr = constants.RELATIVIST_CLOCK_F * eph.ecc * eph.sqrtA * np.sin(E)
        nu = np.arctan2(np.sqrt(1 - eph.ecc ** 2) * np.sin(E), np.cos(E) - eph.ecc)
        phi = np.remainder(nu + eph.omega, 2 * constants.PI)

        u = phi + eph.cuc * np.cos(2 * phi) + eph.cus * np.sin(2 * phi)
        r = a * (1 - eph.ecc * np.cos(E)) + eph.crc * np.cos(2 * phi) + eph.crs * np.sin(2 * phi)
        i = eph.i0 + eph.iDot * tk + eph.cic * np.cos(2 * phi) + eph.cis * np.sin(2 * phi)
        
        Omega = eph.omega0 + (eph.omegaDot - constants.EARTH_ROTATION_RATE) * tk \
            - constants.EARTH_ROTATION_RATE * eph.toe
        Omega = np.remainder(Omega + 2 * constants.PI, 2 * constants.PI)
        
        satellitePosition = np.zeros(3)
        satellitePosition[0] = np.cos(u)*r*np.cos(Omega) - np.sin(u)*r*np.cos(i)*np.sin(Omega)
        satellitePosition[1] = np.cos(u)*r*np.sin(Omega) + np.sin(u)*r*np.cos(i)*np.cos(Omega)
        satellitePosition[2] = np.sin(u)*r*np.sin(i)
        self.lastPosition = satellitePosition

        satelliteClockCorrection = (eph.af2*dt + eph.af1)*dt + eph.af0 + dtr

        # TODO Satellite velocity

        return satellitePosition, satelliteClockCorrection

    def getTGD(self):
        return self.ephemeris.tgd
    
    @staticmethod
    def timeCheck(time):
        """ 
        timeCheck accounting for beginning or end of week crossover.
        corrTime = check_t(time);
          Inputs:
              time        - time in seconds
          Outputs:
              corrTime    - corrected time (seconds)
        Kai Borre 04-01-96
        Copyright (c) by Kai Borre
        """
        half_week = 302400.0
        corrTime = time

        if time > half_week:
            corrTime = time - 2 * half_week
        elif time < - half_week:
            corrTime = time + 2 * half_week
        
        return corrTime

    # ------------------------------------------------------------ 

    def addDSPMeasurement(self, msProcessed, samplesProcessed, chan):
        state      = chan.getState()
        signal     = chan.gnssSignal.signalType
        dspEpoch   = self.dspEpochs[signal]
        navMessage = self.navMessage[signal]
        if state == ChannelState.ACQUIRING:
            dspEpoch.addAcquisition(msProcessed, samplesProcessed, chan)
        elif state == ChannelState.TRACKING:
            dspEpoch.addTracking(msProcessed, samplesProcessed, chan)
            # Decode
            lastMeasurement = self.dspEpochs[signal].getLastMeasurement()
            #navMessage.addMeasurement(msProcessed, lastMeasurement)
            #navMessage.run()
            # Update status
            self.isTOWDecoded = navMessage.isTOWDecoded
            self.isEphemerisDecoded = navMessage.isEphemerisDecoded

            if self.isTOWDecoded:
                self.tow = navMessage.tow
            
            if self.isEphemerisDecoded:
                self.ephemeris = navMessage.ephemeris

        return
        