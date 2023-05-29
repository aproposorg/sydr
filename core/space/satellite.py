from abc import ABC
from typing import Dict
import numpy as np


import core.utils.constants as constants
from core.space.ephemeris import BRDCEphemeris
from core.utils.enumerations import GNSSSystems

class Satellite(ABC):

    systemID    : GNSSSystems
    satelliteID : int
    ephemeris   : list
    navMessage  : dict

    # Flags
    isAcquired         : bool
    isFineTracking     : bool
    isTOWDecoded       : bool
    isEphemerisDecoded : bool

    lastPosition : np.array

    def __init__(self, system:GNSSSystems, svid:int):
        
        self.systemID = system
        self.satelliteID = svid
        self.navMessages  = {}

        self.isTOWDecoded = False
        self.isEphemerisDecoded = False

        self.ephemeris = BRDCEphemeris(system, svid)
        self.partialEphemeris = BRDCEphemeris(system, svid)
        
        self.subframeTOW = 0 # Used when navigation message is decoded from signal

        return

    # -------------------------------------------------------------------------

    def addBRDCEphemeris(self, ephemeris:BRDCEphemeris):
        self.ephemeris = ephemeris
        self.isEphemerisDecoded = True
        return
    
    # -------------------------------------------------------------------------

    def addSubframe(self, subframeBits:np.array):
        self.partialEphemeris.fromSubframeBits(subframeBits)
        if self.partialEphemeris.checkFlags():
            self.addBRDCEphemeris(self.partialEphemeris)
            self.partialEphemeris = BRDCEphemeris(self.systemID, self.satelliteID)
        return

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

        satelliteClockCorrection = (eph.af2*dt + eph.af1)*dt + eph.af0 - dtr

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
        