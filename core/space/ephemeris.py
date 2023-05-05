from abc import ABC, abstractclassmethod
import numpy as np
from enum import Enum, unique
import logging

from core.utils.constants import EARTH_GM, EARTH_ROTATION_RATE, PI, RELATIVIST_CLOCK_F
from core.utils.enumerations import GNSSSystems
from core.utils.time import Time
import core.utils.misc as utils
import core.utils.constants as constants

class Ephemeris(ABC):

    @abstractclassmethod
    def computePosition(self):
        pass

# =============================================================================

class BRDCEphemeris(Ephemeris):
    systemID   : GNSSSystems
    satelliteID: int
    iode     : int
    iodc     : int
    toe      : float
    toc      : float
    tgd      : float
    af2      : float
    af1      : float
    af0      : float
    ecc      : float 
    sqrtA    : float
    crs      : float
    deltan   : float
    m0       : float
    cuc      : float
    cus      : float
    cic      : float
    omega0   : float
    cis      : float
    i0       : float
    crc      : float
    omega    : float
    omegaDot : float
    iDot     : float
    alpha0   : float
    ura      : float
    health   : float
    
    time       : Time()
    tow        : int
    week       : int

    subframe1Flag : bool
    subframe2Flag : bool
    subframe3Flag : bool

    subframe1Bits : np.array
    subframe2Bits : np.array
    subframe3Bits : np.array
    
    # -------------------------------------------------------------------------

    def __init__(self, systemID, satelliteID):
        self.subframe1Flag = False
        self.subframe2Flag = False
        self.subframe3Flag = False
        
        self.systemID = systemID
        self.satelliteID = satelliteID

        return

    # -------------------------------------------------------------------------

    def __eq__(self, other) -> bool:
        try:
            if self.iodc == other.iodc and self.iode == other.iode:
                return True
            else:
                return False 
        except AttributeError:
            # Other object empty
            return False

    # -------------------------------------------------------------------------

    def resetFlags(self):
        self.subframe1Flag = False
        self.subframe2Flag = False
        self.subframe3Flag = False

    # -------------------------------------------------------------------------

    def checkFlags(self):
        return self.subframe1Flag and self.subframe2Flag and self.subframe3Flag

    # -------------------------------------------------------------------------
    
    def fromSubframeBits(self, subframe:np.array):
        
        # Identify the subframe
        subframeID = utils.bin2dec(subframe[49:52])
        if subframeID == 1:
            # It contains WN, SV clock corrections, health and accuracy
            self.week          = utils.bin2dec(subframe[60:70]) + constants.GPS_WEEK_ROLLOVER * 1024
            self.ura           = utils.bin2dec(subframe[72:76])
            self.health        = utils.bin2dec(subframe[76:82])
            self.iodc          = utils.bin2dec(subframe[82:84] + subframe[211:218])  # TODO Check IODC consistency
            self.toc           = utils.bin2dec(subframe[218:234]) * 2 ** 4
            self.tgd           = utils.twosComp2dec(subframe[196:204]) * 2 ** (- 31)
            self.af2           = utils.twosComp2dec(subframe[240:248]) * 2 ** (- 55)
            self.af1           = utils.twosComp2dec(subframe[248:264]) * 2 ** (- 43)
            self.af0           = utils.twosComp2dec(subframe[270:292]) * 2 ** (- 31)
            self.subframe1Flag = True
        elif subframeID == 2:
            # It contains first part of ephemeris parameters
            self.iode          = utils.bin2dec(subframe[60:68]) # TODO Check IODE consistency
            self.ecc           = utils.bin2dec(subframe[166:174] + subframe[180:204]) * 2 ** (- 33)
            self.sqrtA         = utils.bin2dec(subframe[226:234] + subframe[240:264]) * 2 ** (- 19)
            self.toe           = utils.bin2dec(subframe[270:286]) * 2 ** 4
            self.crs           = utils.twosComp2dec(subframe[68:84]) * 2 ** (- 5)
            self.deltan        = utils.twosComp2dec(subframe[90:106]) * 2 ** (- 43) * constants.PI
            self.m0            = utils.twosComp2dec(subframe[106:114] + subframe[120:144]) * 2 ** (- 31) * constants.PI
            self.cuc           = utils.twosComp2dec(subframe[150:166]) * 2 ** (- 29)
            self.cus           = utils.twosComp2dec(subframe[210:226]) * 2 ** (- 29)
            self.subframe2Flag = True
        elif subframeID == 3:
            # It contains second part of ephemeris parameters
            self.iode          = utils.bin2dec(subframe[270:278]) # TODO Check IODE consistency
            self.cic           = utils.twosComp2dec(subframe[60:76]) * 2 ** (- 29)
            self.omega0        = utils.twosComp2dec(subframe[76:84] + subframe[90:114]) * 2 ** (- 31) * constants.PI
            self.cis           = utils.twosComp2dec(subframe[120:136]) * 2 ** (- 29)
            self.i0            = utils.twosComp2dec(subframe[136:144] + subframe[150:174]) * 2 ** (- 31) * constants.PI
            self.crc           = utils.twosComp2dec(subframe[180:196]) * 2 ** (- 5)
            self.omega         = utils.twosComp2dec(subframe[196:204] + subframe[210:234]) * 2 ** (- 31) * constants.PI
            self.omegaDot      = utils.twosComp2dec(subframe[240:264]) * 2 ** (- 43) * constants.PI
            self.iDot          = utils.twosComp2dec(subframe[278:292]) * 2 ** (- 43) * constants.PI
            self.subframe3Flag = True

        elif subframeID == 4:
            # Almanac, ionospheric model, UTC parameters.
            # SV health (PRN: 25-32).
            # Not decoded at the moment.
            # TODO
            # self.ephemeris.alpha0 = utils.twosComp2dec(subframe[60:76]) * 2 ** (-30)
            # self.ephemeris.alpha1 = utils.twosComp2dec(subframe[60:76]) * 2 ** (-27) / constants.PI
            # self.ephemeris.alpha2 = utils.twosComp2dec(subframe[60:76]) * 2 ** (-24) / constants.PI**2
            # self.ephemeris.alpha3 = utils.twosComp2dec(subframe[60:76]) * 2 ** (-24) / constants.PI**3
            # self.ephemeris.beta0  = utils.twosComp2dec(subframe[60:76]) * 2 ** ( 11)
            # self.ephemeris.beta1  = utils.twosComp2dec(subframe[60:76]) * 2 ** ( 14) / constants.PI
            # self.ephemeris.beta2  = utils.twosComp2dec(subframe[60:76]) * 2 ** ( 16) / constants.PI**2
            # self.ephemeris.beta3  = utils.twosComp2dec(subframe[60:76]) * 2 ** ( 16) / constants.PI**3
            pass
        elif subframeID == 5:
            # SV almanac and health (PRN: 1-24).
            # Almanac reference week number and time.
            # Not decoded at the moment.
            # TODO
            pass
        else: 
            logging.getLogger(__name__).warning(f"Unrecognised suframe ID {subframeID} found for satellite G{self.satelliteID}")

        return

    # -------------------------------------------------------------------------
    def computePosition(self, time:Time):
        return

    # -------------------------------------------------------------------------
    
    # def computePosition(self, time:Time):
    #     """
    #     Compute the satellite position based on ephemeris data.
    #     Inputs
    #         time : Time object
    #         The time where the satellite position in computed in GPS seconds
    #         of week.
    #     Outputs
    #         satellitePosition : numpy.array(3)
    #         Satellite position in ECEF 
    #     """
        
    #     gpsTime = time.getGPSSeconds()

    #     # Compute difference between current time and orbit reference time
    #     # Check for week rollover at the same time
    #     dt = self.timeCheck(gpsTime - self.toc)

    #     # Find the satellite clock correction and apply
    #     satClkCorr = (self.af2 * dt + self.af1) * dt + self.af0
    #     gpsTime -= satClkCorr

    #     # Orbit computations
    #     tk = self.timeCheck(gpsTime - self.toe)
    #     a  = self.sqrtA * self.sqrtA
    #     n0 = np.sqrt(EARTH_GM / a ** 3)
    #     n  = n0 + self.deltan
        
    #     ## Eccentricity computation
    #     M = self.m0 + n * tk
    #     M = np.remainder(M + 2 * PI, 2 *  PI)
    #     MDot = n
    #     E = M
    #     for i in range(10):
    #         E_old = E
    #         E = M + self.ecc * np.sin(E)
    #         dE = np.remainder(E - E_old, 2 * PI)
    #         if abs(dE) < 1e-12:
    #             break
    #     E = np.remainder(E + 2 * PI, 2 * PI)
    #     EDot = MDot / (1-self.ecc*np.cos(E))
        
    #     dtr = RELATIVIST_CLOCK_F * self.ecc * self.sqrtA * np.sin(E)
    #     nu = np.arctan2(np.sqrt(1 - self.ecc ** 2) * np.sin(E), np.cos(E) - self.ecc)
    #     phi = np.remainder(nu + self.omega, 2 * PI)

    #     nuDot = EDot*np.sin(E)*(1 + self.ecc*np.cos(nu)) / np.sin(nu)*(1 - self.ecc*np.cos(E))

    #     u = phi + self.cuc * np.cos(2 * phi) + self.cus * np.sin(2 * phi)
    #     r = a * (1 - self.ecc * np.cos(E)) + self.crc * np.cos(2 * phi) + self.crs * np.sin(2 * phi)
    #     i = self.i0 + self.iDot * tk + self.cic * np.cos(2 * phi) + self.cis * np.sin(2 * phi)
    #     uDot = nuDot + 2 * (self.cus*np.cos(2*u) + self.cuc*np.sin(2*u))*nuDot
    #     rDot = a*self.ecc*np.sin(self.ecc)*n / (1-self.ecc*np.cos(self.ecc)) + 2*(self.crs*np.cos(2*u)-self.crc*np.sin(2*u))*nuDot
    #     iDot = self.iDot + (self.cis*np.cos(2*u)-self.cic*np.sin(2*u))*2*nuDot
        
    #     OmegaDot = self.omegaDot - EARTH_ROTATION_RATE
    #     Omega = self.omega0 + OmegaDot * tk - EARTH_ROTATION_RATE * self.toe
    #     Omega = np.remainder(Omega + 2 * PI, 2 * PI)
        
    #     # Position
    #     x = np.cos(u)*r*np.cos(Omega) - np.sin(u)*r*np.cos(i)*np.sin(Omega)
    #     y = np.cos(u)*r*np.sin(Omega) + np.sin(u)*r*np.cos(i)*np.cos(Omega)
    #     z = np.sin(u)*r*np.sin(i)
        
    #     # Velocity
    #     xP = np.cos(u)*r
    #     yP = np.sin(u)*r
    #     xDotP = rDot*np.cos(u) - yP*uDot
    #     yDotP = rDot*np.sin(u) + xP*uDot
    #     vx = (xDotP - yP*np.cos(i)*OmegaDot) * np.cos(Omega) - (xP*OmegaDot+yDotP*np.cos(i)-yP*np.sin(i)*iDot)*np.sin(Omega)
    #     vy = (xDotP - yP*np.cos(i)*OmegaDot) * np.sin(Omega) + (xP*OmegaDot+yDotP*np.cos(i)-yP*np.sin(i)*iDot)*np.cos(Omega)
    #     vz = yDotP*np.sin(i) + yP*np.cos(i)*iDot

    #     self.coordinate.x  = x
    #     self.coordinate.y  = y
    #     self.coordinate.z  = z
    #     self.coordinate.vx = vx
    #     self.coordinate.vy = vy
    #     self.coordinate.vz = vz

    #     self.clockError = self.af0 + self.af1*dt * self.af2*(dt**2) - dtr

    #     return self.coordinate, self.clockError

    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
