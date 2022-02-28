import numpy as np

from gnsstools.acquisition import Acquisition
from gnsstools.tracking import Tracking
from gnsstools.ephemeris import Ephemeris
from gnsstools.decoding import Decoding
import gnsstools.constants as constants

class Satellite:

    def __init__(self):
        self.acquisitionEnabled = False
        self.trackingEnabled    = False
        self.ephemerisEnabled   = False
        self.decodingEnabled    = False

        self.pseudoranges = []
        self.coarsePseudoranges = []

        return

    # =========================================================================
    # SETTER
    # -------------------------------------------------------------------------

    def setAcquisition(self, acquisition:Acquisition):
        self.acquisition = acquisition
        self.acquisitionEnabled = True
        return
    
    # -------------------------------------------------------------------------

    def setTracking(self, tracking:Tracking):
        self.tracking = tracking
        self.trackingEnabled = True
        return
    
    # -------------------------------------------------------------------------

    def setEphemeris(self, ephemeris:Ephemeris):
        self.ephemeris = ephemeris
        self.ephemerisEnabled = True
        return

    # -------------------------------------------------------------------------

    def setDecoding(self, decoding:Decoding):
        self.decoding = decoding
        self.decodingEnabled = True
        return
    
    # =========================================================================
    # GETTER
    # -------------------------------------------------------------------------

    def getAcquisition(self):
        return self.acquisition

    # -------------------------------------------------------------------------

    def getTracking(self):
        return self.tracking

    # -------------------------------------------------------------------------

    def getEphemeris(self):
        return self.ephemeris

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
        eph = self.decoding.ephemeris

        # Compute difference between current time and orbit reference time
        # Check for week rollover at the same time
        dt = self.timeCheck(time - eph.t_oc)

        # Find the satellite clock correction and apply
        satClkCorr = (eph.a_f2 * dt + eph.a_f1) * dt + eph.a_f0 - eph.T_GD
        time -= satClkCorr

        # Orbit computations
        tk = self.timeCheck(time - eph.t_oe)
        a  = eph.sqrtA * eph.sqrtA
        n0 = np.sqrt(constants.EARTH_GM / a ** 3)
        n  = n0 + eph.deltan
        
        ## Eccentricity computation
        M  = eph.M_0 + n * tk
        M  = np.remainder(M + 2 * constants.PI, 2 * constants.PI)
        E  = M
        for i in range(10):
            E_old = E
            E = M + eph.e * np.sin(E)
            dE = np.remainder(E - E_old, 2 * constants.PI)
            if abs(dE) < 1e-12:
                break
        E =np.remainder(E + 2 * constants.PI, 2 * constants.PI)
        
        dtr = constants.RELATIVIST_CLOCK_F * eph.e * eph.sqrtA * np.sin(E)
        nu = np.arctan2(np.sqrt(1 - eph.e ** 2) * np.sin(E), np.cos(E) - eph.e)
        phi = np.remainder(nu + eph.omega, 2 * constants.PI)

        u = phi + eph.C_uc * np.cos(2 * phi) + eph.C_us * np.sin(2 * phi)
        r = a * (1 - eph.e * np.cos(E)) + eph.C_rc * np.cos(2 * phi) + eph.C_rs * np.sin(2 * phi)
        i = eph.i_0 + eph.iDot * tk + eph.C_ic * np.cos(2 * phi) + eph.C_is * np.sin(2 * phi)
        
        Omega = eph.omega_0 + (eph.omegaDot - constants.EARTH_ROTATION_RATE) * tk \
            - constants.EARTH_ROTATION_RATE * eph.t_oe
        Omega = np.remainder(Omega + 2 * constants.PI, 2 * constants.PI)
        
        satellitePosition = np.zeros(3)
        satellitePosition[0] = np.cos(u)*r*np.cos(Omega) - np.sin(u)*r*np.cos(i)*np.sin(Omega)
        satellitePosition[1] = np.cos(u)*r*np.sin(Omega) + np.sin(u)*r*np.cos(i)*np.cos(Omega)
        satellitePosition[2] = np.sin(u)*r*np.sin(i)

        satelliteClockCorrection = (eph.a_f2*dt + eph.a_f1)*dt + eph.a_f0 - eph.T_GD + dtr

        # TODO Satellite velocity

        return satellitePosition, satelliteClockCorrection
    
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