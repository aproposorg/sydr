


#!/usr/bin/env python
# =============================================================================
# pyNavLib
# RINEXObs class 
# Handles observations from RINEX file.
# -----------------------------------------------------------------------------
# Author: Antoine Grenier
# Date  : 2022/07/11
# =============================================================================

from datetime import date, datetime
import numpy as np
import copy
from core.space.ephemeris import BRDCEphemeris
from core.utils.enumerations import GNSSSystems
from core.utils.time import Time, fromDatetime

# =============================================================================

class RINEXNav:

    def __init__(self):

        self.satelliteDict = {}
        self.ionosphericDict = {} # Dictionary to store parameters per systems

        return
    
    # -------------------------------------------------------------------------

    def read(self, filename):
        try:
            with open(filename) as f:
                self._readHeader(f)
                self._readData(f)
        except FileNotFoundError:
            print("File not found in directory.")
            raise FileNotFoundError("File could not be found.")

        return

    # -------------------------------------------------------------------------

    def _readHeader(self, f):
        for line in f:
            if "END OF HEADER" in line:
                break
            elif "RINEX VERSION" in line:
                self.version = float(line[0:8])
                if int(self.version) != 3:
                    raise ValueError("Observation RINEX version is not supported.")
            elif "IONOSPHERIC CORR" in line:
                # TODO Adapt for BDS corrections
                self.ionosphericDict[line[0:4]] = \
                    [float(line[5:17]), float(line[17:29]), float(line[29:41]), float(line[41:53])]
        return

    # -------------------------------------------------------------------------

    def _readData(self, f):
        """
        Read navigation files. See RINEX 3.04 documents for details.
        """
        lineIdx = -1
        for line in f:
            if line[0] != " ":
                if lineIdx != -1:
                    if system == GNSSSystems.GPS:
                        if prn not in self.satelliteDict.keys():
                            self.satelliteDict[prn] = []
                        self.satelliteDict[prn].append(copy.copy(ephemeris))
                prn = line[:3]
                lineIdx = 0
                time = fromDatetime(datetime(int(line[4:8]), int(line[9:11]), int(line[12:14]), \
                    int(line[15:17]), int(line[18:20]), int(line[21:23])))
                system = self._findSystem(line[0])
                ephemeris = BRDCEphemeris()
                ephemeris.time = time
                ephemeris.systemID = system
                ephemeris.satelliteID = int(prn[1:3])

            if system in (GNSSSystems.GPS, GNSSSystems.GALILEO):
                if lineIdx == 0:
                    ephemeris.toc      = time.getGPSSeconds()
                    ephemeris.af0      = float(line[23:42])
                    ephemeris.af1      = float(line[42:61])
                    ephemeris.af2      = float(line[61:80])
                elif lineIdx == 1: 
                    ephemeris.iode     = int(float(line[4:23]))
                    ephemeris.crs      = float(line[23:42])
                    ephemeris.deltan   = float(line[42:61])
                    ephemeris.m0       = float(line[61:80])
                elif lineIdx == 2:   
                    ephemeris.cuc      = float(line[4:23])
                    ephemeris.ecc      = float(line[23:42])
                    ephemeris.cus      = float(line[42:61])
                    ephemeris.sqrtA    = float(line[61:80])
                elif lineIdx == 3:   
                    ephemeris.toe      = float(line[4:23])
                    ephemeris.cic      = float(line[23:42])
                    ephemeris.omega0   = float(line[42:61])
                    ephemeris.cis      = float(line[61:80])
                elif lineIdx == 4: 
                    ephemeris.i0       = float(line[4:23])
                    ephemeris.crc      = float(line[23:42])
                    ephemeris.omega    = float(line[42:61])
                    ephemeris.omegaDot = float(line[61:80])
                elif lineIdx == 5: 
                    ephemeris.iDot     = float(line[4:23])
                    # TODO add system specific stuff
                    ephemeris.week     = float(line[42:61])
                elif lineIdx == 6:
                    N = int(float(line[4:23]))
                    if N <= 6:
                        ephemeris.ura = 2**(1 + N/2)
                    elif N <= 15:
                        ephemeris.ura = 2**(N - 2)
                    else:
                        ephemeris.ura = 8192
                    ephemeris.health   = int(float(line[23:42]))
                    if system is GNSSSystems.GPS:
                        ephemeris.tgd  = float(line[42:61])
                        ephemeris.iodc = int(float(line[61:80]))
                    elif system is GNSSSystems.GALILEO:
                        ephemeris.bgd_e5a = float(line[42:61])
                        ephemeris.bgd_e5b = float(line[42:61])
                elif lineIdx == 7:
                    ephemeris.transmitTime = float(line[4:23])
                    # TODO
                    pass
                # Add the ionospheric corrections if provided in the file
                if system is GNSSSystems.GPS \
                    and "GPSA" in self.ionosphericDict.keys() and "GPSB" in self.ionosphericDict.keys():
                    ephemeris.ionoAlpha = self.ionosphericDict["GPSA"]
                    ephemeris.ionoBeta  = self.ionosphericDict["GPSB"]
                elif system is GNSSSystems.GALILEO \
                    and "GAL" in self.ionosphericDict.keys():
                    ephemeris.ionoAlpha = self.ionosphericDict["GAL"]

                lineIdx += 1
            else:
                # TODO 
                pass
        return

    def _findSystem(self, letter):

        if letter in 'G':
            return GNSSSystems.GPS
        elif letter in 'R':
            return GNSSSystems.GLONASS
        elif letter in 'E':
            return GNSSSystems.GALILEO
        elif letter in 'C':
            return GNSSSystems.BEIDOU
        elif letter in 'J':
            return GNSSSystems.QZSS
        elif letter in 'I':
            return GNSSSystems.IRNSS
        elif letter in 'S':
            return GNSSSystems.SBAS
        else:
            return GNSSSystems.UNKNOWN
