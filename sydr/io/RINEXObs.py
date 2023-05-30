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
from sydr.utils.config import Config
from sydr.utils.coordinate import Coordinate
from sydr.utils.enumerations import GNSSSystems
from sydr.utils.time import Time
from sydr.measurements import Epoch, GNSSMeasurements

# =============================================================================

class RINEXObs:

    def __init__(self, config:Config):

        self.epochsList = []
        self.config = config

        self.approxCoordinates = Coordinate()

        return

    def read(self, filename):
        try:
            with open(filename) as f:
                self._readHeader(f)
                self._readData(f)
        except FileNotFoundError:
            print("File not found in directory.")
            raise FileNotFoundError("File could not be found.")

        return

    def _readHeader(self, f):
        self.obsMap = {}
        for line in f:
            if "END OF HEADER" in line:
                break
            elif "RINEX VERSION" in line:
                self.version = float(line[0:8])
                if self.version < 3:
                    raise ValueError("Observation RINEX version is not supported.")
            elif "APPROX POSITION XYZ" in line:
                self.approxCoordinates.setCoordinates(float(line[0:14]), float(line[14:28]), float(line[28:42]))
            elif "SYS / # / OBS TYPES" in line:
                key = self._findSystem(line[0])
                nbObs = int(line[1:6])
                
                content = line[7:59]

                # Read more lines if needed
                for i in range(int(np.ceil(nbObs/13)) - 1):
                    line = f.readline()
                    content += line[7:60]
                
                values = content.split()

                self.obsMap[key] = values
        return

    def _readData(self, f):
        satelliteCounter = -1
        saveEpoch = False
        for line in f:
            if satelliteCounter == -1 and ">" not in line:
                raise Warning("First epoch not found in file.")
            if satelliteCounter <= 0:
                if satelliteCounter == 0 and saveEpoch:
                    self.epochsList.append(copy.copy(epoch))
                time = Time(datetime(int(line[2:6]), int(line[7:9]), int(line[10:12]), \
                    int(line[13:15]), int(line[16:18]), int(line[19:21]), int(line[22:28])))
                satelliteCounter = int(line[33:35])
                epoch = Epoch(time)
            else:
                if self.config.startTime.getDateTime() <= time.getDateTime() <= self.config.stopTime.getDateTime():
                    saveEpoch = True
                    # Create new measurement
                    system = self._findSystem(line[0:1])
                    if not self.config.systemsEnabled[system]:
                        # System disable, skipped
                        satelliteCounter -= 1
                        continue
                    measurement = GNSSMeasurements(system, int(line[1:3]), time)
                    idx = 4
                    for signal in self.obsMap[system]:
                        if f"L{signal[1:3]}" not in self.config.signalsEnabled[system]:
                            # Measurement disable, skipped
                            idx += 16
                            continue
                        try:
                            value = float(line[idx:idx+13])
                        except ValueError:
                            value = 0.0
                        
                        if signal[0] in 'C':
                            measurement.addPseudorange(signal[1:3], value)
                        elif signal[0] in 'L':
                            measurement.addPhase(signal[1:3], value)
                        elif signal[0] in 'D':
                            measurement.addDoppler(signal[1:3], value)
                        elif signal[0] in 'S':
                            measurement.addSNR(signal[1:3], value)
                        else:
                            raise Warning(f"Invalid signal {signal} uncountered in RINEX, measurement skipped.")
                        idx += 16
                    epoch.addMeasurement(copy.copy(measurement))
                else:
                    saveEpoch = False
                satelliteCounter -= 1
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
    
if __name__=="__main__":
    
    filename = "/mnt/d/Projects/Navigation/Code/pynavlib/example_data/TAUN00FIN_R_20221660000_01D_30S_MO.22o"
    obs = RINEXObs(filename)
