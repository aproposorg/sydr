# -*- coding: utf-8 -*-
# =====================================================================================================================
# Abstract class for tracking process.
# Author: Antoine GRENIER (TAU)
# Date: 2023.03.15
# References: 
# =====================================================================================================================
# PACKAGES

import configparser
import logging
import numpy as np
import math

from core.receiver.receiver import Receiver
from core.channel.channel_l1ca import ChannelStatusL1CA
from core.channel.channel_l1ca import ChannelL1CA
#from core.channel.channel_l1ca_kaplan import ChannelL1CA_Kaplan as ChannelL1CA
from core.channel.channel import ChannelMessage, ChannelStatus
from core.enlightengui import EnlightenGUI
from core.space.satellite import Satellite, GNSSSystems
from core.utils.time import Time
from core.utils.constants import AVG_TRAVEL_TIME_MS, SPEED_OF_LIGHT
from core.utils.enumerations import ReceiverState, GNSSMeasurementType, TrackingFlags
from core.utils.geodesy import correctEarthRotation
from core.measurements import GNSSmeasurements

# =====================================================================================================================

class ReceiverGPSL1CA(Receiver):
    """
    Implementation of receiver for GPS L1 C/A signals. 
    """
    
    configuration : dict
    nextMeasurementTime : Time

    approxPosition : list

    isEphemerisAssited : bool

    def __init__(self, configuration:dict, overwrite=True, gui:EnlightenGUI=None):
        """
        Constructor for ReceiverGPSL1CA class.

        Args:
            None

        Returns:
            None
        
        Raises:
            None
        
        """
        super().__init__(configuration, overwrite, gui)

        self.approxPosition = np.array([
            float(configuration['DEFAULT']['approx_position_x']),\
            float(configuration['DEFAULT']['approx_position_y']),
            float(configuration['DEFAULT']['approx_position_z'])])

        # Set satellites to track
        self.prnList = list(map(int, self.configuration.get('SATELLITES', 'include_prn').split(',')))

        # Assisted GNSS
        self.isEphemerisAssited = eval(configuration['AGNSS']['broadcast_ephemeris_enabled'])
        if self.isEphemerisAssited:
            self.database.importRinexNav(configuration['AGNSS']['broadcast_ephemeris_path'])

        # Add channels in channel manager
        channelConfig = configparser.ConfigParser()
        channelConfig.read(self.configuration['CHANNELS']['gps_l1ca'])
        self.channelManager.addChannel(ChannelL1CA, channelConfig, len(self.prnList))

        # Set satellites to track
        for prn in self.prnList:
            channel = self.channelManager.requestTracking(prn)
            self.addChannelDatabase(channel)
            self.channelsStatus[channel.channelID] = ChannelStatusL1CA(channel.channelID, prn)
            self.satelliteDict[prn] = Satellite(GNSSSystems.GPS, prn)

        self.nextMeasurementTime = Time()

        # Initialise GUI
        self.gui.createReceiverGUI(self)
        self.gui.updateMainStatus(stage=f'Processing {self.name}', status='RUNNING')

        return
    
    # -----------------------------------------------------------------------------------------------------------------

    def _processChannelResults(self, results:list):
        """
        Process the results from channels.

        Args:
            results (list): List of results (dictionnary).

        Returns:
            None

        Raises:
            None
        """

        super()._processChannelResults(results)

        for packet in results:
            channel : ChannelL1CA
            channel = self.channelManager.getChannel(packet['cid'])
            if packet['type'] == ChannelMessage.DECODING_UPDATE:
                satellite : Satellite
                satellite = self.satelliteDict[channel.satelliteID]
                satellite.addSubframe(packet['bits'])
                self.channelsStatus[channel.channelID].subframeFlags[packet['subframe_id']-1] = True
                continue
            elif packet['type'] == ChannelMessage.CHANNEL_UPDATE:
                self.channelsStatus[channel.channelID].channelState  = packet['state']
                self.channelsStatus[channel.channelID].trackFlags    = packet['tracking_flags']
                self.channelsStatus[channel.channelID].tow           = packet['tow']
                self.channelsStatus[channel.channelID].timeSinceTOW  = packet['time_since_tow']
                self.channelsStatus[channel.channelID].unprocessedSamples = packet['unprocessed_samples']
            elif packet['type'] in (ChannelMessage.ACQUISITION_UPDATE, ChannelMessage.TRACKING_UPDATE):
                continue
            else:
                raise ValueError(
                    f"Unknown channel message '{packet['type']}' received from channel {channel.channelID}.")

        return
    
    # -----------------------------------------------------------------------------------------------------------------

    def computeGNSSMeasurements(self):
        """
        Process the channels' results to produce GNSS measurements.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """
        super().computeGNSSMeasurements()

        # # Compute measurements based on receiver time
        # if self.clock.isInitialised or (self.clock. < self.nextMeasurementTime):
        #     return
        
        # # TODO
        
        return
    
    # -----------------------------------------------------------------------------------------------------------------
    
    def computeGNSSMeasurements(self):
        """
        Abstract method to process the channels' results to produce GNSS measurements.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        # Check all channels ready 
        # TODO To be adapted to process with 4 satellites right away
        channel : ChannelStatus
        selectedChannels = {}
        for channel in self.channelsStatus.values():
            if (channel.trackFlags & TrackingFlags.TOW_DECODED) \
                and ((channel.trackFlags & TrackingFlags.EPH_DECODED) or (self.isEphemerisAssited)):
                selectedChannels[channel.channelID] = channel
        
        if len(selectedChannels) < 5:
            return
        
        # Check current receiver state
        if self.receiverState == ReceiverState.NAVIGATION:
            if self.clock < self.nextMeasurementTime:
                # Not yet time to compute measusurement
                return
            
        # Check if ephemeris are provided from external source
        if self.isEphemerisAssited:
            for satellite in self.satelliteDict.values():
                satellite.ephemeris = self.database.fetchBRDC(self.clock, satellite.systemID, satellite.satelliteID)

        # Find earliest signal
        maxTOW = -1
        towlist = []
        for channel in selectedChannels.values():
            # This assumes all the channels were given the same number of samples to process
            if maxTOW < channel.timeSinceTOW:
                maxTOW = channel.timeSinceTOW
                earliestChannel = channel
            towlist.append(int(channel.tow))

        # Check that all the signals have the same TOW
        if not all(x==towlist[0] for x in towlist):
            return
        
        # Update received time
        if not self.clock.isInitialised:
            tow = earliestChannel.tow + earliestChannel.timeSinceTOW / 1e3
            week = self.satelliteDict[earliestChannel.satelliteID].ephemeris.week
            receivedTime = tow + AVG_TRAVEL_TIME_MS / 1e3
            self.clock.fromGPSTime(week, receivedTime)
            self.clock.isInitialised = True
            self.nextMeasurementTime.setGPSTime(week, math.ceil(receivedTime))
        else:
            timeResidual = (self.clock - self.nextMeasurementTime).total_seconds()
            receivedTime = self.clock.getGPSSeconds() - timeResidual
            tow = earliestChannel.tow + earliestChannel.timeSinceTOW / 1e3 - timeResidual
            week = self.clock.gpstime.week_number
            self.nextMeasurementTime.fromGPSTime(week, receivedTime + (1/self.measurementFrequency))
        
        # Compute GNSS measurements
        gnssMeasurementsList = []
        for channel in selectedChannels.values():
            satellite : Satellite 
            satellite = self.satelliteDict[channel.satelliteID]
            
            # Compute transmission time
            relativeTime = (earliestChannel.timeSinceTOW - channel.timeSinceTOW) * 1e-3
            transmitTime = tow - relativeTime

            # Compute pseudoranges
            pseudoranges = (receivedTime - transmitTime) * SPEED_OF_LIGHT

            # Compute satellite positions and clock errors
            satellitePosition, satelliteClock = satellite.computePosition(transmitTime)

            # Apply corrections
            # TODO Ionosphere, troposhere ...
            correctedPseudoranges  = pseudoranges
            correctedPseudoranges += satelliteClock * SPEED_OF_LIGHT # Satellite clock error
            correctedPseudoranges += satellite.getTGD() * SPEED_OF_LIGHT  # Total Group Delay (TODO this is frequency dependant)

            logging.getLogger(__name__).debug(
                f"SVID {channel.satelliteID:02d}, timeSinceLastTOW {channel.timeSinceTOW:.4f}, relativeTime {relativeTime:.6f}, " +\
                f"transmitTime {transmitTime:.4f}, unprocessed samples {channel.unprocessedSamples}, " +\
                f"pseudoranges {pseudoranges:.3f}, correctedPseudoranges {correctedPseudoranges:.3f}")
            
            # Pseudorange
            time = Time()
            time.fromGPSTime(self.clock.gpstime.week_number, receivedTime)
            if self.measurementsEnabled[GNSSMeasurementType.PSEUDORANGE]:
                gnssMeasurements = GNSSmeasurements()
                gnssMeasurements.channel  = channel
                gnssMeasurements.time     = time
                gnssMeasurements.mtype    = GNSSMeasurementType.PSEUDORANGE
                gnssMeasurements.value    = correctedPseudoranges
                gnssMeasurements.rawValue = pseudoranges
                gnssMeasurements.residual = 0.0
                gnssMeasurements.enabled  = self.measurementsEnabled[GNSSMeasurementType.PSEUDORANGE]
                gnssMeasurementsList.append(gnssMeasurements)

            # Doppler
            # TODO

        # Compute position and measurements
        self.computeReceiverPosition(self.clock.gpstime.week_number, receivedTime, gnssMeasurementsList)

        # Update receiver state
        self.receiverState = ReceiverState.NAVIGATION

        coord = self.position.coordinate
        logging.getLogger(__name__).info(f"New measurements computed (Receiver time: {receivedTime:.3f})")
        logging.getLogger(__name__).debug(f"Position=({coord.x:12.4f} {coord.y:12.4f} {coord.z:12.4f}), precision=({coord.xPrecison:8.4f} {coord.yPrecison:8.4f} {coord.zPrecison:8.4f})")
        logging.getLogger(__name__).debug(f"Clock error={self.position.clockError: 12.4f}")
        logging.getLogger(__name__).debug(f"Receiver clock : ({self.clock.gpstime.week_number} {self.clock.gpstime.seconds} {self.clock.gpstime.femtoseconds})")
        logging.getLogger(__name__).debug(f"-------------------------------")

        return
    
    # -----------------------------------------------------------------------------------------------------------------

    def computeReceiverPosition(self, week, time, measurements):
        """
        """
        satellite : Satellite
        meas : GNSSmeasurements

        nbMeasurements = len(measurements)
        G = np.zeros((nbMeasurements, 4))
        W = np.zeros((nbMeasurements, nbMeasurements))
        Ql = np.zeros((nbMeasurements, nbMeasurements))
        y = np.zeros(nbMeasurements)
        self.navigation.setState(self.approxPosition, 0.0)
        for i in range(10):
            x = self.navigation.x

            if np.linalg.norm(self.navigation.dX) < 1e-6:
                break
            # Make matrices
            idx = 0
            for meas in measurements:
                if not meas.enabled:
                    continue
                if meas.mtype == GNSSMeasurementType.PSEUDORANGE:
                    travelTime = meas.value / SPEED_OF_LIGHT
    	            
                    transmitTime = time - travelTime
                    satellite = self.satelliteDict[meas.channel.satelliteID]
                    satpos, satclock = satellite.computePosition(transmitTime)
                    satpos = correctEarthRotation(travelTime, np.transpose(satpos))
                    
                    # Geometric range
                    p = np.sqrt((x[0] - satpos[0])**2 + (x[1] - satpos[1])**2 + (x[2] - satpos[2])**2)

                    # Observation vector
                    y[idx] = meas.value - p - x[3]

                    # Design matrix
                    G[idx, 0] = (x[0] - satpos[0]) / p
                    G[idx, 1] = (x[1] - satpos[1]) / p
                    G[idx, 2] = (x[2] - satpos[2]) / p
                    G[idx, 3] = 1

                    # Weight matrix
                    # TODO Implement sigma for each measurements
                    _SIGMA_PSEUDORANGE = 1.0
                    Ql[idx, idx] = _SIGMA_PSEUDORANGE
                else:
                    # TODO Adapt to other measurement types
                    continue

                idx += 1 
            
            # Least Squares
            self.navigation.G = G
            self.navigation.y = y
            self.navigation.W = W
            self.navigation.Ql = Ql

            success = self.navigation.compute()
        
        # Correct after minimisation
        idx = 0
        for meas in measurements:
            meas.residual = self.navigation.v[idx]
            if meas.mtype == GNSSMeasurementType.PSEUDORANGE:
                meas.value -= self.navigation.x[3]
            else:
                # TODO Adapt to other measurement types
                pass
            if meas.enabled:
                self.position.measurements.append(meas)
            
            logging.getLogger(__name__).debug(
                f"CID {meas.channel.channelID} SVID {meas.channel.satelliteID:02d} {meas.mtype:11} " + \
                f"{meas.value:13.4f} (residual: {meas.residual: .4f}, enabled: {meas.enabled})")

            idx += 1

        if success:
            _time = Time()
            _time.fromGPSTime(week, time)
            state = self.navigation.x
            statePrecision = self.navigation.getStatePrecision()
            self.position.time = _time
            self.position.coordinate.setCoordinates(state[0], state[1], state[2])
            self.position.coordinate.setPrecision(statePrecision[0], statePrecision[1], statePrecision[2])
            self.position.clockError = self.navigation.x[3]
            self.position.id += 1

            self.clock.applyCorrection(-self.position.clockError / SPEED_OF_LIGHT)
            self.addPositionDatabase(self.position, measurements)
        
        return

    # -------------------------------------------------------------------------


