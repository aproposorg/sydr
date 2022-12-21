
from datetime import datetime, timedelta
import sqlite3
from gps_time import GPSTime

import core.utils.constants as constants

class Time(object):

    datetime : datetime

    # GPS Time
    gpsTime: GPSTime

    def __init__(self):
        self.datetime = datetime(1970,1,1,0,0,0)
        self.gpsTime = GPSTime.from_datetime(self.datetime)
        return
    
    def __repr__(self):
        return self.datetime.strftime("%Y-%m-%d %H:%M:%S.%f")

    def __conform__(self, protocol):
        """
        Definition of the object reperesentation.
        This is used for interaction with the database. 
        """
        if protocol is sqlite3.PrepareProtocol:
            return self.datetime.strftime("%Y-%m-%d %H:%M:%S.%f")
    
    def __add__(self, other):
        return self.datetime + other.datetime

    def __sub__(self, other):
        return self.datetime - other.datetime
    
    def __gt__(self, other):
        return self.datetime > other.datetime
    
    def __ge__(self, other):
        return self.datetime >= other.datetime
    
    def __lt__(self, other):
        return self.datetime < other.datetime
    
    def __le__(self, other):
        return self.datetime <= other.datetime

    def __eq__(self, other):
        return self.datetime == other.datetime

    @classmethod
    def fromDatetime(self, _datetime:datetime):

        time = Time()
        time.datetime = _datetime
        time.gpsTime = GPSTime.from_datetime(_datetime)

        return time
    
    @classmethod
    def fromGPSTime(self, gpsWeek:int, gpsSeconds:float):

        time = Time()
        time.gpsTime = GPSTime(week_number=gpsWeek, time_of_week=gpsSeconds)
        time.datetime = time.gpsTime.to_datetime()

        return time

    def applyCorrection(self, seconds):
        self.datetime += timedelta(seconds=seconds)
        self.gpsTime  += seconds 
        return

    # -------------------------------------------------------------------------

    def getDateTime(self):
        return self.datetime

    def getGPSSeconds(self):
        return self.gpsTime.time_of_week

    def getGPSWeek(self):
        return self.gpsTime.week_number

    def getTransmitTime(self, pseudorange, satClock):
        dt = pseudorange/constants.SPEED_OF_LIGHT - satClock
        datetime = self.datetime - timedelta(seconds=dt)
        return Time(datetime), dt
    
    def getDOY(self):
        return self.datetime.timetuple().tm_yday
    
    # -------------------------------------------------------------------------

    def setGPSTime(self, gpsWeek:int, gpsSeconds:float):
        self.gpsTime = GPSTime(week_number=gpsWeek, time_of_week=gpsSeconds)
        self.datetime = self.gpsTime.to_datetime()
        return

    def setDatetime(self, dateTime:datetime):

        self.datetime = dateTime
        self.gpsTime = GPSTime.from_datetime(dateTime)

        return