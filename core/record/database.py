from datetime import datetime
from distutils.log import error
import os
import sqlite3
import numpy as np
import pickle

from core.io.RINEXNav import RINEXNav
from core.measurements import GNSSPosition
from core.satellite.ephemeris import BRDCEphemeris
from core.utils.enumerations import GNSSSystems
from core.utils.time import Time

#from core.io.RINEXNav import RINEXNav

class DatabaseHandler:

    dictBuffer : dict

    def __init__(self, dbPath,  overwrite=False):

        # Defining the database
        if overwrite:
            if os.path.exists(dbPath):
                os.remove(dbPath)
        
        self.connection = sqlite3.connect(dbPath, detect_types=sqlite3.PARSE_DECLTYPES)
        #self.connection = sqlite3.connect(':memory:')
        self.cursor = self.connection.cursor()

        self.columns = {} # Dictionnary containing the column list for each table
        self.dictBuffer = {} # Dictionnary buffering the object to be send to DB, this is to limit DB interations

        # Initialise database content
        self._initialise()

        return
    
    # -------------------------------------------------------------------------

    def addData(self, table, data):

        if table not in [*self.dictBuffer]:
            self.dictBuffer[table] = []
        
        self.dictBuffer[table].append(data)

        return


    # -------------------------------------------------------------------------

    def commit(self):
        """ 
        """

        for table, inserts in self.dictBuffer.items(): 
            for data in inserts: 
                columns = ""
                qmarks = ""
                values  = []
                insertList = []
                for key, val in data.items():
                    # Check if column exist, otherwise add it
                    if key not in self.columns[table]:
                        if isinstance(val, int):
                            mtype = "INTEGER"
                        elif isinstance(val, float):
                            mtype = "FLOAT"
                        elif isinstance(val, str):
                            mtype = "TEXT"
                        elif isinstance(val, list) or isinstance(val, np.ndarray):
                            mtype = "BLOB"
                        else:
                            raise TypeError("Unknown type given in database.")
                        
                        self.addColumn(table, {key: mtype})
                        self.columns[table].append(key)

                    # Check if data need to be serialise first
                    if isinstance(val, list) or isinstance(val, np.ndarray):
                        # Using the BLOB type of sqlite to store this type
                        # We need to serialise the array with Pickle first
                        # We use sqlite3.Binary to ;ake it fit in the BLOB
                        # Reference: https://stackoverflow.com/questions/198692/can-i-pickle-a-python-dictionary-into-a-sqlite3-text-field
                        val = sqlite3.Binary(pickle.dumps(val, pickle.HIGHEST_PROTOCOL))

                    # Build SQL string
                    columns += f"{key},"
                    qmarks += f"?,"
                    values.append(val)
                insertList.append(values)
                sqlstr = f"INSERT INTO {table} ({columns[:-1]}) VALUES ({qmarks[:-1]});"
                self.cursor.executemany(sqlstr, insertList)
        self.connection.commit()
        self.dictBuffer = {}

        return
    
    # -------------------------------------------------------------------------


    def _initialise(self):
        """
        Initialise the database by adding the default tables inside. 
        """

        # Define the standard tables
        # Channel
        sqlstr = """CREATE TABLE IF NOT EXISTS channel (
                        id INTEGER PRIMARY KEY,
                        physical_id INTEGER,
                        system TEXT,
                        satellite_id INTEGER,
                        signal TEXT,
                        start_time FLOAT,
                        stop_time FLOAT,
                        start_sample INTEGER
                        );
                    """
        self.cursor.execute(sqlstr)
        self.columns["channel"] = ["id", "physical_id", "system", "satellite_id", "signal", "start_time", "stop_time",\
            "start_sample"]

        # Acquisition
        sqlstr = """CREATE TABLE IF NOT EXISTS acquisition (
                        id INTEGER PRIMARY KEY,
                        channel_id INTEGER,
                        time FLOAT,
                        time_sample INTEGER,
                        FOREIGN KEY (channel_id) REFERENCES channels(id)
                        );
                    """
        self.cursor.execute(sqlstr)
        self.columns["acquisition"] = ["id", "channel_id", "time", "time_sample"]

        # Tracking
        sqlstr = """CREATE TABLE IF NOT EXISTS tracking (
                        id INTEGER PRIMARY KEY,
                        channel_id INTEGER,
                        time FLOAT,
                        time_sample INTEGER,
                        FOREIGN KEY (channel_id) REFERENCES channels(id)
                        );
                    """
        self.cursor.execute(sqlstr)
        self.columns["tracking"] = ["id", "channel_id", "time", "time_sample"]

        # Decoding
        sqlstr = """CREATE TABLE IF NOT EXISTS decoding (
                        id INTEGER PRIMARY KEY,
                        channel_id INTEGER,
                        time FLOAT,
                        time_sample INTEGER,
                        FOREIGN KEY (channel_id) REFERENCES channels(id)
                        );
                    """
        self.cursor.execute(sqlstr)
        self.columns["decoding"] = ["id", "channel_id", "time", "time_sample"]

        # Position
        sqlstr = """CREATE TABLE IF NOT EXISTS position (
                        id INTEGER PRIMARY KEY,
                        time FLOAT,
                        time_sample INTEGER,
                        time_receiver TEXT,
                        x FLOAT,
                        y FLOAT,
                        z FLOAT,
                        clock FLOAT
                        );
                    """
        self.cursor.execute(sqlstr)
        self.columns["position"] = ["id", "time", "time_sample", "time_receiver", "x", "y", "z", "clock"]

        # Measurement
        sqlstr = """CREATE TABLE IF NOT EXISTS measurement (
                        id INTEGER PRIMARY KEY,
                        channel_id INTEGER,
                        time FLOAT,
                        time_sample FLOAT,
                        position_id INTEGER,
                        type FLOAT,
                        value FLOAT,
                        raw_value FLOAT,
                        residuals FLOAT,
                        FOREIGN KEY (channel_id) REFERENCES channels(id),
                        FOREIGN KEY (position_id) REFERENCES positions(id)
                        );
                    """
        self.cursor.execute(sqlstr)
        self.columns["measurement"] = ["id", "channel_id", "time", "time_sample", "position_id", "type", "value", "raw_value", "residuals"]

        # Broadcast ephemeris
        self._addTableGPSBRDC()

        self.connection.commit()
        return

    # -------------------------------------------------------------------------

    def addColumn(self, table, columnDict):
        """
        Alter table to one or multiple column. In columnDict, The dict.key will
        be the column name, the associated value will be the type. Key and 
        value should be Strings.

        Args:
            table (String): Name of the table to alter
            columnDict (Dictionnary): Columns to be added ({String : String})
        """

        for key, value in columnDict.items():
            sqlstr = f"ALTER TABLE {table} ADD {key} {value}"
            self.cursor.execute(sqlstr)

        return
    
    # -------------------------------------------------------------------------

    def _addTableGPSBRDC(self):

        sqlstr = """CREATE TABLE IF NOT EXISTS gpsbrdc (
                        id INTEGER PRIMARY KEY,
                        system TEXT,
                        svid INTEGER,
                        datetime TEXT,
                        ura INTEGER,
                        health INTEGER,
                        weeknumber INTEGER,
                        iode INTEGER,
                        iodc INTEGER,
                        toe INTEGER,
                        toc INTEGER,
                        tgd FLOAT,
                        af0 FLOAT,    
                        af1 FLOAT, 
                        af2 FLOAT,
                        ecc FLOAT,
                        sqrtA FLOAT,
                        crs FLOAT,
                        deltan FLOAT,
                        m0 FLOAT,
                        cuc FLOAT,
                        cus FLOAT,
                        cic FLOAT,
                        omega0 FLOAT,
                        cis FLOAT,
                        i0 FLOAT,
                        crc FLOAT,
                        omega FLOAT,
                        omegaDot FLOAT,
                        iDot FLOAT
                        );
                    """
        self.columns["gpsbrdc"] = ["id", "system", "svid", "datetime", "ura", "health", "weeknumber", "iode", "iodc", "toe", "toc", "tgd", \
            "af2", "af1", "af0", "ecc", "sqrtA", "crs", "deltan", "m0", "cuc", "cus", "cic", "omega0", "cis", \
            "i0", "crc", "omega", "omegaDot", "iDot"]
        self.cursor.execute(sqlstr)
        self.connection.commit()

        return

    # -------------------------------------------------------------------------

    def importRinexNav(self, filepath):
        nav = RINEXNav()
        nav.read(filepath)

        # Add entries
        data = {}
        for key, satellite in nav.satelliteDict.items():
            for ephemeris in satellite.ephemeris:
                data = {}
                data["svid"]       = ephemeris.svid
                data["system"]     = ephemeris.system
                data["datetime"]   = ephemeris.time
                data["ura"]        = ephemeris.ura   
                data["iode"]       = ephemeris.iode    
                data["iodc"]       = ephemeris.iodc    
                data["weeknumber"] = ephemeris.weekNumber
                data["toe"]        = ephemeris.toe     
                data["toc"]        = ephemeris.toc     
                data["tgd"]        = ephemeris.tgd     
                data["af2"]        = ephemeris.af2     
                data["af1"]        = ephemeris.af1     
                data["af0"]        = ephemeris.af0     
                data["ecc"]        = ephemeris.ecc      
                data["sqrtA"]      = ephemeris.sqrtA   
                data["crs"]        = ephemeris.crs     
                data["deltan"]     = ephemeris.deltan  
                data["m0"]         = ephemeris.m0      
                data["cuc"]        = ephemeris.cuc     
                data["cus"]        = ephemeris.cus     
                data["cic"]        = ephemeris.cic     
                data["omega0"]     = ephemeris.omega0  
                data["cis"]        = ephemeris.cis     
                data["i0"]         = ephemeris.i0      
                data["crc"]        = ephemeris.crc     
                data["omega"]      = ephemeris.omega   
                data["omegaDot"]   = ephemeris.omegaDot
                data["iDot"]       = ephemeris.iDot
                data["health"]     = ephemeris.health 

                self.addData("gpsbrdc", data)
        
        self.commit()

        # Debug
        # self.fetchBRDC(Time.fromGPSTime(2186, 180000), "GPS", 2)
        
        return
    
    # -------------------------------------------------------------------------

    def fetchBRDC(self, time:Time, system:GNSSSystems, svid:int):

        str = f"SELECT * FROM gpsbrdc WHERE svid={svid} AND system='{system}';"
        fetchedData = self.cursor.execute(str).fetchall()

        idx = 0
        for eph in fetchedData:
            if time <= Time.fromDatetime(datetime.strptime(eph[3], ("%Y-%m-%d %H:%M:%S.%f"))):
                break
            idx += 1
        try:
            data = fetchedData[idx]
        except IndexError:
            error("No broadcast ephemeris found in database.")

        ephemeris            = BRDCEphemeris()
        ephemeris.system     = GNSSSystems[data[1]]
        ephemeris.svid       = data[2]
        ephemeris.time       = Time.fromDatetime(datetime.strptime(data[3], ("%Y-%m-%d %H:%M:%S.%f")))
        ephemeris.ura        = data[4]
        ephemeris.health     = data[5]
        ephemeris.weekNumber = data[6]
        ephemeris.iode       = data[7]
        ephemeris.iodc       = data[8]
        ephemeris.toe        = data[9]
        ephemeris.toc        = data[10]
        ephemeris.tgd        = data[11]
        ephemeris.af0        = data[12]
        ephemeris.af1        = data[13]
        ephemeris.af2        = data[14]
        ephemeris.ecc        = data[15]
        ephemeris.sqrtA      = data[16]
        ephemeris.crs        = data[17]
        ephemeris.deltan     = data[18]
        ephemeris.m0         = data[19]
        ephemeris.cuc        = data[20]
        ephemeris.cus        = data[21]
        ephemeris.cic        = data[22]
        ephemeris.omega0     = data[23]
        ephemeris.cis        = data[24]
        ephemeris.i0         = data[25]
        ephemeris.crc        = data[26]
        ephemeris.omega      = data[27]
        ephemeris.omegaDot   = data[28]
        ephemeris.iDot       = data[29]

        return ephemeris

    # -------------------------------------------------------------------------

    def fetchMeasurements(self, channelID=None, mtype=None):
        if channelID is None:
            str = f"SELECT * FROM measurement;"
        else:
            str = f"SELECT * FROM measurement WHERE channel_id={channelID} AND type='{mtype}';"
        
        fetchedData = self.cursor.execute(str).fetchall()
        dataList = self._unpackData(fetchedData)
        
        return dataList

    # -------------------------------------------------------------------------

    def fetchTracking(self, channelID=None):

        if channelID is None:
            str = f"SELECT * FROM tracking;"
        else:
            str = f"SELECT * FROM tracking WHERE channel_id={channelID};"
        
        fetchedData = self.cursor.execute(str).fetchall()
        dataList = self._unpackData(fetchedData)

        return dataList

    # -------------------------------------------------------------------------

    def fetchAcquisition(self, channelID=None):

        if channelID is None:
            str = f"SELECT * FROM acquisition;"
        else:
            str = f"SELECT * FROM acquisition WHERE channel_id={channelID};"
        
        fetchedData = self.cursor.execute(str).fetchall()
        dataList = self._unpackData(fetchedData)

        return dataList

    # -------------------------------------------------------------------------

    def fetchPositions(self):

        str = f"SELECT * FROM position;"
        fetchedData = self.cursor.execute(str).fetchall()

        positionList = []
        for data in fetchedData:
            position              = GNSSPosition()
            position.id           = data[0]
            if len(data[3]) < 20:
                position.time = Time.fromDatetime(datetime.strptime(data[3], ("%Y-%m-%d %H:%M:%S")))
            else:
                position.time = Time.fromDatetime(datetime.strptime(data[3], ("%Y-%m-%d %H:%M:%S.%f")))
            position.coordinate.x = data[4]
            position.coordinate.y = data[5]
            position.coordinate.z = data[6]
            position.clockError   = data[7]
            positionList.append(position)

        return positionList

    # -------------------------------------------------------------------------

    def fetchTable(self, tableName):

        str = f"SELECT * FROM {tableName};"
        fetchedData = self.cursor.execute(str).fetchall()
        columnNames = list(map(lambda x: x[0], self.cursor.description))

        dataList = []
        for data in fetchedData:
            dataDict = {}
            idx = 0
            for name in columnNames:
                dataDict[name] = data[idx]
                idx += 1
            
            dataList.append(dataDict)

        return dataList

    # -------------------------------------------------------------------------

    def sqlRequest(self, request):

        fetchedData = self.cursor.execute(request).fetchall()

        dataList = self._unpackData(fetchedData)

        return dataList

    # -------------------------------------------------------------------------

    def _unpackData(self, fetchedData):
        columnNames = list(map(lambda x: x[0], self.cursor.description))
        dataList = []
        for data in fetchedData:
            dataDict = {}
            idx = 0
            for name in columnNames:
                if isinstance(data[idx], bytes):
                    element = pickle.loads(data[idx])
                else:
                    element = data[idx]
                dataDict[name] = element
                idx += 1
            
            dataList.append(dataDict)
        return dataList
    
if __name__=="__main__":
    db = DatabaseHandler('./.results/REC1.db', overwrite=False)

    db.fetchMeasurements()