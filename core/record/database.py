import os
import sqlite3
import numpy as np
import pickle

class DatabaseHandler:

    def __init__(self, dbPath,  overwrite=False):

        # Defining the database
        if overwrite:
            if os.path.exists(dbPath):
                os.remove(dbPath)
        self.connection = sqlite3.connect(dbPath)
        self.cursor = self.connection.cursor()

        self.columns = {} # Dictionnary containing the column list for each table

        # Initialise database content
        self._initialise()

        return
    
    # -------------------------------------------------------------------------

    def addData(self, table, data):
        """ 
        """

        columns = ""
        values  = ""
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
                data[key] = val

            # Build SQL string
            columns += f"{key},"
            values  += f":{key},"

        sqlstr = f"INSERT INTO {table} ({columns[:-1]}) VALUES ({values[:-1]});"
        self.cursor.execute(sqlstr, data)

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
                        x FLOAT,
                        y FLOAT,
                        z FLOAT,
                        clock FLOAT,
                        sigma_x FLOAT,
                        sigma_y FLOAT,
                        sigma_z FLOAT,
                        sigma_clock FLOAT,
                        gdop FLOAT,
                        pdop FLOAT,
                        hdop FLOAT
                        );
                    """
        self.cursor.execute(sqlstr)
        self.columns["position"] = []

        # Measurement
        sqlstr = """CREATE TABLE IF NOT EXISTS measurement (
                        id INTEGER PRIMARY KEY,
                        channel_id INTEGER,
                        time FLOAT,
                        position_id INTEGER,
                        type INTEGER,
                        value FLOAT,
                        residual FLOAT,
                        FOREIGN KEY (channel_id) REFERENCES channels(id),
                        FOREIGN KEY (position_id) REFERENCES positions(id)
                        );
                    """
        self.cursor.execute(sqlstr)
        self.columns["measurement"] = []

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

    def commit(self):
        self.connection.commit()
        return
    
    # -------------------------------------------------------------------------
    
if __name__=="__main__":
    db = DatabaseHandler('test.db', overwrite=True)
    db.addColumn("acquisition", {"test" : "Integer"})
    db.commit()