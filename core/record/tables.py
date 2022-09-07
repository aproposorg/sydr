from sqlalchemy import Column, Integer, String, ForeignKey, Table, Float, Text
from sqlalchemy.dialects.sqlite import DATETIME
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session

import time

# Defining the database
Base = declarative_base()

# Channels table
class Channel(Base):
    __tablename__ = "channel"

    # Columns
    id           = Column(Integer, primary_key=True) # The channel's ID, unique and new for each new signal 
    physical_id  = Column(Integer)                   # The "physical" channel's ID 
    system_id    = Column(Integer)                   # Satellite system ID, similar to core.enumerations.GNSSSystems
    signal_id    = Column(Integer)                   # Signal ID, similar to core.signal.GNSSSignalType
    start_time   = Column(Float())           # Starting/creation time
    stop_time    = Column(Float())           # Stoping/destruction time
    start_sample = Column(Integer)           # Current sample of the file when channel created

# Measurements table
class Measurement(Base):
    __tablename__ = "measurement"

    # Columns
    id           = Column(Integer, primary_key=True)          # The measurement's ID, unique 
    time         = Column(Float())
    channel_id   = Column(Integer, ForeignKey("channel.id"))  # FK to channel's ID
    position_id  = Column(Integer, ForeignKey("position.id")) # FK to position's ID
    type_id      = Column(Integer)                            # Measurement type ID, similar to core.enumerations.GNSSMeasurementType
    value        = Column(Float())                            # Measurement value 
    residual     = Column(Float())                            # Measurement residual after LSE

# Measurements table
class Position(Base):
    __tablename__ = "position"

    # Columns
    id           = Column(Integer, primary_key=True)          # The position's ID, unique 
    time         = Column(Float())
    x            = Column(Float())
    y            = Column(Float())
    z            = Column(Float())
    clock        = Column(Float())
    sigma_x      = Column(Float())
    sigma_y      = Column(Float())
    sigma_z      = Column(Float())  
    sigma_clock  = Column(Float())
    gdop         = Column(Float())
    pdop         = Column(Float())
    tdop         = Column(Float())

# Measurements table
class Acquisition(Base):
    __tablename__ = "acquisition"

    # Columns
    id           = Column(Integer, primary_key=True)          # The acquisition's ID, unique
    channel_id   = Column(Integer, ForeignKey("channel.id"))  # FK to channel's ID 
    time         = Column(Float())
    frequency    = Column(Float())
    code         = Column(Float())

class Tracking(Base):
    __tablename__ = "tracking"

    # Columns
    id           = Column(Integer, primary_key=True)          # The tracking's ID, unique
    channel_id   = Column(Integer, ForeignKey("channel.id"))  # FK to channel's ID 
    time         = Column(Float())
    i_prompt     = Column(Float())
    q_prompt     = Column(Float())

    # i_early      = Column(Float())
    # q_early      = Column(Float())
    # i_late       = Column(Float())
    # q_late       = Column(Float())
    # remain_phase = Column(Float())

class Decoding(Base):
    __tablename__ = "decoding"

    # Columns
    id           = Column(Integer, primary_key=True)          # The decoding's ID, unique
    channel_id   = Column(Integer, ForeignKey("channel.id"))  # FK to channel's ID 
    time         = Column(Float())
    bits         = Column(Text())

class Logs(Base):
    __tablename__ = "logs"

    # Columns
    id           = Column(Integer, primary_key=True)          # The log's ID, unique
    time         = Column(Float())
    scenario_time= Column(Integer)
    level        = Column(Integer) # TBD
    type         = Column(Integer) # TBD
    comment      = Column(Text)    # TBD


if __name__=="__main__":
    engine = create_engine("sqlite:///test.db", echo=True, future=True)
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        channel1 = Channel(id=1, physical_id=1, system_id=1, signal_id=1, start_time=time.time(), stop_time=time.time(), start_sample=12)
        decoding1 = Decoding(id=1, channel_id=1, time=time.time(), bits="011010101010101010101101010101")
        decoding2 = Decoding(id=2, channel_id=1, time=time.time(), bits="011010101010101010101")

        session.add(channel1)
        session.add(decoding1)
        session.add(decoding2)
        session.commit()

