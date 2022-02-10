import configparser
from tqdm import tqdm
from gnsstools.ephemeris import Ephemeris
from gnsstools.gnsssignal import GNSSSignal
from gnsstools.acquisition import Acquisition
from gnsstools.rffile import RFFile
from gnsstools.tracking import Tracking

class GNSS:
    def __init__(self, configfile):
        # Initialise from config file
        config = configparser.ConfigParser()
        config.read(configfile)
    	
        self.configfile = configfile
        self.signalfile = config.get('DEFAULT', 'signalpath')
        
        self.signal_type = config.get('DEFAULT', 'signal_type')

        # Instanciate variables
        self.data_file = RFFile(self.configfile)
        self.signal = GNSSSignal(self.signalfile, self.signal_type)

        return

    def doAcquisition(self, prnlist):
        
        self.acquisition_results = {}
        for prn in tqdm(prnlist, desc="Acquisition progress"):
            acquisition = Acquisition(self.configfile, prn, self.signal)
            acquisition.acquire(self.data_file)
            self.acquisition_results[prn] = acquisition
        return

    def doTracking(self, prnlist, ms):

        self.tracking_results= {}
        for prn in tqdm(prnlist, desc="Tracking progress"):
            tracking = Tracking(self.configfile, self.acquisition_results[prn])
            tracking.track(self.data_file, ms)
            self.tracking_results[prn] = tracking

        return

    def doDecoding(self, prnlist):
        
        self.decoding_results = {}
        for prn in tqdm(prnlist, desc="Decoding progress"):
            ephemeris = Ephemeris(self.tracking_results[prn])
            ephemeris.fromRawNavigationMessage()
            self.decoding_results[prn] = ephemeris
        
        return

